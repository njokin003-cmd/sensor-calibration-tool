/*
 * ============================================================
 *  SENSOR CALIBRATION TOOL — EXTENDED (Pure C++17)
 *  Features:
 *    1. Linear OLS regression (baseline)
 *    2. Polynomial regression (degree N) via Gaussian elimination
 *    3. Weighted least squares regression
 *    4. Confidence intervals (t-distribution, pure C++)
 *    5. Multi-sensor batch mode (directory scan)
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────
//  DATA STRUCTURES
// ─────────────────────────────────────────────────────────────────

struct CalibrationPoint {
    double raw;
    double reference;
    double weight = 1.0;
};

// Polynomial model: y = c[0] + c[1]*x + c[2]*x^2 + ...
struct PolynomialModel {
    std::vector<double> coeffs;
    double r_squared = 0.0;
    int    degree    = 1;

    double predict(double x) const {
        double result = 0.0, xpow = 1.0;
        for (double c : coeffs) { result += c * xpow; xpow *= x; }
        return result;
    }
};

struct ResidualEntry {
    double raw, reference, predicted, residual, abs_residual;
    double weight = 1.0;
};

struct ConfidenceInterval {
    double slope_lo, slope_hi;
    double intercept_lo, intercept_hi;
    double prediction_se;
    double alpha;
};

// ─────────────────────────────────────────────────────────────────
//  DATA LOADER
// ─────────────────────────────────────────────────────────────────

class CalibrationLoader {
public:
    // CSV columns: raw, reference [, weight]  (comma or tab)
    static std::vector<CalibrationPoint> fromCSV(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath);

        std::vector<CalibrationPoint> points;
        std::string line;
        bool firstLine = true;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), '\t', ',');

            if (firstLine) {
                firstLine = false;
                char c = line[0];
                if (!std::isdigit(c) && c != '-' && c != '+' && c != '.') continue;
            }

            std::istringstream ss(line);
            std::string token;
            CalibrationPoint pt;

            if (!std::getline(ss, token, ',')) throw std::runtime_error("Bad line: " + line);
            pt.raw = std::stod(token);

            if (!std::getline(ss, token, ',')) throw std::runtime_error("Bad line: " + line);
            pt.reference = std::stod(token);

            if (std::getline(ss, token, ','))
                pt.weight = std::stod(token);

            points.push_back(pt);
        }

        if (points.size() < 2)
            throw std::runtime_error("Need at least 2 data points.");
        return points;
    }
};

// ─────────────────────────────────────────────────────────────────
//  GAUSSIAN ELIMINATION  (used by polynomial solver)
// ─────────────────────────────────────────────────────────────────

static std::vector<double> gaussianElimination(
    std::vector<std::vector<double>> A,
    std::vector<double> b)
{
    const int n = static_cast<int>(A.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col+1; row < n; ++row)
            if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);
        if (std::abs(A[col][col]) < 1e-14)
            throw std::runtime_error("Singular matrix — reduce polynomial degree or add more data.");
        for (int row = col+1; row < n; ++row) {
            double f = A[row][col] / A[col][col];
            for (int k = col; k < n; ++k) A[row][k] -= f * A[col][k];
            b[row] -= f * b[col];
        }
    }
    std::vector<double> x(n, 0.0);
    for (int i = n-1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i+1; j < n; ++j) x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────
//  EXTENSION 1 — POLYNOMIAL REGRESSION  (degree N, weighted)
// ─────────────────────────────────────────────────────────────────
//
//  Solves the weighted normal equations: (XᵀWX) c = XᵀWy
//  where X is the Vandermonde matrix and W is the diagonal weight
//  matrix.  With all weights = 1 this reduces to standard OLS.
//  Degree=1 gives the same result as simple linear regression.

class PolynomialRegression {
public:
    static PolynomialModel fit(const std::vector<CalibrationPoint>& data,
                               int degree,
                               bool useWeights = false)
    {
        if (degree < 1) throw std::invalid_argument("Degree must be >= 1.");
        const int n = static_cast<int>(data.size());
        const int m = degree + 1;
        if (n < m) throw std::runtime_error("Need at least degree+1 data points.");

        std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
        std::vector<double>              b(m, 0.0);

        for (const auto& p : data) {
            double w = useWeights ? p.weight : 1.0;
            std::vector<double> xpow(2*degree+1, 1.0);
            for (int k = 1; k <= 2*degree; ++k) xpow[k] = xpow[k-1] * p.raw;
            for (int i = 0; i < m; ++i) {
                b[i] += w * xpow[i] * p.reference;
                for (int j = 0; j < m; ++j) A[i][j] += w * xpow[i+j];
            }
        }

        PolynomialModel model;
        model.degree = degree;
        model.coeffs = gaussianElimination(A, b);

        // R² (weighted if applicable)
        double sw = 0, swy = 0;
        for (const auto& p : data) {
            double w = useWeights ? p.weight : 1.0;
            sw += w; swy += w * p.reference;
        }
        double mean_y = swy / sw;
        double ss_tot = 0, ss_res = 0;
        for (const auto& p : data) {
            double w    = useWeights ? p.weight : 1.0;
            double pred = model.predict(p.raw);
            ss_res += w * std::pow(p.reference - pred, 2);
            ss_tot += w * std::pow(p.reference - mean_y, 2);
        }
        model.r_squared = (ss_tot < 1e-12) ? 1.0 : 1.0 - ss_res / ss_tot;

        return model;
    }

    static std::vector<ResidualEntry> computeResiduals(
        const std::vector<CalibrationPoint>& data,
        const PolynomialModel& model)
    {
        std::vector<ResidualEntry> entries;
        for (const auto& p : data) {
            ResidualEntry e;
            e.raw          = p.raw;
            e.reference    = p.reference;
            e.weight       = p.weight;
            e.predicted    = model.predict(p.raw);
            e.residual     = p.reference - e.predicted;
            e.abs_residual = std::abs(e.residual);
            entries.push_back(e);
        }
        return entries;
    }
};

// ─────────────────────────────────────────────────────────────────
//  EXTENSION 2 — CONFIDENCE INTERVALS  (pure C++, no Boost)
// ─────────────────────────────────────────────────────────────────
//
//  Computes (1-alpha)*100% CIs for slope, intercept, and prediction
//  using a fully self-contained t-distribution implementation:
//    - Normal quantile via Abramowitz & Stegun rational approximation
//    - t-PDF via log-gamma
//    - t-CDF via regularised incomplete beta (Lentz continued fraction)
//    - t quantile via Newton-Raphson on the CDF

class ConfidenceIntervalEstimator {
public:
    static ConfidenceInterval compute(
        const std::vector<CalibrationPoint>& data,
        const PolynomialModel& model,
        double alpha = 0.05)
    {
        const int n  = static_cast<int>(data.size());
        const int df = n - 2;
        if (df < 1) throw std::runtime_error("Need at least 3 points for CI.");

        double slope     = model.coeffs[1];
        double intercept = model.coeffs[0];
        double sum_x = 0, sum_xx = 0;
        for (const auto& p : data) { sum_x += p.raw; sum_xx += p.raw * p.raw; }
        double mean_x = sum_x / n;

        double ss_res = 0;
        for (const auto& p : data) {
            double pred = model.predict(p.raw);
            ss_res += std::pow(p.reference - pred, 2);
        }
        double s   = std::sqrt(ss_res / df);
        double sxx = sum_xx - n * mean_x * mean_x;

        double se_slope     = s / std::sqrt(sxx);
        double se_intercept = s * std::sqrt(sum_xx / (n * sxx));
        double tc = tQuantile(1.0 - alpha / 2.0, df);

        ConfidenceInterval ci;
        ci.alpha         = alpha;
        ci.slope_lo      = slope     - tc * se_slope;
        ci.slope_hi      = slope     + tc * se_slope;
        ci.intercept_lo  = intercept - tc * se_intercept;
        ci.intercept_hi  = intercept + tc * se_intercept;
        ci.prediction_se = s;
        return ci;
    }

private:
    // Standard normal quantile (rational approx, A&S 26.2.17)
    static double normalQuantile(double p) {
        double sign = 1.0;
        if (p < 0.5) { sign = -1.0; p = 1.0 - p; }
        double t = std::sqrt(-2.0 * std::log(1.0 - p));
        const double c[] = {2.515517, 0.802853, 0.010328};
        const double d[] = {1.432788, 0.189269, 0.001308};
        double num = c[0] + c[1]*t + c[2]*t*t;
        double den = 1.0 + d[0]*t + d[1]*t*t + d[2]*t*t*t;
        return sign * (t - num / den);
    }

    // t-distribution PDF (via log-gamma)
    static double tPDF(double t, int df) {
        double d = static_cast<double>(df);
        double log_val = std::lgamma((d+1)/2) - std::lgamma(d/2)
                       - 0.5*std::log(d * M_PI)
                       - ((d+1)/2) * std::log(1 + t*t/d);
        return std::exp(log_val);
    }

    // Regularised incomplete beta I_x(a,b) via Lentz continued fraction
    static double incompleteBeta(double x, double a, double b) {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a+b);
        double front = std::exp(std::log(x)*a + std::log(1-x)*b - lbeta) / a;
        double f=1, c=1, d2=1-(a+b)*x/(a+1);
        if (std::abs(d2)<1e-30) { d2=1e-30; } d2=1/d2; f=d2;
        for (int m=1; m<=200; ++m) {
            double dm=m;
            auto step = [&](double num) {
                d2=1+num*d2; if(std::abs(d2)<1e-30)d2=1e-30; d2=1/d2;
                c =1+num/c;  if(std::abs(c) <1e-30)c =1e-30;
                double delta=c*d2; f*=delta; return delta;
            };
            step(dm*(b-dm)*x/((a+2*dm-1)*(a+2*dm)));
            double delta = step(-(a+dm)*(a+b+dm)*x/((a+2*dm)*(a+2*dm+1)));
            if (std::abs(delta-1)<1e-10) break;
        }
        return front*f;
    }

    // t-distribution CDF
    static double tCDF(double t, int df) {
        double d = static_cast<double>(df);
        double x = d / (d + t*t);
        double ib = incompleteBeta(x, d/2.0, 0.5);
        return (t >= 0) ? 1.0 - 0.5*ib : 0.5*ib;
    }

    // t quantile via Newton-Raphson on tCDF
    static double tQuantile(double p, int df) {
        if (df > 200) return normalQuantile(p);
        double t = normalQuantile(p);
        for (int iter = 0; iter < 50; ++iter) {
            double f  = tCDF(t, df) - p;
            double fp = tPDF(t, df);
            if (std::abs(fp) < 1e-15) break;
            double dt = f / fp;
            t -= dt;
            if (std::abs(dt) < 1e-10) break;
        }
        return t;
    }
};

// ─────────────────────────────────────────────────────────────────
//  REPORT GENERATOR
// ─────────────────────────────────────────────────────────────────

class CalibrationReport {
public:
    static void print(
        const PolynomialModel&            model,
        const std::vector<ResidualEntry>& residuals,
        const std::string&                sensorName = "Sensor",
        std::ostream&                     out        = std::cout,
        const ConfidenceInterval*         ci         = nullptr)
    {
        const int W = 65;
        const std::string sep(W, '=');
        const std::string thin(W, '-');

        out << "\n" << sep << "\n";
        out << "  CALIBRATION REPORT  —  " << sensorName << "\n";
        out << sep << "\n\n";

        // ── Equation ──────────────────────────────────────────
        out << "  CALIBRATION EQUATION  (degree " << model.degree << ")\n" << thin << "\n";
        out << std::fixed << std::setprecision(6);
        out << "  calibrated = ";
        for (int i = model.degree; i >= 0; --i) {
            double c = model.coeffs[i];
            if (i < model.degree) out << (c >= 0 ? " + " : " - ");
            out << std::abs(c);
            if      (i == 1) out << "·x";
            else if (i >  1) out << "·x^" << i;
        }
        out << "\n\n  Coefficients:\n";
        for (int i = 0; i <= model.degree; ++i)
            out << "    c[" << i << "] = " << model.coeffs[i]
                << (i==0 ? "  (intercept)" : i==1 ? "  (slope)" : "") << "\n";
        out << "\n  R² = " << model.r_squared << "\n\n";

        // ── Confidence Intervals ──────────────────────────────
        if (ci) {
            int pct = static_cast<int>((1.0 - ci->alpha) * 100);
            out << "  CONFIDENCE INTERVALS  (" << pct << " %)\n" << thin << "\n";
            out << "  Slope     : [" << ci->slope_lo     << ",  " << ci->slope_hi     << "]\n";
            out << "  Intercept : [" << ci->intercept_lo << ",  " << ci->intercept_hi << "]\n";
            out << "  Prediction SE (at mean x) : "  << ci->prediction_se << "\n\n";
        }

        // ── Error Statistics ──────────────────────────────────
        double mae=0, mse=0, max_err=0;
        for (const auto& e : residuals) {
            mae     += e.abs_residual;
            mse     += e.residual * e.residual;
            max_err  = std::max(max_err, e.abs_residual);
        }
        mae /= residuals.size();
        mse /= residuals.size();

        out << "  RESIDUAL ERROR STATISTICS\n" << thin << "\n";
        out << "  MAE   (Mean Absolute Error)     : " << mae            << "\n";
        out << "  RMSE  (Root Mean Squared Error)  : " << std::sqrt(mse) << "\n";
        out << "  Max   Absolute Residual          : " << max_err        << "\n\n";

        // ── Per-point table ───────────────────────────────────
        bool weighted = std::any_of(residuals.begin(), residuals.end(),
                                    [](const ResidualEntry& e){ return e.weight != 1.0; });
        out << "  PER-POINT RESIDUAL TABLE\n" << thin << "\n";
        out << std::setw(6)  << "  #"
            << std::setw(13) << "Raw"
            << std::setw(13) << "Reference"
            << std::setw(13) << "Predicted"
            << std::setw(12) << "Residual";
        if (weighted) out << std::setw(10) << "Weight";
        out << "\n" << thin << "\n";

        int idx = 1;
        for (const auto& e : residuals) {
            out << std::setw(6)  << ("  "+std::to_string(idx++))
                << std::setw(13) << e.raw
                << std::setw(13) << e.reference
                << std::setw(13) << e.predicted
                << std::setw(12) << e.residual;
            if (weighted) out << std::setw(10) << e.weight;
            out << "\n";
        }
        out << sep << "\n\n";
    }

    static void exportCSV(const std::vector<ResidualEntry>& residuals,
                          const std::string& filepath)
    {
        std::ofstream f(filepath);
        if (!f.is_open()) throw std::runtime_error("Cannot write: " + filepath);
        f << "raw,reference,predicted,residual,abs_residual,weight\n";
        f << std::fixed << std::setprecision(6);
        for (const auto& e : residuals)
            f << e.raw << "," << e.reference << "," << e.predicted
              << "," << e.residual << "," << e.abs_residual
              << "," << e.weight << "\n";
    }
};

// ─────────────────────────────────────────────────────────────────
//  EXTENSION 3 — MULTI-SENSOR BATCH MODE
// ─────────────────────────────────────────────────────────────────
//
//  Scans a directory for *.csv files using std::filesystem (C++17).
//  Fits the chosen model to each file, prints individual reports,
//  exports per-sensor residual CSVs, and prints a summary table.

class BatchProcessor {
public:
    struct SensorResult {
        std::string     name;
        PolynomialModel model;
        double          mae=0, rmse=0, max_err=0;
        bool            ok=true;
        std::string     error;
    };

    static std::vector<SensorResult> processDirectory(
        const std::string& dir,
        int    degree      = 1,
        bool   useWeighted = false,
        bool   doCI        = false,
        double alpha       = 0.05,
        const  std::string& outDir = "")
    {
        if (!fs::exists(dir) || !fs::is_directory(dir))
            throw std::runtime_error("Not a valid directory: " + dir);

        std::vector<SensorResult> results;

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".csv") continue;

            SensorResult res;
            res.name = entry.path().stem().string();
            std::cout << "Processing: " << res.name << ".csv\n";

            try {
                auto data = CalibrationLoader::fromCSV(entry.path().string());
                res.model = PolynomialRegression::fit(data, degree, useWeighted);
                auto residuals = PolynomialRegression::computeResiduals(data, res.model);

                // Stats
                for (const auto& e : residuals) {
                    res.mae     += e.abs_residual;
                    res.rmse    += e.residual * e.residual;
                    res.max_err  = std::max(res.max_err, e.abs_residual);
                }
                res.mae  /= residuals.size();
                res.rmse  = std::sqrt(res.rmse / residuals.size());

                // Per-sensor report
                ConfidenceInterval ci;
                const ConfidenceInterval* pci = nullptr;
                if (doCI && degree == 1) {
                    ci  = ConfidenceIntervalEstimator::compute(data, res.model, alpha);
                    pci = &ci;
                }
                CalibrationReport::print(res.model, residuals, res.name, std::cout, pci);

                // Export residuals CSV
                if (!outDir.empty()) {
                    CalibrationReport::exportCSV(residuals, outDir + "/" + res.name + "_residuals.csv");
                }

            } catch (const std::exception& ex) {
                res.ok    = false;
                res.error = ex.what();
                std::cerr << "  [SKIP] " << res.name << ": " << ex.what() << "\n";
            }
            results.push_back(res);
        }
        return results;
    }

    static void printSummary(const std::vector<SensorResult>& results,
                             std::ostream& out = std::cout)
    {
        const std::string sep(70, '=');
        out << "\n" << sep << "\n";
        out << "  BATCH CALIBRATION SUMMARY\n" << sep << "\n";
        out << std::left
            << std::setw(22) << "  Sensor"
            << std::setw(8)  << "Degree"
            << std::setw(10) << "R²"
            << std::setw(12) << "MAE"
            << std::setw(12) << "RMSE"
            << std::setw(12) << "MaxErr"
            << "Status\n" << std::string(70,'-') << "\n";

        for (const auto& r : results) {
            out << std::left << std::setw(22) << ("  "+r.name);
            if (r.ok) {
                out << std::setw(8)  << r.model.degree
                    << std::fixed << std::setprecision(4)
                    << std::setw(10) << r.model.r_squared
                    << std::setw(12) << r.mae
                    << std::setw(12) << r.rmse
                    << std::setw(12) << r.max_err
                    << "OK\n";
            } else {
                out << std::setw(8)<<"-"<<std::setw(10)<<"-"
                    << std::setw(12)<<"-"<<std::setw(12)<<"-"
                    << std::setw(12)<<"-"
                    << "FAILED: " << r.error << "\n";
            }
        }
        out << sep << "\n\n";
    }
};

// ─────────────────────────────────────────────────────────────────
//  MAIN — CLI DISPATCHER
// ─────────────────────────────────────────────────────────────────

static void printUsage(const char* prog) {
    std::cerr
        << "\nUsage:\n"
        << "  Single file — linear (default):\n"
        << "    " << prog << " <data.csv>\n\n"
        << "  Polynomial fit (degree N):\n"
        << "    " << prog << " <data.csv> --degree <N>\n\n"
        << "  Weighted regression (reads 3rd CSV column as weight):\n"
        << "    " << prog << " <data.csv> --weighted\n\n"
        << "  95% confidence intervals:\n"
        << "    " << prog << " <data.csv> --ci [--alpha 0.05]\n\n"
        << "  Export residuals:\n"
        << "    " << prog << " <data.csv> --export residuals.csv\n\n"
        << "  Batch mode:\n"
        << "    " << prog << " --batch <dir/> [--degree N] [--weighted] [--ci] [--out <outdir/>]\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    std::string inputFile, batchDir, exportFile, outDir;
    int    degree    = 1;
    bool   weighted  = false;
    bool   doBatch   = false;
    bool   doCI      = false;
    double alpha     = 0.05;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--batch"    && i+1<argc) { doBatch=true; batchDir=argv[++i]; }
        else if (arg == "--degree"   && i+1<argc) { degree=std::stoi(argv[++i]); }
        else if (arg == "--export"   && i+1<argc) { exportFile=argv[++i]; }
        else if (arg == "--out"      && i+1<argc) { outDir=argv[++i]; }
        else if (arg == "--alpha"    && i+1<argc) { alpha=std::stod(argv[++i]); }
        else if (arg == "--weighted")              { weighted=true; }
        else if (arg == "--ci")                    { doCI=true; }
        else if (arg[0] != '-')                    { inputFile=arg; }
    }

    try {
        // ── BATCH MODE ────────────────────────────────────────
        if (doBatch) {
            if (!outDir.empty()) fs::create_directories(outDir);
            auto results = BatchProcessor::processDirectory(
                batchDir, degree, weighted, doCI, alpha, outDir);
            BatchProcessor::printSummary(results);
            return 0;
        }

        // ── SINGLE FILE MODE ──────────────────────────────────
        if (inputFile.empty()) { printUsage(argv[0]); return 1; }

        std::cout << "Loading: " << inputFile << "\n";
        auto data = CalibrationLoader::fromCSV(inputFile);
        std::cout << data.size() << " points loaded.  "
                  << "Mode: " << (weighted ? "Weighted " : "")
                  << "Polynomial (degree=" << degree << ")\n";

        auto model     = PolynomialRegression::fit(data, degree, weighted);
        auto residuals = PolynomialRegression::computeResiduals(data, model);

        ConfidenceInterval ci;
        const ConfidenceInterval* pci = nullptr;
        if (doCI && degree == 1) {
            ci  = ConfidenceIntervalEstimator::compute(data, model, alpha);
            pci = &ci;
        }

        CalibrationReport::print(model, residuals, "Sensor", std::cout, pci);

        if (!exportFile.empty()) {
            CalibrationReport::exportCSV(residuals, exportFile);
            std::cout << "Residuals exported -> " << exportFile << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
