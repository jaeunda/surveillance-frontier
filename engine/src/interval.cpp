#include "sf/interval.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sf {

namespace {
// Continued fraction for I_x(a, b) (modified Lentz), valid for x < (a + 1) / (a + b + 2).
double beta_cf(double a, double b, double x) {
  constexpr double kTiny = 1e-300, kEps = std::numeric_limits<double>::epsilon();
  const double qab = a + b, qap = a + 1, qam = a - 1;
  double c = 1, d = 1 - qab * x / qap;
  if (std::fabs(d) < kTiny) d = kTiny;
  d = 1 / d;
  double h = d;
  for (int m = 1; m < 100000; ++m) {
    const int m2 = 2 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) <= kEps) return h;
  }
  throw std::runtime_error("ibeta: continued fraction did not converge");
}

// log of x^a (1 - x)^b / (a B(a, b)), the prefactor of the continued fraction.
double log_front(double a, double b, double x) {
  return a * std::log(x) + b * std::log1p(-x) - (std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b)) - std::log(a);
}
}  // namespace

double ibeta(double a, double b, double x) {
  if (!(a > 0 && b > 0) || !(x >= 0 && x <= 1)) throw std::invalid_argument("ibeta: argument out of range");
  if (x == 0) return 0;
  if (x == 1) return 1;
  if (x < (a + 1) / (a + b + 2)) return std::exp(log_front(a, b, x)) * beta_cf(a, b, x);
  return 1 - std::exp(log_front(b, a, 1 - x)) * beta_cf(b, a, 1 - x);
}

double ibeta_inv(double a, double b, double p) {
  if (!(p > 0 && p < 1)) throw std::invalid_argument("ibeta_inv: p out of (0, 1)");
  // I_x is increasing in x. Newton steps on I_x - p, safeguarded by a bracket [lo, hi] that always holds the root:
  // a step that leaves the bracket is replaced by bisection. Stops when the bracket or the step reaches a few ulp.
  const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  double lo = 0, hi = 1, x = a / (a + b);
  for (int it = 0; it < 400; ++it) {
    const double f = ibeta(a, b, x) - p;
    if (f == 0) return x;
    (f < 0 ? lo : hi) = x;
    const double pdf = std::exp((a - 1) * std::log(x) + (b - 1) * std::log1p(-x) - lbeta);
    double next = x - f / pdf;
    if (!(next > lo && next < hi) || !std::isfinite(next)) next = lo + (hi - lo) / 2;
    const double tol = 4 * std::numeric_limits<double>::epsilon() * std::max(x, 1e-300);
    if (std::fabs(next - x) <= tol || hi - lo <= tol) return next;
    x = next;
  }
  throw std::runtime_error("ibeta_inv did not converge");
}

Interval clopper_pearson(int64_t k, int64_t n, double alpha) {
  if (n <= 0 || k < 0 || k > n || !(alpha > 0 && alpha < 1)) throw std::invalid_argument("clopper_pearson");
  const double kd = static_cast<double>(k), nd = static_cast<double>(n);
  Interval r{0.0, 1.0};
  if (k > 0) r.lower = ibeta_inv(kd, nd - kd + 1, alpha / 2);
  if (k < n) r.upper = ibeta_inv(kd + 1, nd - kd, 1 - alpha / 2);
  return r;
}

}  // namespace sf
