// Clopper-Pearson intervals for a binomial proportion (Phase 2 SPEC §2.4), shared by every back end.
//
// lower = B^-1(alpha/2; k, n - k + 1) for k > 0 (else 0), upper = B^-1(1 - alpha/2; k + 1, n - k) for k < n (else 1),
// where B^-1 inverts the regularised incomplete beta function I_x(a, b). Validated against SciPy beta.ppf by
// experiments/phase2-gpu/check_intervals.py.
#pragma once

#include <cstdint>

namespace sf {

// I_x(a, b), a, b > 0, 0 <= x <= 1.
double ibeta(double a, double b, double x);

// The x with I_x(a, b) = p, 0 < p < 1.
double ibeta_inv(double a, double b, double p);

struct Interval {
  double lower, upper;
  double half_width() const { return (upper - lower) / 2; }
};

Interval clopper_pearson(int64_t k, int64_t n, double alpha);

}  // namespace sf
