// Scenarios for stress-testing a detector: the base series with one synthetic test event added.
//
// The test event is the Phase 0 shape: over L bars, prices are scaled by 1 + q * r * shape(t), where r is the median
// relative range of length-L windows in the base series and shape rises linearly over the first 2/3 and returns over
// the last 1/3; volume is scaled by 1 + q. It measures detector sensitivity; it is not a model of any real behaviour.
// The price-only and volume-only variants are controls: they show how much of a detection needs both features.
#pragma once

#include <cstdint>

#include "sf/series.hpp"

namespace sf {

enum class EventKind { Both, Price, Volume };

struct TestEvent {
  int64_t start;
  int32_t length;
  double strength;  // q; q = 0 leaves the series unchanged (null control)
  EventKind kind = EventKind::Both;
};

// Median relative range of all length-L windows (the scale r above).
double median_range(const SeriesView& x, int32_t length);

// dst = base with the event applied; dst must already be a copy of base (only [start, start + L) is rewritten).
void apply_event(const Series& base, const TestEvent& e, double median_range_L, Series& dst);

// Restore [start, start + L) of dst from base, so one buffer can be reused across scenarios.
void restore(const Series& base, const TestEvent& e, Series& dst);

}  // namespace sf
