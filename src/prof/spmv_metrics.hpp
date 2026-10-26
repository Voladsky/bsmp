#ifndef BSMP_SPMV_METRICS_H
#define BSMP_SPMV_METRICS_H

struct SpmvMetrics {
    double elapsed_time;
    // ... add whatever you want to track
};

inline SpmvMetrics make_metrics(double elapsed_time) {
    SpmvMetrics metrics;
    metrics.elapsed_time = elapsed_time;
    return metrics;
}

#endif  // BSMP_SPMV_METRICS_H
