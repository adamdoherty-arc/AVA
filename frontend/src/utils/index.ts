// ============================================
// CENTRALIZED UTILITIES EXPORT
// Import all utilities from this single file
// ============================================

// Performance Monitoring
export {
    performanceCollector,
    measureTime,
    timed,
    startMark,
    onRenderCallback,
    getProfilerData,
    clearProfilerData,
    trackBundleLoad,
    getBundleInfo,
    getMemoryUsage,
    logMemory,
    getNetworkTimings,
    getSlowRequests,
    frameRateMonitor,
    generatePerformanceReport,
    sendPerformanceReport,
} from './performance'

// Safe Formatters
export {
    // Currency
    formatCurrency,
    formatCompactCurrency,

    // Percentages
    formatPercent,
    formatDecimalAsPercent,

    // Numbers
    formatNumber,
    formatNumberWithCommas,
    formatCompactNumber,
    formatSignedNumber,

    // Dates & Times
    formatDate,
    formatDateTime,
    formatRelativeTime,
    formatTime,

    // Options/Trading
    formatDelta,
    formatTheta,
    formatIV,
    formatStrike,
    formatDTE,
    formatPOP,

    // Sports Betting
    formatOdds,
    decimalToAmerican,
    formatImpliedProbability,
    formatEV,

    // Utilities
    truncate,
    formatFileSize,
    formatDuration,
    safeNumber,
} from './formatters'
