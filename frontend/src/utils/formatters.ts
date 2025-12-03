/**
 * Safe Formatting Utilities
 * Robust formatters that handle null/undefined values gracefully
 */

// ============================================
// CURRENCY FORMATTERS
// ============================================

/**
 * Format a number as USD currency
 */
export function formatCurrency(
    value: number | null | undefined,
    options: {
        decimals?: number
        showSign?: boolean
        compact?: boolean
        fallback?: string
    } = {}
): string {
    const { decimals = 2, showSign = false, compact = false, fallback = '$0.00' } = options

    if (value === null || value === undefined || isNaN(value)) {
        return fallback
    }

    const sign = showSign && value >= 0 ? '+' : ''

    if (compact && Math.abs(value) >= 1000) {
        return sign + new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            notation: 'compact',
            maximumFractionDigits: 1,
        }).format(value)
    }

    return sign + new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    }).format(value)
}

/**
 * Format a number as a short dollar amount (e.g., $1.2M, $500K)
 */
export function formatCompactCurrency(value: number | null | undefined): string {
    return formatCurrency(value, { compact: true })
}

// ============================================
// PERCENTAGE FORMATTERS
// ============================================

/**
 * Format a number as a percentage
 */
export function formatPercent(
    value: number | null | undefined,
    options: {
        decimals?: number
        showSign?: boolean
        fallback?: string
    } = {}
): string {
    const { decimals = 2, showSign = false, fallback = '0%' } = options

    if (value === null || value === undefined || isNaN(value)) {
        return fallback
    }

    const sign = showSign && value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(decimals)}%`
}

/**
 * Format a decimal as a percentage (0.15 -> 15%)
 */
export function formatDecimalAsPercent(
    value: number | null | undefined,
    decimals: number = 1
): string {
    if (value === null || value === undefined || isNaN(value)) {
        return '0%'
    }
    return `${(value * 100).toFixed(decimals)}%`
}

// ============================================
// NUMBER FORMATTERS
// ============================================

/**
 * Format a number with fixed decimals, handling null/undefined
 */
export function formatNumber(
    value: number | null | undefined,
    decimals: number = 2,
    fallback: string = '-'
): string {
    if (value === null || value === undefined || isNaN(value)) {
        return fallback
    }
    return value.toFixed(decimals)
}

/**
 * Format a number with locale string (commas)
 */
export function formatNumberWithCommas(
    value: number | null | undefined,
    decimals: number = 0,
    fallback: string = '0'
): string {
    if (value === null || value === undefined || isNaN(value)) {
        return fallback
    }
    return value.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    })
}

/**
 * Format a number compactly (e.g., 1.5M, 200K)
 */
export function formatCompactNumber(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) {
        return '0'
    }
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 1,
    }).format(value)
}

/**
 * Format a number with sign prefix
 */
export function formatSignedNumber(
    value: number | null | undefined,
    decimals: number = 2
): string {
    if (value === null || value === undefined || isNaN(value)) {
        return '-'
    }
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(decimals)}`
}

// ============================================
// DATE & TIME FORMATTERS
// ============================================

/**
 * Format a date string or Date object
 */
export function formatDate(
    value: string | Date | null | undefined,
    options: Intl.DateTimeFormatOptions = {}
): string {
    if (!value) return '-'

    try {
        const date = typeof value === 'string' ? new Date(value) : value
        if (isNaN(date.getTime())) return '-'

        const defaultOptions: Intl.DateTimeFormatOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            ...options,
        }

        return date.toLocaleDateString('en-US', defaultOptions)
    } catch {
        return '-'
    }
}

/**
 * Format a date with time
 */
export function formatDateTime(
    value: string | Date | null | undefined,
    includeSeconds: boolean = false
): string {
    return formatDate(value, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        ...(includeSeconds && { second: '2-digit' }),
    })
}

/**
 * Format a date as relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(value: string | Date | null | undefined): string {
    if (!value) return '-'

    try {
        const date = typeof value === 'string' ? new Date(value) : value
        if (isNaN(date.getTime())) return '-'

        const now = new Date()
        const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

        if (diffInSeconds < 60) return 'just now'
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`
        if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`

        return formatDate(date)
    } catch {
        return '-'
    }
}

/**
 * Format time only
 */
export function formatTime(value: string | Date | null | undefined): string {
    return formatDate(value, {
        hour: '2-digit',
        minute: '2-digit',
    })
}

// ============================================
// OPTIONS/TRADING SPECIFIC FORMATTERS
// ============================================

/**
 * Format a delta value
 */
export function formatDelta(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    return value.toFixed(2)
}

/**
 * Format a theta value
 */
export function formatTheta(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    return value.toFixed(3)
}

/**
 * Format IV (implied volatility)
 */
export function formatIV(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    // IV is often stored as decimal, convert to percentage
    const ivPercent = value > 1 ? value : value * 100
    return `${ivPercent.toFixed(0)}%`
}

/**
 * Format a strike price
 */
export function formatStrike(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    return `$${value.toFixed(value % 1 === 0 ? 0 : 2)}`
}

/**
 * Format days to expiration
 */
export function formatDTE(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    return `${Math.round(value)}d`
}

/**
 * Format probability of profit
 */
export function formatPOP(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    const popPercent = value > 1 ? value : value * 100
    return `${popPercent.toFixed(0)}%`
}

// ============================================
// SPORTS BETTING FORMATTERS
// ============================================

/**
 * Format American odds
 */
export function formatOdds(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'

    if (value >= 2) {
        return `+${((value - 1) * 100).toFixed(0)}`
    }
    return `-${(100 / (value - 1)).toFixed(0)}`
}

/**
 * Format decimal odds to American
 */
export function decimalToAmerican(value: number | null | undefined): string {
    return formatOdds(value)
}

/**
 * Format implied probability
 */
export function formatImpliedProbability(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    const percent = value > 1 ? value : value * 100
    return `${percent.toFixed(1)}%`
}

/**
 * Format expected value
 */
export function formatEV(value: number | null | undefined): string {
    if (value === null || value === undefined || isNaN(value)) return '-'
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(1)}%`
}

// ============================================
// UTILITY FORMATTERS
// ============================================

/**
 * Truncate a string with ellipsis
 */
export function truncate(value: string | null | undefined, maxLength: number): string {
    if (!value) return ''
    if (value.length <= maxLength) return value
    return `${value.slice(0, maxLength - 3)}...`
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number | null | undefined): string {
    if (bytes === null || bytes === undefined || isNaN(bytes)) return '0 B'

    const units = ['B', 'KB', 'MB', 'GB', 'TB']
    let unitIndex = 0

    while (bytes >= 1024 && unitIndex < units.length - 1) {
        bytes /= 1024
        unitIndex++
    }

    return `${bytes.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

/**
 * Format duration in milliseconds
 */
export function formatDuration(ms: number | null | undefined): string {
    if (ms === null || ms === undefined || isNaN(ms)) return '-'

    if (ms < 1000) return `${ms.toFixed(0)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`

    const hours = Math.floor(ms / 3600000)
    const minutes = Math.floor((ms % 3600000) / 60000)
    return `${hours}h ${minutes}m`
}

/**
 * Safe value accessor with default
 */
export function safeNumber(value: unknown, fallback: number = 0): number {
    if (typeof value === 'number' && !isNaN(value)) return value
    if (typeof value === 'string') {
        const parsed = parseFloat(value)
        if (!isNaN(parsed)) return parsed
    }
    return fallback
}
