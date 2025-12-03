import React, { memo, useMemo, useCallback } from 'react'
import clsx from 'clsx'

// ============================================
// MEMOIZED STAT CARD - Prevents unnecessary re-renders
// ============================================

interface StatCardProps {
    label: string
    value: string | number
    icon: React.ReactNode
    trend?: number
    trendLabel?: string
    color?: 'emerald' | 'amber' | 'red' | 'blue' | 'purple' | 'cyan'
    onClick?: () => void
    className?: string
}

export const StatCard = memo(function StatCard({
    label,
    value,
    icon,
    trend,
    trendLabel,
    color = 'emerald',
    onClick,
    className
}: StatCardProps) {
    const colorClasses = useMemo(() => ({
        emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
        amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
        red: 'text-red-400 bg-red-500/10 border-red-500/20',
        blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
        purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
        cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
    }), [])

    return (
        <div
            className={clsx(
                "glass-card p-4 transition-all",
                onClick && "cursor-pointer hover:border-slate-600",
                className
            )}
            onClick={onClick}
        >
            <div className="flex items-center gap-4">
                <div className={clsx(
                    "w-12 h-12 rounded-xl flex items-center justify-center border",
                    colorClasses[color]
                )}>
                    {icon}
                </div>
                <div className="flex-1">
                    <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
                    <p className="text-xl font-bold text-white">{value}</p>
                    {trend !== undefined && (
                        <p className={clsx(
                            "text-xs mt-1",
                            trend >= 0 ? "text-emerald-400" : "text-red-400"
                        )}>
                            {trend >= 0 ? '+' : ''}{trend}% {trendLabel}
                        </p>
                    )}
                </div>
            </div>
        </div>
    )
})

// ============================================
// MEMOIZED DATA TABLE - High-performance table
// ============================================

interface Column<T> {
    key: keyof T | string
    header: string
    render?: (row: T) => React.ReactNode
    className?: string
    align?: 'left' | 'center' | 'right'
}

interface DataTableProps<T> {
    data: T[]
    columns: Column<T>[]
    keyExtractor: (row: T, index: number) => string | number
    onRowClick?: (row: T) => void
    isLoading?: boolean
    emptyMessage?: string
    className?: string
}

function DataTableComponent<T>({
    data,
    columns,
    keyExtractor,
    onRowClick,
    isLoading,
    emptyMessage = 'No data available',
    className
}: DataTableProps<T>) {
    const alignClasses = useMemo(() => ({
        left: 'text-left',
        center: 'text-center',
        right: 'text-right'
    }), [])

    const renderCell = useCallback((row: T, column: Column<T>) => {
        if (column.render) return column.render(row)
        const key = column.key as keyof T
        return String(row[key] ?? '-')
    }, [])

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-8">
                <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            </div>
        )
    }

    if (data.length === 0) {
        return (
            <div className="text-center py-8 text-slate-400">
                {emptyMessage}
            </div>
        )
    }

    return (
        <div className={clsx("overflow-x-auto", className)}>
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-slate-700/50">
                        {columns.map(column => (
                            <th
                                key={String(column.key)}
                                className={clsx(
                                    "p-3 text-slate-400 font-medium",
                                    alignClasses[column.align || 'left'],
                                    column.className
                                )}
                            >
                                {column.header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row, idx) => (
                        <tr
                            key={keyExtractor(row, idx)}
                            className={clsx(
                                idx % 2 === 0 && "bg-slate-800/20",
                                onRowClick && "cursor-pointer hover:bg-slate-700/30"
                            )}
                            onClick={() => onRowClick?.(row)}
                        >
                            {columns.map(column => (
                                <td
                                    key={String(column.key)}
                                    className={clsx(
                                        "p-3",
                                        alignClasses[column.align || 'left'],
                                        column.className
                                    )}
                                >
                                    {renderCell(row, column)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

export const DataTable = memo(DataTableComponent) as typeof DataTableComponent

// ============================================
// MEMOIZED BADGE - Status indicators
// ============================================

interface BadgeProps {
    variant: 'success' | 'warning' | 'danger' | 'info' | 'neutral'
    children: React.ReactNode
    size?: 'sm' | 'md' | 'lg'
    className?: string
}

export const Badge = memo(function Badge({
    variant,
    children,
    size = 'md',
    className
}: BadgeProps) {
    const variantClasses = useMemo(() => ({
        success: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
        danger: 'bg-red-500/20 text-red-400 border-red-500/30',
        info: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
        neutral: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    }), [])

    const sizeClasses = useMemo(() => ({
        sm: 'px-1.5 py-0.5 text-xs',
        md: 'px-2 py-1 text-sm',
        lg: 'px-3 py-1.5 text-base',
    }), [])

    return (
        <span className={clsx(
            "inline-flex items-center rounded-full border font-medium",
            variantClasses[variant],
            sizeClasses[size],
            className
        )}>
            {children}
        </span>
    )
})

// ============================================
// MEMOIZED PROGRESS BAR - Animated progress
// ============================================

interface ProgressBarProps {
    value: number
    max?: number
    color?: 'emerald' | 'amber' | 'red' | 'blue' | 'purple' | 'cyan'
    showLabel?: boolean
    size?: 'sm' | 'md' | 'lg'
    className?: string
}

export const ProgressBar = memo(function ProgressBar({
    value,
    max = 100,
    color = 'emerald',
    showLabel = false,
    size = 'md',
    className
}: ProgressBarProps) {
    const percentage = useMemo(() =>
        Math.min(Math.max((value / max) * 100, 0), 100),
        [value, max]
    )

    const colorClasses = useMemo(() => ({
        emerald: 'bg-emerald-500',
        amber: 'bg-amber-500',
        red: 'bg-red-500',
        blue: 'bg-blue-500',
        purple: 'bg-purple-500',
        cyan: 'bg-cyan-500',
    }), [])

    const sizeClasses = useMemo(() => ({
        sm: 'h-1',
        md: 'h-2',
        lg: 'h-3',
    }), [])

    return (
        <div className={clsx("w-full", className)}>
            {showLabel && (
                <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Progress</span>
                    <span className="font-medium">{percentage.toFixed(0)}%</span>
                </div>
            )}
            <div className={clsx("bg-slate-700 rounded-full overflow-hidden", sizeClasses[size])}>
                <div
                    className={clsx(
                        "h-full rounded-full transition-all duration-500 ease-out",
                        colorClasses[color]
                    )}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    )
})

// ============================================
// MEMOIZED METRIC DISPLAY - Formatted numbers
// ============================================

interface MetricDisplayProps {
    value: number
    format?: 'currency' | 'percent' | 'number' | 'compact'
    decimals?: number
    prefix?: string
    suffix?: string
    showSign?: boolean
    className?: string
    colorize?: boolean
}

export const MetricDisplay = memo(function MetricDisplay({
    value,
    format = 'number',
    decimals = 2,
    prefix,
    suffix,
    showSign = false,
    className,
    colorize = false
}: MetricDisplayProps) {
    const formattedValue = useMemo(() => {
        const safeValue = value ?? 0

        switch (format) {
            case 'currency':
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    minimumFractionDigits: decimals,
                    maximumFractionDigits: decimals,
                }).format(safeValue)

            case 'percent':
                return `${safeValue.toFixed(decimals)}%`

            case 'compact':
                return new Intl.NumberFormat('en-US', {
                    notation: 'compact',
                    maximumFractionDigits: 1,
                }).format(safeValue)

            default:
                return safeValue.toFixed(decimals)
        }
    }, [value, format, decimals])

    const sign = useMemo(() => {
        if (!showSign) return ''
        return (value ?? 0) >= 0 ? '+' : ''
    }, [value, showSign])

    const colorClass = useMemo(() => {
        if (!colorize) return ''
        return (value ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
    }, [value, colorize])

    return (
        <span className={clsx("font-mono", colorClass, className)}>
            {prefix}{sign}{formattedValue}{suffix}
        </span>
    )
})

// ============================================
// MEMOIZED LOADING BUTTON - With loading state
// ============================================

interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    isLoading?: boolean
    loadingText?: string
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
    size?: 'sm' | 'md' | 'lg'
    leftIcon?: React.ReactNode
    rightIcon?: React.ReactNode
}

export const LoadingButton = memo(function LoadingButton({
    children,
    isLoading,
    loadingText,
    variant = 'primary',
    size = 'md',
    leftIcon,
    rightIcon,
    disabled,
    className,
    ...props
}: LoadingButtonProps) {
    const variantClasses = useMemo(() => ({
        primary: 'bg-primary hover:bg-primary/90 text-white',
        secondary: 'bg-slate-700 hover:bg-slate-600 text-white',
        danger: 'bg-red-500 hover:bg-red-600 text-white',
        ghost: 'bg-transparent hover:bg-slate-700/50 text-slate-300',
    }), [])

    const sizeClasses = useMemo(() => ({
        sm: 'px-3 py-1.5 text-sm',
        md: 'px-4 py-2 text-base',
        lg: 'px-6 py-3 text-lg',
    }), [])

    return (
        <button
            disabled={disabled || isLoading}
            className={clsx(
                "inline-flex items-center justify-center gap-2 rounded-lg font-medium",
                "transition-all duration-200",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                variantClasses[variant],
                sizeClasses[size],
                className
            )}
            {...props}
        >
            {isLoading ? (
                <>
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    {loadingText || children}
                </>
            ) : (
                <>
                    {leftIcon}
                    {children}
                    {rightIcon}
                </>
            )}
        </button>
    )
})

// ============================================
// MEMOIZED EMPTY STATE - Placeholder component
// ============================================

interface EmptyStateProps {
    icon?: React.ReactNode
    title: string
    description?: string
    action?: React.ReactNode
    className?: string
}

export const EmptyState = memo(function EmptyState({
    icon,
    title,
    description,
    action,
    className
}: EmptyStateProps) {
    return (
        <div className={clsx("text-center py-12", className)}>
            {icon && (
                <div className="w-16 h-16 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-6 border border-slate-700/50">
                    <div className="text-slate-500">{icon}</div>
                </div>
            )}
            <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
            {description && (
                <p className="text-slate-400 max-w-md mx-auto mb-6">{description}</p>
            )}
            {action}
        </div>
    )
})

// ============================================
// VIRTUALIZED LIST - For large datasets
// ============================================

interface VirtualizedListProps<T> {
    items: T[]
    itemHeight: number
    renderItem: (item: T, index: number) => React.ReactNode
    containerHeight: number
    overscan?: number
    className?: string
}

export function VirtualizedList<T>({
    items,
    itemHeight,
    renderItem,
    containerHeight,
    overscan = 3,
    className
}: VirtualizedListProps<T>) {
    const [scrollTop, setScrollTop] = React.useState(0)

    const startIndex = useMemo(() =>
        Math.max(0, Math.floor(scrollTop / itemHeight) - overscan),
        [scrollTop, itemHeight, overscan]
    )

    const endIndex = useMemo(() =>
        Math.min(
            items.length,
            Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
        ),
        [scrollTop, containerHeight, itemHeight, items.length, overscan]
    )

    const visibleItems = useMemo(() =>
        items.slice(startIndex, endIndex),
        [items, startIndex, endIndex]
    )

    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        setScrollTop(e.currentTarget.scrollTop)
    }, [])

    const totalHeight = items.length * itemHeight
    const offsetY = startIndex * itemHeight

    return (
        <div
            className={clsx("overflow-auto", className)}
            style={{ height: containerHeight }}
            onScroll={handleScroll}
        >
            <div style={{ height: totalHeight, position: 'relative' }}>
                <div style={{ transform: `translateY(${offsetY}px)` }}>
                    {visibleItems.map((item, idx) => (
                        <div key={startIndex + idx} style={{ height: itemHeight }}>
                            {renderItem(item, startIndex + idx)}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
