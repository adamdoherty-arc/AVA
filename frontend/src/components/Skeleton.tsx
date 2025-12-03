import { ReactNode } from 'react'
import clsx from 'clsx'

interface SkeletonProps {
    className?: string
    variant?: 'text' | 'circular' | 'rectangular' | 'rounded'
    width?: string | number
    height?: string | number
    animation?: 'pulse' | 'wave' | 'none'
}

/**
 * Modern skeleton loader component with multiple variants
 */
export function Skeleton({
    className,
    variant = 'text',
    width,
    height,
    animation = 'pulse'
}: SkeletonProps) {
    const baseClasses = 'bg-slate-700/50'

    const variantClasses = {
        text: 'rounded h-4',
        circular: 'rounded-full',
        rectangular: 'rounded-none',
        rounded: 'rounded-xl'
    }

    const animationClasses = {
        pulse: 'animate-pulse',
        wave: 'skeleton-wave',
        none: ''
    }

    const style: React.CSSProperties = {
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height
    }

    return (
        <div
            className={clsx(baseClasses, variantClasses[variant], animationClasses[animation], className)}
            style={style}
        />
    )
}

/**
 * Card skeleton for loading states
 */
export function CardSkeleton({ className }: { className?: string }) {
    return (
        <div className={clsx('glass-card p-5 space-y-4', className)}>
            <div className="flex items-center gap-4">
                <Skeleton variant="circular" width={48} height={48} />
                <div className="flex-1 space-y-2">
                    <Skeleton width="60%" height={16} />
                    <Skeleton width="40%" height={12} />
                </div>
            </div>
            <Skeleton variant="rounded" height={100} />
            <div className="flex gap-2">
                <Skeleton variant="rounded" width={80} height={32} />
                <Skeleton variant="rounded" width={80} height={32} />
            </div>
        </div>
    )
}

/**
 * Table row skeleton
 */
export function TableRowSkeleton({ columns = 5 }: { columns?: number }) {
    return (
        <tr className="border-b border-slate-700/30">
            {Array.from({ length: columns }).map((_, i) => (
                <td key={i} className="p-4">
                    <Skeleton width={i === 0 ? '80%' : '60%'} />
                </td>
            ))}
        </tr>
    )
}

/**
 * Table skeleton with multiple rows
 */
export function TableSkeleton({ rows = 5, columns = 5 }: { rows?: number; columns?: number }) {
    return (
        <div className="glass-card overflow-hidden">
            <table className="w-full">
                <thead>
                    <tr className="border-b border-slate-700/50">
                        {Array.from({ length: columns }).map((_, i) => (
                            <th key={i} className="p-4 text-left">
                                <Skeleton width="70%" height={12} />
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {Array.from({ length: rows }).map((_, i) => (
                        <TableRowSkeleton key={i} columns={columns} />
                    ))}
                </tbody>
            </table>
        </div>
    )
}

/**
 * Stats card skeleton
 */
export function StatsSkeleton() {
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="glass-card p-4">
                    <Skeleton width="40%" height={12} className="mb-2" />
                    <Skeleton width="60%" height={28} />
                </div>
            ))}
        </div>
    )
}

/**
 * Chart skeleton
 */
export function ChartSkeleton({ height = 300 }: { height?: number }) {
    return (
        <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
                <Skeleton width={150} height={20} />
                <div className="flex gap-2">
                    <Skeleton variant="rounded" width={60} height={28} />
                    <Skeleton variant="rounded" width={60} height={28} />
                </div>
            </div>
            <Skeleton variant="rounded" height={height} />
        </div>
    )
}

/**
 * Options chain skeleton
 */
export function OptionsChainSkeleton() {
    return (
        <div className="glass-card overflow-hidden">
            <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                <Skeleton width={180} height={24} />
                <Skeleton variant="rounded" width={150} height={36} />
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr>
                            <th colSpan={6} className="p-3 bg-emerald-500/10">
                                <Skeleton width="100%" height={16} />
                            </th>
                            <th className="p-3 bg-slate-800">
                                <Skeleton width="100%" height={16} />
                            </th>
                            <th colSpan={6} className="p-3 bg-red-500/10">
                                <Skeleton width="100%" height={16} />
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {Array.from({ length: 10 }).map((_, i) => (
                            <tr key={i}>
                                {Array.from({ length: 13 }).map((_, j) => (
                                    <td key={j} className="p-3">
                                        <Skeleton width="100%" height={14} />
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

/**
 * Stock analysis skeleton
 */
export function StockAnalysisSkeleton() {
    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="glass-card p-5">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Skeleton variant="circular" width={56} height={56} />
                        <div>
                            <Skeleton width={80} height={28} className="mb-2" />
                            <Skeleton width={120} height={16} />
                        </div>
                    </div>
                    <div className="text-right">
                        <Skeleton width={100} height={28} className="mb-2" />
                        <Skeleton width={60} height={16} />
                    </div>
                </div>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="glass-card p-4">
                        <Skeleton width="50%" height={12} className="mb-2" />
                        <Skeleton width="70%" height={24} />
                    </div>
                ))}
            </div>

            {/* Chart */}
            <ChartSkeleton height={350} />
        </div>
    )
}

/**
 * List item skeleton
 */
export function ListItemSkeleton() {
    return (
        <div className="flex items-center gap-4 p-4 border-b border-slate-700/30">
            <Skeleton variant="circular" width={40} height={40} />
            <div className="flex-1">
                <Skeleton width="60%" height={16} className="mb-2" />
                <Skeleton width="40%" height={12} />
            </div>
            <Skeleton variant="rounded" width={60} height={24} />
        </div>
    )
}

/**
 * Dashboard skeleton combining multiple elements
 */
export function DashboardSkeleton() {
    return (
        <div className="space-y-6">
            <StatsSkeleton />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ChartSkeleton />
                <div className="glass-card p-5">
                    <Skeleton width={150} height={20} className="mb-4" />
                    {Array.from({ length: 5 }).map((_, i) => (
                        <ListItemSkeleton key={i} />
                    ))}
                </div>
            </div>
        </div>
    )
}

/**
 * AI response skeleton with typing animation
 */
export function AIResponseSkeleton() {
    return (
        <div className="glass-card p-5">
            <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <span className="text-white text-xs font-bold">AI</span>
                </div>
                <Skeleton width={120} height={16} />
            </div>
            <div className="space-y-2">
                <Skeleton width="100%" height={16} />
                <Skeleton width="95%" height={16} />
                <Skeleton width="85%" height={16} />
                <Skeleton width="90%" height={16} />
                <Skeleton width="70%" height={16} />
            </div>
            <div className="flex items-center gap-1 mt-4">
                <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
        </div>
    )
}

export default Skeleton
