import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Calendar, RefreshCw, TrendingUp, Star, Clock, DollarSign,
    AlertCircle, ChevronDown, ChevronRight, Building2, Target,
    Percent, Award, Filter, BarChart3, Download, CheckCircle, Database
} from 'lucide-react'
import clsx from 'clsx'

interface SyncStatus {
    upcoming_events: number
    has_data: boolean
    date_range: string
    message: string
}

interface EarningsEvent {
    symbol: string
    company_name: string
    earnings_date: string
    earnings_time: string
    expected_move_pct: number | null
    iv_rank: number | null
    beat_rate: number | null
    avg_surprise: number | null
    quality_score: number | null
    sector: string
}

interface EarningsData {
    opportunities: EarningsEvent[]
    upcoming: EarningsEvent[]
    total_count: number
}

const TIME_FILTERS = ['All', 'Before Market', 'After Hours', 'During Market']
const QUALITY_FILTERS = ['All', 'High Quality (70+)', 'Medium (50-69)', 'Any Score']

export default function EarningsCalendar() {
    const [timeFilter, setTimeFilter] = useState('All')
    const [qualityFilter, setQualityFilter] = useState('All')
    const [daysAhead, setDaysAhead] = useState(7)
    const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null)
    const queryClient = useQueryClient()

    // Sync status query
    const { data: syncStatus } = useQuery<SyncStatus>({
        queryKey: ['earnings-sync-status'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/earnings/sync/status')
            return data
        },
        staleTime: 60000, // 1 minute
    })

    // Sync mutation
    const syncMutation = useMutation({
        mutationFn: async (includeHistory: boolean) => {
            const { data } = await axiosInstance.post('/earnings/sync', null, {
                params: {
                    days_ahead: 30,
                    include_history: includeHistory
                }
            })
            return data
        },
        onSuccess: () => {
            // Invalidate all earnings queries to refresh data
            queryClient.invalidateQueries({ queryKey: ['earnings'] })
            queryClient.invalidateQueries({ queryKey: ['earnings-sync-status'] })
        }
    })

    const { data, isLoading, refetch, error } = useQuery<EarningsData>({
        queryKey: ['earnings', daysAhead, timeFilter, qualityFilter],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/earnings/upcoming', {
                params: {
                    days: daysAhead,
                    time_filter: timeFilter !== 'All' ? timeFilter : undefined,
                    min_quality: qualityFilter === 'High Quality (70+)' ? 70 :
                                 qualityFilter === 'Medium (50-69)' ? 50 : undefined
                }
            })
            return data
        },
        staleTime: 300000, // 5 minutes
    })

    const getQualityColor = (score: number | null) => {
        if (score === null) return 'text-slate-500'
        if (score >= 70) return 'text-emerald-400'
        if (score >= 50) return 'text-amber-400'
        return 'text-red-400'
    }

    const getQualityBadge = (score: number | null) => {
        if (score === null) return { label: 'N/A', color: 'bg-slate-500/20 text-slate-400' }
        if (score >= 70) return { label: 'Excellent', color: 'bg-emerald-500/20 text-emerald-400' }
        if (score >= 50) return { label: 'Good', color: 'bg-amber-500/20 text-amber-400' }
        return { label: 'Caution', color: 'bg-red-500/20 text-red-400' }
    }

    const formatTime = (time: string) => {
        if (time === 'bmo') return 'Before Market'
        if (time === 'amc') return 'After Hours'
        return 'During Market'
    }

    const opportunities = data?.opportunities || []
    const upcoming = data?.upcoming || []

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <Calendar className="w-5 h-5 text-white" />
                        </div>
                        Earnings Calendar
                    </h1>
                    <p className="page-subtitle">Track upcoming earnings and find IV expansion opportunities</p>
                </div>
                <div className="flex items-center gap-2">
                    {/* Sync Status */}
                    {syncStatus && (
                        <div className={clsx(
                            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm",
                            syncStatus.has_data
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-amber-500/20 text-amber-400"
                        )}>
                            {syncStatus.has_data ? (
                                <>
                                    <CheckCircle className="w-4 h-4" />
                                    <span>{syncStatus.upcoming_events} events</span>
                                </>
                            ) : (
                                <>
                                    <AlertCircle className="w-4 h-4" />
                                    <span>No data</span>
                                </>
                            )}
                        </div>
                    )}
                    {/* Sync Button */}
                    <button
                        onClick={() => syncMutation.mutate(false)}
                        disabled={syncMutation.isPending}
                        className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg font-medium transition-all disabled:opacity-50"
                    >
                        {syncMutation.isPending ? (
                            <>
                                <RefreshCw className="w-4 h-4 animate-spin" />
                                Syncing...
                            </>
                        ) : (
                            <>
                                <Download className="w-4 h-4" />
                                Sync Data
                            </>
                        )}
                    </button>
                    <button
                        onClick={() => refetch()}
                        disabled={isLoading}
                        className="btn-icon"
                    >
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </header>

            {/* Filters */}
            <div className="card p-4">
                <div className="flex flex-wrap items-center gap-4">
                    {/* Days Ahead */}
                    <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Days Ahead:</span>
                        <div className="flex gap-1">
                            {[7, 14, 30].map(days => (
                                <button
                                    key={days}
                                    onClick={() => setDaysAhead(days)}
                                    className={clsx(
                                        "px-3 py-1 rounded-lg text-sm font-medium transition-all",
                                        daysAhead === days
                                            ? "bg-primary text-white"
                                            : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50"
                                    )}
                                >
                                    {days}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Time Filter */}
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <select
                            value={timeFilter}
                            onChange={e => setTimeFilter(e.target.value)}
                            className="bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                            {TIME_FILTERS.map(filter => (
                                <option key={filter} value={filter}>{filter}</option>
                            ))}
                        </select>
                    </div>

                    {/* Quality Filter */}
                    <div className="flex items-center gap-2">
                        <Star className="w-4 h-4 text-slate-400" />
                        <select
                            value={qualityFilter}
                            onChange={e => setQualityFilter(e.target.value)}
                            className="bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                            {QUALITY_FILTERS.map(filter => (
                                <option key={filter} value={filter}>{filter}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {/* High-Quality Opportunities */}
            {opportunities.length > 0 && (
                <section>
                    <div className="flex items-center gap-2 mb-4">
                        <Award className="w-5 h-5 text-amber-400" />
                        <h2 className="text-lg font-semibold">High-Quality Opportunities</h2>
                        <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-medium">
                            {opportunities.length} found
                        </span>
                    </div>
                    <div className="grid gap-3">
                        {opportunities.map(event => (
                            <EarningsCard
                                key={event.symbol}
                                event={event}
                                isExpanded={expandedSymbol === event.symbol}
                                onToggle={() => setExpandedSymbol(
                                    expandedSymbol === event.symbol ? null : event.symbol
                                )}
                                getQualityBadge={getQualityBadge}
                                formatTime={formatTime}
                            />
                        ))}
                    </div>
                </section>
            )}

            {/* Upcoming Earnings Table */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                    <Calendar className="w-5 h-5 text-primary" />
                    <h2 className="text-lg font-semibold">All Upcoming Earnings</h2>
                    <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-medium">
                        Next {daysAhead} days
                    </span>
                </div>

                {isLoading ? (
                    <div className="card p-8 flex items-center justify-center">
                        <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                        <span className="ml-2 text-slate-400">Loading earnings data...</span>
                    </div>
                ) : error ? (
                    <div className="card p-8 flex items-center justify-center text-red-400">
                        <AlertCircle className="w-6 h-6 mr-2" />
                        Failed to load earnings calendar
                    </div>
                ) : upcoming.length === 0 ? (
                    <div className="card p-8 text-center text-slate-400">
                        <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No earnings scheduled in the next {daysAhead} days</p>
                    </div>
                ) : (
                    <div className="card overflow-hidden">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-700/50">
                                    <th className="text-left p-4 text-sm font-medium text-slate-400">Symbol</th>
                                    <th className="text-left p-4 text-sm font-medium text-slate-400">Company</th>
                                    <th className="text-left p-4 text-sm font-medium text-slate-400">Date</th>
                                    <th className="text-left p-4 text-sm font-medium text-slate-400">Time</th>
                                    <th className="text-right p-4 text-sm font-medium text-slate-400">Expected Move</th>
                                    <th className="text-right p-4 text-sm font-medium text-slate-400">IV Rank</th>
                                    <th className="text-right p-4 text-sm font-medium text-slate-400">Beat Rate</th>
                                    <th className="text-center p-4 text-sm font-medium text-slate-400">Quality</th>
                                </tr>
                            </thead>
                            <tbody>
                                {upcoming.map((event, idx) => {
                                    const badge = getQualityBadge(event.quality_score)
                                    return (
                                        <tr
                                            key={event.symbol}
                                            className={clsx(
                                                "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                                idx % 2 === 0 && "bg-slate-800/20"
                                            )}
                                        >
                                            <td className="p-4">
                                                <span className="font-mono font-semibold text-primary">
                                                    {event.symbol}
                                                </span>
                                            </td>
                                            <td className="p-4 text-sm text-slate-300">{event.company_name || '-'}</td>
                                            <td className="p-4 text-sm text-slate-300">
                                                {new Date(event.earnings_date).toLocaleDateString()}
                                            </td>
                                            <td className="p-4 text-sm">
                                                <span className={clsx(
                                                    "px-2 py-0.5 rounded text-xs",
                                                    event.earnings_time === 'bmo' && "bg-blue-500/20 text-blue-400",
                                                    event.earnings_time === 'amc' && "bg-purple-500/20 text-purple-400",
                                                    !['bmo', 'amc'].includes(event.earnings_time) && "bg-slate-500/20 text-slate-400"
                                                )}>
                                                    {formatTime(event.earnings_time)}
                                                </span>
                                            </td>
                                            <td className="p-4 text-right font-mono text-sm">
                                                {event.expected_move_pct !== null
                                                    ? `±${event.expected_move_pct.toFixed(1)}%`
                                                    : '-'}
                                            </td>
                                            <td className="p-4 text-right font-mono text-sm">
                                                {event.iv_rank !== null
                                                    ? `${event.iv_rank.toFixed(0)}%`
                                                    : '-'}
                                            </td>
                                            <td className="p-4 text-right font-mono text-sm">
                                                {event.beat_rate !== null
                                                    ? `${event.beat_rate.toFixed(0)}%`
                                                    : '-'}
                                            </td>
                                            <td className="p-4 text-center">
                                                <span className={clsx("px-2 py-1 rounded-full text-xs font-medium", badge.color)}>
                                                    {event.quality_score !== null ? event.quality_score.toFixed(0) : '-'}
                                                </span>
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </section>

            {/* Info Section */}
            <section className="card p-4">
                <details className="group">
                    <summary className="flex items-center gap-2 cursor-pointer list-none">
                        <ChevronRight className="w-4 h-4 text-slate-400 group-open:rotate-90 transition-transform" />
                        <span className="text-sm font-medium text-slate-300">What is Quality Score?</span>
                    </summary>
                    <div className="mt-3 pl-6 text-sm text-slate-400 space-y-2">
                        <p><strong className="text-white">Quality Score (0-100)</strong> measures earnings predictability and opportunity:</p>
                        <ul className="list-disc list-inside space-y-1">
                            <li><span className="text-emerald-400">70-100</span>: Excellent - Consistent beaters with strong patterns</li>
                            <li><span className="text-amber-400">50-69</span>: Good - Reliable earnings with moderate consistency</li>
                            <li><span className="text-red-400">Below 50</span>: Caution - Unpredictable or inconsistent results</li>
                        </ul>
                        <p className="mt-2"><strong className="text-white">Components:</strong></p>
                        <ul className="list-disc list-inside space-y-1">
                            <li>Beat Rate (40%): Percentage of earnings beats</li>
                            <li>Avg Surprise (30%): Average EPS surprise magnitude</li>
                            <li>Consistency (30%): Standard deviation of surprises (lower is better)</li>
                        </ul>
                        <p className="mt-2 text-xs">
                            <strong>Expected Move</strong> is calculated from options pricing (ATM straddle × 0.85)
                        </p>
                    </div>
                </details>
            </section>
        </div>
    )
}

// Earnings Card Component for opportunities
function EarningsCard({
    event,
    isExpanded,
    onToggle,
    getQualityBadge,
    formatTime
}: {
    event: EarningsEvent
    isExpanded: boolean
    onToggle: () => void
    getQualityBadge: (score: number | null) => { label: string; color: string }
    formatTime: (time: string) => string
}) {
    const badge = getQualityBadge(event.quality_score)

    return (
        <div className="card overflow-hidden">
            <button
                onClick={onToggle}
                className="w-full p-4 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
            >
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center">
                        <span className="font-mono font-bold text-primary">{event.symbol.slice(0, 2)}</span>
                    </div>
                    <div className="text-left">
                        <div className="flex items-center gap-2">
                            <span className="font-semibold">{event.symbol}</span>
                            <span className={clsx("px-2 py-0.5 rounded-full text-xs font-medium", badge.color)}>
                                {event.quality_score?.toFixed(0) || '-'} - {badge.label}
                            </span>
                        </div>
                        <p className="text-sm text-slate-400">{event.company_name || 'Unknown Company'}</p>
                    </div>
                </div>

                <div className="flex items-center gap-6">
                    <div className="text-right">
                        <p className="text-sm text-slate-400">Date</p>
                        <p className="font-medium">{new Date(event.earnings_date).toLocaleDateString()}</p>
                    </div>
                    <div className="text-right">
                        <p className="text-sm text-slate-400">Expected Move</p>
                        <p className="font-mono font-medium text-amber-400">
                            {event.expected_move_pct !== null ? `±${event.expected_move_pct.toFixed(1)}%` : 'N/A'}
                        </p>
                    </div>
                    <ChevronDown className={clsx(
                        "w-5 h-5 text-slate-400 transition-transform",
                        isExpanded && "rotate-180"
                    )} />
                </div>
            </button>

            {isExpanded && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700/50 grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Earnings Time</p>
                        <p className="font-medium">{formatTime(event.earnings_time)}</p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">IV Rank</p>
                        <p className="font-mono font-medium">
                            {event.iv_rank !== null ? `${event.iv_rank.toFixed(0)}%` : 'N/A'}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Beat Rate (8Q)</p>
                        <p className="font-mono font-medium text-emerald-400">
                            {event.beat_rate !== null ? `${event.beat_rate.toFixed(0)}%` : 'N/A'}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Avg Surprise</p>
                        <p className="font-mono font-medium">
                            {event.avg_surprise !== null ? `${event.avg_surprise > 0 ? '+' : ''}${event.avg_surprise.toFixed(1)}%` : 'N/A'}
                        </p>
                    </div>
                    {event.sector && (
                        <div className="col-span-2">
                            <p className="text-xs text-slate-400 mb-1">Sector</p>
                            <p className="flex items-center gap-1">
                                <Building2 className="w-3 h-3" />
                                {event.sector}
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
