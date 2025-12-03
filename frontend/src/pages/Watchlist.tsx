import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Search, RefreshCw, TrendingUp, TrendingDown, BarChart2,
    Eye, Target, DollarSign, Activity, Zap, ChevronRight
} from 'lucide-react'
import clsx from 'clsx'

interface Strategy {
    ticker: string
    current_price?: number
    iv_rank?: number
    recommendation?: string
    strategy_type?: string
    expected_return?: number
    risk_level?: string
    strike?: number
    expiration?: string
    premium?: number
    breakeven?: number
    pop?: number  // Probability of profit
}

export default function Watchlist() {
    const [watchlistName, setWatchlistName] = useState('NVDA')
    const [searchInput, setSearchInput] = useState('')

    // Fetch strategies
    const { data: strategies, isLoading, refetch, isFetching } = useQuery({
        queryKey: ['strategies', watchlistName],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/strategies/${watchlistName}`)
            return data as Strategy[]
        },
        enabled: !!watchlistName
    })

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault()
        if (searchInput.trim()) {
            setWatchlistName(searchInput.toUpperCase().trim())
        }
    }

    const quickSymbols = ['NVDA', 'AAPL', 'TSLA', 'AMD', 'SPY', 'QQQ', 'META', 'MSFT']

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                        <Eye className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Strategy Analyzer</h1>
                        <p className="text-sm text-slate-400">AI-powered options strategy recommendations</p>
                    </div>
                </div>
                <button
                    onClick={() => refetch()}
                    disabled={isFetching}
                    className="btn-primary flex items-center gap-2"
                >
                    <RefreshCw size={16} className={isFetching ? 'animate-spin' : ''} />
                    {isFetching ? 'Analyzing...' : 'Refresh'}
                </button>
            </header>

            {/* Search Card */}
            <div className="glass-card p-6">
                <form onSubmit={handleSearch} className="flex gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                        <input
                            type="text"
                            value={searchInput}
                            onChange={(e) => setSearchInput(e.target.value)}
                            placeholder="Enter symbol to analyze (e.g., NVDA, AAPL)"
                            className="input-field pl-12 text-lg"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={!searchInput.trim() || isLoading}
                        className="btn-primary px-8 flex items-center gap-2"
                    >
                        <Zap size={20} />
                        Analyze
                    </button>
                </form>

                {/* Quick Symbols */}
                <div className="flex flex-wrap items-center gap-2 mt-4">
                    <span className="text-sm text-slate-400">Popular:</span>
                    {quickSymbols.map(symbol => (
                        <button
                            key={symbol}
                            onClick={() => {
                                setSearchInput(symbol)
                                setWatchlistName(symbol)
                            }}
                            className={clsx(
                                "px-3 py-1.5 text-sm rounded-lg font-medium transition-all duration-200",
                                watchlistName === symbol
                                    ? "bg-cyan-500 text-white shadow-lg shadow-cyan-500/20"
                                    : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white border border-slate-700/50"
                            )}
                        >
                            {symbol}
                        </button>
                    ))}
                </div>
            </div>

            {/* Current Symbol Stats */}
            {watchlistName && (
                <div className="grid grid-cols-4 gap-4">
                    <StatCard
                        label="Symbol"
                        value={watchlistName}
                        icon={<Target className="w-5 h-5" />}
                        color="cyan"
                    />
                    <StatCard
                        label="Strategies"
                        value={strategies?.length || 0}
                        icon={<BarChart2 className="w-5 h-5" />}
                        color="emerald"
                    />
                    <StatCard
                        label="Best Return"
                        value={strategies?.length ? `${Math.max(...strategies.map(s => s.expected_return || 0)).toFixed(1)}%` : '-'}
                        icon={<TrendingUp className="w-5 h-5" />}
                        color="green"
                    />
                    <StatCard
                        label="IV Rank"
                        value={strategies?.[0]?.iv_rank ? `${strategies[0].iv_rank}%` : '-'}
                        icon={<Activity className="w-5 h-5" />}
                        color="purple"
                    />
                </div>
            )}

            {/* Loading State */}
            {isLoading && (
                <div className="glass-card p-12 text-center">
                    <div className="relative w-16 h-16 mx-auto mb-6">
                        <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                        <div className="absolute inset-0 rounded-full border-4 border-t-cyan-500 border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">Analyzing {watchlistName}</h3>
                    <p className="text-slate-400">Generating optimal options strategies...</p>
                </div>
            )}

            {/* Strategy Cards */}
            {!isLoading && strategies && strategies.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                    {strategies.map((strategy, idx) => (
                        <StrategyCard key={`${strategy.ticker}-${idx}`} strategy={strategy} />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {!isLoading && (!strategies || strategies.length === 0) && watchlistName && (
                <div className="glass-card p-16 text-center">
                    <div className="w-20 h-20 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-6 border border-slate-700/50">
                        <BarChart2 className="w-10 h-10 text-slate-500" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">No Strategies Found</h3>
                    <p className="text-slate-400 max-w-md mx-auto">
                        No options strategies are currently available for {watchlistName}. Try a different symbol or check back later.
                    </p>
                </div>
            )}
        </div>
    )
}

// Stat Card Component
interface StatCardProps {
    label: string
    value: string | number
    icon: React.ReactNode
    color: 'cyan' | 'emerald' | 'green' | 'purple' | 'amber'
}

function StatCard({ label, value, icon, color }: StatCardProps) {
    const colorClasses = {
        cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
        emerald: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
        green: 'text-green-400 bg-green-500/10 border-green-500/20',
        purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
        amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    }

    return (
        <div className="glass-card p-4 flex items-center gap-4">
            <div className={clsx("w-12 h-12 rounded-xl flex items-center justify-center border", colorClasses[color])}>
                {icon}
            </div>
            <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
                <p className="text-xl font-bold text-white">{value}</p>
            </div>
        </div>
    )
}

// Strategy Card Component
function StrategyCard({ strategy }: { strategy: Strategy }) {
    const getRiskColor = (risk?: string) => {
        switch (risk?.toLowerCase()) {
            case 'low': return 'badge-success'
            case 'medium': return 'badge-warning'
            case 'high': return 'badge-danger'
            default: return 'badge-neutral'
        }
    }

    const getStrategyIcon = (type?: string) => {
        switch (type?.toLowerCase()) {
            case 'covered call': return <TrendingUp className="w-4 h-4" />
            case 'cash secured put': return <TrendingDown className="w-4 h-4" />
            case 'iron condor': return <Activity className="w-4 h-4" />
            default: return <BarChart2 className="w-4 h-4" />
        }
    }

    return (
        <div className="glass-card p-5 hover:border-cyan-500/30 transition-all duration-200 group">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center border border-cyan-500/20">
                        {getStrategyIcon(strategy.strategy_type)}
                    </div>
                    <div>
                        <h3 className="font-bold text-white">{strategy.ticker}</h3>
                        <p className="text-xs text-slate-400">{strategy.strategy_type || 'Options Strategy'}</p>
                    </div>
                </div>
                <span className={getRiskColor(strategy.risk_level)}>
                    {strategy.risk_level || 'N/A'}
                </span>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-slate-800/30 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Strike</p>
                    <p className="font-semibold text-white">${strategy.strike || '-'}</p>
                </div>
                <div className="bg-slate-800/30 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Premium</p>
                    <p className="font-semibold text-emerald-400">${strategy.premium?.toFixed(2) || '-'}</p>
                </div>
                <div className="bg-slate-800/30 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">Breakeven</p>
                    <p className="font-semibold text-white">${strategy.breakeven?.toFixed(2) || '-'}</p>
                </div>
                <div className="bg-slate-800/30 rounded-lg p-3">
                    <p className="text-xs text-slate-400 mb-1">POP</p>
                    <p className="font-semibold text-cyan-400">{strategy.pop ? `${strategy.pop}%` : '-'}</p>
                </div>
            </div>

            {/* Expected Return */}
            {strategy.expected_return !== undefined && (
                <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-lg p-3 border border-emerald-500/20 mb-4">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-slate-300">Expected Return</span>
                        <span className={clsx(
                            "text-lg font-bold",
                            strategy.expected_return >= 0 ? "text-emerald-400" : "text-red-400"
                        )}>
                            {(strategy.expected_return ?? 0) >= 0 ? '+' : ''}{(strategy.expected_return ?? 0).toFixed(1)}%
                        </span>
                    </div>
                </div>
            )}

            {/* Expiration */}
            <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Expires</span>
                <span className="text-white font-medium">{strategy.expiration || 'N/A'}</span>
            </div>

            {/* Action */}
            <button className="w-full mt-4 flex items-center justify-center gap-2 py-2.5 rounded-lg bg-slate-800/50 text-slate-300 hover:bg-cyan-500/20 hover:text-cyan-400 transition-all duration-200 border border-slate-700/50 hover:border-cyan-500/30 group-hover:border-cyan-500/20">
                <span className="text-sm font-medium">View Details</span>
                <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
        </div>
    )
}
