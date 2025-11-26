import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    PieChart, RefreshCw, TrendingUp, TrendingDown, BarChart3,
    ArrowUpRight, ArrowDownRight, Zap, Target, Activity
} from 'lucide-react'
import clsx from 'clsx'

interface Sector {
    name: string
    symbol: string
    performance_1d: number
    performance_1w: number
    performance_1m: number
    performance_ytd: number
    relative_strength: number
    momentum: 'bullish' | 'bearish' | 'neutral'
    top_stocks: string[]
    flow_sentiment: number
}

const TIMEFRAMES = ['1D', '1W', '1M', 'YTD']

export default function SectorAnalysis() {
    const [timeframe, setTimeframe] = useState('1W')
    const [sortBy, setSortBy] = useState<'performance' | 'strength'>('performance')

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['sectors', timeframe],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/research/sectors', { params: { timeframe } })
            return data
        },
        staleTime: 300000,
    })

    const sectors: Sector[] = data?.sectors || []
    const sortedSectors = [...sectors].sort((a, b) => {
        if (sortBy === 'performance') {
            const perfKey = `performance_${timeframe.toLowerCase()}` as keyof Sector
            return (b[perfKey] as number) - (a[perfKey] as number)
        }
        return b.relative_strength - a.relative_strength
    })

    const bullishCount = sectors.filter(s => s.momentum === 'bullish').length
    const bearishCount = sectors.filter(s => s.momentum === 'bearish').length

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <PieChart className="w-5 h-5 text-white" />
                        </div>
                        Sector Analysis
                    </h1>
                    <p className="page-subtitle">Market sector rotation and relative strength analysis</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* Timeframe Selection */}
            <div className="flex gap-2">
                {TIMEFRAMES.map(tf => (
                    <button
                        key={tf}
                        onClick={() => setTimeframe(tf)}
                        className={clsx(
                            "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                            timeframe === tf
                                ? "bg-primary text-white"
                                : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50"
                        )}
                    >
                        {tf}
                    </button>
                ))}
            </div>

            {/* Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <PieChart className="w-4 h-4" />
                        <span className="text-sm">Total Sectors</span>
                    </div>
                    <p className="text-2xl font-bold">{sectors.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Bullish</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{bullishCount}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingDown className="w-4 h-4" />
                        <span className="text-sm">Bearish</span>
                    </div>
                    <p className="text-2xl font-bold text-rose-400">{bearishCount}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-sm">Market Breadth</span>
                    </div>
                    <p className={clsx(
                        "text-2xl font-bold",
                        bullishCount > bearishCount ? "text-emerald-400" : "text-rose-400"
                    )}>
                        {((bullishCount / sectors.length) * 100).toFixed(0)}%
                    </p>
                </div>
            </div>

            {/* Sort Options */}
            <div className="flex gap-2">
                <button
                    onClick={() => setSortBy('performance')}
                    className={clsx(
                        "px-3 py-1.5 rounded-lg text-sm",
                        sortBy === 'performance' ? "bg-primary/20 text-primary" : "bg-slate-800/50 text-slate-400"
                    )}
                >
                    Sort by Performance
                </button>
                <button
                    onClick={() => setSortBy('strength')}
                    className={clsx(
                        "px-3 py-1.5 rounded-lg text-sm",
                        sortBy === 'strength' ? "bg-primary/20 text-primary" : "bg-slate-800/50 text-slate-400"
                    )}
                >
                    Sort by Relative Strength
                </button>
            </div>

            {/* Sectors Grid */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading sectors...</span>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {sortedSectors.map(sector => {
                        const perfKey = `performance_${timeframe.toLowerCase()}` as keyof Sector
                        const perf = sector[perfKey] as number

                        return (
                            <div key={sector.symbol} className="card p-4 hover:border-primary/50 transition-colors">
                                <div className="flex items-start justify-between mb-3">
                                    <div>
                                        <h3 className="font-semibold">{sector.name}</h3>
                                        <p className="text-sm text-slate-400">{sector.symbol}</p>
                                    </div>
                                    <div className={clsx(
                                        "flex items-center gap-1 px-2 py-1 rounded text-sm font-bold",
                                        sector.momentum === 'bullish' ? "bg-emerald-500/20 text-emerald-400" :
                                        sector.momentum === 'bearish' ? "bg-rose-500/20 text-rose-400" :
                                        "bg-slate-700 text-slate-400"
                                    )}>
                                        {sector.momentum === 'bullish' ? <ArrowUpRight className="w-4 h-4" /> :
                                         sector.momentum === 'bearish' ? <ArrowDownRight className="w-4 h-4" /> : null}
                                        {sector.momentum}
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-2 mb-3">
                                    <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                                        <p className="text-xs text-slate-400">Performance</p>
                                        <p className={clsx(
                                            "text-lg font-bold",
                                            perf >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            {perf >= 0 ? '+' : ''}{perf.toFixed(2)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-2 text-center">
                                        <p className="text-xs text-slate-400">RS Rating</p>
                                        <p className={clsx(
                                            "text-lg font-bold",
                                            sector.relative_strength >= 70 ? "text-emerald-400" :
                                            sector.relative_strength >= 50 ? "text-amber-400" : "text-rose-400"
                                        )}>
                                            {sector.relative_strength}
                                        </p>
                                    </div>
                                </div>

                                <div>
                                    <p className="text-xs text-slate-400 mb-1">Top Holdings</p>
                                    <div className="flex flex-wrap gap-1">
                                        {sector.top_stocks.slice(0, 5).map(stock => (
                                            <span key={stock} className="px-2 py-0.5 bg-slate-700/50 rounded text-xs font-mono">
                                                {stock}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                <div className="mt-3 pt-3 border-t border-slate-700/50">
                                    <div className="flex items-center justify-between text-sm">
                                        <span className="text-slate-400">Flow Sentiment</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-20 h-2 bg-slate-700 rounded-full overflow-hidden">
                                                <div
                                                    className={clsx(
                                                        "h-full rounded-full",
                                                        sector.flow_sentiment >= 60 ? "bg-emerald-500" :
                                                        sector.flow_sentiment >= 40 ? "bg-amber-500" : "bg-rose-500"
                                                    )}
                                                    style={{ width: `${sector.flow_sentiment}%` }}
                                                />
                                            </div>
                                            <span className="font-mono">{sector.flow_sentiment}%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
