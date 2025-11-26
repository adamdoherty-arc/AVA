import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Zap, RefreshCw, TrendingUp, DollarSign, Percent, Clock,
    Trophy, Target, ArrowUpRight, ArrowDownRight, Filter
} from 'lucide-react'
import clsx from 'clsx'

interface KalshiMarket {
    id: string
    title: string
    category: string
    yes_price: number
    no_price: number
    volume: number
    open_interest: number
    close_time: string
    ai_prediction: number
    edge: number
    confidence: number
    recommendation?: string
    home_team?: string
    away_team?: string
    game_time?: string
    reasoning?: string
}

const CATEGORIES = ['All', 'NFL', 'NBA', 'Politics', 'Finance', 'Weather', 'Entertainment']

export default function KalshiMarkets() {
    const [category, setCategory] = useState('NFL')
    const [minEdge, setMinEdge] = useState(5)
    const [showOnlyEdge, setShowOnlyEdge] = useState(true)

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['kalshi-markets', category],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/predictions/kalshi', {
                params: { category, min_edge: minEdge }
            })
            return data
        },
        staleTime: 60000,
    })

    const markets: KalshiMarket[] = data?.markets || []
    const filteredMarkets = showOnlyEdge ? markets.filter(m => m.edge >= minEdge) : markets
    const totalVolume = markets.reduce((a, m) => a + m.volume, 0)
    const avgEdge = filteredMarkets.length ? filteredMarkets.reduce((a, m) => a + m.edge, 0) / filteredMarkets.length : 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Zap className="w-5 h-5 text-white" />
                        </div>
                        Kalshi Prediction Markets
                    </h1>
                    <p className="page-subtitle">AI-powered edge detection on prediction markets</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Target className="w-4 h-4" />
                        <span className="text-sm">Markets</span>
                    </div>
                    <p className="text-2xl font-bold">{filteredMarkets.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-sm">Total Volume</span>
                    </div>
                    <p className="text-2xl font-bold">${(totalVolume / 1000).toFixed(0)}K</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Avg Edge</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{avgEdge.toFixed(1)}%</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Trophy className="w-4 h-4" />
                        <span className="text-sm">Best Edge</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">
                        {filteredMarkets.length ? Math.max(...filteredMarkets.map(m => m.edge)).toFixed(1) : 0}%
                    </p>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Category</label>
                        <select
                            value={category}
                            onChange={e => setCategory(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Min Edge: {minEdge}%</label>
                        <input
                            type="range"
                            min={0}
                            max={20}
                            value={minEdge}
                            onChange={e => setMinEdge(Number(e.target.value))}
                            className="w-full"
                        />
                    </div>
                    <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="showEdge"
                            checked={showOnlyEdge}
                            onChange={e => setShowOnlyEdge(e.target.checked)}
                            className="w-4 h-4 rounded"
                        />
                        <label htmlFor="showEdge" className="text-sm">Only show markets with edge</label>
                    </div>
                </div>
            </div>

            {/* Markets */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading markets...</span>
                </div>
            ) : filteredMarkets.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Zap className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No markets found with sufficient edge</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {filteredMarkets.map(market => (
                        <div key={market.id} className="card p-4 hover:border-primary/50 transition-colors">
                            <div className="flex items-start justify-between">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="px-2 py-0.5 bg-slate-700 rounded text-xs">{market.category}</span>
                                        {market.recommendation && (
                                            <span className={clsx(
                                                "px-2 py-0.5 rounded text-xs font-bold",
                                                market.recommendation === 'YES' ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                                            )}>
                                                {market.recommendation}
                                            </span>
                                        )}
                                        <span className="text-xs text-slate-500">
                                            <Clock className="w-3 h-3 inline mr-1" />
                                            {market.game_time || market.close_time || 'TBD'}
                                        </span>
                                    </div>
                                    <h3 className="font-semibold text-lg">
                                        {market.home_team && market.away_team
                                            ? `${market.away_team} @ ${market.home_team}`
                                            : market.title}
                                    </h3>
                                    {market.reasoning && (
                                        <p className="text-sm text-slate-400 mt-1">{market.reasoning}</p>
                                    )}
                                </div>
                                <div className={clsx(
                                    "px-3 py-1 rounded-full text-sm font-bold",
                                    market.edge >= 10 ? "bg-emerald-500/20 text-emerald-400" :
                                    market.edge >= 5 ? "bg-amber-500/20 text-amber-400" :
                                    "bg-slate-700 text-slate-400"
                                )}>
                                    {market.edge > 0 ? '+' : ''}{market.edge.toFixed(1)}% edge
                                </div>
                            </div>

                            <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
                                <div className="bg-emerald-500/10 rounded-lg p-3 text-center">
                                    <p className="text-xs text-slate-400 mb-1">YES Price</p>
                                    <p className="text-xl font-bold text-emerald-400">{market.yes_price}¢</p>
                                </div>
                                <div className="bg-rose-500/10 rounded-lg p-3 text-center">
                                    <p className="text-xs text-slate-400 mb-1">NO Price</p>
                                    <p className="text-xl font-bold text-rose-400">{market.no_price}¢</p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                                    <p className="text-xs text-slate-400 mb-1">AI Prediction</p>
                                    <p className="text-xl font-bold text-primary">{market.ai_prediction}%</p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                                    <p className="text-xs text-slate-400 mb-1">Volume</p>
                                    <p className="text-xl font-bold">${market.volume.toLocaleString()}</p>
                                </div>
                                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                                    <p className="text-xs text-slate-400 mb-1">Confidence</p>
                                    <p className="text-xl font-bold">{market.confidence}%</p>
                                </div>
                            </div>

                            <div className="mt-3 flex justify-end gap-2">
                                <button className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-1",
                                    market.recommendation === 'YES'
                                        ? "bg-emerald-600 hover:bg-emerald-500 ring-2 ring-emerald-400"
                                        : "bg-emerald-600/50 hover:bg-emerald-500"
                                )}>
                                    <ArrowUpRight className="w-4 h-4" />
                                    Buy YES
                                </button>
                                <button className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-1",
                                    market.recommendation === 'NO'
                                        ? "bg-rose-600 hover:bg-rose-500 ring-2 ring-rose-400"
                                        : "bg-rose-600/50 hover:bg-rose-500"
                                )}>
                                    <ArrowDownRight className="w-4 h-4" />
                                    Buy NO
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
