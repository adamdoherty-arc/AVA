import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Activity, RefreshCw, TrendingUp, TrendingDown, DollarSign,
    Zap, Filter, ArrowUpRight, ArrowDownRight, Clock, BarChart3
} from 'lucide-react'
import clsx from 'clsx'

interface FlowOrder {
    id: string
    symbol: string
    timestamp: string
    type: 'call' | 'put'
    strike: number
    expiration: string
    premium: number
    size: number
    spot_price: number
    sentiment: 'bullish' | 'bearish' | 'neutral'
    unusual: boolean
    sweep: boolean
    block: boolean
    oi_ratio: number
}

const FLOW_TYPES = ['All', 'Sweeps', 'Blocks', 'Unusual']
const SENTIMENTS = ['All', 'Bullish', 'Bearish']

export default function OptionsFlow() {
    const [flowType, setFlowType] = useState('All')
    const [sentiment, setSentiment] = useState('All')
    const [minPremium, setMinPremium] = useState(100000)
    const [symbols, setSymbols] = useState('')

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['options-flow', flowType, sentiment, minPremium],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/flow', {
                params: { flow_type: flowType, sentiment, min_premium: minPremium, symbols }
            })
            return data
        },
        staleTime: 30000,
        refetchInterval: 30000,
    })

    const orders: FlowOrder[] = data?.orders || []
    const bullishCount = orders.filter(o => o.sentiment === 'bullish').length
    const bearishCount = orders.filter(o => o.sentiment === 'bearish').length
    const totalPremium = orders.reduce((a, o) => a + o.premium, 0)

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg">
                            <Activity className="w-5 h-5 text-white" />
                        </div>
                        Options Flow
                    </h1>
                    <p className="page-subtitle">Real-time unusual options activity and smart money tracking</p>
                </div>
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1 px-3 py-1 bg-emerald-500/20 rounded-full">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-xs text-emerald-400">Live</span>
                    </div>
                    <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-sm">Total Orders</span>
                    </div>
                    <p className="text-2xl font-bold">{orders.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-sm">Total Premium</span>
                    </div>
                    <p className="text-2xl font-bold">${(totalPremium / 1000000).toFixed(1)}M</p>
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
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-sm">Bull/Bear Ratio</span>
                    </div>
                    <p className={clsx(
                        "text-2xl font-bold",
                        bullishCount > bearishCount ? "text-emerald-400" : "text-rose-400"
                    )}>
                        {bearishCount ? (bullishCount / bearishCount).toFixed(2) : '∞'}
                    </p>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Flow Type</label>
                        <select
                            value={flowType}
                            onChange={e => setFlowType(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {FLOW_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Sentiment</label>
                        <select
                            value={sentiment}
                            onChange={e => setSentiment(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {SENTIMENTS.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Min Premium</label>
                        <select
                            value={minPremium}
                            onChange={e => setMinPremium(Number(e.target.value))}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            <option value={50000}>$50K+</option>
                            <option value={100000}>$100K+</option>
                            <option value={250000}>$250K+</option>
                            <option value={500000}>$500K+</option>
                            <option value={1000000}>$1M+</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Symbol Filter</label>
                        <input
                            type="text"
                            value={symbols}
                            onChange={e => setSymbols(e.target.value)}
                            placeholder="AAPL, TSLA..."
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        />
                    </div>
                </div>
            </div>

            {/* Flow Table */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading flow data...</span>
                </div>
            ) : orders.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No options flow matching criteria</p>
                </div>
            ) : (
                <div className="card overflow-hidden">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-slate-700/50">
                                <th className="text-left p-4 text-sm font-medium text-slate-400">Time</th>
                                <th className="text-left p-4 text-sm font-medium text-slate-400">Symbol</th>
                                <th className="text-center p-4 text-sm font-medium text-slate-400">Type</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Strike</th>
                                <th className="text-center p-4 text-sm font-medium text-slate-400">Exp</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Premium</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Size</th>
                                <th className="text-center p-4 text-sm font-medium text-slate-400">Sentiment</th>
                                <th className="text-center p-4 text-sm font-medium text-slate-400">Tags</th>
                            </tr>
                        </thead>
                        <tbody>
                            {orders.map((order, idx) => (
                                <tr
                                    key={order.id}
                                    className={clsx(
                                        "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                        idx % 2 === 0 && "bg-slate-800/20"
                                    )}
                                >
                                    <td className="p-4 text-sm text-slate-400">
                                        <Clock className="w-3 h-3 inline mr-1" />
                                        {order.timestamp}
                                    </td>
                                    <td className="p-4">
                                        <span className="font-mono font-bold text-primary">{order.symbol}</span>
                                    </td>
                                    <td className="p-4 text-center">
                                        <span className={clsx(
                                            "px-2 py-1 rounded text-xs font-bold",
                                            order.type === 'call' ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                                        )}>
                                            {order.type.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="p-4 text-right font-mono">${order.strike}</td>
                                    <td className="p-4 text-center text-sm">{order.expiration}</td>
                                    <td className="p-4 text-right font-mono font-bold">
                                        ${(order.premium / 1000).toFixed(0)}K
                                    </td>
                                    <td className="p-4 text-right font-mono">{order.size}</td>
                                    <td className="p-4 text-center">
                                        {order.sentiment === 'bullish' ? (
                                            <ArrowUpRight className="w-5 h-5 text-emerald-400 mx-auto" />
                                        ) : order.sentiment === 'bearish' ? (
                                            <ArrowDownRight className="w-5 h-5 text-rose-400 mx-auto" />
                                        ) : (
                                            <span className="text-slate-400">—</span>
                                        )}
                                    </td>
                                    <td className="p-4 text-center">
                                        <div className="flex items-center justify-center gap-1">
                                            {order.sweep && (
                                                <span className="px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">SWEEP</span>
                                            )}
                                            {order.block && (
                                                <span className="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">BLOCK</span>
                                            )}
                                            {order.unusual && (
                                                <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded text-xs">
                                                    <Zap className="w-3 h-3 inline" />
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
