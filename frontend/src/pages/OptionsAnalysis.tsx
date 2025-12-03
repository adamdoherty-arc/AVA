import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    LineChart, Search, RefreshCw, TrendingUp, TrendingDown,
    Activity, DollarSign, Percent, Target, AlertCircle, Loader2
} from 'lucide-react'
import {
    ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, Area, AreaChart
} from 'recharts'

interface OptionsData {
    symbol: string
    underlying_price: number
    change_pct: number
    iv_rank: number
    iv_percentile: number
    historical_iv: { date: string; iv: number }[]
    put_call_ratio: number
    max_pain: number
    expected_move: number
    volume_analysis: {
        total_call_volume: number
        total_put_volume: number
        unusual_activity: boolean
    }
    risk_metrics: {
        beta: number
        correlation_spy: number
        avg_daily_range: number
    }
    price_targets: {
        analyst_high: number
        analyst_low: number
        analyst_mean: number
    }
}

export default function OptionsAnalysis() {
    const [symbol, setSymbol] = useState('AAPL')
    const [searchInput, setSearchInput] = useState('AAPL')

    const { data: analysis, isLoading, refetch } = useQuery<OptionsData>({
        queryKey: ['options-analysis', symbol],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/options/analysis/${symbol}`)
            return data
        },
        staleTime: 60000
    })

    const handleSearch = () => {
        if (searchInput.trim()) {
            setSymbol(searchInput.trim().toUpperCase())
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                            <LineChart className="w-5 h-5 text-white" />
                        </div>
                        Options Analysis
                    </h1>
                    <p className="page-subtitle">Deep dive into options data and volatility metrics</p>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Search */}
            <div className="glass-card p-5">
                <div className="flex items-center gap-4">
                    <div className="relative flex-1 max-w-md">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            value={searchInput}
                            onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="Enter symbol..."
                            className="input-field pl-10"
                        />
                    </div>
                    <button onClick={handleSearch} className="btn-primary px-6">
                        Analyze
                    </button>
                </div>
            </div>

            {isLoading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
            ) : analysis ? (
                <>
                    {/* Symbol Header */}
                    <div className="glass-card p-5">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-6">
                                <div>
                                    <div className="text-3xl font-bold text-white">{analysis.symbol}</div>
                                    <div className="text-sm text-slate-400">Options Analysis</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-mono text-white">${analysis.underlying_price?.toFixed(2) ?? 'N/A'}</div>
                                    <div className={`flex items-center gap-1 text-sm ${
                                        (analysis.change_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                        {(analysis.change_pct ?? 0) >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                        {Math.abs(analysis.change_pct ?? 0).toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                            {analysis.volume_analysis?.unusual_activity && (
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/20 border border-amber-500/30 rounded-lg text-amber-400">
                                    <AlertCircle className="w-4 h-4" />
                                    <span className="text-sm font-medium">Unusual Activity</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">IV Rank</div>
                            <div className={`text-2xl font-bold ${
                                (analysis.iv_rank ?? 0) >= 50 ? 'text-amber-400' : 'text-slate-300'
                            }`}>
                                {analysis.iv_rank?.toFixed(0) ?? 'N/A'}%
                            </div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">IV Percentile</div>
                            <div className="text-2xl font-bold text-white">{analysis.iv_percentile?.toFixed(0) ?? 'N/A'}%</div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Expected Move</div>
                            <div className="text-2xl font-bold text-purple-400">±${typeof analysis.expected_move === 'number' ? analysis.expected_move.toFixed(2) : 'N/A'}</div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Put/Call Ratio</div>
                            <div className={`text-2xl font-bold ${
                                (analysis.put_call_ratio ?? 0) > 1 ? 'text-red-400' : 'text-emerald-400'
                            }`}>
                                {analysis.put_call_ratio?.toFixed(2) ?? 'N/A'}
                            </div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Max Pain</div>
                            <div className="text-2xl font-bold text-white">${analysis.max_pain?.toFixed(0) ?? 'N/A'}</div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Beta</div>
                            <div className="text-2xl font-bold text-blue-400">{analysis.risk_metrics?.beta?.toFixed(2) ?? 'N/A'}</div>
                        </div>
                    </div>

                    {/* Charts */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* IV History Chart */}
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4">Historical IV</h3>
                            {analysis.historical_iv?.length ? (
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                                        <AreaChart data={analysis.historical_iv}>
                                            <defs>
                                                <linearGradient id="ivGradient" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#A855F7" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#A855F7" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                            <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                                            <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#1e293b',
                                                    border: '1px solid #334155',
                                                    borderRadius: '0.5rem'
                                                }}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="iv"
                                                stroke="#A855F7"
                                                fill="url(#ivGradient)"
                                                strokeWidth={2}
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            ) : (
                                <div className="h-64 flex items-center justify-center text-slate-400">
                                    Historical IV data not available
                                </div>
                            )}
                        </div>

                        {/* Volume Analysis */}
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4">Volume Analysis</h3>
                            {analysis.volume_analysis ? (
                                <div className="space-y-4">
                                    <div className="bg-slate-800/40 rounded-xl p-4">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-slate-400">Call Volume</span>
                                            <span className="text-emerald-400 font-mono">{(analysis.volume_analysis.total_call_volume ?? 0).toLocaleString()}</span>
                                        </div>
                                        <div className="h-2 bg-slate-700 rounded-full">
                                            <div
                                                className="h-full bg-emerald-500 rounded-full"
                                                style={{
                                                    width: `${((analysis.volume_analysis.total_call_volume ?? 0) / ((analysis.volume_analysis.total_call_volume ?? 0) + (analysis.volume_analysis.total_put_volume ?? 1))) * 100}%`
                                                }}
                                            />
                                        </div>
                                    </div>
                                    <div className="bg-slate-800/40 rounded-xl p-4">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-slate-400">Put Volume</span>
                                            <span className="text-red-400 font-mono">{(analysis.volume_analysis.total_put_volume ?? 0).toLocaleString()}</span>
                                        </div>
                                        <div className="h-2 bg-slate-700 rounded-full">
                                            <div
                                                className="h-full bg-red-500 rounded-full"
                                                style={{
                                                    width: `${((analysis.volume_analysis.total_put_volume ?? 0) / ((analysis.volume_analysis.total_call_volume ?? 0) + (analysis.volume_analysis.total_put_volume ?? 1))) * 100}%`
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center py-8 text-slate-400">
                                    Volume data not available
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Risk Metrics & Price Targets */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Risk Metrics */}
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-blue-400" />
                                Risk Metrics
                            </h3>
                            {analysis.risk_metrics ? (
                                <div className="grid grid-cols-3 gap-4">
                                    <div className="bg-slate-800/40 rounded-xl p-4">
                                        <div className="text-slate-400 text-sm mb-1">Beta</div>
                                        <div className="text-xl font-bold text-white">{analysis.risk_metrics.beta?.toFixed(2) ?? 'N/A'}</div>
                                        <div className="text-xs text-slate-500">Market sensitivity</div>
                                    </div>
                                    <div className="bg-slate-800/40 rounded-xl p-4">
                                        <div className="text-slate-400 text-sm mb-1">SPY Correlation</div>
                                        <div className="text-xl font-bold text-white">{analysis.risk_metrics.correlation_spy?.toFixed(2) ?? 'N/A'}</div>
                                        <div className="text-xs text-slate-500">Index correlation</div>
                                    </div>
                                    <div className="bg-slate-800/40 rounded-xl p-4">
                                        <div className="text-slate-400 text-sm mb-1">Avg Daily Range</div>
                                        <div className="text-xl font-bold text-white">{analysis.risk_metrics.avg_daily_range?.toFixed(2) ?? 'N/A'}%</div>
                                        <div className="text-xs text-slate-500">Price volatility</div>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center py-8 text-slate-400">
                                    Risk metrics not available
                                </div>
                            )}
                        </div>

                        {/* Price Targets */}
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Target className="w-5 h-5 text-amber-400" />
                                Analyst Price Targets
                            </h3>
                            {analysis.price_targets ? (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-slate-400">High</span>
                                        <span className="text-emerald-400 font-mono text-lg">${analysis.price_targets.analyst_high?.toFixed(2) ?? 'N/A'}</span>
                                    </div>
                                    <div className="relative">
                                        <div className="h-2 bg-slate-700 rounded-full">
                                            {analysis.price_targets.analyst_high && analysis.price_targets.analyst_low && (
                                                <div
                                                    className="absolute h-4 w-1 bg-white rounded -top-1"
                                                    style={{
                                                        left: `${Math.max(0, Math.min(100, ((analysis.underlying_price - analysis.price_targets.analyst_low) / (analysis.price_targets.analyst_high - analysis.price_targets.analyst_low)) * 100))}%`
                                                    }}
                                                />
                                            )}
                                        </div>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <span className="text-slate-400">Low: </span>
                                            <span className="text-red-400 font-mono">${analysis.price_targets.analyst_low?.toFixed(2) ?? 'N/A'}</span>
                                        </div>
                                        <div>
                                            <span className="text-slate-400">Mean: </span>
                                            <span className="text-blue-400 font-mono">${analysis.price_targets.analyst_mean?.toFixed(2) ?? 'N/A'}</span>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center py-8 text-slate-400">
                                    Price target data not available
                                </div>
                            )}
                        </div>
                    </div>
                </>
            ) : (
                <div className="glass-card p-12 text-center">
                    <LineChart className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Enter a Symbol</h3>
                    <p className="text-slate-400">Search for a stock symbol to view options analysis</p>
                </div>
            )}
        </div>
    )
}
