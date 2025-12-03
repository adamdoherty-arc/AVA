import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    BarChart3, Search, RefreshCw, TrendingUp, TrendingDown,
    Activity, DollarSign, Percent, Clock, Target, Zap,
    ArrowUpRight, ArrowDownRight, Loader2
} from 'lucide-react'
import {
    ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend
} from 'recharts'

interface OptionsAnalysis {
    symbol: string
    underlying_price: number
    change_pct: number
    iv_rank: number
    iv_percentile: number
    expected_move: number
    put_call_ratio: number
    max_pain: number
    options: {
        calls: OptionChain[]
        puts: OptionChain[]
    }
    greeks_summary: {
        total_delta: number
        total_gamma: number
        total_theta: number
        total_vega: number
    }
    strategy_suggestions: {
        strategy: string
        description: string
        score: number
        profit_potential: string
        risk_level: string
    }[]
}

interface OptionChain {
    strike: number
    expiration: string
    dte: number
    bid: number
    ask: number
    last: number
    volume: number
    open_interest: number
    iv: number
    delta: number
    gamma: number
    theta: number
    vega: number
}

export default function OptionsAnalysisHub() {
    const [symbol, setSymbol] = useState('AAPL')
    const [searchInput, setSearchInput] = useState('AAPL')
    const [selectedExpiry, setSelectedExpiry] = useState<string>('')

    const { data: analysis, isLoading, refetch } = useQuery<OptionsAnalysis>({
        queryKey: ['options-analysis-hub', symbol],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/options/analysis-hub/${symbol}`)
            return data
        },
        staleTime: 60000
    })

    const handleSearch = () => {
        if (searchInput.trim()) {
            setSymbol(searchInput.trim().toUpperCase())
        }
    }

    const expirations = [...new Set([
        ...(analysis?.options?.calls?.map(o => o.expiration) || []),
        ...(analysis?.options?.puts?.map(o => o.expiration) || [])
    ])].sort()

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <BarChart3 className="w-5 h-5 text-white" />
                        </div>
                        Options Analysis Hub
                    </h1>
                    <p className="page-subtitle">Comprehensive options chain analysis with Greeks</p>
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
                    {/* Symbol Overview */}
                    <div className="glass-card p-5">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-6">
                                <div>
                                    <div className="text-3xl font-bold text-white">{analysis.symbol}</div>
                                    <div className="text-sm text-slate-400">Underlying</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-mono text-white">${(analysis.underlying_price ?? 0).toFixed(2)}</div>
                                    <div className={`flex items-center gap-1 text-sm ${
                                        (analysis.change_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                        {(analysis.change_pct ?? 0) >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                                        {Math.abs(analysis.change_pct ?? 0).toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">IV Rank</div>
                            <div className={`text-2xl font-bold ${
                                (analysis.iv_rank ?? 0) >= 50 ? 'text-amber-400' : 'text-slate-300'
                            }`}>
                                {(analysis.iv_rank ?? 0).toFixed(0)}%
                            </div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">IV Percentile</div>
                            <div className="text-2xl font-bold text-white">{(analysis.iv_percentile ?? 0).toFixed(0)}%</div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Expected Move</div>
                            <div className="text-2xl font-bold text-purple-400">
                                ±${(analysis.expected_move ?? 0).toFixed(2)}
                            </div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Put/Call Ratio</div>
                            <div className={`text-2xl font-bold ${
                                (analysis.put_call_ratio ?? 0) > 1 ? 'text-red-400' : 'text-emerald-400'
                            }`}>
                                {(analysis.put_call_ratio ?? 0).toFixed(2)}
                            </div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Max Pain</div>
                            <div className="text-2xl font-bold text-white">${(analysis.max_pain ?? 0).toFixed(0)}</div>
                        </div>
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Total Theta</div>
                            <div className={`text-2xl font-bold ${
                                (analysis.greeks_summary?.total_theta ?? 0) < 0 ? 'text-red-400' : 'text-emerald-400'
                            }`}>
                                ${(analysis.greeks_summary?.total_theta ?? 0).toFixed(0)}
                            </div>
                        </div>
                    </div>

                    {/* Greeks Summary */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-400" />
                            Portfolio Greeks
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-slate-800/40 rounded-xl p-4">
                                <div className="text-slate-400 text-sm mb-1">Delta</div>
                                <div className="text-xl font-bold text-white">{(analysis.greeks_summary?.total_delta ?? 0).toFixed(2)}</div>
                                <div className="text-xs text-slate-500">Directional exposure</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-4">
                                <div className="text-slate-400 text-sm mb-1">Gamma</div>
                                <div className="text-xl font-bold text-white">{(analysis.greeks_summary?.total_gamma ?? 0).toFixed(4)}</div>
                                <div className="text-xs text-slate-500">Rate of delta change</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-4">
                                <div className="text-slate-400 text-sm mb-1">Theta</div>
                                <div className={`text-xl font-bold ${
                                    (analysis.greeks_summary?.total_theta ?? 0) < 0 ? 'text-red-400' : 'text-emerald-400'
                                }`}>
                                    ${(analysis.greeks_summary?.total_theta ?? 0).toFixed(2)}
                                </div>
                                <div className="text-xs text-slate-500">Daily time decay</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-4">
                                <div className="text-slate-400 text-sm mb-1">Vega</div>
                                <div className="text-xl font-bold text-white">${(analysis.greeks_summary?.total_vega ?? 0).toFixed(2)}</div>
                                <div className="text-xs text-slate-500">IV sensitivity</div>
                            </div>
                        </div>
                    </div>

                    {/* Strategy Suggestions */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Zap className="w-5 h-5 text-amber-400" />
                            Strategy Suggestions
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {analysis.strategy_suggestions.map((strategy, idx) => (
                                <div key={idx} className="bg-slate-800/40 rounded-xl p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-semibold text-white">{strategy.strategy}</span>
                                        <span className={`px-2 py-0.5 rounded-lg text-xs font-medium ${
                                            strategy.score >= 80 ? 'bg-emerald-500/20 text-emerald-400' :
                                            strategy.score >= 60 ? 'bg-blue-500/20 text-blue-400' :
                                            'bg-slate-500/20 text-slate-400'
                                        }`}>
                                            Score: {strategy.score}
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-400 mb-3">{strategy.description}</p>
                                    <div className="flex items-center justify-between text-xs">
                                        <span className="text-slate-500">Profit: <span className="text-emerald-400">{strategy.profit_potential}</span></span>
                                        <span className={`${
                                            strategy.risk_level === 'Low' ? 'text-emerald-400' :
                                            strategy.risk_level === 'Medium' ? 'text-amber-400' : 'text-red-400'
                                        }`}>
                                            Risk: {strategy.risk_level}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Options Chain */}
                    <div className="glass-card overflow-hidden">
                        <div className="p-5 border-b border-slate-700/50">
                            <div className="flex items-center justify-between">
                                <h3 className="text-lg font-semibold text-white">Options Chain</h3>
                                <select
                                    value={selectedExpiry}
                                    onChange={(e) => setSelectedExpiry(e.target.value)}
                                    className="input-field w-48"
                                >
                                    <option value="">All Expirations</option>
                                    {expirations.map(exp => (
                                        <option key={exp} value={exp}>{exp}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                        <div className="overflow-x-auto max-h-96">
                            <table className="data-table">
                                <thead className="sticky top-0 bg-slate-900">
                                    <tr>
                                        <th colSpan={6} className="text-center bg-emerald-500/10 text-emerald-400">CALLS</th>
                                        <th className="bg-slate-800">Strike</th>
                                        <th colSpan={6} className="text-center bg-red-500/10 text-red-400">PUTS</th>
                                    </tr>
                                    <tr>
                                        <th>Bid</th>
                                        <th>Ask</th>
                                        <th>IV</th>
                                        <th>Delta</th>
                                        <th>Volume</th>
                                        <th>OI</th>
                                        <th className="bg-slate-800">$</th>
                                        <th>Bid</th>
                                        <th>Ask</th>
                                        <th>IV</th>
                                        <th>Delta</th>
                                        <th>Volume</th>
                                        <th>OI</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {analysis.options.calls
                                        .filter(c => !selectedExpiry || c.expiration === selectedExpiry)
                                        .slice(0, 20)
                                        .map((call, idx) => {
                                            const put = analysis.options.puts.find(p => p.strike === call.strike && p.expiration === call.expiration)
                                            const isATM = Math.abs(call.strike - analysis.underlying_price) < analysis.underlying_price * 0.02
                                            return (
                                                <tr key={idx} className={isATM ? 'bg-primary/10' : ''}>
                                                    <td className="text-emerald-400 font-mono">{(call.bid ?? 0).toFixed(2)}</td>
                                                    <td className="text-emerald-400 font-mono">{(call.ask ?? 0).toFixed(2)}</td>
                                                    <td className="text-slate-400">{((call.iv ?? 0) * 100).toFixed(0)}%</td>
                                                    <td className="text-slate-400">{(call.delta ?? 0).toFixed(2)}</td>
                                                    <td className="text-slate-400">{(call.volume ?? 0).toLocaleString()}</td>
                                                    <td className="text-slate-400">{(call.open_interest ?? 0).toLocaleString()}</td>
                                                    <td className="text-white font-bold bg-slate-800">${call.strike ?? 0}</td>
                                                    <td className="text-red-400 font-mono">{put ? (put.bid ?? 0).toFixed(2) : '-'}</td>
                                                    <td className="text-red-400 font-mono">{put ? (put.ask ?? 0).toFixed(2) : '-'}</td>
                                                    <td className="text-slate-400">{put ? ((put.iv ?? 0) * 100).toFixed(0) + '%' : '-'}</td>
                                                    <td className="text-slate-400">{put ? (put.delta ?? 0).toFixed(2) : '-'}</td>
                                                    <td className="text-slate-400">{put ? (put.volume ?? 0).toLocaleString() : '-'}</td>
                                                    <td className="text-slate-400">{put ? (put.open_interest ?? 0).toLocaleString() : '-'}</td>
                                                </tr>
                                            )
                                        })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            ) : (
                <div className="glass-card p-12 text-center">
                    <BarChart3 className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Enter a Symbol</h3>
                    <p className="text-slate-400">Search for a stock symbol to view options analysis</p>
                </div>
            )}
        </div>
    )
}
