import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Calendar, RefreshCw, TrendingUp, DollarSign, Percent, Target,
    ArrowRight, AlertCircle, Sparkles, BarChart3, Clock, Zap
} from 'lucide-react'
import clsx from 'clsx'

interface CalendarSpread {
    id: string
    symbol: string
    company_name: string
    current_price: number
    front_leg: {
        expiration: string
        strike: number
        premium: number
        iv: number
        dte: number
    }
    back_leg: {
        expiration: string
        strike: number
        premium: number
        iv: number
        dte: number
    }
    net_debit: number
    max_profit: number
    max_profit_pct: number
    iv_skew: number
    breakeven: number
    ai_score: number
    ai_reasoning: string
    risk_reward: number
}

const STRATEGIES = ['Calendar Put', 'Calendar Call', 'Diagonal Put', 'Diagonal Call']
const DTE_RANGES = ['7-14 / 30-45', '14-21 / 45-60', '21-30 / 60-90']

export default function CalendarSpreads() {
    const [strategy, setStrategy] = useState('Calendar Put')
    const [dteRange, setDteRange] = useState('7-14 / 30-45')
    const [minIVSkew, setMinIVSkew] = useState(5)
    const [symbols, setSymbols] = useState('TSLA,NVDA,AMD,AAPL,META')

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['calendar-spreads', strategy, dteRange, symbols],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/strategy/calendar-spreads', {
                params: { strategy, dte_range: dteRange, symbols, min_iv_skew: minIVSkew }
            })
            return data
        },
        staleTime: 120000,
    })

    const spreads: CalendarSpread[] = data?.spreads || []
    const avgScore = spreads.length ? spreads.reduce((a, s) => a + s.ai_score, 0) / spreads.length : 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg">
                            <Calendar className="w-5 h-5 text-white" />
                        </div>
                        Calendar Spreads
                    </h1>
                    <p className="page-subtitle">AI-powered calendar spread analysis with IV skew exploitation</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* AI Analysis Banner */}
            <div className="card p-4 bg-gradient-to-r from-primary/10 to-secondary/10 border-primary/30">
                <div className="flex items-center gap-3">
                    <Sparkles className="w-6 h-6 text-primary" />
                    <div>
                        <p className="font-semibold">AI-Powered Analysis</p>
                        <p className="text-sm text-slate-400">
                            Our AI analyzes IV term structure, theta decay curves, and historical performance
                        </p>
                    </div>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Strategy</label>
                        <select
                            value={strategy}
                            onChange={e => setStrategy(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">DTE Range (Front / Back)</label>
                        <select
                            value={dteRange}
                            onChange={e => setDteRange(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {DTE_RANGES.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Symbols</label>
                        <input
                            type="text"
                            value={symbols}
                            onChange={e => setSymbols(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                            placeholder="AAPL,MSFT,..."
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Min IV Skew %</label>
                        <div className="flex items-center gap-2">
                            <input
                                type="range"
                                min={0}
                                max={20}
                                value={minIVSkew}
                                onChange={e => setMinIVSkew(Number(e.target.value))}
                                className="flex-1"
                            />
                            <span className="text-sm font-mono w-10">{minIVSkew}%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Target className="w-4 h-4" />
                        <span className="text-sm">Opportunities</span>
                    </div>
                    <p className="text-2xl font-bold">{spreads.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Sparkles className="w-4 h-4" />
                        <span className="text-sm">Avg AI Score</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">{avgScore.toFixed(0)}/100</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Best Return</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">
                        {spreads.length ? Math.max(...spreads.map(s => s.max_profit_pct)).toFixed(0) : 0}%
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-sm">Avg IV Skew</span>
                    </div>
                    <p className="text-2xl font-bold text-amber-400">
                        {spreads.length ? (spreads.reduce((a, s) => a + s.iv_skew, 0) / spreads.length).toFixed(1) : 0}%
                    </p>
                </div>
            </div>

            {/* Results */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Analyzing spreads...</span>
                </div>
            ) : spreads.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No calendar spread opportunities found</p>
                    <p className="text-sm mt-1">Try adjusting filters or adding more symbols</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {spreads.map(spread => (
                        <div key={spread.id} className="card p-4 hover:border-primary/50 transition-colors">
                            <div className="flex items-start justify-between">
                                <div className="flex items-center gap-4">
                                    <div>
                                        <span className="font-mono font-bold text-lg text-primary">{spread.symbol}</span>
                                        <p className="text-sm text-slate-400">{spread.company_name}</p>
                                        <p className="text-xs text-slate-500">Stock: ${(spread.current_price ?? 0).toFixed(2)}</p>
                                    </div>
                                    <div className="flex items-center gap-2 text-slate-400">
                                        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
                                            <p className="text-xs text-slate-500">Front Leg</p>
                                            <p className="font-mono">${spread.front_leg.strike}</p>
                                            <p className="text-xs">{spread.front_leg.dte}d</p>
                                        </div>
                                        <ArrowRight className="w-4 h-4" />
                                        <div className="text-center p-2 bg-slate-800/50 rounded-lg">
                                            <p className="text-xs text-slate-500">Back Leg</p>
                                            <p className="font-mono">${spread.back_leg.strike}</p>
                                            <p className="text-xs">{spread.back_leg.dte}d</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={clsx(
                                        "inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-bold",
                                        spread.ai_score >= 80 ? "bg-emerald-500/20 text-emerald-400" :
                                        spread.ai_score >= 60 ? "bg-amber-500/20 text-amber-400" :
                                        "bg-slate-700 text-slate-400"
                                    )}>
                                        <Sparkles className="w-3 h-3" />
                                        {spread.ai_score}/100
                                    </div>
                                </div>
                            </div>
                            <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                                <div>
                                    <p className="text-slate-400">Net Debit</p>
                                    <p className="font-mono font-bold">${(spread.net_debit ?? 0).toFixed(2)}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Max Profit</p>
                                    <p className="font-mono font-bold text-emerald-400">${(spread.max_profit ?? 0).toFixed(2)}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Return %</p>
                                    <p className="font-mono font-bold text-emerald-400">{(spread.max_profit_pct ?? 0).toFixed(0)}%</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">IV Skew</p>
                                    <p className="font-mono font-bold text-amber-400">{(spread.iv_skew ?? 0).toFixed(1)}%</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Risk/Reward</p>
                                    <p className="font-mono font-bold">1:{(spread.risk_reward ?? 0).toFixed(1)}</p>
                                </div>
                            </div>
                            <div className="mt-3 p-2 bg-slate-800/30 rounded-lg">
                                <p className="text-xs text-slate-400">
                                    <Sparkles className="w-3 h-3 inline mr-1" />
                                    AI Analysis: {spread.ai_reasoning}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Info Section */}
            <div className="card p-4 text-sm text-slate-400">
                <p className="font-medium text-white mb-2">About Calendar Spreads</p>
                <ul className="list-disc list-inside space-y-1">
                    <li><strong>IV Skew:</strong> Profit when near-term IV is higher than far-term IV</li>
                    <li><strong>Theta Decay:</strong> Front month decays faster, earning the spread value</li>
                    <li><strong>Best For:</strong> Low volatility environments expecting IV crush</li>
                    <li><strong>Risk:</strong> Stock moves significantly away from strike price</li>
                </ul>
            </div>
        </div>
    )
}
