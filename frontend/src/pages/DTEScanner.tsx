import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Clock, RefreshCw, Search, TrendingUp, DollarSign, Percent,
    AlertCircle, Filter, Target, Zap, ChevronDown, Activity,
    ArrowUpRight, ArrowDownRight
} from 'lucide-react'
import clsx from 'clsx'

interface DTEOpportunity {
    symbol: string
    company_name: string
    current_price: number
    strike: number
    expiration: string
    dte: number
    bid: number
    ask: number
    premium_pct: number
    annual_return: number
    delta: number
    theta: number
    iv: number
    volume: number
    open_interest: number
}

interface DTEScannerData {
    opportunities: DTEOpportunity[]
    scanned_at: string
    symbols_scanned: number
}

const WATCHLISTS = {
    'High Theta': ['TSLA', 'NVDA', 'AMD', 'PLTR', 'SOFI', 'SNAP', 'GME', 'AMC'],
    'Blue Chips': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'JPM', 'V', 'MA'],
    'Tech': ['AAPL', 'MSFT', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'TSLA'],
    'Under $30': ['F', 'PLTR', 'SOFI', 'SNAP', 'T', 'BAC', 'INTC', 'CCL', 'AAL'],
}

const MAX_DTE_OPTIONS = [3, 5, 7]

export default function DTEScanner() {
    const [maxDTE, setMaxDTE] = useState(7)
    const [selectedWatchlist, setSelectedWatchlist] = useState('High Theta')
    const [customSymbols, setCustomSymbols] = useState('')
    const [minPremium, setMinPremium] = useState(0.5)
    const [maxStrike, setMaxStrike] = useState(100)
    const [sortBy, setSortBy] = useState<'premium' | 'annual' | 'theta'>('annual')

    const { data, isLoading, refetch, error } = useQuery<DTEScannerData>({
        queryKey: ['dte-scanner', maxDTE, selectedWatchlist],
        queryFn: async () => {
            const symbols = customSymbols.trim()
                ? customSymbols.split(',').map(s => s.trim().toUpperCase())
                : WATCHLISTS[selectedWatchlist as keyof typeof WATCHLISTS]

            const { data } = await axiosInstance.get('/scanner/dte', {
                params: {
                    symbols: symbols.join(','),
                    max_dte: maxDTE,
                    min_premium_pct: minPremium,
                    max_strike: maxStrike
                }
            })
            return data
        },
        staleTime: 60000,
    })

    const opportunities = data?.opportunities || []

    const sortedOpportunities = [...opportunities].sort((a, b) => {
        if (sortBy === 'premium') return b.premium_pct - a.premium_pct
        if (sortBy === 'annual') return b.annual_return - a.annual_return
        return Math.abs(b.theta) - Math.abs(a.theta)
    })

    const filteredOpportunities = sortedOpportunities.filter(opp =>
        opp.premium_pct >= minPremium && opp.strike <= maxStrike
    )

    const avgAnnualReturn = filteredOpportunities.length > 0
        ? filteredOpportunities.reduce((acc, o) => acc + o.annual_return, 0) / filteredOpportunities.length
        : 0

    const avgPremium = filteredOpportunities.length > 0
        ? filteredOpportunities.reduce((acc, o) => acc + o.premium_pct, 0) / filteredOpportunities.length
        : 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center shadow-lg">
                            <Clock className="w-5 h-5 text-white" />
                        </div>
                        7-Day DTE Scanner
                    </h1>
                    <p className="page-subtitle">Find short-term theta capture opportunities (0-7 DTE)</p>
                </div>
                <button
                    onClick={() => refetch()}
                    disabled={isLoading}
                    className="btn-icon"
                >
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* DTE Selection */}
            <div className="flex gap-2">
                {MAX_DTE_OPTIONS.map(dte => (
                    <button
                        key={dte}
                        onClick={() => setMaxDTE(dte)}
                        className={clsx(
                            "flex-1 p-4 rounded-xl border-2 transition-all",
                            maxDTE === dte
                                ? "border-primary bg-primary/10"
                                : "border-slate-700 bg-slate-800/50 hover:border-slate-600"
                        )}
                    >
                        <div className="text-2xl font-bold">{dte}</div>
                        <div className="text-sm text-slate-400">DTE Max</div>
                    </button>
                ))}
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {/* Watchlist */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Watchlist</label>
                        <select
                            value={selectedWatchlist}
                            onChange={e => setSelectedWatchlist(e.target.value)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            {Object.keys(WATCHLISTS).map(wl => (
                                <option key={wl} value={wl}>{wl}</option>
                            ))}
                        </select>
                    </div>

                    {/* Custom Symbols */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Custom Symbols</label>
                        <input
                            type="text"
                            value={customSymbols}
                            onChange={e => setCustomSymbols(e.target.value)}
                            placeholder="AAPL, MSFT, ..."
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        />
                    </div>

                    {/* Min Premium */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Min Premium %</label>
                        <div className="flex items-center gap-2">
                            <input
                                type="range"
                                min={0}
                                max={5}
                                step={0.1}
                                value={minPremium}
                                onChange={e => setMinPremium(Number(e.target.value))}
                                className="flex-1"
                            />
                            <span className="text-sm font-mono w-12 text-right">{minPremium}%</span>
                        </div>
                    </div>

                    {/* Sort By */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Sort By</label>
                        <select
                            value={sortBy}
                            onChange={e => setSortBy(e.target.value as any)}
                            className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                        >
                            <option value="annual">Annual Return</option>
                            <option value="premium">Premium %</option>
                            <option value="theta">Theta</option>
                        </select>
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
                    <p className="text-2xl font-bold">{filteredOpportunities.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Avg Annual Return</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{avgAnnualReturn.toFixed(0)}%</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-sm">Avg Premium</span>
                    </div>
                    <p className="text-2xl font-bold text-amber-400">{avgPremium.toFixed(2)}%</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Clock className="w-4 h-4" />
                        <span className="text-sm">Max DTE</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">{maxDTE} days</p>
                </div>
            </div>

            {/* Results Table */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Scanning options...</span>
                </div>
            ) : error ? (
                <div className="card p-8 flex items-center justify-center text-red-400">
                    <AlertCircle className="w-6 h-6 mr-2" />
                    Failed to scan options
                </div>
            ) : filteredOpportunities.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No opportunities found matching your criteria</p>
                    <p className="text-sm mt-1">Try lowering minimum premium or increasing max DTE</p>
                </div>
            ) : (
                <div className="card overflow-hidden">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-slate-700/50">
                                <th className="text-left p-4 text-sm font-medium text-slate-400">Symbol</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Price</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Strike</th>
                                <th className="text-center p-4 text-sm font-medium text-slate-400">DTE</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Premium</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Annual %</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Delta</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">Theta</th>
                                <th className="text-right p-4 text-sm font-medium text-slate-400">IV</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredOpportunities.map((opp, idx) => (
                                <tr
                                    key={`${opp.symbol}-${opp.strike}-${opp.expiration}`}
                                    className={clsx(
                                        "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                        idx % 2 === 0 && "bg-slate-800/20"
                                    )}
                                >
                                    <td className="p-4">
                                        <div>
                                            <span className="font-mono font-semibold text-primary">{opp.symbol}</span>
                                            <p className="text-xs text-slate-400">{opp.company_name}</p>
                                        </div>
                                    </td>
                                    <td className="p-4 text-right font-mono">${(opp.current_price ?? 0).toFixed(2)}</td>
                                    <td className="p-4 text-right font-mono">${(opp.strike ?? 0).toFixed(0)}</td>
                                    <td className="p-4 text-center">
                                        <span className={clsx(
                                            "px-2 py-1 rounded text-xs font-bold",
                                            opp.dte <= 3 ? "bg-rose-500/20 text-rose-400" :
                                            opp.dte <= 5 ? "bg-amber-500/20 text-amber-400" :
                                            "bg-emerald-500/20 text-emerald-400"
                                        )}>
                                            {opp.dte}d
                                        </span>
                                    </td>
                                    <td className="p-4 text-right">
                                        <div className="font-mono">${(((opp.bid ?? 0) + (opp.ask ?? 0)) / 2).toFixed(2)}</div>
                                        <div className="text-xs text-amber-400">{(opp.premium_pct ?? 0).toFixed(2)}%</div>
                                    </td>
                                    <td className="p-4 text-right font-mono text-emerald-400 font-bold">
                                        {(opp.annual_return ?? 0).toFixed(0)}%
                                    </td>
                                    <td className="p-4 text-right font-mono text-sm">
                                        {(opp.delta ?? 0).toFixed(2)}
                                    </td>
                                    <td className="p-4 text-right font-mono text-sm text-rose-400">
                                        {(opp.theta ?? 0).toFixed(3)}
                                    </td>
                                    <td className="p-4 text-right font-mono text-sm">
                                        {((opp.iv ?? 0) * 100).toFixed(0)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Info Section */}
            <div className="card p-4 text-sm text-slate-400">
                <p className="font-medium text-white mb-2">About 0-7 DTE Options</p>
                <ul className="list-disc list-inside space-y-1">
                    <li><strong>Theta Decay:</strong> Options lose value fastest in the final week</li>
                    <li><strong>Risk:</strong> Higher gamma risk - prices can move quickly against you</li>
                    <li><strong>Best For:</strong> Experienced traders comfortable with quick adjustments</li>
                    <li><strong>Strategy:</strong> Sell puts/calls on stocks you're comfortable owning</li>
                </ul>
            </div>
        </div>
    )
}
