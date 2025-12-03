import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Calculator, RefreshCw, DollarSign, TrendingUp, TrendingDown,
    Calendar, AlertTriangle, CheckCircle, Filter, BarChart3
} from 'lucide-react'
import clsx from 'clsx'

interface TaxLot {
    id: string
    symbol: string
    purchase_date: string
    quantity: number
    cost_basis: number
    current_value: number
    gain_loss: number
    gain_loss_pct: number
    holding_period: 'short' | 'long'
    days_held: number
    days_to_long: number
}

interface TaxSummary {
    total_unrealized_gain: number
    short_term_gain: number
    long_term_gain: number
    short_term_loss: number
    long_term_loss: number
    wash_sale_risk: number
    estimated_tax: number
}

export default function TaxLotOptimizer() {
    const [sortBy, setSortBy] = useState<'gain' | 'date' | 'days_to_long'>('gain')
    const [filterHolding, setFilterHolding] = useState<'all' | 'short' | 'long'>('all')
    const [showLosses, setShowLosses] = useState(false)

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['tax-lots', sortBy, filterHolding],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/tax-lots', {
                params: { sort_by: sortBy, holding_period: filterHolding }
            })
            return data
        },
        staleTime: 60000,
    })

    const lots: TaxLot[] = data?.lots || []
    const summary: TaxSummary = data?.summary || {
        total_unrealized_gain: 0,
        short_term_gain: 0,
        long_term_gain: 0,
        short_term_loss: 0,
        long_term_loss: 0,
        wash_sale_risk: 0,
        estimated_tax: 0
    }

    const filteredLots = lots.filter(lot => {
        if (filterHolding === 'short' && lot.holding_period !== 'short') return false
        if (filterHolding === 'long' && lot.holding_period !== 'long') return false
        if (showLosses && lot.gain_loss >= 0) return false
        return true
    })

    const sortedLots = [...filteredLots].sort((a, b) => {
        if (sortBy === 'gain') return b.gain_loss - a.gain_loss
        if (sortBy === 'date') return new Date(a.purchase_date).getTime() - new Date(b.purchase_date).getTime()
        return a.days_to_long - b.days_to_long
    })

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-lg">
                            <Calculator className="w-5 h-5 text-white" />
                        </div>
                        Tax Lot Optimizer
                    </h1>
                    <p className="page-subtitle">Optimize tax lots for tax-loss harvesting and gain management</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* Tax Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-sm">Unrealized Gain</span>
                    </div>
                    <p className={clsx(
                        "text-2xl font-bold",
                        summary.total_unrealized_gain >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}>
                        ${(summary.total_unrealized_gain ?? 0).toLocaleString()}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Short-Term Gain</span>
                    </div>
                    <p className="text-2xl font-bold text-amber-400">
                        ${(summary.short_term_gain ?? 0).toLocaleString()}
                    </p>
                    <p className="text-xs text-slate-500">Taxed at income rate</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Long-Term Gain</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">
                        ${(summary.long_term_gain ?? 0).toLocaleString()}
                    </p>
                    <p className="text-xs text-slate-500">Lower tax rate</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Calculator className="w-4 h-4" />
                        <span className="text-sm">Est. Tax Liability</span>
                    </div>
                    <p className="text-2xl font-bold text-rose-400">
                        ${(summary.estimated_tax ?? 0).toLocaleString()}
                    </p>
                </div>
            </div>

            {/* Harvesting Opportunities */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card p-4 bg-gradient-to-br from-rose-500/10 to-orange-500/10 border-rose-500/30">
                    <h3 className="font-semibold mb-2 flex items-center gap-2">
                        <TrendingDown className="w-5 h-5 text-rose-400" />
                        Tax-Loss Harvesting Available
                    </h3>
                    <p className="text-3xl font-bold text-rose-400 mb-1">
                        ${((summary.short_term_loss ?? 0) + (summary.long_term_loss ?? 0)).toLocaleString()}
                    </p>
                    <p className="text-sm text-slate-400">
                        Potential losses available to offset gains
                    </p>
                </div>
                <div className="card p-4 bg-gradient-to-br from-amber-500/10 to-yellow-500/10 border-amber-500/30">
                    <h3 className="font-semibold mb-2 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-amber-400" />
                        Wash Sale Risk
                    </h3>
                    <p className="text-3xl font-bold text-amber-400 mb-1">
                        ${(summary.wash_sale_risk ?? 0).toLocaleString()}
                    </p>
                    <p className="text-sm text-slate-400">
                        Positions at risk of wash sale if sold
                    </p>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Filters:</span>
                    </div>
                    <div className="flex gap-2">
                        {(['all', 'short', 'long'] as const).map(h => (
                            <button
                                key={h}
                                onClick={() => setFilterHolding(h)}
                                className={clsx(
                                    "px-3 py-1.5 rounded-lg text-sm",
                                    filterHolding === h ? "bg-primary/20 text-primary" : "bg-slate-800/50 text-slate-400"
                                )}
                            >
                                {h === 'all' ? 'All' : h === 'short' ? 'Short-Term' : 'Long-Term'}
                            </button>
                        ))}
                    </div>
                    <div className="flex gap-2">
                        {([
                            { id: 'gain', label: 'By Gain/Loss' },
                            { id: 'date', label: 'By Date' },
                            { id: 'days_to_long', label: 'Days to Long-Term' }
                        ] as const).map(s => (
                            <button
                                key={s.id}
                                onClick={() => setSortBy(s.id)}
                                className={clsx(
                                    "px-3 py-1.5 rounded-lg text-sm",
                                    sortBy === s.id ? "bg-blue-500/20 text-blue-400" : "bg-slate-800/50 text-slate-400"
                                )}
                            >
                                {s.label}
                            </button>
                        ))}
                    </div>
                    <label className="flex items-center gap-2 ml-auto cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showLosses}
                            onChange={e => setShowLosses(e.target.checked)}
                            className="w-4 h-4 rounded"
                        />
                        <span className="text-sm">Show Only Losses</span>
                    </label>
                </div>
            </div>

            {/* Tax Lots Table */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading tax lots...</span>
                </div>
            ) : sortedLots.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Calculator className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No tax lots found</p>
                </div>
            ) : (
                <div className="card overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-slate-700/50">
                                <th className="text-left p-4 text-slate-400">Symbol</th>
                                <th className="text-center p-4 text-slate-400">Purchase Date</th>
                                <th className="text-right p-4 text-slate-400">Qty</th>
                                <th className="text-right p-4 text-slate-400">Cost Basis</th>
                                <th className="text-right p-4 text-slate-400">Current Value</th>
                                <th className="text-right p-4 text-slate-400">Gain/Loss</th>
                                <th className="text-center p-4 text-slate-400">Holding</th>
                                <th className="text-center p-4 text-slate-400">Days to LT</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedLots.map((lot, idx) => (
                                <tr key={lot.id} className={clsx(
                                    "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                    idx % 2 === 0 && "bg-slate-800/20"
                                )}>
                                    <td className="p-4">
                                        <span className="font-mono font-bold text-primary">{lot.symbol}</span>
                                    </td>
                                    <td className="p-4 text-center text-slate-400">{lot.purchase_date}</td>
                                    <td className="p-4 text-right font-mono">{lot.quantity}</td>
                                    <td className="p-4 text-right font-mono">${(lot.cost_basis ?? 0).toLocaleString()}</td>
                                    <td className="p-4 text-right font-mono">${(lot.current_value ?? 0).toLocaleString()}</td>
                                    <td className="p-4 text-right">
                                        <div className={clsx(
                                            "font-mono font-bold",
                                            (lot.gain_loss ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            {(lot.gain_loss ?? 0) >= 0 ? '+' : ''}${(lot.gain_loss ?? 0).toLocaleString()}
                                        </div>
                                        <div className={clsx(
                                            "text-xs",
                                            (lot.gain_loss_pct ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            ({(lot.gain_loss_pct ?? 0).toFixed(1)}%)
                                        </div>
                                    </td>
                                    <td className="p-4 text-center">
                                        <span className={clsx(
                                            "px-2 py-1 rounded text-xs font-medium",
                                            lot.holding_period === 'long'
                                                ? "bg-emerald-500/20 text-emerald-400"
                                                : "bg-amber-500/20 text-amber-400"
                                        )}>
                                            {lot.holding_period === 'long' ? 'Long' : 'Short'}
                                        </span>
                                    </td>
                                    <td className="p-4 text-center">
                                        {lot.holding_period === 'long' ? (
                                            <CheckCircle className="w-4 h-4 text-emerald-400 mx-auto" />
                                        ) : (
                                            <span className={clsx(
                                                "font-mono",
                                                lot.days_to_long <= 30 ? "text-amber-400" : "text-slate-400"
                                            )}>
                                                {lot.days_to_long}d
                                            </span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Info Section */}
            <div className="card p-4 text-sm text-slate-400">
                <p className="font-medium text-white mb-2">Tax Optimization Tips</p>
                <ul className="list-disc list-inside space-y-1">
                    <li><strong>Hold 365+ days:</strong> Long-term capital gains are taxed at lower rates (0-20%)</li>
                    <li><strong>Tax-Loss Harvesting:</strong> Sell losing positions to offset gains</li>
                    <li><strong>Wash Sale Rule:</strong> Cannot buy substantially identical security within 30 days</li>
                    <li><strong>Specific ID:</strong> Sell highest-cost lots first to minimize gains</li>
                </ul>
            </div>
        </div>
    )
}
