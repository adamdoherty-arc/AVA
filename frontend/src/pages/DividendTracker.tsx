import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Coins, RefreshCw, Calendar, DollarSign, Percent, TrendingUp,
    ArrowUpRight, Clock, Filter, BarChart3
} from 'lucide-react'
import clsx from 'clsx'

interface DividendStock {
    symbol: string
    name: string
    price: number
    dividend_yield: number
    annual_dividend: number
    ex_date: string
    pay_date: string
    frequency: 'monthly' | 'quarterly' | 'annual'
    years_growth: number
    payout_ratio: number
    sector: string
}

interface DividendIncome {
    month: string
    amount: number
    stocks: { symbol: string; amount: number }[]
}

export default function DividendTracker() {
    const [filter, setFilter] = useState<'all' | 'upcoming' | 'high_yield'>('all')
    const [minYield, setMinYield] = useState(2)

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['dividends', filter, minYield],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/dividends', {
                params: { filter, min_yield: minYield }
            })
            return data
        },
        staleTime: 300000,
    })

    const stocks: DividendStock[] = data?.stocks || []
    const income: DividendIncome[] = data?.income || []
    const totalAnnualDividends = stocks.reduce((a, s) => a + s.annual_dividend, 0)
    const avgYield = stocks.length ? stocks.reduce((a, s) => a + s.dividend_yield, 0) / stocks.length : 0

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg">
                            <Coins className="w-5 h-5 text-white" />
                        </div>
                        Dividend Tracker
                    </h1>
                    <p className="page-subtitle">Track dividend income and upcoming payments</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Coins className="w-4 h-4" />
                        <span className="text-sm">Dividend Stocks</span>
                    </div>
                    <p className="text-2xl font-bold">{stocks.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-sm">Annual Income</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">${totalAnnualDividends.toLocaleString()}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Percent className="w-4 h-4" />
                        <span className="text-sm">Avg Yield</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">{avgYield.toFixed(2)}%</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Calendar className="w-4 h-4" />
                        <span className="text-sm">Monthly Income</span>
                    </div>
                    <p className="text-2xl font-bold">${(totalAnnualDividends / 12).toFixed(0)}</p>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Filter:</span>
                    </div>
                    <div className="flex gap-2">
                        {(['all', 'upcoming', 'high_yield'] as const).map(f => (
                            <button
                                key={f}
                                onClick={() => setFilter(f)}
                                className={clsx(
                                    "px-3 py-1.5 rounded-lg text-sm",
                                    filter === f ? "bg-primary/20 text-primary" : "bg-slate-800/50 text-slate-400"
                                )}
                            >
                                {f === 'all' ? 'All' : f === 'upcoming' ? 'Upcoming Ex-Date' : 'High Yield'}
                            </button>
                        ))}
                    </div>
                    <div className="flex items-center gap-2 ml-auto">
                        <span className="text-sm text-slate-400">Min Yield:</span>
                        <input
                            type="range"
                            min={0}
                            max={10}
                            step={0.5}
                            value={minYield}
                            onChange={e => setMinYield(Number(e.target.value))}
                            className="w-24"
                        />
                        <span className="text-sm font-mono w-10">{minYield}%</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Dividend Stocks */}
                <div className="lg:col-span-2">
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4">Dividend Holdings</h3>
                        {isLoading ? (
                            <div className="flex items-center justify-center py-8">
                                <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                            </div>
                        ) : stocks.length === 0 ? (
                            <div className="text-center py-8 text-slate-400">
                                <Coins className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                <p>No dividend stocks found</p>
                            </div>
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-slate-700/50">
                                            <th className="text-left p-3 text-slate-400">Symbol</th>
                                            <th className="text-right p-3 text-slate-400">Price</th>
                                            <th className="text-right p-3 text-slate-400">Yield</th>
                                            <th className="text-right p-3 text-slate-400">Annual Div</th>
                                            <th className="text-center p-3 text-slate-400">Ex-Date</th>
                                            <th className="text-center p-3 text-slate-400">Freq</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {stocks.map((stock, idx) => (
                                            <tr key={stock.symbol} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                <td className="p-3">
                                                    <div>
                                                        <span className="font-mono font-bold text-primary">{stock.symbol}</span>
                                                        <p className="text-xs text-slate-400">{stock.name}</p>
                                                    </div>
                                                </td>
                                                <td className="p-3 text-right font-mono">${(stock.price ?? 0).toFixed(2)}</td>
                                                <td className="p-3 text-right">
                                                    <span className={clsx(
                                                        "font-bold",
                                                        (stock.dividend_yield ?? 0) >= 4 ? "text-emerald-400" :
                                                        (stock.dividend_yield ?? 0) >= 2 ? "text-amber-400" : "text-slate-400"
                                                    )}>
                                                        {(stock.dividend_yield ?? 0).toFixed(2)}%
                                                    </span>
                                                </td>
                                                <td className="p-3 text-right font-mono">${(stock.annual_dividend ?? 0).toFixed(2)}</td>
                                                <td className="p-3 text-center text-xs">{stock.ex_date}</td>
                                                <td className="p-3 text-center">
                                                    <span className="px-2 py-0.5 bg-slate-700 rounded text-xs">
                                                        {stock.frequency}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>

                {/* Income Calendar */}
                <div className="space-y-4">
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Calendar className="w-5 h-5 text-emerald-400" />
                            Income Calendar
                        </h3>
                        <div className="space-y-3">
                            {income.map(month => (
                                <div key={month.month} className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-medium">{month.month}</span>
                                        <span className="font-mono font-bold text-emerald-400">
                                            ${(month.amount ?? 0).toFixed(0)}
                                        </span>
                                    </div>
                                    <div className="flex flex-wrap gap-1">
                                        {month.stocks.map(s => (
                                            <span key={s.symbol} className="text-xs px-1.5 py-0.5 bg-slate-700 rounded">
                                                {s.symbol}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Clock className="w-5 h-5 text-amber-400" />
                            Upcoming Ex-Dates
                        </h3>
                        <div className="space-y-2">
                            {stocks
                                .filter(s => s.ex_date)
                                .sort((a, b) => new Date(a.ex_date).getTime() - new Date(b.ex_date).getTime())
                                .slice(0, 5)
                                .map(stock => (
                                    <div key={stock.symbol} className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
                                        <span className="font-mono text-primary">{stock.symbol}</span>
                                        <span className="text-sm text-slate-400">{stock.ex_date}</span>
                                    </div>
                                ))
                            }
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
