import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    History, Play, RefreshCw, TrendingUp, TrendingDown, DollarSign,
    Percent, BarChart3, Calendar, Settings, Target
} from 'lucide-react'
import clsx from 'clsx'

interface BacktestResult {
    strategy_name: string
    start_date: string
    end_date: string
    initial_capital: number
    final_capital: number
    total_return: number
    cagr: number
    max_drawdown: number
    sharpe_ratio: number
    sortino_ratio: number
    win_rate: number
    total_trades: number
    profit_factor: number
    avg_trade: number
    best_trade: number
    worst_trade: number
    trades: {
        date: string
        symbol: string
        type: 'buy' | 'sell'
        price: number
        pnl: number
    }[]
    equity_curve: { date: string; value: number }[]
}

const STRATEGIES = [
    { id: 'csp', name: 'Cash Secured Puts', description: 'Sell puts on quality stocks' },
    { id: 'wheel', name: 'Wheel Strategy', description: 'CSP + CC rotation' },
    { id: 'momentum', name: 'Momentum', description: 'Buy high RS stocks' },
    { id: 'mean-reversion', name: 'Mean Reversion', description: 'Buy oversold, sell overbought' },
]

export default function Backtesting() {
    const [strategy, setStrategy] = useState('csp')
    const [symbols, setSymbols] = useState('AAPL,MSFT,NVDA,AMD,TSLA')
    const [startDate, setStartDate] = useState('2023-01-01')
    const [endDate, setEndDate] = useState('2024-01-01')
    const [initialCapital, setInitialCapital] = useState(100000)
    const [result, setResult] = useState<BacktestResult | null>(null)

    const backtestMutation = useMutation({
        mutationFn: async (params: any) => {
            const { data } = await axiosInstance.post('/strategy/backtest', params)
            return data as BacktestResult
        },
        onSuccess: (data) => setResult(data)
    })

    const handleRun = () => {
        backtestMutation.mutate({
            strategy,
            symbols: symbols.split(',').map(s => s.trim()),
            start_date: startDate,
            end_date: endDate,
            initial_capital: initialCapital
        })
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header>
                <h1 className="page-title flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-yellow-600 flex items-center justify-center shadow-lg">
                        <History className="w-5 h-5 text-white" />
                    </div>
                    Strategy Backtesting
                </h1>
                <p className="page-subtitle">Test trading strategies against historical data</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Configuration */}
                <div className="space-y-4">
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Settings className="w-5 h-5 text-slate-400" />
                            Configuration
                        </h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Strategy</label>
                                <select
                                    value={strategy}
                                    onChange={e => setStrategy(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                >
                                    {STRATEGIES.map(s => (
                                        <option key={s.id} value={s.id}>{s.name}</option>
                                    ))}
                                </select>
                                <p className="text-xs text-slate-500 mt-1">
                                    {STRATEGIES.find(s => s.id === strategy)?.description}
                                </p>
                            </div>

                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Symbols</label>
                                <input
                                    type="text"
                                    value={symbols}
                                    onChange={e => setSymbols(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    placeholder="AAPL,MSFT,..."
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className="block text-sm text-slate-400 mb-2">Start Date</label>
                                    <input
                                        type="date"
                                        value={startDate}
                                        onChange={e => setStartDate(e.target.value)}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm text-slate-400 mb-2">End Date</label>
                                    <input
                                        type="date"
                                        value={endDate}
                                        onChange={e => setEndDate(e.target.value)}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Initial Capital</label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
                                    <input
                                        type="number"
                                        value={initialCapital}
                                        onChange={e => setInitialCapital(Number(e.target.value))}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-8 pr-3 py-2"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={handleRun}
                                disabled={backtestMutation.isPending}
                                className="w-full bg-primary hover:bg-primary/80 disabled:opacity-50 px-4 py-3 rounded-lg font-semibold flex items-center justify-center gap-2"
                            >
                                {backtestMutation.isPending ? (
                                    <>
                                        <RefreshCw className="w-5 h-5 animate-spin" />
                                        Running Backtest...
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-5 h-5" />
                                        Run Backtest
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Results */}
                <div className="lg:col-span-2 space-y-4">
                    {result ? (
                        <>
                            {/* Performance Summary */}
                            <div className="card p-4">
                                <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Total Return</p>
                                        <p className={clsx(
                                            "text-xl font-bold",
                                            result.total_return >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            {result.total_return >= 0 ? '+' : ''}{result.total_return.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">CAGR</p>
                                        <p className={clsx(
                                            "text-xl font-bold",
                                            result.cagr >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            {result.cagr.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Max Drawdown</p>
                                        <p className="text-xl font-bold text-rose-400">
                                            {result.max_drawdown.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Sharpe Ratio</p>
                                        <p className={clsx(
                                            "text-xl font-bold",
                                            result.sharpe_ratio >= 1 ? "text-emerald-400" : "text-amber-400"
                                        )}>
                                            {result.sharpe_ratio.toFixed(2)}
                                        </p>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Win Rate</p>
                                        <p className="text-xl font-bold">{result.win_rate.toFixed(0)}%</p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Total Trades</p>
                                        <p className="text-xl font-bold">{result.total_trades}</p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Profit Factor</p>
                                        <p className={clsx(
                                            "text-xl font-bold",
                                            result.profit_factor >= 1.5 ? "text-emerald-400" : "text-amber-400"
                                        )}>
                                            {result.profit_factor.toFixed(2)}
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/50 rounded-lg p-3">
                                        <p className="text-xs text-slate-400">Avg Trade</p>
                                        <p className={clsx(
                                            "text-xl font-bold",
                                            result.avg_trade >= 0 ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            ${result.avg_trade.toFixed(0)}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Capital Growth */}
                            <div className="card p-4">
                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                    <DollarSign className="w-5 h-5 text-emerald-400" />
                                    Capital Growth
                                </h3>
                                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
                                    <div>
                                        <p className="text-sm text-slate-400">Initial</p>
                                        <p className="text-2xl font-bold">${result.initial_capital.toLocaleString()}</p>
                                    </div>
                                    <TrendingUp className={clsx(
                                        "w-8 h-8",
                                        result.final_capital >= result.initial_capital ? "text-emerald-400" : "text-rose-400"
                                    )} />
                                    <div className="text-right">
                                        <p className="text-sm text-slate-400">Final</p>
                                        <p className={clsx(
                                            "text-2xl font-bold",
                                            result.final_capital >= result.initial_capital ? "text-emerald-400" : "text-rose-400"
                                        )}>
                                            ${result.final_capital.toLocaleString()}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Trade History */}
                            <div className="card p-4">
                                <h3 className="text-lg font-semibold mb-4">Recent Trades</h3>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b border-slate-700/50">
                                                <th className="text-left p-2 text-slate-400">Date</th>
                                                <th className="text-left p-2 text-slate-400">Symbol</th>
                                                <th className="text-center p-2 text-slate-400">Type</th>
                                                <th className="text-right p-2 text-slate-400">Price</th>
                                                <th className="text-right p-2 text-slate-400">P/L</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {result.trades.slice(0, 10).map((trade, idx) => (
                                                <tr key={idx} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                    <td className="p-2">{trade.date}</td>
                                                    <td className="p-2 font-mono text-primary">{trade.symbol}</td>
                                                    <td className="p-2 text-center">
                                                        <span className={clsx(
                                                            "px-2 py-0.5 rounded text-xs",
                                                            trade.type === 'buy' ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"
                                                        )}>
                                                            {trade.type.toUpperCase()}
                                                        </span>
                                                    </td>
                                                    <td className="p-2 text-right font-mono">${trade.price.toFixed(2)}</td>
                                                    <td className={clsx(
                                                        "p-2 text-right font-mono font-bold",
                                                        trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                                    )}>
                                                        {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(0)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="card p-8 text-center text-slate-400">
                            <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                            <p>Configure your strategy and run a backtest</p>
                            <p className="text-sm mt-1">Results will appear here</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
