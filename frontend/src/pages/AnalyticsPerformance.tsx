import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    BarChart2, RefreshCw, TrendingUp, TrendingDown, DollarSign,
    Percent, Target, Calendar, Activity, Award, AlertCircle
} from 'lucide-react'
import {
    ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, AreaChart, Area, PieChart, Pie, Cell
} from 'recharts'

interface PerformanceData {
    total_pnl: number
    total_pnl_pct: number
    win_rate: number
    total_trades: number
    winning_trades: number
    losing_trades: number
    avg_win: number
    avg_loss: number
    profit_factor: number
    max_drawdown: number
    sharpe_ratio: number
    sortino_ratio: number
    daily_pnl: { date: string; pnl: number; cumulative: number }[]
    monthly_pnl: { month: string; pnl: number }[]
    strategy_breakdown: { strategy: string; pnl: number; trades: number; win_rate: number }[]
    best_trade: { symbol: string; pnl: number; date: string }
    worst_trade: { symbol: string; pnl: number; date: string }
}

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#A855F7', '#06B6D4']

export default function AnalyticsPerformance() {
    const [timeframe, setTimeframe] = useState('30d')

    const { data: performance, isLoading, refetch } = useQuery<PerformanceData>({
        queryKey: ['analytics-performance', timeframe],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/analytics/performance?timeframe=${timeframe}`)
            return data
        },
        staleTime: 60000
    })

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
                            <BarChart2 className="w-5 h-5 text-white" />
                        </div>
                        Analytics & Performance
                    </h1>
                    <p className="page-subtitle">Track your trading performance and portfolio metrics</p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={timeframe}
                        onChange={(e) => setTimeframe(e.target.value)}
                        className="input-field w-32"
                    >
                        <option value="7d">7 Days</option>
                        <option value="30d">30 Days</option>
                        <option value="90d">90 Days</option>
                        <option value="1y">1 Year</option>
                        <option value="all">All Time</option>
                    </select>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <DollarSign className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Total P&L</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        (performance?.total_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                        ${performance?.total_pnl.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? '0.00'}
                    </div>
                    <div className={`text-sm ${
                        (performance?.total_pnl_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                        {(performance?.total_pnl_pct ?? 0) >= 0 ? '+' : ''}{(performance?.total_pnl_pct ?? 0).toFixed(2)}%
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Percent className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Win Rate</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        (performance?.win_rate ?? 0) >= 50 ? 'text-emerald-400' : 'text-amber-400'
                    }`}>
                        {performance?.win_rate.toFixed(1) ?? 0}%
                    </div>
                    <div className="text-sm text-slate-400">
                        {performance?.winning_trades ?? 0}W / {performance?.losing_trades ?? 0}L
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Total Trades</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {performance?.total_trades ?? 0}
                    </div>
                    <div className="text-sm text-slate-400">Executed</div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Target className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Profit Factor</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        (performance?.profit_factor ?? 0) >= 1.5 ? 'text-emerald-400' :
                        (performance?.profit_factor ?? 0) >= 1 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                        {performance?.profit_factor.toFixed(2) ?? '0.00'}
                    </div>
                    <div className="text-sm text-slate-400">Gross Win/Loss</div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <TrendingDown className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Max Drawdown</span>
                    </div>
                    <div className="text-2xl font-bold text-red-400">
                        {performance?.max_drawdown.toFixed(2) ?? 0}%
                    </div>
                    <div className="text-sm text-slate-400">Peak to trough</div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Award className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Sharpe Ratio</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        (performance?.sharpe_ratio ?? 0) >= 1 ? 'text-emerald-400' : 'text-amber-400'
                    }`}>
                        {performance?.sharpe_ratio.toFixed(2) ?? '0.00'}
                    </div>
                    <div className="text-sm text-slate-400">Risk-adjusted</div>
                </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Cumulative P&L */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">Cumulative P&L</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <AreaChart data={performance?.daily_pnl || []}>
                                <defs>
                                    <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                                <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v}`} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '0.5rem'
                                    }}
                                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="cumulative"
                                    stroke="#10B981"
                                    fill="url(#pnlGradient)"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Monthly P&L */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">Monthly P&L</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <ComposedChart data={performance?.monthly_pnl || []}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="month" stroke="#94a3b8" fontSize={12} />
                                <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v}`} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '0.5rem'
                                    }}
                                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
                                />
                                <Bar
                                    dataKey="pnl"
                                    fill="#3B82F6"
                                    radius={[4, 4, 0, 0]}
                                />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Strategy Breakdown */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Strategy Performance Table */}
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50">
                        <h3 className="text-lg font-semibold text-white">Strategy Breakdown</h3>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Strategy</th>
                                    <th>P&L</th>
                                    <th>Trades</th>
                                    <th>Win Rate</th>
                                </tr>
                            </thead>
                            <tbody>
                                {performance?.strategy_breakdown?.map((strategy, idx) => (
                                    <tr key={strategy.strategy}>
                                        <td>
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className="w-3 h-3 rounded-full"
                                                    style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                                                />
                                                <span className="font-medium text-white">{strategy.strategy}</span>
                                            </div>
                                        </td>
                                        <td className={strategy.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                            ${strategy.pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                        </td>
                                        <td className="text-slate-300">{strategy.trades}</td>
                                        <td className={strategy.win_rate >= 50 ? 'text-emerald-400' : 'text-amber-400'}>
                                            {strategy.win_rate.toFixed(1)}%
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Strategy Pie Chart */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">P&L Distribution</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <PieChart>
                                <Pie
                                    data={performance?.strategy_breakdown?.filter(s => s.pnl > 0) || []}
                                    dataKey="pnl"
                                    nameKey="strategy"
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    label={({ name, value }) => `${name}: $${value.toFixed(0)}`}
                                    labelLine={false}
                                >
                                    {performance?.strategy_breakdown?.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    formatter={(value: number) => `$${value.toFixed(2)}`}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Best/Worst Trades */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-emerald-400" />
                        Best Trade
                    </h3>
                    {performance?.best_trade ? (
                        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="text-xl font-bold text-white">{performance.best_trade.symbol}</div>
                                    <div className="text-sm text-slate-400">{performance.best_trade.date}</div>
                                </div>
                                <div className="text-2xl font-bold text-emerald-400">
                                    +${performance.best_trade.pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-slate-400 py-4">No trades yet</div>
                    )}
                </div>

                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <TrendingDown className="w-5 h-5 text-red-400" />
                        Worst Trade
                    </h3>
                    {performance?.worst_trade ? (
                        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="text-xl font-bold text-white">{performance.worst_trade.symbol}</div>
                                    <div className="text-sm text-slate-400">{performance.worst_trade.date}</div>
                                </div>
                                <div className="text-2xl font-bold text-red-400">
                                    ${performance.worst_trade.pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-slate-400 py-4">No trades yet</div>
                    )}
                </div>
            </div>

            {/* Additional Metrics */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Additional Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="text-slate-400 text-sm mb-1">Avg Win</div>
                        <div className="text-xl font-bold text-emerald-400">
                            ${performance?.avg_win.toFixed(2) ?? '0.00'}
                        </div>
                    </div>
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="text-slate-400 text-sm mb-1">Avg Loss</div>
                        <div className="text-xl font-bold text-red-400">
                            ${performance?.avg_loss.toFixed(2) ?? '0.00'}
                        </div>
                    </div>
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="text-slate-400 text-sm mb-1">Sortino Ratio</div>
                        <div className={`text-xl font-bold ${
                            (performance?.sortino_ratio ?? 0) >= 1 ? 'text-emerald-400' : 'text-amber-400'
                        }`}>
                            {performance?.sortino_ratio.toFixed(2) ?? '0.00'}
                        </div>
                    </div>
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="text-slate-400 text-sm mb-1">Win/Loss Ratio</div>
                        <div className="text-xl font-bold text-white">
                            {performance?.avg_win && performance?.avg_loss
                                ? (Math.abs(performance.avg_win / performance.avg_loss)).toFixed(2)
                                : '0.00'}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
