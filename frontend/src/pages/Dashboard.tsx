import { useState, memo } from 'react'
import { useDashboardSummary, usePositions, useLiveGames, useUpcomingGames, usePerformanceHistory } from '../hooks/useMagnusApi'
import { AlertCircle, TrendingUp, TrendingDown, DollarSign, Briefcase, Activity, RefreshCw, Zap, ArrowUpRight, ArrowDownRight, Clock, Target } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Link } from 'react-router-dom'

const COLORS = ['#10B981', '#3B82F6', '#8B5CF6']

export function Dashboard() {
    const [selectedPeriod, setSelectedPeriod] = useState('1W')
    const { data: summary, isLoading: summaryLoading, error: summaryError, refetch } = useDashboardSummary()
    const { data: positions, isLoading: positionsLoading } = usePositions()
    const { data: liveGames } = useLiveGames()
    const { data: upcomingGames } = useUpcomingGames(5)
    const { data: performanceData } = usePerformanceHistory(selectedPeriod)

    // Use real performance data or fallback to empty array
    const chartData = performanceData?.history || []

    const isLoading = summaryLoading || positionsLoading

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-slate-400 text-sm">Loading your dashboard...</p>
                </div>
            </div>
        )
    }

    if (summaryError) {
        return (
            <div className="glass-card p-6 border-red-500/30 bg-red-500/5">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-red-400">Connection Error</h3>
                        <p className="text-sm text-slate-400">Failed to load dashboard data. Ensure the backend is running.</p>
                    </div>
                </div>
            </div>
        )
    }

    const allocations = summary?.allocations || { stocks: 0, options: 0, cash: 100 }
    const allocationData = [
        { name: 'Stocks', value: allocations.stocks || 0 },
        { name: 'Options', value: allocations.options || 0 },
        { name: 'Cash', value: allocations.cash || 0 },
    ].filter(d => d.value > 0)

    const stockPositions = positions?.stocks || []
    const optionPositions = positions?.options || []
    const totalPositions = stockPositions.length + optionPositions.length

    return (
        <div className="space-y-8">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title">Dashboard</h1>
                    <p className="page-subtitle">Welcome back. Here's your portfolio overview.</p>
                </div>
                <button
                    onClick={() => refetch()}
                    className="btn-icon flex items-center gap-2 px-4"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span className="text-sm font-medium">Refresh</span>
                </button>
            </header>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
                <StatCard
                    title="Portfolio Value"
                    value={formatCurrency(summary?.total_value || 0)}
                    change={summary?.day_change_pct || 0}
                    icon={Briefcase}
                    iconColor="text-primary"
                    iconBg="bg-primary/10"
                />
                <StatCard
                    title="Day's P&L"
                    value={formatCurrency(summary?.day_change || 0)}
                    change={summary?.day_change_pct || 0}
                    icon={(summary?.day_change || 0) >= 0 ? TrendingUp : TrendingDown}
                    iconColor={(summary?.day_change || 0) >= 0 ? "text-emerald-400" : "text-red-400"}
                    iconBg={(summary?.day_change || 0) >= 0 ? "bg-emerald-500/10" : "bg-red-500/10"}
                />
                <StatCard
                    title="Buying Power"
                    value={formatCurrency(summary?.buying_power || 0)}
                    subtitle="Available"
                    icon={DollarSign}
                    iconColor="text-blue-400"
                    iconBg="bg-blue-500/10"
                />
                <StatCard
                    title="Active Positions"
                    value={String(totalPositions)}
                    subtitle={`${optionPositions.length} options`}
                    icon={Activity}
                    iconColor="text-purple-400"
                    iconBg="bg-purple-500/10"
                />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Performance Chart */}
                <div className="xl:col-span-2 glass-card p-6">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h3 className="text-lg font-semibold text-white">Performance</h3>
                            <p className="text-sm text-slate-400">Portfolio value over time</p>
                        </div>
                        <div className="tabs">
                            {['1W', '1M', '3M', '1Y'].map(period => (
                                <button
                                    key={period}
                                    onClick={() => setSelectedPeriod(period)}
                                    className={`tab ${selectedPeriod === period ? 'active' : ''}`}
                                >
                                    {period}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="h-72 w-full" style={{ minHeight: 288, minWidth: 200 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData}>
                                <defs>
                                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                                        <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#64748b"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                />
                                <YAxis
                                    stroke="#64748b"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '12px',
                                        boxShadow: '0 10px 40px -10px rgba(0,0,0,0.5)'
                                    }}
                                    formatter={(value: number) => [`$${value.toLocaleString()}`, 'Value']}
                                    labelStyle={{ color: '#94a3b8' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#10B981"
                                    strokeWidth={2}
                                    fill="url(#colorValue)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Allocation Chart */}
                <div className="glass-card p-6">
                    <div className="mb-4">
                        <h3 className="text-lg font-semibold text-white">Allocation</h3>
                        <p className="text-sm text-slate-400">Portfolio breakdown</p>
                    </div>
                    <div className="h-52 w-full" style={{ minHeight: 208, minWidth: 200 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={allocationData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={55}
                                    outerRadius={80}
                                    paddingAngle={4}
                                    dataKey="value"
                                    strokeWidth={0}
                                >
                                    {allocationData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '12px'
                                    }}
                                    formatter={(value: number) => `${value.toFixed(1)}%`}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="space-y-3 mt-4">
                        {allocationData.map((item, index) => (
                            <div key={item.name} className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index] }} />
                                    <span className="text-sm text-slate-300">{item.name}</span>
                                </div>
                                <span className="text-sm font-medium text-white">{item.value.toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Bottom Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Top Positions */}
                <div className="glass-card p-6">
                    <div className="flex items-center justify-between mb-5">
                        <div>
                            <h3 className="text-lg font-semibold text-white">Top Positions</h3>
                            <p className="text-sm text-slate-400">Your largest holdings</p>
                        </div>
                        <Link to="/positions" className="text-sm text-primary hover:text-primary/80 font-medium flex items-center gap-1">
                            View all
                            <ArrowUpRight className="w-4 h-4" />
                        </Link>
                    </div>
                    {totalPositions === 0 ? (
                        <div className="text-center py-10">
                            <div className="w-14 h-14 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-4">
                                <Briefcase className="w-6 h-6 text-slate-500" />
                            </div>
                            <p className="text-slate-400 text-sm">No positions found</p>
                            <p className="text-slate-500 text-xs mt-1">Connect to Robinhood to sync</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {[...stockPositions, ...optionPositions].slice(0, 5).map((pos: Position | OptionPosition, idx: number) => (
                                <div key={idx} className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 hover:bg-slate-800/50 transition-colors">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-xl bg-slate-700/50 flex items-center justify-center font-bold text-sm text-white">
                                            {pos.symbol.slice(0, 2)}
                                        </div>
                                        <div>
                                            <span className="font-semibold text-white">{pos.symbol}</span>
                                            <p className="text-xs text-slate-400 mt-0.5">
                                                {'strategy' in pos ? pos.strategy : pos.type}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="font-semibold text-white">{formatCurrency(pos.current_value || 0)}</div>
                                        <div className={`text-sm flex items-center justify-end gap-1 ${(pos.pl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {(pos.pl || 0) >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                            {formatPercent(pos.pl_pct || 0)}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Live Games */}
                <div className="glass-card p-6">
                    <div className="flex items-center justify-between mb-5">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
                                <Zap className="w-5 h-5 text-amber-400" />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-white">Sports</h3>
                                <p className="text-sm text-slate-400">Live & upcoming games</p>
                            </div>
                        </div>
                        <Link to="/betting" className="text-sm text-primary hover:text-primary/80 font-medium flex items-center gap-1">
                            Betting Hub
                            <ArrowUpRight className="w-4 h-4" />
                        </Link>
                    </div>
                    {(!liveGames || liveGames.length === 0) ? (
                        <div>
                            <div className="text-center py-6 mb-4">
                                <p className="text-slate-400 text-sm">No live games right now</p>
                            </div>
                            {upcomingGames && upcomingGames.length > 0 && (
                                <div>
                                    <div className="flex items-center gap-2 mb-3">
                                        <Clock className="w-4 h-4 text-slate-500" />
                                        <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Upcoming</span>
                                    </div>
                                    <div className="space-y-2">
                                        {upcomingGames.slice(0, 3).map((game: Game, idx: number) => (
                                            <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30">
                                                <span className="text-sm text-slate-300">{game.away_team} @ {game.home_team}</span>
                                                <span className="text-xs text-slate-500">{game.game_time}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {liveGames.slice(0, 4).map((game: Game, idx: number) => (
                                <div key={idx} className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30">
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="live-indicator">LIVE</span>
                                        </div>
                                        <div className="font-medium text-white">{game.away_team} @ {game.home_team}</div>
                                        <div className="text-xs text-slate-500">{game.league}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-2xl font-bold text-white">
                                            {game.away_score || 0} - {game.home_score || 0}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <QuickAction to="/scanner" icon={Target} label="Premium Scanner" color="emerald" />
                <QuickAction to="/research" icon={TrendingUp} label="Research" color="blue" />
                <QuickAction to="/chat" icon={Zap} label="Ask AVA" color="purple" />
                <QuickAction to="/betting" icon={Activity} label="Betting Hub" color="amber" />
            </div>
        </div>
    )
}

// Types
interface Position {
    symbol: string
    type: string
    current_value: number
    pl: number
    pl_pct: number
}

interface OptionPosition {
    symbol: string
    strategy: string
    current_value: number
    pl: number
    pl_pct: number
}

interface Game {
    away_team: string
    home_team: string
    league: string
    game_time: string
    away_score?: number
    home_score?: number
}

// Helper functions
function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value)
}

function formatPercent(value: number): string {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
}

// Stat Card Component
interface StatCardProps {
    title: string
    value: string
    change?: number
    subtitle?: string
    icon: React.ComponentType<{ className?: string }>
    iconColor: string
    iconBg: string
}

const StatCard = memo(function StatCard({ title, value, change, subtitle, icon: Icon, iconColor, iconBg }: StatCardProps) {
    return (
        <div className="stat-card group">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm text-slate-400 mb-1">{title}</p>
                    <p className="text-2xl font-bold text-white">{value}</p>
                    {change !== undefined && (
                        <div className={`flex items-center gap-1 mt-2 text-sm font-medium ${change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {change >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                            {formatPercent(change)}
                        </div>
                    )}
                    {subtitle && (
                        <p className="text-sm text-slate-500 mt-2">{subtitle}</p>
                    )}
                </div>
                <div className={`w-12 h-12 rounded-xl ${iconBg} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                    <Icon className={`w-6 h-6 ${iconColor}`} />
                </div>
            </div>
        </div>
    )
})

// Quick Action Component - colorMap moved outside for performance
const QUICK_ACTION_COLORS = {
    emerald: 'from-emerald-500/10 to-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/40 text-emerald-400',
    blue: 'from-blue-500/10 to-blue-500/5 border-blue-500/20 hover:border-blue-500/40 text-blue-400',
    purple: 'from-purple-500/10 to-purple-500/5 border-purple-500/20 hover:border-purple-500/40 text-purple-400',
    amber: 'from-amber-500/10 to-amber-500/5 border-amber-500/20 hover:border-amber-500/40 text-amber-400',
} as const

interface QuickActionProps {
    to: string
    icon: React.ComponentType<{ className?: string }>
    label: string
    color: keyof typeof QUICK_ACTION_COLORS
}

const QuickAction = memo(function QuickAction({ to, icon: Icon, label, color }: QuickActionProps) {
    return (
        <Link
            to={to}
            className={`flex items-center gap-3 p-4 rounded-xl bg-gradient-to-br border transition-all duration-200 hover:scale-[1.02] ${QUICK_ACTION_COLORS[color]}`}
        >
            <Icon className="w-5 h-5" />
            <span className="font-medium text-sm text-white">{label}</span>
        </Link>
    )
})
