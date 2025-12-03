import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Smartphone, RefreshCw, Users, Activity, TrendingUp, TrendingDown,
    DollarSign, Percent, CheckCircle, XCircle,
    Plus, Settings, History, BarChart3, Play, Pause,
    Bell, Send, Trash2, RotateCcw
} from 'lucide-react'
import clsx from 'clsx'

// Types
interface XTradesTrade {
    id: number
    profile_id: number
    profile_name: string
    ticker: string
    strategy: string
    entry_price: number
    exit_price?: number
    strike_price?: number
    expiration_date: string
    quantity: number
    status: 'open' | 'closed' | 'expired'
    pnl?: number
    pnl_percent?: number
    alert_timestamp: string
    exit_date?: string
    days_open?: number
}

interface XTradesProfile {
    id: number
    username: string
    display_name?: string
    active: boolean
    last_sync?: string
    last_sync_status?: string
    total_trades_scraped: number
    added_date: string
    notes?: string
}

interface SyncLog {
    id: number
    sync_timestamp: string
    profiles_synced: number
    trades_found: number
    new_trades: number
    updated_trades: number
    duration_seconds: number
    status: 'success' | 'partial' | 'failed' | 'running'
    errors?: string
}

interface XTradesStats {
    total_profiles: number
    total_trades: number
    total_pnl: number
    win_rate: number
}

interface ProfileStats {
    profile_id: number
    profile_name: string
    total_trades: number
    open_trades: number
    closed_trades: number
    total_pnl: number
    win_rate: number
    avg_pnl: number
}

interface StrategyStats {
    strategy: string
    total_trades: number
    total_pnl: number
    avg_pnl: number
    win_rate: number
    best_trade: number
    worst_trade: number
}

const TABS = [
    { id: 'active', label: 'Active Trades', icon: Activity },
    { id: 'closed', label: 'Closed Trades', icon: CheckCircle },
    { id: 'analytics', label: 'Performance Analytics', icon: BarChart3 },
    { id: 'profiles', label: 'Manage Profiles', icon: Users },
    { id: 'history', label: 'Sync History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings },
]

const STRATEGIES = ['All', 'CSP', 'CC', 'Long Call', 'Long Put', 'Put Credit Spread', 'Call Credit Spread']
const SORT_OPTIONS = ['Date (Newest)', 'Date (Oldest)', 'Ticker', 'P/L']

export default function XTradesWatchlists() {
    const [activeTab, setActiveTab] = useState('active')
    const queryClient = useQueryClient()

    // Filters state
    const [profileFilter, setProfileFilter] = useState('All')
    const [strategyFilter, setStrategyFilter] = useState('All')
    const [tickerFilter, setTickerFilter] = useState('All')
    const [sortBy, setSortBy] = useState('Date (Newest)')

    // Add profile form
    const [newUsername, setNewUsername] = useState('')
    const [newDisplayName, setNewDisplayName] = useState('')

    // Settings state
    const [syncInterval, setSyncInterval] = useState('15 minutes')
    const [maxAlerts, setMaxAlerts] = useState(50)
    const [telegramEnabled, setTelegramEnabled] = useState(true)
    const [headlessMode, setHeadlessMode] = useState(true)

    // Queries
    const { data: activeTrades, isLoading: loadingActive, refetch: refetchActive } = useQuery<XTradesTrade[]>({
        queryKey: ['xtrades-active'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/trades', { params: { status: 'open' } })
            return data.trades || []
        },
        staleTime: 60000,
    })

    const { data: closedTrades, isLoading: loadingClosed, refetch: refetchClosed } = useQuery<XTradesTrade[]>({
        queryKey: ['xtrades-closed'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/trades', { params: { status: 'closed' } })
            return data.trades || []
        },
        staleTime: 60000,
    })

    const { data: profiles, isLoading: loadingProfiles, refetch: refetchProfiles } = useQuery<XTradesProfile[]>({
        queryKey: ['xtrades-profiles'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/profiles')
            return data.profiles || []
        },
        staleTime: 300000,
    })

    const { data: stats } = useQuery<XTradesStats>({
        queryKey: ['xtrades-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/stats')
            return data
        },
        staleTime: 300000,
    })

    const { data: syncHistory } = useQuery<SyncLog[]>({
        queryKey: ['xtrades-sync-history'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/sync/history')
            return data.logs || []
        },
        staleTime: 120000,
    })

    const { data: profileStats } = useQuery<ProfileStats[]>({
        queryKey: ['xtrades-profile-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/profiles/stats/all')
            return data.profiles || []
        },
        staleTime: 300000,
    })

    const { data: strategyStats } = useQuery<StrategyStats[]>({
        queryKey: ['xtrades-strategy-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/xtrades/stats/by-strategy')
            return data.strategies || []
        },
        staleTime: 300000,
    })

    // Mutations
    const addProfileMutation = useMutation({
        mutationFn: async (data: { username: string, display_name?: string }) => {
            const response = await axiosInstance.post('/xtrades/profiles', data)
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['xtrades-profiles'] })
            setNewUsername('')
            setNewDisplayName('')
        },
    })

    const toggleProfileMutation = useMutation({
        mutationFn: async ({ id, active }: { id: number, active: boolean }) => {
            const response = await axiosInstance.patch(`/xtrades/profiles/${id}`, { active })
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['xtrades-profiles'] })
        },
    })

    const syncProfileMutation = useMutation({
        mutationFn: async (profileId: number) => {
            const response = await axiosInstance.post(`/xtrades/profiles/${profileId}/sync`)
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['xtrades-active'] })
            queryClient.invalidateQueries({ queryKey: ['xtrades-closed'] })
            queryClient.invalidateQueries({ queryKey: ['xtrades-sync-history'] })
        },
    })

    // Filter and sort trades
    const filterTrades = (trades: XTradesTrade[]) => {
        let filtered = [...trades]

        if (profileFilter !== 'All') {
            filtered = filtered.filter(t => t.profile_name === profileFilter)
        }
        if (strategyFilter !== 'All') {
            filtered = filtered.filter(t => t.strategy === strategyFilter)
        }
        if (tickerFilter !== 'All') {
            filtered = filtered.filter(t => t.ticker === tickerFilter)
        }

        // Sort
        filtered.sort((a, b) => {
            if (sortBy === 'Date (Newest)') return new Date(b.alert_timestamp).getTime() - new Date(a.alert_timestamp).getTime()
            if (sortBy === 'Date (Oldest)') return new Date(a.alert_timestamp).getTime() - new Date(b.alert_timestamp).getTime()
            if (sortBy === 'Ticker') return a.ticker.localeCompare(b.ticker)
            if (sortBy === 'P/L') return (b.pnl || 0) - (a.pnl || 0)
            return 0
        })

        return filtered
    }

    const filteredActive = filterTrades(activeTrades || [])
    const filteredClosed = filterTrades(closedTrades || [])

    // Computed stats
    const uniqueProfiles = [...new Set(activeTrades?.map(t => t.profile_name) || [])]
    const uniqueTickers = [...new Set(activeTrades?.map(t => t.ticker) || [])]
    const totalContracts = activeTrades?.reduce((acc, t) => acc + (t.quantity || 1), 0) || 0

    const closedStats = {
        totalPL: closedTrades?.reduce((acc, t) => acc + (t.pnl || 0), 0) || 0,
        winRate: closedTrades?.length ?
            (closedTrades.filter(t => (t.pnl || 0) > 0).length / closedTrades.length * 100) : 0,
        avgPL: closedTrades?.length ?
            (closedTrades.reduce((acc, t) => acc + (t.pnl || 0), 0) / closedTrades.length) : 0,
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <Smartphone className="w-5 h-5 text-white" />
                        </div>
                        XTrades Watchlists
                    </h1>
                    <p className="page-subtitle">Monitor option trades from Discord-connected Xtrades.net profiles</p>
                </div>
                <button
                    onClick={() => {
                        refetchActive()
                        refetchClosed()
                        refetchProfiles()
                    }}
                    className="btn-icon"
                >
                    <RefreshCw className={clsx("w-5 h-5", (loadingActive || loadingClosed) && "animate-spin")} />
                </button>
            </header>

            {/* Tab Navigation */}
            <div className="flex gap-2 overflow-x-auto pb-2">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={clsx(
                            "flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all",
                            activeTab === tab.id
                                ? "bg-primary text-white"
                                : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50"
                        )}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Active Trades Tab */}
            {activeTab === 'active' && (
                <div className="space-y-4">
                    {/* Summary Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <Activity className="w-4 h-4" />
                                <span className="text-sm">Total Active</span>
                            </div>
                            <p className="text-2xl font-bold">{activeTrades?.length || 0}</p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <Users className="w-4 h-4" />
                                <span className="text-sm">Unique Profiles</span>
                            </div>
                            <p className="text-2xl font-bold">{uniqueProfiles.length}</p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <BarChart3 className="w-4 h-4" />
                                <span className="text-sm">Unique Tickers</span>
                            </div>
                            <p className="text-2xl font-bold">{uniqueTickers.length}</p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <DollarSign className="w-4 h-4" />
                                <span className="text-sm">Total Contracts</span>
                            </div>
                            <p className="text-2xl font-bold">{totalContracts}</p>
                        </div>
                    </div>

                    {/* Filters */}
                    <div className="card p-4">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Profile</label>
                                <select
                                    value={profileFilter}
                                    onChange={e => setProfileFilter(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                                >
                                    <option value="All">All Profiles</option>
                                    {profiles?.filter(p => p.active).map(p => (
                                        <option key={p.id} value={p.display_name || p.username}>
                                            {p.display_name || p.username}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Strategy</label>
                                <select
                                    value={strategyFilter}
                                    onChange={e => setStrategyFilter(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                                >
                                    {STRATEGIES.map(s => (
                                        <option key={s} value={s}>{s}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Ticker</label>
                                <select
                                    value={tickerFilter}
                                    onChange={e => setTickerFilter(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                                >
                                    <option value="All">All Tickers</option>
                                    {uniqueTickers.sort().map(t => (
                                        <option key={t} value={t}>{t}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Sort By</label>
                                <select
                                    value={sortBy}
                                    onChange={e => setSortBy(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                                >
                                    {SORT_OPTIONS.map(s => (
                                        <option key={s} value={s}>{s}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Trades Table */}
                    {loadingActive ? (
                        <div className="card p-8 flex items-center justify-center">
                            <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                            <span className="ml-2 text-slate-400">Loading trades...</span>
                        </div>
                    ) : filteredActive.length === 0 ? (
                        <div className="card p-8 text-center text-slate-400">
                            <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                            <p>No active trades found</p>
                            <p className="text-sm mt-1">Sync profiles to load trades</p>
                        </div>
                    ) : (
                        <div className="card overflow-hidden">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-slate-700/50">
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Profile</th>
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Ticker</th>
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Strategy</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">Entry</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">Strike</th>
                                        <th className="text-center p-4 text-sm font-medium text-slate-400">Expiration</th>
                                        <th className="text-center p-4 text-sm font-medium text-slate-400">Days Open</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">Est P/L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredActive.map((trade, idx) => (
                                        <tr
                                            key={trade.id}
                                            className={clsx(
                                                "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                                idx % 2 === 0 && "bg-slate-800/20"
                                            )}
                                        >
                                            <td className="p-4 text-sm">{trade.profile_name}</td>
                                            <td className="p-4">
                                                <span className="font-mono font-semibold text-primary">{trade.ticker}</span>
                                            </td>
                                            <td className="p-4">
                                                <span className="px-2 py-1 rounded text-xs bg-slate-700/50">
                                                    {trade.strategy}
                                                </span>
                                            </td>
                                            <td className="p-4 text-right font-mono">
                                                ${trade.entry_price?.toFixed(2) || 'N/A'}
                                            </td>
                                            <td className="p-4 text-right font-mono">
                                                ${trade.strike_price?.toFixed(0) || 'N/A'}
                                            </td>
                                            <td className="p-4 text-center text-sm">{trade.expiration_date}</td>
                                            <td className="p-4 text-center">
                                                <span className={clsx(
                                                    "px-2 py-1 rounded text-xs font-bold",
                                                    (trade.days_open || 0) <= 3 ? "bg-emerald-500/20 text-emerald-400" :
                                                    (trade.days_open || 0) <= 7 ? "bg-amber-500/20 text-amber-400" :
                                                    "bg-rose-500/20 text-rose-400"
                                                )}>
                                                    {trade.days_open || 0}d
                                                </span>
                                            </td>
                                            <td className={clsx(
                                                "p-4 text-right font-mono font-bold",
                                                (trade.pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                            )}>
                                                ${(trade.pnl || 0).toFixed(2)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* Closed Trades Tab */}
            {activeTab === 'closed' && (
                <div className="space-y-4">
                    {/* Summary Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <CheckCircle className="w-4 h-4" />
                                <span className="text-sm">Total Closed</span>
                            </div>
                            <p className="text-2xl font-bold">{closedTrades?.length || 0}</p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <DollarSign className="w-4 h-4" />
                                <span className="text-sm">Total P/L</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                closedStats.totalPL >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                ${(closedStats.totalPL ?? 0).toFixed(2)}
                            </p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <Percent className="w-4 h-4" />
                                <span className="text-sm">Win Rate</span>
                            </div>
                            <p className="text-2xl font-bold text-amber-400">{(closedStats.winRate ?? 0).toFixed(1)}%</p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <TrendingUp className="w-4 h-4" />
                                <span className="text-sm">Avg P/L</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                closedStats.avgPL >= 0 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                ${(closedStats.avgPL ?? 0).toFixed(2)}
                            </p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <BarChart3 className="w-4 h-4" />
                                <span className="text-sm">Best / Worst</span>
                            </div>
                            <p className="text-lg font-bold">
                                <span className="text-emerald-400">
                                    ${Math.max(...(closedTrades?.map(t => t.pnl || 0) || [0])).toFixed(0)}
                                </span>
                                {' / '}
                                <span className="text-rose-400">
                                    ${Math.min(...(closedTrades?.map(t => t.pnl || 0) || [0])).toFixed(0)}
                                </span>
                            </p>
                        </div>
                    </div>

                    {/* Closed Trades Table */}
                    {loadingClosed ? (
                        <div className="card p-8 flex items-center justify-center">
                            <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                            <span className="ml-2 text-slate-400">Loading trades...</span>
                        </div>
                    ) : filteredClosed.length === 0 ? (
                        <div className="card p-8 text-center text-slate-400">
                            <CheckCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                            <p>No closed trades found</p>
                        </div>
                    ) : (
                        <div className="card overflow-hidden">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-slate-700/50">
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Profile</th>
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Ticker</th>
                                        <th className="text-left p-4 text-sm font-medium text-slate-400">Strategy</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">Entry</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">Exit</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">P/L</th>
                                        <th className="text-right p-4 text-sm font-medium text-slate-400">P/L %</th>
                                        <th className="text-center p-4 text-sm font-medium text-slate-400">Close Date</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredClosed.map((trade, idx) => (
                                        <tr
                                            key={trade.id}
                                            className={clsx(
                                                "border-b border-slate-700/30 hover:bg-slate-800/50 transition-colors",
                                                idx % 2 === 0 && "bg-slate-800/20"
                                            )}
                                        >
                                            <td className="p-4 text-sm">{trade.profile_name}</td>
                                            <td className="p-4">
                                                <span className="font-mono font-semibold text-primary">{trade.ticker}</span>
                                            </td>
                                            <td className="p-4">
                                                <span className="px-2 py-1 rounded text-xs bg-slate-700/50">
                                                    {trade.strategy}
                                                </span>
                                            </td>
                                            <td className="p-4 text-right font-mono">
                                                ${trade.entry_price?.toFixed(2) || 'N/A'}
                                            </td>
                                            <td className="p-4 text-right font-mono">
                                                ${trade.exit_price?.toFixed(2) || 'N/A'}
                                            </td>
                                            <td className={clsx(
                                                "p-4 text-right font-mono font-bold",
                                                (trade.pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                            )}>
                                                ${(trade.pnl || 0).toFixed(2)}
                                            </td>
                                            <td className={clsx(
                                                "p-4 text-right font-mono",
                                                (trade.pnl_percent || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                            )}>
                                                {(trade.pnl_percent || 0).toFixed(1)}%
                                            </td>
                                            <td className="p-4 text-center text-sm">{trade.exit_date || 'N/A'}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* Performance Analytics Tab */}
            {activeTab === 'analytics' && (
                <div className="space-y-6">
                    {/* Overall Performance */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-primary" />
                            Overall Performance
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <p className="text-slate-400 text-sm">Total Profiles</p>
                                <p className="text-2xl font-bold">{stats?.total_profiles || 0}</p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <p className="text-slate-400 text-sm">Total Trades</p>
                                <p className="text-2xl font-bold">{stats?.total_trades || 0}</p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <p className="text-slate-400 text-sm">Total P/L</p>
                                <p className={clsx(
                                    "text-2xl font-bold",
                                    (stats?.total_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                                )}>
                                    ${(stats?.total_pnl || 0).toLocaleString()}
                                </p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <p className="text-slate-400 text-sm">Win Rate</p>
                                <p className="text-2xl font-bold text-amber-400">
                                    {(stats?.win_rate || 0).toFixed(1)}%
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Performance by Profile */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Users className="w-5 h-5 text-violet-400" />
                            Performance by Profile
                        </h3>
                        {profileStats && profileStats.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-slate-700/50">
                                            <th className="text-left p-3 text-sm text-slate-400">Profile</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Total Trades</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Open</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Closed</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Total P/L</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Win Rate</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {profileStats.map((profile, idx) => (
                                            <tr key={profile.profile_id} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                <td className="p-3 font-medium">{profile.profile_name}</td>
                                                <td className="p-3 text-right">{profile.total_trades}</td>
                                                <td className="p-3 text-right">{profile.open_trades}</td>
                                                <td className="p-3 text-right">{profile.closed_trades}</td>
                                                <td className={clsx(
                                                    "p-3 text-right font-mono",
                                                    profile.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                                )}>
                                                    ${(profile.total_pnl ?? 0).toFixed(2)}
                                                </td>
                                                <td className="p-3 text-right">{profile.win_rate?.toFixed(1) || 0}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : profiles && profiles.filter(p => p.active).length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-slate-700/50">
                                            <th className="text-left p-3 text-sm text-slate-400">Profile</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Total Trades</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Open</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Closed</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Total P/L</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Win Rate</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {profiles.filter(p => p.active).map((profile, idx) => (
                                            <tr key={profile.id} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                <td className="p-3 font-medium">{profile.display_name || profile.username}</td>
                                                <td className="p-3 text-right">{profile.total_trades_scraped}</td>
                                                <td className="p-3 text-right text-slate-500">-</td>
                                                <td className="p-3 text-right text-slate-500">-</td>
                                                <td className="p-3 text-right font-mono text-slate-500">-</td>
                                                <td className="p-3 text-right text-slate-500">-</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <p className="text-slate-400 text-center py-4">No profiles to analyze</p>
                        )}
                    </div>

                    {/* Performance by Strategy */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-emerald-400" />
                            Performance by Strategy
                        </h3>
                        {strategyStats && strategyStats.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-slate-700/50">
                                            <th className="text-left p-3 text-sm text-slate-400">Strategy</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Trades</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Total P/L</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Avg P/L</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Win Rate</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Best / Worst</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {strategyStats.map((strategy, idx) => (
                                            <tr key={strategy.strategy} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                <td className="p-3 font-medium">
                                                    <span className="px-2 py-1 rounded text-xs bg-slate-700/50">
                                                        {strategy.strategy}
                                                    </span>
                                                </td>
                                                <td className="p-3 text-right">{strategy.total_trades}</td>
                                                <td className={clsx(
                                                    "p-3 text-right font-mono",
                                                    strategy.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                                )}>
                                                    ${(strategy.total_pnl ?? 0).toFixed(2)}
                                                </td>
                                                <td className={clsx(
                                                    "p-3 text-right font-mono",
                                                    strategy.avg_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                                )}>
                                                    ${(strategy.avg_pnl ?? 0).toFixed(2)}
                                                </td>
                                                <td className="p-3 text-right">{strategy.win_rate?.toFixed(1) || 0}%</td>
                                                <td className="p-3 text-right font-mono">
                                                    <span className="text-emerald-400">${strategy.best_trade?.toFixed(0) || 0}</span>
                                                    {' / '}
                                                    <span className="text-rose-400">${strategy.worst_trade?.toFixed(0) || 0}</span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <p className="text-slate-400 text-center py-8">
                                Strategy analytics will appear here once closed trades are available
                            </p>
                        )}
                    </div>
                </div>
            )}

            {/* Manage Profiles Tab */}
            {activeTab === 'profiles' && (
                <div className="space-y-6">
                    {/* Add New Profile */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Plus className="w-5 h-5 text-emerald-400" />
                            Add New Profile
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Xtrades Username</label>
                                <input
                                    type="text"
                                    value={newUsername}
                                    onChange={e => setNewUsername(e.target.value)}
                                    placeholder="e.g., behappy"
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Display Name (Optional)</label>
                                <input
                                    type="text"
                                    value={newDisplayName}
                                    onChange={e => setNewDisplayName(e.target.value)}
                                    placeholder="e.g., BeHappy Trader"
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                />
                            </div>
                            <div className="flex items-end">
                                <button
                                    onClick={() => newUsername && addProfileMutation.mutate({
                                        username: newUsername,
                                        display_name: newDisplayName || undefined
                                    })}
                                    disabled={!newUsername || addProfileMutation.isPending}
                                    className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 rounded-lg flex items-center justify-center gap-2"
                                >
                                    <Plus className="w-4 h-4" />
                                    {addProfileMutation.isPending ? 'Adding...' : 'Add Profile'}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Existing Profiles */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Users className="w-5 h-5 text-violet-400" />
                            Existing Profiles ({profiles?.length || 0})
                        </h3>
                        {loadingProfiles ? (
                            <div className="flex items-center justify-center py-8">
                                <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                            </div>
                        ) : profiles && profiles.length > 0 ? (
                            <div className="space-y-3">
                                {profiles.map(profile => (
                                    <div key={profile.id} className="bg-slate-800/50 rounded-lg p-4">
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className={clsx(
                                                    "w-10 h-10 rounded-full flex items-center justify-center",
                                                    profile.active ? "bg-emerald-500/20" : "bg-slate-700"
                                                )}>
                                                    {profile.active ? (
                                                        <CheckCircle className="w-5 h-5 text-emerald-400" />
                                                    ) : (
                                                        <XCircle className="w-5 h-5 text-slate-500" />
                                                    )}
                                                </div>
                                                <div>
                                                    <p className="font-semibold">
                                                        {profile.display_name || profile.username}
                                                    </p>
                                                    <p className="text-sm text-slate-400">@{profile.username}</p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => syncProfileMutation.mutate(profile.id)}
                                                    disabled={syncProfileMutation.isPending}
                                                    className="btn-icon text-sm"
                                                    title="Sync Now"
                                                >
                                                    <RefreshCw className={clsx(
                                                        "w-4 h-4",
                                                        syncProfileMutation.isPending && "animate-spin"
                                                    )} />
                                                </button>
                                                <button
                                                    onClick={() => toggleProfileMutation.mutate({
                                                        id: profile.id,
                                                        active: !profile.active
                                                    })}
                                                    className={clsx(
                                                        "btn-icon text-sm",
                                                        profile.active ? "text-amber-400" : "text-emerald-400"
                                                    )}
                                                    title={profile.active ? "Deactivate" : "Activate"}
                                                >
                                                    {profile.active ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                                                </button>
                                            </div>
                                        </div>
                                        <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                                            <div>
                                                <p className="text-slate-400">Last Sync</p>
                                                <p>{profile.last_sync || 'Never'}</p>
                                            </div>
                                            <div>
                                                <p className="text-slate-400">Status</p>
                                                <p className={clsx(
                                                    profile.last_sync_status === 'success' && "text-emerald-400",
                                                    profile.last_sync_status === 'error' && "text-rose-400"
                                                )}>
                                                    {profile.last_sync_status || 'N/A'}
                                                </p>
                                            </div>
                                            <div>
                                                <p className="text-slate-400">Total Trades</p>
                                                <p>{profile.total_trades_scraped}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-slate-400 text-center py-8">
                                No profiles added yet. Add your first profile above!
                            </p>
                        )}
                    </div>
                </div>
            )}

            {/* Sync History Tab */}
            {activeTab === 'history' && (
                <div className="space-y-4">
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <History className="w-5 h-5 text-blue-400" />
                            Recent Sync Operations
                        </h3>
                        {syncHistory && syncHistory.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-slate-700/50">
                                            <th className="text-left p-3 text-sm text-slate-400">Timestamp</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Profiles</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Found</th>
                                            <th className="text-right p-3 text-sm text-slate-400">New</th>
                                            <th className="text-right p-3 text-sm text-slate-400">Duration</th>
                                            <th className="text-center p-3 text-sm text-slate-400">Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {syncHistory.map((log, idx) => (
                                            <tr key={log.id} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                                <td className="p-3 text-sm">{log.sync_timestamp}</td>
                                                <td className="p-3 text-right">{log.profiles_synced}</td>
                                                <td className="p-3 text-right">{log.trades_found}</td>
                                                <td className="p-3 text-right text-emerald-400">{log.new_trades}</td>
                                                <td className="p-3 text-right">{log.duration_seconds?.toFixed(1)}s</td>
                                                <td className="p-3 text-center">
                                                    <span className={clsx(
                                                        "px-2 py-1 rounded text-xs",
                                                        log.status === 'success' && "bg-emerald-500/20 text-emerald-400",
                                                        log.status === 'partial' && "bg-amber-500/20 text-amber-400",
                                                        log.status === 'failed' && "bg-rose-500/20 text-rose-400",
                                                        log.status === 'running' && "bg-blue-500/20 text-blue-400"
                                                    )}>
                                                        {log.status}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <p className="text-slate-400 text-center py-8">
                                No sync history yet. Sync profiles to see history here.
                            </p>
                        )}
                    </div>
                </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
                <div className="space-y-6">
                    {/* Sync Configuration */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <RefreshCw className="w-5 h-5 text-blue-400" />
                            Sync Configuration
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Auto-Sync Interval</label>
                                <select
                                    value={syncInterval}
                                    onChange={e => setSyncInterval(e.target.value)}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                >
                                    <option>Disabled</option>
                                    <option>5 minutes</option>
                                    <option>10 minutes</option>
                                    <option>15 minutes</option>
                                    <option>30 minutes</option>
                                    <option>1 hour</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Max Alerts per Profile</label>
                                <input
                                    type="number"
                                    value={maxAlerts}
                                    onChange={e => setMaxAlerts(Number(e.target.value))}
                                    min={10}
                                    max={500}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Telegram Notifications */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Bell className="w-5 h-5 text-amber-400" />
                            Telegram Notifications
                        </h3>
                        <div className="space-y-3">
                            <label className="flex items-center gap-3 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={telegramEnabled}
                                    onChange={e => setTelegramEnabled(e.target.checked)}
                                    className="w-5 h-5 rounded border-slate-600 bg-slate-700/50"
                                />
                                <span>Enable Telegram Notifications</span>
                            </label>
                            {telegramEnabled && (
                                <div className="pl-8 space-y-2">
                                    <label className="flex items-center gap-3 cursor-pointer">
                                        <input type="checkbox" defaultChecked className="w-4 h-4 rounded" />
                                        <span className="text-sm">Notify on New Trades</span>
                                    </label>
                                    <label className="flex items-center gap-3 cursor-pointer">
                                        <input type="checkbox" defaultChecked className="w-4 h-4 rounded" />
                                        <span className="text-sm">Notify on Closed Trades</span>
                                    </label>
                                    <label className="flex items-center gap-3 cursor-pointer">
                                        <input type="checkbox" defaultChecked className="w-4 h-4 rounded" />
                                        <span className="text-sm">Notify on Large P/L (&gt;$500)</span>
                                    </label>
                                    <button className="mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg flex items-center gap-2">
                                        <Send className="w-4 h-4" />
                                        Send Test Notification
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Scraper Settings */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Settings className="w-5 h-5 text-slate-400" />
                            Scraper Settings
                        </h3>
                        <div className="space-y-3">
                            <label className="flex items-center gap-3 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={headlessMode}
                                    onChange={e => setHeadlessMode(e.target.checked)}
                                    className="w-5 h-5 rounded border-slate-600 bg-slate-700/50"
                                />
                                <span>Run Scraper in Headless Mode</span>
                            </label>
                            <p className="text-sm text-slate-400 pl-8">
                                Headless mode runs browser in background (recommended for automation)
                            </p>
                            <button className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2">
                                <Activity className="w-4 h-4" />
                                Test Scraper Connection
                            </button>
                        </div>
                    </div>

                    {/* Maintenance */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Trash2 className="w-5 h-5 text-rose-400" />
                            Maintenance
                        </h3>
                        <div className="flex flex-wrap gap-3">
                            <button className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2">
                                <Trash2 className="w-4 h-4" />
                                Clear Cache
                            </button>
                            <button className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2">
                                <RotateCcw className="w-4 h-4" />
                                Reset Sync History
                            </button>
                            <button className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2">
                                <BarChart3 className="w-4 h-4" />
                                Recalculate Stats
                            </button>
                        </div>
                    </div>

                    {/* Save Button */}
                    <button className="w-full bg-primary hover:bg-primary/80 px-6 py-3 rounded-lg font-semibold">
                        Save All Settings
                    </button>
                </div>
            )}
        </div>
    )
}
