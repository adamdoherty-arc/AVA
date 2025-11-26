import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    BookOpen, RefreshCw, Plus, TrendingUp, TrendingDown, DollarSign,
    Calendar, Tag, MessageSquare, Star, Filter, BarChart3
} from 'lucide-react'
import clsx from 'clsx'

interface JournalEntry {
    id: string
    date: string
    symbol: string
    strategy: string
    entry_price: number
    exit_price: number
    pnl: number
    pnl_pct: number
    notes: string
    emotions: string
    mistakes: string[]
    lessons: string[]
    rating: number
    tags: string[]
}

interface JournalStats {
    total_trades: number
    win_rate: number
    total_pnl: number
    avg_pnl: number
    best_trade: number
    worst_trade: number
    avg_hold_time: string
}

export default function TradeJournal() {
    const queryClient = useQueryClient()
    const [showAddForm, setShowAddForm] = useState(false)
    const [filterTag, setFilterTag] = useState('')
    const [filterStrategy, setFilterStrategy] = useState('')

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['journal-entries', filterTag, filterStrategy],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/journal', {
                params: { tag: filterTag, strategy: filterStrategy }
            })
            return data
        },
        staleTime: 60000,
    })

    const entries: JournalEntry[] = data?.entries || []
    const stats: JournalStats = data?.stats || {
        total_trades: 0, win_rate: 0, total_pnl: 0, avg_pnl: 0,
        best_trade: 0, worst_trade: 0, avg_hold_time: '0d'
    }

    const allTags = [...new Set(entries.flatMap(e => e.tags))]
    const allStrategies = [...new Set(entries.map(e => e.strategy))]

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center shadow-lg">
                            <BookOpen className="w-5 h-5 text-white" />
                        </div>
                        Trade Journal
                    </h1>
                    <p className="page-subtitle">Track trades, analyze patterns, and improve your strategy</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowAddForm(!showAddForm)}
                        className="bg-primary hover:bg-primary/80 px-4 py-2 rounded-lg flex items-center gap-2"
                    >
                        <Plus className="w-4 h-4" />
                        Add Entry
                    </button>
                    <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </header>

            {/* Stats Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-xs">Total Trades</span>
                    </div>
                    <p className="text-xl font-bold">{stats.total_trades}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-xs">Win Rate</span>
                    </div>
                    <p className={clsx(
                        "text-xl font-bold",
                        stats.win_rate >= 50 ? "text-emerald-400" : "text-rose-400"
                    )}>
                        {stats.win_rate.toFixed(0)}%
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-xs">Total P/L</span>
                    </div>
                    <p className={clsx(
                        "text-xl font-bold",
                        stats.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}>
                        ${stats.total_pnl.toLocaleString()}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <DollarSign className="w-4 h-4" />
                        <span className="text-xs">Avg P/L</span>
                    </div>
                    <p className={clsx(
                        "text-xl font-bold",
                        stats.avg_pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}>
                        ${stats.avg_pnl.toFixed(0)}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-xs">Best</span>
                    </div>
                    <p className="text-xl font-bold text-emerald-400">${stats.best_trade.toLocaleString()}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingDown className="w-4 h-4" />
                        <span className="text-xs">Worst</span>
                    </div>
                    <p className="text-xl font-bold text-rose-400">${stats.worst_trade.toLocaleString()}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Calendar className="w-4 h-4" />
                        <span className="text-xs">Avg Hold</span>
                    </div>
                    <p className="text-xl font-bold">{stats.avg_hold_time}</p>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Filters:</span>
                    </div>
                    <select
                        value={filterStrategy}
                        onChange={e => setFilterStrategy(e.target.value)}
                        className="bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-1.5 text-sm"
                    >
                        <option value="">All Strategies</option>
                        {allStrategies.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <select
                        value={filterTag}
                        onChange={e => setFilterTag(e.target.value)}
                        className="bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-1.5 text-sm"
                    >
                        <option value="">All Tags</option>
                        {allTags.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                </div>
            </div>

            {/* Journal Entries */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading journal...</span>
                </div>
            ) : entries.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No journal entries yet</p>
                    <p className="text-sm mt-1">Start tracking your trades to improve your strategy</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {entries.map(entry => (
                        <div key={entry.id} className="card p-4 hover:border-primary/50 transition-colors">
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-4">
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <span className="font-mono font-bold text-lg text-primary">{entry.symbol}</span>
                                            <span className="px-2 py-0.5 bg-slate-700 rounded text-xs">{entry.strategy}</span>
                                        </div>
                                        <p className="text-sm text-slate-400">{entry.date}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={clsx(
                                        "text-xl font-bold",
                                        entry.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                    )}>
                                        {entry.pnl >= 0 ? '+' : ''}${entry.pnl.toFixed(0)}
                                    </p>
                                    <p className={clsx(
                                        "text-sm",
                                        entry.pnl_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                                    )}>
                                        {entry.pnl_pct >= 0 ? '+' : ''}{entry.pnl_pct.toFixed(1)}%
                                    </p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                                <div>
                                    <p className="text-slate-400">Entry</p>
                                    <p className="font-mono">${entry.entry_price.toFixed(2)}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Exit</p>
                                    <p className="font-mono">${entry.exit_price.toFixed(2)}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Emotion</p>
                                    <p>{entry.emotions}</p>
                                </div>
                                <div>
                                    <p className="text-slate-400">Rating</p>
                                    <div className="flex">
                                        {[1, 2, 3, 4, 5].map(star => (
                                            <Star
                                                key={star}
                                                className={clsx(
                                                    "w-4 h-4",
                                                    star <= entry.rating ? "text-amber-400 fill-amber-400" : "text-slate-600"
                                                )}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {entry.notes && (
                                <div className="p-3 bg-slate-800/50 rounded-lg mb-3">
                                    <p className="text-sm text-slate-400 flex items-center gap-1 mb-1">
                                        <MessageSquare className="w-3 h-3" /> Notes
                                    </p>
                                    <p className="text-sm">{entry.notes}</p>
                                </div>
                            )}

                            <div className="flex flex-wrap gap-1">
                                {entry.tags.map(tag => (
                                    <span key={tag} className="px-2 py-0.5 bg-primary/20 text-primary rounded text-xs">
                                        <Tag className="w-3 h-3 inline mr-1" />
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
