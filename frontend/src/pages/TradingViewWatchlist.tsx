import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    LineChart, RefreshCw, List, ChevronDown, ChevronRight,
    Search, Plus, Trash2, Clock, Globe, ExternalLink,
    TrendingUp
} from 'lucide-react'
import clsx from 'clsx'

interface TradingViewWatchlist {
    id: number
    name: string
    description: string | null
    symbol_count: number
    created_at: string | null
    last_synced: string | null
}

interface TVSymbol {
    symbol: string
    company_name: string
    sector: string
    current_price: number
    volume: number
    market_cap: number
    added_at: string | null
}

interface WatchlistsResponse {
    watchlists: TradingViewWatchlist[]
    total: number
    source: string
    generated_at: string
    error?: string
    message?: string
}

interface SymbolsResponse {
    watchlist_id: number
    watchlist_name: string
    symbols: TVSymbol[]
    count: number
    generated_at: string
}

export default function TradingViewWatchlist() {
    const [selectedWatchlist, setSelectedWatchlist] = useState<TradingViewWatchlist | null>(null)
    const [searchQuery, setSearchQuery] = useState('')
    const [showCreateModal, setShowCreateModal] = useState(false)
    const [newWatchlistName, setNewWatchlistName] = useState('')
    const [newSymbols, setNewSymbols] = useState('')

    const queryClient = useQueryClient()

    // Fetch all TradingView watchlists
    const { data: watchlistsData, isLoading, refetch, isFetching } = useQuery<WatchlistsResponse>({
        queryKey: ['tradingview-watchlists'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/watchlist/tradingview')
            return data
        }
    })

    // Fetch symbols for selected watchlist
    const { data: symbolsData, isLoading: symbolsLoading } = useQuery<SymbolsResponse>({
        queryKey: ['tv-watchlist-symbols', selectedWatchlist?.id],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/watchlist/tradingview/${selectedWatchlist?.id}/symbols`)
            return data
        },
        enabled: !!selectedWatchlist?.id
    })

    // Create watchlist mutation
    const createWatchlistMutation = useMutation({
        mutationFn: async (data: { name: string; symbols: string[] }) => {
            const response = await axiosInstance.post('/watchlist/create', data)
            return response.data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['tradingview-watchlists'] })
            setShowCreateModal(false)
            setNewWatchlistName('')
            setNewSymbols('')
        }
    })

    const watchlists = watchlistsData?.watchlists || []

    // Filter symbols based on search
    const filteredSymbols = symbolsData?.symbols?.filter(s =>
        s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.company_name.toLowerCase().includes(searchQuery.toLowerCase())
    ) || []

    const formatMarketCap = (cap: number) => {
        if (cap >= 1e12) return `$${(cap / 1e12).toFixed(1)}T`
        if (cap >= 1e9) return `$${(cap / 1e9).toFixed(1)}B`
        if (cap >= 1e6) return `$${(cap / 1e6).toFixed(1)}M`
        return cap > 0 ? `$${cap.toLocaleString()}` : '-'
    }

    const formatVolume = (vol: number) => {
        if (vol >= 1e6) return `${(vol / 1e6).toFixed(1)}M`
        if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`
        return vol > 0 ? vol.toLocaleString() : '-'
    }

    const formatDate = (dateStr: string | null) => {
        if (!dateStr) return 'Never'
        const date = new Date(dateStr)
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    const handleCreateWatchlist = () => {
        const symbols = newSymbols.split(/[,\s\n]+/).filter(s => s.trim()).map(s => s.trim().toUpperCase())
        createWatchlistMutation.mutate({
            name: newWatchlistName,
            symbols
        })
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
                        <LineChart className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">TradingView Watchlists</h1>
                        <p className="text-sm text-slate-400">Synced watchlists from TradingView</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="btn-secondary flex items-center gap-2"
                    >
                        <Plus size={16} />
                        Create Watchlist
                    </button>
                    <button
                        onClick={() => refetch()}
                        disabled={isFetching}
                        className="btn-primary flex items-center gap-2"
                    >
                        <RefreshCw size={16} className={isFetching ? 'animate-spin' : ''} />
                        {isFetching ? 'Syncing...' : 'Refresh'}
                    </button>
                </div>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <List className="w-4 h-4" />
                        <span className="text-sm">Total Watchlists</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{watchlistsData?.total || 0}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Total Symbols</span>
                    </div>
                    <p className="text-2xl font-bold text-orange-400">
                        {watchlists.reduce((sum, w) => sum + (w.symbol_count || 0), 0).toLocaleString()}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Globe className="w-4 h-4" />
                        <span className="text-sm">Source</span>
                    </div>
                    <p className="text-lg font-semibold text-emerald-400">TradingView</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Clock className="w-4 h-4" />
                        <span className="text-sm">Last Sync</span>
                    </div>
                    <p className="text-sm font-medium text-slate-300">
                        {watchlists[0]?.last_synced ? formatDate(watchlists[0].last_synced) : 'Never'}
                    </p>
                </div>
            </div>

            {/* No data message */}
            {watchlistsData?.message && (
                <div className="card p-4 bg-amber-500/10 border-amber-500/20">
                    <p className="text-amber-400">{watchlistsData.message}</p>
                </div>
            )}

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Watchlist List */}
                <div className="lg:col-span-1">
                    <div className="card">
                        <div className="p-4 border-b border-slate-700/50">
                            <h2 className="font-semibold text-white flex items-center gap-2">
                                <List className="w-4 h-4" />
                                Available Watchlists
                            </h2>
                        </div>
                        <div className="max-h-[600px] overflow-y-auto">
                            {isLoading ? (
                                <div className="p-4 flex items-center justify-center">
                                    <RefreshCw className="w-5 h-5 animate-spin text-orange-400" />
                                    <span className="ml-2 text-slate-400">Loading watchlists...</span>
                                </div>
                            ) : watchlists.length === 0 ? (
                                <div className="p-8 text-center text-slate-400">
                                    <LineChart className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                    <p>No TradingView watchlists found</p>
                                    <p className="text-sm mt-1">Create a watchlist or sync from TradingView</p>
                                </div>
                            ) : (
                                <div className="divide-y divide-slate-700/50">
                                    {watchlists.map((watchlist) => (
                                        <button
                                            key={watchlist.id}
                                            onClick={() => setSelectedWatchlist(watchlist)}
                                            className={clsx(
                                                "w-full p-3 text-left hover:bg-slate-800/50 transition-colors",
                                                selectedWatchlist?.id === watchlist.id && "bg-orange-500/10 border-l-2 border-orange-500"
                                            )}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="font-medium text-white">{watchlist.name}</p>
                                                    <div className="flex items-center gap-2 mt-1">
                                                        <span className="text-xs text-slate-400">
                                                            {watchlist.symbol_count} symbols
                                                        </span>
                                                        {watchlist.last_synced && (
                                                            <>
                                                                <span className="text-slate-600">|</span>
                                                                <span className="text-xs text-slate-500">
                                                                    Last sync: {new Date(watchlist.last_synced).toLocaleDateString()}
                                                                </span>
                                                            </>
                                                        )}
                                                    </div>
                                                </div>
                                                <span className="px-2 py-1 text-xs rounded bg-orange-500/20 text-orange-400">
                                                    {watchlist.symbol_count}
                                                </span>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Symbol Details */}
                <div className="lg:col-span-2">
                    <div className="card">
                        <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                            <h2 className="font-semibold text-white flex items-center gap-2">
                                <TrendingUp className="w-4 h-4" />
                                {selectedWatchlist ? `Symbols in "${selectedWatchlist.name}"` : 'Select a Watchlist'}
                            </h2>
                            {selectedWatchlist && (
                                <div className="flex items-center gap-2">
                                    <div className="relative">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search symbols..."
                                            className="pl-9 pr-4 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-400 focus:outline-none focus:border-orange-500"
                                        />
                                    </div>
                                    <a
                                        href={`https://www.tradingview.com/watchlists/`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                                        title="Open in TradingView"
                                    >
                                        <ExternalLink className="w-4 h-4 text-slate-400" />
                                    </a>
                                </div>
                            )}
                        </div>

                        <div className="max-h-[600px] overflow-y-auto">
                            {!selectedWatchlist ? (
                                <div className="p-12 text-center text-slate-400">
                                    <LineChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                    <p>Select a watchlist to view symbols</p>
                                </div>
                            ) : symbolsLoading ? (
                                <div className="p-8 flex items-center justify-center">
                                    <RefreshCw className="w-5 h-5 animate-spin text-orange-400" />
                                    <span className="ml-2 text-slate-400">Loading symbols...</span>
                                </div>
                            ) : filteredSymbols.length === 0 ? (
                                <div className="p-8 text-center text-slate-400">
                                    <Search className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                    <p>No symbols found</p>
                                </div>
                            ) : (
                                <table className="w-full">
                                    <thead className="bg-slate-800/50 sticky top-0">
                                        <tr>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Symbol</th>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Company</th>
                                            <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Price</th>
                                            <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Volume</th>
                                            <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Market Cap</th>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Sector</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-700/50">
                                        {filteredSymbols.map((symbol) => (
                                            <tr key={symbol.symbol} className="hover:bg-slate-800/30 transition-colors">
                                                <td className="px-4 py-3">
                                                    <a
                                                        href={`https://www.tradingview.com/symbols/${symbol.symbol}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="font-bold text-white hover:text-orange-400 transition-colors"
                                                    >
                                                        {symbol.symbol}
                                                    </a>
                                                </td>
                                                <td className="px-4 py-3 text-slate-300 text-sm max-w-[200px] truncate">
                                                    {symbol.company_name}
                                                </td>
                                                <td className="px-4 py-3 text-right">
                                                    <span className={clsx(
                                                        "font-mono font-medium",
                                                        symbol.current_price > 0 ? "text-emerald-400" : "text-slate-400"
                                                    )}>
                                                        {symbol.current_price > 0 ? `$${symbol.current_price.toFixed(2)}` : '-'}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3 text-right text-slate-300 text-sm">
                                                    {formatVolume(symbol.volume)}
                                                </td>
                                                <td className="px-4 py-3 text-right text-slate-300 text-sm">
                                                    {formatMarketCap(symbol.market_cap)}
                                                </td>
                                                <td className="px-4 py-3">
                                                    <span className="px-2 py-1 text-xs rounded bg-slate-700/50 text-slate-300">
                                                        {symbol.sector || 'Unknown'}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>

                        {selectedWatchlist && symbolsData && (
                            <div className="p-3 border-t border-slate-700/50 bg-slate-800/30 text-sm text-slate-400">
                                Showing {filteredSymbols.length} of {symbolsData.count} symbols
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Create Watchlist Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="card w-full max-w-md mx-4">
                        <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                            <h3 className="font-semibold text-white">Create New Watchlist</h3>
                            <button
                                onClick={() => setShowCreateModal(false)}
                                className="text-slate-400 hover:text-white"
                            >
                                &times;
                            </button>
                        </div>
                        <div className="p-4 space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Watchlist Name</label>
                                <input
                                    type="text"
                                    value={newWatchlistName}
                                    onChange={(e) => setNewWatchlistName(e.target.value)}
                                    placeholder="My Custom Watchlist"
                                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-orange-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Symbols (comma or space separated)</label>
                                <textarea
                                    value={newSymbols}
                                    onChange={(e) => setNewSymbols(e.target.value)}
                                    placeholder="AAPL, MSFT, GOOG, NVDA..."
                                    rows={4}
                                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-orange-500 resize-none"
                                />
                            </div>
                        </div>
                        <div className="p-4 border-t border-slate-700/50 flex justify-end gap-2">
                            <button
                                onClick={() => setShowCreateModal(false)}
                                className="btn-secondary"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleCreateWatchlist}
                                disabled={!newWatchlistName.trim() || createWatchlistMutation.isPending}
                                className="btn-primary flex items-center gap-2"
                            >
                                {createWatchlistMutation.isPending ? (
                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Plus className="w-4 h-4" />
                                )}
                                Create
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
