import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Database, RefreshCw, List, ChevronDown, ChevronRight,
    Search, BarChart2, ExternalLink
} from 'lucide-react'
import clsx from 'clsx'

interface WatchlistItem {
    id: number | string
    name: string
    symbol_count: number
    created_at: string | null
    updated_at: string | null
}

interface SymbolDetail {
    symbol: string
    exchange: string
    full_symbol: string
    added_at: string | null
}

interface WatchlistsResponse {
    watchlists: WatchlistItem[]
    total: number
    source: string
    generated_at: string
    error?: string
}

interface SymbolsResponse {
    watchlist_name: string
    symbols: SymbolDetail[]
    count: number
    generated_at: string
}

export default function DatabaseWatchlist() {
    const [selectedWatchlist, setSelectedWatchlist] = useState<string | null>(null)
    const [expandedWatchlist, setExpandedWatchlist] = useState<string | null>(null)
    const [searchQuery, setSearchQuery] = useState('')
    const [stocksOnly, setStocksOnly] = useState(false)

    // Fetch all database watchlists
    const { data: watchlistsData, isLoading, refetch, isFetching } = useQuery<WatchlistsResponse>({
        queryKey: ['database-watchlists'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/watchlist/database')
            return data
        }
    })

    // Fetch symbols for selected watchlist
    const { data: symbolsData, isLoading: symbolsLoading, refetch: refetchSymbols } = useQuery<SymbolsResponse>({
        queryKey: ['watchlist-symbols', selectedWatchlist, stocksOnly],
        queryFn: async () => {
            const { data } = await axiosInstance.get(
                `/watchlist/database/${selectedWatchlist}/symbols?stocks_only=${stocksOnly}`
            )
            return data
        },
        enabled: !!selectedWatchlist
    })

    const watchlists = watchlistsData?.watchlists || []

    // Filter symbols based on search
    const filteredSymbols = symbolsData?.symbols?.filter(s =>
        s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.exchange.toLowerCase().includes(searchQuery.toLowerCase())
    ) || []

    const toggleWatchlist = (name: string) => {
        if (expandedWatchlist === name) {
            setExpandedWatchlist(null)
            setSelectedWatchlist(null)
        } else {
            setExpandedWatchlist(name)
            setSelectedWatchlist(name)
        }
    }

    const openTradingView = (fullSymbol: string) => {
        // Open in TradingView
        window.open(`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(fullSymbol)}`, '_blank')
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <Database className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Database Watchlists</h1>
                        <p className="text-sm text-slate-400">Stocks from synced TradingView watchlists in database</p>
                    </div>
                </div>
                <button
                    onClick={() => refetch()}
                    disabled={isFetching}
                    className="btn-primary flex items-center gap-2"
                >
                    <RefreshCw size={16} className={isFetching ? 'animate-spin' : ''} />
                    {isFetching ? 'Loading...' : 'Refresh'}
                </button>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <List className="w-4 h-4" />
                        <span className="text-sm">Total Watchlists</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{watchlistsData?.total || 0}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <BarChart2 className="w-4 h-4" />
                        <span className="text-sm">Total Symbols</span>
                    </div>
                    <p className="text-2xl font-bold text-blue-400">
                        {watchlists.reduce((sum, w) => sum + (w.symbol_count || 0), 0).toLocaleString()}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Database className="w-4 h-4" />
                        <span className="text-sm">Data Source</span>
                    </div>
                    <p className="text-lg font-semibold text-emerald-400">{watchlistsData?.source || 'Database'}</p>
                </div>
            </div>

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
                                    <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
                                    <span className="ml-2 text-slate-400">Loading watchlists...</span>
                                </div>
                            ) : watchlists.length === 0 ? (
                                <div className="p-8 text-center text-slate-400">
                                    <Database className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                    <p>No watchlists found in database</p>
                                    <p className="text-sm mt-1">Run TradingView sync to populate</p>
                                </div>
                            ) : (
                                <div className="divide-y divide-slate-700/50">
                                    {watchlists.map((watchlist) => (
                                        <button
                                            key={watchlist.name}
                                            onClick={() => toggleWatchlist(watchlist.name)}
                                            className={clsx(
                                                "w-full p-3 text-left hover:bg-slate-800/50 transition-colors flex items-center justify-between",
                                                expandedWatchlist === watchlist.name && "bg-blue-500/10 border-l-2 border-blue-500"
                                            )}
                                        >
                                            <div className="flex items-center gap-3">
                                                {expandedWatchlist === watchlist.name ? (
                                                    <ChevronDown className="w-4 h-4 text-blue-400" />
                                                ) : (
                                                    <ChevronRight className="w-4 h-4 text-slate-400" />
                                                )}
                                                <div>
                                                    <p className="font-medium text-white">{watchlist.name}</p>
                                                    <p className="text-xs text-slate-400">
                                                        {watchlist.symbol_count} symbols
                                                    </p>
                                                </div>
                                            </div>
                                            <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                                                {watchlist.symbol_count}
                                            </span>
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
                        <div className="p-4 border-b border-slate-700/50 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                            <h2 className="font-semibold text-white flex items-center gap-2">
                                <BarChart2 className="w-4 h-4" />
                                {selectedWatchlist ? `Symbols in "${selectedWatchlist}"` : 'Select a Watchlist'}
                            </h2>
                            {selectedWatchlist && (
                                <div className="flex items-center gap-3">
                                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={stocksOnly}
                                            onChange={(e) => {
                                                setStocksOnly(e.target.checked)
                                            }}
                                            className="rounded bg-slate-700 border-slate-600 text-blue-500 focus:ring-blue-500"
                                        />
                                        Stocks Only
                                    </label>
                                    <div className="relative">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search symbols..."
                                            className="pl-9 pr-4 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="max-h-[600px] overflow-y-auto">
                            {!selectedWatchlist ? (
                                <div className="p-12 text-center text-slate-400">
                                    <List className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                    <p>Select a watchlist to view symbols</p>
                                </div>
                            ) : symbolsLoading ? (
                                <div className="p-8 flex items-center justify-center">
                                    <RefreshCw className="w-5 h-5 animate-spin text-blue-400" />
                                    <span className="ml-2 text-slate-400">Loading symbols...</span>
                                </div>
                            ) : filteredSymbols.length === 0 ? (
                                <div className="p-8 text-center text-slate-400">
                                    <Search className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                    <p>No symbols found</p>
                                    {stocksOnly && (
                                        <p className="text-sm mt-1">Try unchecking "Stocks Only" to see crypto symbols</p>
                                    )}
                                </div>
                            ) : (
                                <table className="w-full">
                                    <thead className="bg-slate-800/50 sticky top-0">
                                        <tr>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Symbol</th>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Exchange</th>
                                            <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Full Symbol</th>
                                            <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-700/50">
                                        {filteredSymbols.map((symbol) => (
                                            <tr key={symbol.full_symbol} className="hover:bg-slate-800/30 transition-colors">
                                                <td className="px-4 py-3">
                                                    <span className="font-bold text-white">{symbol.symbol}</span>
                                                </td>
                                                <td className="px-4 py-3">
                                                    <span className={clsx(
                                                        "px-2 py-1 text-xs rounded",
                                                        symbol.exchange === 'NASDAQ' ? 'bg-blue-500/20 text-blue-400' :
                                                        symbol.exchange === 'NYSE' ? 'bg-emerald-500/20 text-emerald-400' :
                                                        symbol.exchange === 'AMEX' ? 'bg-purple-500/20 text-purple-400' :
                                                        'bg-slate-700/50 text-slate-300'
                                                    )}>
                                                        {symbol.exchange}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3 text-slate-300 text-sm font-mono">
                                                    {symbol.full_symbol}
                                                </td>
                                                <td className="px-4 py-3 text-center">
                                                    <button
                                                        onClick={() => openTradingView(symbol.full_symbol)}
                                                        className="p-1.5 rounded hover:bg-slate-700 text-slate-400 hover:text-blue-400 transition-colors"
                                                        title="Open in TradingView"
                                                    >
                                                        <ExternalLink className="w-4 h-4" />
                                                    </button>
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
        </div>
    )
}
