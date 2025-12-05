import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Database, RefreshCw, Search, Filter, ChevronDown, ChevronUp,
    ExternalLink, TrendingUp, TrendingDown, Building2, BarChart2,
    DollarSign, Activity, Layers, X, ChevronLeft, ChevronRight
} from 'lucide-react'
import clsx from 'clsx'

interface Stock {
    symbol: string
    company_name: string | null
    exchange: string | null
    sector: string | null
    industry: string | null
    current_price: number | null
    market_cap: number | null
    volume: number | null
    avg_volume_10d: number | null
    pe_ratio: number | null
    dividend_yield: number | null
    beta: number | null
    week_52_high: number | null
    week_52_low: number | null
    sma_50: number | null
    sma_200: number | null
    rsi_14: number | null
    revenue_growth: number | null
    earnings_growth: number | null
    profit_margin: number | null
    has_options: boolean
    recommendation_key: string | null
    last_updated: string | null
}

interface StocksResponse {
    stocks: Stock[]
    total: number
    count: number
    limit: number
    offset: number
    has_more: boolean
    generated_at: string
    error?: string
}

interface SectorsResponse {
    sectors: Array<{
        sector: string
        stock_count: number
        avg_market_cap_billions: number | null
        avg_pe_ratio: number | null
        avg_dividend_yield_pct: number | null
    }>
}

interface StatsResponse {
    stocks: {
        total: number
        optionable: number
        active: number
        sectors: number
    }
    etfs: {
        total: number
        optionable: number
        active: number
    }
}

const formatNumber = (num: number | null, decimals = 2): string => {
    if (num === null || num === undefined) return '-'
    return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

const formatMarketCap = (cap: number | null): string => {
    if (cap === null || cap === undefined) return '-'
    if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`
    return `$${cap.toLocaleString()}`
}

const formatVolume = (vol: number | null): string => {
    if (vol === null || vol === undefined) return '-'
    if (vol >= 1e6) return `${(vol / 1e6).toFixed(2)}M`
    if (vol >= 1e3) return `${(vol / 1e3).toFixed(1)}K`
    return vol.toLocaleString()
}

const formatPercent = (pct: number | null): string => {
    if (pct === null || pct === undefined) return '-'
    return `${(pct * 100).toFixed(2)}%`
}

export default function DatabaseWatchlist() {
    // Filters
    const [search, setSearch] = useState('')
    const [sector, setSector] = useState('')
    const [exchange, setExchange] = useState('')
    const [minPrice, setMinPrice] = useState('')
    const [maxPrice, setMaxPrice] = useState('')
    const [minMarketCap, setMinMarketCap] = useState('')
    const [hasOptions, setHasOptions] = useState<boolean | null>(null)
    const [showFilters, setShowFilters] = useState(false)

    // Sorting
    const [sortBy, setSortBy] = useState('market_cap')
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

    // Pagination
    const [page, setPage] = useState(0)
    const pageSize = 50

    // Build query params
    const queryParams = useMemo(() => {
        const params: Record<string, string> = {
            sort_by: sortBy,
            sort_order: sortOrder,
            limit: String(pageSize),
            offset: String(page * pageSize)
        }
        if (search) params.search = search
        if (sector) params.sector = sector
        if (exchange) params.exchange = exchange
        if (minPrice) params.min_price = minPrice
        if (maxPrice) params.max_price = maxPrice
        if (minMarketCap) params.min_market_cap = String(parseFloat(minMarketCap) * 1e9) // Convert billions to dollars
        if (hasOptions !== null) params.has_options = String(hasOptions)
        return params
    }, [search, sector, exchange, minPrice, maxPrice, minMarketCap, hasOptions, sortBy, sortOrder, page])

    // Fetch stocks
    const { data: stocksData, isLoading, refetch, isFetching } = useQuery<StocksResponse>({
        queryKey: ['stock-universe', queryParams],
        queryFn: async () => {
            const queryString = new URLSearchParams(queryParams).toString()
            const { data } = await axiosInstance.get(`/universe/stocks?${queryString}`)
            return data
        }
    })

    // Fetch sectors for filter dropdown
    const { data: sectorsData } = useQuery<SectorsResponse>({
        queryKey: ['universe-sectors'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/universe/sectors')
            return data
        }
    })

    // Fetch stats
    const { data: statsData } = useQuery<StatsResponse>({
        queryKey: ['universe-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/universe/stats')
            return data
        }
    })

    const stocks = stocksData?.stocks || []
    const totalStocks = stocksData?.total || 0
    const sectors = sectorsData?.sectors || []
    const totalPages = Math.ceil(totalStocks / pageSize)

    const handleSort = (field: string) => {
        if (sortBy === field) {
            setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
        } else {
            setSortBy(field)
            setSortOrder('desc')
        }
        setPage(0)
    }

    const clearFilters = () => {
        setSearch('')
        setSector('')
        setExchange('')
        setMinPrice('')
        setMaxPrice('')
        setMinMarketCap('')
        setHasOptions(null)
        setPage(0)
    }

    const openTradingView = (symbol: string) => {
        window.open(`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(symbol)}`, '_blank')
    }

    const SortIcon = ({ field }: { field: string }) => {
        if (sortBy !== field) return null
        return sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
                        <Database className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Stock Universe</h1>
                        <p className="text-sm text-slate-400">Complete database of {statsData?.stocks?.total?.toLocaleString() || '...'} stocks with comprehensive data</p>
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

            {/* Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Building2 className="w-4 h-4" />
                        <span className="text-sm">Total Stocks</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{statsData?.stocks?.total?.toLocaleString() || '-'}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-sm">With Options</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{statsData?.stocks?.optionable?.toLocaleString() || '-'}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Layers className="w-4 h-4" />
                        <span className="text-sm">Sectors</span>
                    </div>
                    <p className="text-2xl font-bold text-blue-400">{statsData?.stocks?.sectors || '-'}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <BarChart2 className="w-4 h-4" />
                        <span className="text-sm">Total ETFs</span>
                    </div>
                    <p className="text-2xl font-bold text-purple-400">{statsData?.etfs?.total?.toLocaleString() || '-'}</p>
                </div>
            </div>

            {/* Search and Filters */}
            <div className="card p-4">
                <div className="flex flex-col md:flex-row gap-4">
                    {/* Search */}
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                            type="text"
                            value={search}
                            onChange={(e) => { setSearch(e.target.value); setPage(0) }}
                            placeholder="Search by symbol or company name..."
                            className="w-full pl-10 pr-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-purple-500"
                        />
                    </div>

                    {/* Sector Filter */}
                    <select
                        value={sector}
                        onChange={(e) => { setSector(e.target.value); setPage(0) }}
                        className="px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    >
                        <option value="">All Sectors</option>
                        {sectors.map(s => (
                            <option key={s.sector} value={s.sector}>{s.sector} ({s.stock_count})</option>
                        ))}
                    </select>

                    {/* Toggle Filters */}
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={clsx(
                            "px-4 py-2.5 rounded-lg flex items-center gap-2 transition-colors",
                            showFilters ? "bg-purple-500/20 text-purple-400 border border-purple-500" : "bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600"
                        )}
                    >
                        <Filter className="w-4 h-4" />
                        Filters
                    </button>
                </div>

                {/* Advanced Filters */}
                {showFilters && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50 grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div>
                            <label className="text-xs text-slate-400 mb-1 block">Exchange</label>
                            <select
                                value={exchange}
                                onChange={(e) => { setExchange(e.target.value); setPage(0) }}
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                            >
                                <option value="">All</option>
                                <option value="NYSE">NYSE</option>
                                <option value="NASDAQ">NASDAQ</option>
                                <option value="AMEX">AMEX</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 mb-1 block">Min Price ($)</label>
                            <input
                                type="number"
                                value={minPrice}
                                onChange={(e) => { setMinPrice(e.target.value); setPage(0) }}
                                placeholder="0"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 mb-1 block">Max Price ($)</label>
                            <input
                                type="number"
                                value={maxPrice}
                                onChange={(e) => { setMaxPrice(e.target.value); setPage(0) }}
                                placeholder="No limit"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 mb-1 block">Min Market Cap (B)</label>
                            <input
                                type="number"
                                value={minMarketCap}
                                onChange={(e) => { setMinMarketCap(e.target.value); setPage(0) }}
                                placeholder="0"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 mb-1 block">Has Options</label>
                            <select
                                value={hasOptions === null ? '' : String(hasOptions)}
                                onChange={(e) => {
                                    const val = e.target.value
                                    setHasOptions(val === '' ? null : val === 'true')
                                    setPage(0)
                                }}
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
                            >
                                <option value="">All</option>
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>

                        <div className="col-span-2 md:col-span-5 flex justify-end">
                            <button
                                onClick={clearFilters}
                                className="text-sm text-slate-400 hover:text-white flex items-center gap-1"
                            >
                                <X className="w-4 h-4" />
                                Clear All Filters
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Results Info */}
            <div className="flex items-center justify-between text-sm text-slate-400">
                <span>
                    Showing {stocks.length} of {totalStocks.toLocaleString()} stocks
                    {search && ` matching "${search}"`}
                    {sector && ` in ${sector}`}
                </span>
                <span>
                    Page {page + 1} of {totalPages || 1}
                </span>
            </div>

            {/* Stock Table */}
            <div className="card overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-slate-800/80 sticky top-0">
                            <tr>
                                <th
                                    className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('symbol')}
                                >
                                    <div className="flex items-center gap-1">Symbol <SortIcon field="symbol" /></div>
                                </th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Company</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Sector</th>
                                <th
                                    className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('current_price')}
                                >
                                    <div className="flex items-center justify-end gap-1">Price <SortIcon field="current_price" /></div>
                                </th>
                                <th
                                    className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('market_cap')}
                                >
                                    <div className="flex items-center justify-end gap-1">Mkt Cap <SortIcon field="market_cap" /></div>
                                </th>
                                <th
                                    className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('avg_volume_10d')}
                                >
                                    <div className="flex items-center justify-end gap-1">Avg Vol <SortIcon field="avg_volume_10d" /></div>
                                </th>
                                <th
                                    className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('pe_ratio')}
                                >
                                    <div className="flex items-center justify-end gap-1">P/E <SortIcon field="pe_ratio" /></div>
                                </th>
                                <th
                                    className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase cursor-pointer hover:text-white"
                                    onClick={() => handleSort('dividend_yield')}
                                >
                                    <div className="flex items-center justify-end gap-1">Div Yield <SortIcon field="dividend_yield" /></div>
                                </th>
                                <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">52W Range</th>
                                <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Options</th>
                                <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/50">
                            {isLoading ? (
                                <tr>
                                    <td colSpan={11} className="px-4 py-12 text-center">
                                        <RefreshCw className="w-6 h-6 animate-spin mx-auto text-purple-400 mb-2" />
                                        <span className="text-slate-400">Loading stocks...</span>
                                    </td>
                                </tr>
                            ) : stocks.length === 0 ? (
                                <tr>
                                    <td colSpan={11} className="px-4 py-12 text-center text-slate-400">
                                        <Database className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                        <p>No stocks found matching your criteria</p>
                                    </td>
                                </tr>
                            ) : (
                                stocks.map((stock) => (
                                    <tr key={stock.symbol} className="hover:bg-slate-800/30 transition-colors">
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <span className="font-bold text-white">{stock.symbol}</span>
                                                <span className={clsx(
                                                    "px-1.5 py-0.5 text-xs rounded",
                                                    stock.exchange === 'NASDAQ' ? 'bg-blue-500/20 text-blue-400' :
                                                    stock.exchange === 'NYSE' ? 'bg-emerald-500/20 text-emerald-400' :
                                                    'bg-slate-700/50 text-slate-300'
                                                )}>
                                                    {stock.exchange || '-'}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="text-slate-300 text-sm truncate max-w-[200px] block" title={stock.company_name || ''}>
                                                {stock.company_name || '-'}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="text-slate-400 text-sm">{stock.sector || '-'}</span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <span className="font-medium text-white">${formatNumber(stock.current_price)}</span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <span className="text-slate-300">{formatMarketCap(stock.market_cap)}</span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <span className="text-slate-400">{formatVolume(stock.avg_volume_10d)}</span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <span className="text-slate-300">{stock.pe_ratio ? formatNumber(stock.pe_ratio, 1) : '-'}</span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <span className={clsx(
                                                stock.dividend_yield && stock.dividend_yield > 0 ? 'text-emerald-400' : 'text-slate-400'
                                            )}>
                                                {stock.dividend_yield ? formatPercent(stock.dividend_yield) : '-'}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            {stock.week_52_low !== null && stock.week_52_high !== null && stock.current_price !== null ? (
                                                <div className="flex items-center gap-2 justify-end">
                                                    <span className="text-xs text-slate-500">${formatNumber(stock.week_52_low, 0)}</span>
                                                    <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-purple-500 rounded-full"
                                                            style={{
                                                                width: `${Math.min(100, Math.max(0, ((stock.current_price - stock.week_52_low) / (stock.week_52_high - stock.week_52_low)) * 100))}%`
                                                            }}
                                                        />
                                                    </div>
                                                    <span className="text-xs text-slate-500">${formatNumber(stock.week_52_high, 0)}</span>
                                                </div>
                                            ) : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            {stock.has_options ? (
                                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-emerald-500/20 text-emerald-400">
                                                    Yes
                                                </span>
                                            ) : (
                                                <span className="text-slate-500">-</span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3 text-center">
                                            <button
                                                onClick={() => openTradingView(stock.symbol)}
                                                className="p-1.5 rounded hover:bg-slate-700 text-slate-400 hover:text-purple-400 transition-colors"
                                                title="Open in TradingView"
                                            >
                                                <ExternalLink className="w-4 h-4" />
                                            </button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="p-4 border-t border-slate-700/50 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => setPage(0)}
                                disabled={page === 0}
                                className="px-3 py-1.5 text-sm bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                First
                            </button>
                            <button
                                onClick={() => setPage(p => Math.max(0, p - 1))}
                                disabled={page === 0}
                                className="p-1.5 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <ChevronLeft className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="flex items-center gap-2">
                            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                                let pageNum: number
                                if (totalPages <= 5) {
                                    pageNum = i
                                } else if (page < 3) {
                                    pageNum = i
                                } else if (page > totalPages - 4) {
                                    pageNum = totalPages - 5 + i
                                } else {
                                    pageNum = page - 2 + i
                                }
                                return (
                                    <button
                                        key={pageNum}
                                        onClick={() => setPage(pageNum)}
                                        className={clsx(
                                            "w-8 h-8 text-sm rounded",
                                            page === pageNum
                                                ? "bg-purple-500 text-white"
                                                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                                        )}
                                    >
                                        {pageNum + 1}
                                    </button>
                                )
                            })}
                        </div>

                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                                disabled={page >= totalPages - 1}
                                className="p-1.5 bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <ChevronRight className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setPage(totalPages - 1)}
                                disabled={page >= totalPages - 1}
                                className="px-3 py-1.5 text-sm bg-slate-800 text-slate-300 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Last
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
