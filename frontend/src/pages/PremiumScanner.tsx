import { useState, useMemo, useCallback, useRef, useEffect, memo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    AlertCircle, RefreshCw, Search, Filter, TrendingUp,
    Clock, DollarSign, Target, Percent, Activity, ChevronDown,
    Download, Zap, BarChart3, Sparkles, Cloud, Briefcase, List,
    ArrowUpDown, ArrowUp, ArrowDown, ExternalLink, CheckSquare, Square,
    History, Loader2, CheckCircle, XCircle, Database, Brain, Star
} from 'lucide-react'
import { AIPickCard } from '../components/AIPickCard'
import type { AIPick } from '../components/AIPickCard'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface Watchlist {
    id: string
    name: string
    source: 'predefined' | 'tradingview' | 'robinhood' | 'database'
    symbols: string[]
}

interface WatchlistsResponse {
    watchlists: Watchlist[]
    total: number
    generated_at: string
}

interface ScanResult {
    symbol: string
    stock_price: number
    strike: number
    expiration: string
    dte: number
    premium: number
    premium_pct: number
    monthly_return: number
    annual_return: number
    iv: number
    volume: number
    open_interest: number
    bid_ask_spread: number
}

interface ScanHistory {
    scan_id: string
    symbols: string[]
    symbol_count: number
    dte: number
    max_price: number
    min_premium_pct: number
    result_count: number
    created_at: string
}

interface ScanProgress {
    type: 'start' | 'progress' | 'batch' | 'complete' | 'error' | 'symbol_start' | 'symbol_complete' | 'symbol_error'
    scan_id?: string
    total?: number
    current?: number
    percent?: number
    symbols?: string[]
    symbol?: string
    index?: number
    found?: number
    count?: number
    total_so_far?: number
    results?: ScanResult[]
    error?: string
    saved?: boolean
}

interface AIPicksResponse {
    picks: AIPick[]
    market_context: string
    generated_at: string
    model: string
    total_candidates: number
    cached: boolean
}

interface SymbolStatus {
    status: 'pending' | 'scanning' | 'complete' | 'error'
    found?: number
    error?: string
}

type SortField = 'symbol' | 'stock_price' | 'strike' | 'expiration' | 'dte' | 'premium' | 'premium_pct' | 'monthly_return' | 'annual_return' | 'iv' | 'volume'
type SortDirection = 'asc' | 'desc'

const DTE_OPTIONS = [7, 14, 30, 45]
const DTE_FILTER_OPTIONS = [0, 7, 14, 30, 45]
const COLORS = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']

// PRESET WATCHLISTS - Always available immediately without API calls
const PRESET_WATCHLISTS: Watchlist[] = [
    { id: 'popular', name: 'Popular Stocks', source: 'predefined', symbols: ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD'] },
    { id: 'tech', name: 'Tech Leaders', source: 'predefined', symbols: ['GOOGL', 'META', 'AMZN', 'NFLX', 'CRM', 'ORCL', 'INTC', 'QCOM'] },
    { id: 'finance', name: 'Financials', source: 'predefined', symbols: ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'] },
    { id: 'retail', name: 'Retail & Consumer', source: 'predefined', symbols: ['WMT', 'COST', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD'] },
    { id: 'healthcare', name: 'Healthcare', source: 'predefined', symbols: ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'] },
    { id: 'energy', name: 'Energy', source: 'predefined', symbols: ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'VLO', 'PSX'] },
    { id: 'high-iv', name: 'High IV Plays', source: 'predefined', symbols: ['GME', 'AMC', 'RIVN', 'LCID', 'MARA', 'COIN', 'HOOD', 'PLTR'] },
]

// PLACEHOLDER WATCHLISTS - Show in dropdown immediately, symbols load from API
const PLACEHOLDER_WATCHLISTS: Watchlist[] = [
    { id: 'tradingview-main', name: 'TradingView Watchlist', source: 'tradingview', symbols: [] },
    { id: 'database-universe', name: 'Database Universe', source: 'database', symbols: [] },
    { id: 'robinhood-portfolio', name: 'Robinhood Portfolio', source: 'robinhood', symbols: [] },
]

const sourceIcons: Record<string, React.ReactNode> = {
    predefined: <List className="w-3.5 h-3.5" />,
    tradingview: <Activity className="w-3.5 h-3.5" />,
    robinhood: <Cloud className="w-3.5 h-3.5" />,
    database: <Briefcase className="w-3.5 h-3.5" />
}

const sourceColors: Record<string, string> = {
    predefined: 'text-blue-400',
    tradingview: 'text-purple-400',
    robinhood: 'text-green-400',
    database: 'text-amber-400'
}

const COLUMNS: { key: SortField; label: string; minWidth: number; defaultWidth: number }[] = [
    { key: 'symbol', label: 'Symbol', minWidth: 80, defaultWidth: 100 },
    { key: 'stock_price', label: 'Price', minWidth: 70, defaultWidth: 90 },
    { key: 'strike', label: 'Strike', minWidth: 70, defaultWidth: 90 },
    { key: 'expiration', label: 'Expiration', minWidth: 100, defaultWidth: 120 },
    { key: 'dte', label: 'DTE', minWidth: 60, defaultWidth: 80 },
    { key: 'premium', label: 'Premium', minWidth: 80, defaultWidth: 100 },
    { key: 'premium_pct', label: 'Premium %', minWidth: 90, defaultWidth: 110 },
    { key: 'monthly_return', label: 'Monthly', minWidth: 90, defaultWidth: 110 },
    { key: 'annual_return', label: 'Annual', minWidth: 80, defaultWidth: 100 },
    { key: 'iv', label: 'IV', minWidth: 60, defaultWidth: 80 },
    { key: 'volume', label: 'Volume', minWidth: 80, defaultWidth: 100 },
]

export default function PremiumScanner() {
    const queryClient = useQueryClient()
    const [selectedDTE, setSelectedDTE] = useState(30)
    const [selectedWatchlistId, setSelectedWatchlistId] = useState<string>('popular')
    const [customSymbols, setCustomSymbols] = useState('')
    const [maxPrice, setMaxPrice] = useState(0)  // 0 = no filter (show all prices)
    const [minPremium, setMinPremium] = useState(1.0)
    const [showFilters, setShowFilters] = useState(false)
    const [showWatchlistDropdown, setShowWatchlistDropdown] = useState(false)
    const [showHistoryDropdown, setShowHistoryDropdown] = useState(false)

    // Scan state
    const [isScanning, setIsScanning] = useState(false)
    const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null)
    const [scanResults, setScanResults] = useState<ScanResult[]>([])
    const [scanId, setScanId] = useState<string | null>(null)
    const [scanError, setScanError] = useState<string | null>(null)
    const [symbolStatuses, setSymbolStatuses] = useState<Record<string, SymbolStatus>>({})
    const [allSymbols, setAllSymbols] = useState<string[]>([])

    // Sort state
    const [sortField, setSortField] = useState<SortField>('monthly_return')
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

    // Column widths state for resizing
    const [columnWidths, setColumnWidths] = useState<Record<string, number>>(() =>
        COLUMNS.reduce((acc, col) => ({ ...acc, [col.key]: col.defaultWidth }), {})
    )

    // DTE filter for results (0 = All)
    const [resultDTEFilter, setResultDTEFilter] = useState(0)

    // Unique stocks filter
    const [showUniqueOnly, setShowUniqueOnly] = useState(false)

    // Resizing refs
    const resizingRef = useRef<{ column: string; startX: number; startWidth: number } | null>(null)
    const eventSourceRef = useRef<EventSource | null>(null)
    const abortControllerRef = useRef<AbortController | null>(null)

    // Fetch additional watchlists from API (TradingView, Robinhood, database)
    // PRESET_WATCHLISTS are always available immediately - API watchlists merge in when ready
    // Timeout auto-detected by axios interceptor (60s for /scanner/watchlists)
    const { data: apiWatchlistsData, isFetching: watchlistsFetching } = useQuery<WatchlistsResponse>({
        queryKey: ['scanner-watchlists'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/watchlists')
            return data
        },
        staleTime: 1000 * 60 * 30,  // 30 minutes - watchlists rarely change
        gcTime: 1000 * 60 * 60,     // Keep in cache for 1 hour
        retry: 1,                    // Don't retry too much - presets work fine
        refetchOnWindowFocus: false, // Don't refetch on focus
    })

    // Merge preset watchlists with API watchlists - presets always first and always available
    const watchlistsData = useMemo<WatchlistsResponse>(() => {
        const apiWatchlists = apiWatchlistsData?.watchlists || []

        // Filter out duplicates (API might return same preset watchlists)
        const uniqueApiWatchlists = apiWatchlists.filter(
            apiWl => !PRESET_WATCHLISTS.some(preset => preset.id === apiWl.id)
        )

        // For placeholder watchlists, use API data if available, otherwise show placeholder
        // This ensures TradingView/Database/Robinhood options always appear in dropdown
        const placeholdersWithData = PLACEHOLDER_WATCHLISTS.map(placeholder => {
            // Find matching API watchlist by source type
            const apiMatch = uniqueApiWatchlists.find(
                apiWl => apiWl.source === placeholder.source
            )
            if (apiMatch) {
                return apiMatch  // Use API data with actual symbols
            }
            return placeholder  // Show placeholder (loading state)
        })

        // Remove API watchlists that were used in placeholders
        const remainingApiWatchlists = uniqueApiWatchlists.filter(
            apiWl => !PLACEHOLDER_WATCHLISTS.some(ph => ph.source === apiWl.source)
        )

        // Final order: Presets first, then placeholders (with data), then any other API watchlists
        const allWatchlists = [...PRESET_WATCHLISTS, ...placeholdersWithData, ...remainingApiWatchlists]
        return {
            watchlists: allWatchlists,
            total: allWatchlists.length,
            generated_at: new Date().toISOString()
        }
    }, [apiWatchlistsData])

    // Fetch scan history - non-critical, can fail gracefully
    // Timeout auto-detected by axios interceptor (30s standard)
    const { data: historyData, refetch: refetchHistory } = useQuery<{ history: ScanHistory[]; count: number }>({
        queryKey: ['scan-history'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/history?limit=10')
            return data
        },
        staleTime: 60000,            // 1 minute
        retry: 0,                    // Don't retry - page works without history
        refetchOnWindowFocus: false,
    })

    // Fetch pre-computed stored premiums (background scanner data)
    // This is nice-to-have on initial load - user can trigger scan manually
    // Timeout auto-detected by axios interceptor (60s for /scanner/stored-premiums)
    const { data: storedPremiums, isLoading: storedPremiumsLoading, refetch: refetchStoredPremiums } = useQuery<{
        results: ScanResult[];
        count: number;
        last_updated: string;
    }>({
        queryKey: ['stored-premiums'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/stored-premiums?limit=500')
            return data
        },
        staleTime: 120000,           // 2 minutes
        retry: 0,                    // Don't retry - user can refresh manually
        refetchOnWindowFocus: false,
    })

    // DTE comparison chart data - not critical for page functionality
    // Timeout auto-detected by axios interceptor (30s standard)
    const { data: comparisonData, isLoading: comparisonLoading, refetch: refetchComparison } = useQuery({
        queryKey: ['dte-comparison'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/dte-comparison')
            return data
        },
        staleTime: 300000,           // 5 minutes
        retry: 0,                    // Don't retry - chart is optional
        refetchOnWindowFocus: false,
    })

    // Fetch AI-powered CSP recommendations - expensive operation
    // Timeout auto-detected by axios interceptor (120s for /ai-picks)
    const {
        data: aiPicksData,
        isLoading: aiPicksLoading,
        error: aiPicksError,
        refetch: refetchAIPicks,
        isFetching: aiPicksFetching
    } = useQuery<AIPicksResponse>({
        queryKey: ['ai-csp-picks', selectedDTE],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/ai-picks', {
                params: {
                    min_dte: 7,
                    max_dte: selectedDTE + 15,  // Slightly wider range for AI analysis
                    min_premium_pct: 0.5
                },
            })
            return data
        },
        staleTime: 300000,           // 5 minutes - matches backend cache
        gcTime: 600000,              // 10 minutes cache
        retry: 0,                    // Don't retry expensive AI calls
        refetchOnWindowFocus: false, // Don't refetch expensive AI on focus
        enabled: true,               // Always fetch on load
    })

    // Cleanup EventSource and AbortController on unmount
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close()
            }
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
            }
        }
    }, [])

    // Auto-load stored premiums on mount (pre-computed by background scanner)
    useEffect(() => {
        if (storedPremiums?.results?.length && scanResults.length === 0 && !isScanning) {
            // Map stored premium fields to ScanResult format
            const mappedResults = storedPremiums.results.map(opp => ({
                ...opp,
                annual_return: opp.annual_return || (opp as any).annualized_return || 0,
                premium_pct: opp.premium_pct || 0,
                monthly_return: opp.monthly_return || 0,
                iv: (opp as any).implied_volatility || opp.iv || 0,
            }))
            setScanResults(mappedResults)
            setScanProgress({ type: 'complete', count: mappedResults.length })
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [storedPremiums, isScanning])

    // Auto-select first watchlist if default 'popular' doesn't exist
    useEffect(() => {
        if (watchlistsData?.watchlists?.length && !watchlistsData.watchlists.find(wl => wl.id === selectedWatchlistId)) {
            setSelectedWatchlistId(watchlistsData.watchlists[0].id)
        }
    }, [watchlistsData, selectedWatchlistId])

    // Group watchlists by source
    const watchlistsBySource = useMemo(() => {
        if (!watchlistsData?.watchlists) return {}
        return watchlistsData.watchlists.reduce((acc, wl) => {
            const source = wl.source
            if (!acc[source]) acc[source] = []
            acc[source].push(wl)
            return acc
        }, {} as Record<string, Watchlist[]>)
    }, [watchlistsData])

    // Get selected watchlist
    const selectedWatchlist = useMemo(() => {
        return watchlistsData?.watchlists.find(wl => wl.id === selectedWatchlistId)
    }, [watchlistsData, selectedWatchlistId])

    // Cancel an in-progress scan
    const handleCancelScan = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort()
            abortControllerRef.current = null
        }
        setIsScanning(false)
        setScanProgress(null)
        setScanError('Scan cancelled by user')
    }, [])

    // Handle scan by filtering database results (instant, uses pre-computed data from Robinhood background scanner)
    // Uses AbortController for cancellation and request deduplication
    const handleScan = useCallback(async () => {
        // Prevent duplicate scans - abort any in-progress scan first
        if (abortControllerRef.current) {
            abortControllerRef.current.abort()
        }

        const symbols = customSymbols.trim()
            ? customSymbols.split(',').map(s => s.trim().toUpperCase())
            : selectedWatchlist?.symbols || []

        if (symbols.length === 0) return

        // Create new AbortController for this request
        abortControllerRef.current = new AbortController()
        const signal = abortControllerRef.current.signal

        // Reset state
        setIsScanning(true)
        setScanProgress(null)
        setScanResults([])
        setScanError(null)
        setScanId(null)
        setSymbolStatuses({})
        setAllSymbols(symbols)

        // Initialize all symbols as scanning
        const initialStatus: Record<string, SymbolStatus> = {}
        symbols.forEach(s => {
            initialStatus[s] = { status: 'scanning' }
        })
        setSymbolStatuses(initialStatus)

        try {
            // Query stored premiums - use watchlist_id for predefined watchlists (avoids 431 error),
            // or symbols param for custom input
            // Note: max_price is filtered client-side in the table, not in the API call
            const isCustomInput = customSymbols.trim().length > 0
            const params: Record<string, string | number> = {
                min_premium_pct: minPremium,
                max_dte: selectedDTE,
                min_dte: 0,
                limit: 500
            }

            if (isCustomInput) {
                // Custom symbols: send in URL (typically small list)
                params.symbols = symbols.join(',')
            } else if (selectedWatchlistId) {
                // Predefined watchlist: let backend look up symbols (avoids 431 for large watchlists)
                params.watchlist_id = selectedWatchlistId
            }

            const { data } = await axiosInstance.get('/scanner/stored-premiums', {
                params,
                signal,  // Pass AbortController signal for cancellation
                timeout: 120000  // 2 minute timeout (matches backend MAX_SCAN_TIMEOUT)
            })

            // Map response to ScanResult format
            const mappedResults: ScanResult[] = (data.results || []).map((opp: any) => ({
                symbol: opp.symbol,
                stock_price: opp.stock_price || 0,
                strike: opp.strike || 0,
                expiration: opp.expiration || '',
                dte: opp.dte || 0,
                premium: opp.premium || opp.mid || 0,
                premium_pct: opp.premium_pct || 0,
                monthly_return: opp.monthly_return || 0,
                annual_return: opp.annualized_return || opp.annual_return || 0,
                iv: opp.implied_volatility || opp.iv || 0,
                volume: opp.volume || 0,
                open_interest: opp.open_interest || 0,
                bid_ask_spread: (opp.ask && opp.bid) ? (opp.ask - opp.bid) : 0
            }))

            // Update symbol statuses based on found opportunities
            const finalStatus: Record<string, SymbolStatus> = {}
            symbols.forEach(s => {
                const foundCount = mappedResults.filter(r => r.symbol === s).length
                finalStatus[s] = { status: 'complete', found: foundCount }
            })
            setSymbolStatuses(finalStatus)

            setScanResults(mappedResults)
            setScanProgress({ type: 'complete', count: mappedResults.length })
            setIsScanning(false)

        } catch (error: any) {
            // Don't show error if request was intentionally cancelled
            if (error.name === 'CanceledError' || error.code === 'ERR_CANCELED') {
                console.log('Scan cancelled')
                return  // State already handled by handleCancelScan
            }

            console.error('Scan error:', error)

            // Provide user-friendly error messages
            let errorMessage = 'Failed to fetch stored premiums'
            if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
                errorMessage = 'Request timed out - try scanning fewer symbols or check backend status'
            } else if (error.response?.status === 500) {
                errorMessage = `Server error: ${error.response?.data?.detail || 'Internal server error'}`
            } else if (error.response?.status === 503) {
                errorMessage = 'Service temporarily unavailable - backend may be restarting'
            } else if (error.message) {
                errorMessage = error.message
            }

            setScanError(errorMessage)
            setIsScanning(false)
            abortControllerRef.current = null  // Clean up abort controller

            // Mark all symbols as error
            const errorStatus: Record<string, SymbolStatus> = {}
            symbols.forEach(s => {
                errorStatus[s] = { status: 'error', error: errorMessage }
            })
            setSymbolStatuses(errorStatus)
        }
    }, [customSymbols, selectedWatchlist, minPremium, maxPrice, selectedDTE, selectedWatchlistId])

    // Load a previous scan
    const loadPreviousScan = useCallback(async (scanHistoryId: string) => {
        try {
            const { data } = await axiosInstance.get(`/scanner/history/${scanHistoryId}`)
            setScanResults(data.results || [])
            setScanId(data.scan_id)
            setScanProgress({ type: 'complete', count: data.result_count })
            setShowHistoryDropdown(false)
        } catch (error) {
            console.error('Failed to load scan:', error)
        }
    }, [])

    // Handle column sort
    const handleSort = useCallback((field: SortField) => {
        if (sortField === field) {
            setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
        } else {
            setSortField(field)
            setSortDirection('desc')
        }
    }, [sortField])

    // Handle column resize
    const handleResizeStart = useCallback((e: React.MouseEvent, column: string) => {
        e.preventDefault()
        resizingRef.current = {
            column,
            startX: e.clientX,
            startWidth: columnWidths[column]
        }

        const handleMouseMove = (e: MouseEvent) => {
            if (!resizingRef.current) return
            const diff = e.clientX - resizingRef.current.startX
            const colConfig = COLUMNS.find(c => c.key === resizingRef.current?.column)
            const minWidth = colConfig?.minWidth || 60
            const newWidth = Math.max(minWidth, resizingRef.current.startWidth + diff)
            setColumnWidths(prev => ({ ...prev, [resizingRef.current!.column]: newWidth }))
        }

        const handleMouseUp = () => {
            resizingRef.current = null
            document.removeEventListener('mousemove', handleMouseMove)
            document.removeEventListener('mouseup', handleMouseUp)
        }

        document.addEventListener('mousemove', handleMouseMove)
        document.addEventListener('mouseup', handleMouseUp)
    }, [columnWidths])

    // Open TradingView SuperChart
    const openTradingView = (symbol: string) => {
        window.open(`https://www.tradingview.com/chart/?symbol=${symbol}`, '_blank')
    }

    const chartData = comparisonData?.summary ? DTE_OPTIONS.map((dte, index) => ({
        dte: `${dte} DTE`,
        avgReturn: comparisonData.summary[dte]?.avg_monthly_return || 0,
        count: comparisonData.summary[dte]?.count || 0,
        color: COLORS[index]
    })) : []

    // Process results: filter by max price, DTE, unique, then sort
    const processedResults = useMemo(() => {
        let results = [...scanResults]

        // Filter by max stock price (client-side filter for table display)
        if (maxPrice > 0) {
            results = results.filter((r: ScanResult) => r.stock_price <= maxPrice)
        }

        if (resultDTEFilter > 0) {
            results = results.filter((r: ScanResult) => r.dte <= resultDTEFilter)
        }

        if (showUniqueOnly) {
            const symbolMap = new Map<string, ScanResult>()
            for (const result of results) {
                const existing = symbolMap.get(result.symbol)
                if (!existing || result.monthly_return > existing.monthly_return) {
                    symbolMap.set(result.symbol, result)
                }
            }
            results = Array.from(symbolMap.values())
        }

        results.sort((a: ScanResult, b: ScanResult) => {
            let aVal = a[sortField]
            let bVal = b[sortField]

            if (sortField === 'symbol' || sortField === 'expiration') {
                const comparison = String(aVal).localeCompare(String(bVal))
                return sortDirection === 'asc' ? comparison : -comparison
            }

            const numA = Number(aVal) || 0
            const numB = Number(bVal) || 0
            return sortDirection === 'asc' ? numA - numB : numB - numA
        })

        return results
    }, [scanResults, maxPrice, resultDTEFilter, showUniqueOnly, sortField, sortDirection])

    const getSortIcon = (field: SortField) => {
        if (sortField !== field) {
            return <ArrowUpDown className="w-3.5 h-3.5 text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity" />
        }
        return sortDirection === 'asc'
            ? <ArrowUp className="w-3.5 h-3.5 text-primary" />
            : <ArrowDown className="w-3.5 h-3.5 text-primary" />
    }

    // Format relative time
    const formatRelativeTime = (dateStr: string) => {
        const date = new Date(dateStr)
        const now = new Date()
        const diffMs = now.getTime() - date.getTime()
        const diffMins = Math.floor(diffMs / 60000)
        if (diffMins < 1) return 'Just now'
        if (diffMins < 60) return `${diffMins}m ago`
        const diffHours = Math.floor(diffMins / 60)
        if (diffHours < 24) return `${diffHours}h ago`
        return date.toLocaleDateString()
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        Premium Scanner
                    </h1>
                    <p className="page-subtitle">Find the best CSP opportunities by DTE</p>
                </div>
                <div className="flex items-center gap-2">
                    {/* Scan History Button */}
                    <div className="relative">
                        <button
                            onClick={() => setShowHistoryDropdown(!showHistoryDropdown)}
                            className={`btn-icon relative ${showHistoryDropdown ? 'bg-primary/20' : ''}`}
                            title="Previous Scans"
                        >
                            <History className="w-5 h-5" />
                            {historyData?.count ? (
                                <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary text-white text-[10px] rounded-full flex items-center justify-center">
                                    {historyData.count}
                                </span>
                            ) : null}
                        </button>

                        {/* History Dropdown */}
                        {showHistoryDropdown && (
                            <div className="absolute right-0 mt-2 w-80 bg-slate-900 border border-slate-700 rounded-xl shadow-xl z-50 max-h-96 overflow-y-auto">
                                <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
                                    <Database className="w-4 h-4 text-slate-400" />
                                    <span className="font-semibold text-white">Previous Scans</span>
                                </div>
                                {historyData?.history?.length ? (
                                    historyData.history.map((scan) => (
                                        <button
                                            key={scan.scan_id}
                                            onClick={() => loadPreviousScan(scan.scan_id)}
                                            className="w-full px-4 py-3 text-left hover:bg-slate-800/60 border-b border-slate-700/50 last:border-b-0"
                                        >
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-sm font-medium text-white">
                                                    {scan.symbol_count} symbols • {scan.dte} DTE
                                                </span>
                                                <span className="text-xs text-slate-500">
                                                    {formatRelativeTime(scan.created_at)}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-emerald-400">
                                                    {scan.result_count} opportunities
                                                </span>
                                                <span className="text-xs text-slate-500">
                                                    ID: {scan.scan_id}
                                                </span>
                                            </div>
                                        </button>
                                    ))
                                ) : (
                                    <div className="px-4 py-6 text-center text-slate-500">
                                        No previous scans
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <button
                        onClick={() => refetchComparison()}
                        className="btn-icon"
                    >
                        <RefreshCw className="w-5 h-5" />
                    </button>
                </div>
            </header>

            {/* Scan Progress Indicator with Per-Symbol Grid */}
            {isScanning && allSymbols.length > 0 && (
                <div className="glass-card p-5 border-primary/30 bg-primary/5">
                    {/* Header with overall progress */}
                    <div className="flex items-center gap-4 mb-4">
                        <Loader2 className="w-6 h-6 text-primary animate-spin" />
                        <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-white">
                                    Scanning {allSymbols.length} Symbols
                                </span>
                                <span className="text-sm text-primary font-mono">
                                    {scanProgress?.percent || Math.round((Object.values(symbolStatuses).filter(s => s.status === 'complete' || s.status === 'error').length / allSymbols.length) * 100)}%
                                </span>
                            </div>
                            {/* Overall progress bar */}
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-primary to-violet-500 transition-all duration-300"
                                    style={{ width: `${scanProgress?.percent || Math.round((Object.values(symbolStatuses).filter(s => s.status === 'complete' || s.status === 'error').length / allSymbols.length) * 100)}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Stats row */}
                    <div className="flex items-center gap-6 mb-4 text-sm">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-slate-600" />
                            <span className="text-slate-400">
                                Pending: {Object.values(symbolStatuses).filter(s => s.status === 'pending').length}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-amber-500 animate-pulse" />
                            <span className="text-amber-400">
                                Scanning: {Object.values(symbolStatuses).filter(s => s.status === 'scanning').length}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-emerald-500" />
                            <span className="text-emerald-400">
                                Complete: {Object.values(symbolStatuses).filter(s => s.status === 'complete').length}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded bg-red-500" />
                            <span className="text-red-400">
                                Errors: {Object.values(symbolStatuses).filter(s => s.status === 'error').length}
                            </span>
                        </div>
                        <div className="ml-auto text-emerald-400 font-medium">
                            {Object.values(symbolStatuses).reduce((sum, s) => sum + (s.found || 0), 0)} opportunities found
                        </div>
                    </div>

                    {/* Symbol Grid */}
                    <div className="grid grid-cols-8 sm:grid-cols-10 md:grid-cols-12 lg:grid-cols-16 xl:grid-cols-20 gap-1.5">
                        {allSymbols.map(symbol => {
                            const status = symbolStatuses[symbol]
                            const statusClasses = {
                                pending: 'bg-slate-700/50 text-slate-500 border-slate-600/50',
                                scanning: 'bg-amber-500/20 text-amber-400 border-amber-500/50 animate-pulse',
                                complete: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50',
                                error: 'bg-red-500/20 text-red-400 border-red-500/50'
                            }
                            const statusClass = status?.status ? statusClasses[status.status] : statusClasses.pending

                            return (
                                <div
                                    key={symbol}
                                    className={`relative px-1.5 py-1 text-xs font-medium rounded border text-center truncate ${statusClass}`}
                                    title={status?.error || (status?.found !== undefined ? `${symbol}: ${status.found} found` : symbol)}
                                >
                                    {symbol}
                                    {status?.status === 'complete' && status.found !== undefined && status.found > 0 && (
                                        <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-emerald-500 text-white text-[8px] rounded-full flex items-center justify-center font-bold">
                                            {status.found > 9 ? '9+' : status.found}
                                        </span>
                                    )}
                                    {status?.status === 'error' && (
                                        <XCircle className="absolute -top-1 -right-1 w-3.5 h-3.5 text-red-500" />
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Scan Complete Success */}
            {!isScanning && scanProgress?.type === 'complete' && scanId && (
                <div className="glass-card p-4 border-emerald-500/30 bg-emerald-500/5 flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-emerald-400">
                        Scan complete! {scanProgress.count} opportunities found and saved.
                    </span>
                    <span className="text-xs text-slate-500 ml-auto">ID: {scanId}</span>
                </div>
            )}

            {/* Stored Premiums Info - shows when data loaded from background scanner */}
            {!isScanning && storedPremiums?.last_updated && scanResults.length > 0 && !scanId && (
                <div className="glass-card p-4 border-blue-500/30 bg-blue-500/5 flex items-center justify-between flex-wrap gap-3">
                    <div className="flex items-center gap-3">
                        <Database className="w-5 h-5 text-blue-400" />
                        <span className="text-blue-400">
                            Loaded {storedPremiums.count} opportunities from database
                        </span>
                    </div>
                    <div className="flex items-center gap-4">
                        <span className="text-xs text-slate-400">
                            Updated: {formatRelativeTime(storedPremiums.last_updated)}
                        </span>
                        <button
                            onClick={() => {
                                refetchStoredPremiums()
                                setScanResults([])  // Clear to re-trigger load
                            }}
                            className="text-xs bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors"
                        >
                            <RefreshCw className="w-3.5 h-3.5" />
                            Refresh from DB
                        </button>
                    </div>
                </div>
            )}

            {/* Loading Stored Premiums */}
            {storedPremiumsLoading && scanResults.length === 0 && (
                <div className="glass-card p-4 border-slate-500/30 flex items-center gap-3">
                    <Loader2 className="w-5 h-5 text-slate-400 animate-spin" />
                    <span className="text-slate-400">Loading pre-computed premiums...</span>
                </div>
            )}

            {/* Scan Error */}
            {scanError && (
                <div className="glass-card p-4 border-red-500/30 bg-red-500/5 flex items-center gap-3">
                    <XCircle className="w-5 h-5 text-red-400" />
                    <span className="text-red-400">{scanError}</span>
                </div>
            )}

            {/* DTE Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {DTE_OPTIONS.map((dte, index) => {
                    const summary = comparisonData?.summary?.[dte]
                    const isSelected = selectedDTE === dte
                    return (
                        <button
                            key={dte}
                            onClick={() => setSelectedDTE(dte)}
                            className={`glass-card p-5 text-left transition-all group ${
                                isSelected
                                    ? 'border-primary ring-2 ring-primary/20'
                                    : 'hover:border-slate-600'
                            }`}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-lg font-bold text-white">{dte} DTE</span>
                                <div
                                    className="w-3 h-3 rounded-full ring-2 ring-white/10"
                                    style={{ backgroundColor: COLORS[index] }}
                                />
                            </div>
                            {comparisonLoading ? (
                                <div className="space-y-2">
                                    <div className="animate-pulse h-8 bg-slate-700 rounded" />
                                    <div className="animate-pulse h-4 bg-slate-700/50 rounded w-2/3" />
                                </div>
                            ) : (
                                <>
                                    <div className="text-2xl font-bold text-emerald-400">
                                        {summary?.avg_monthly_return?.toFixed(1) || 0}%
                                    </div>
                                    <div className="text-sm text-slate-400">
                                        Monthly Return
                                    </div>
                                    <div className="mt-2 pt-2 border-t border-slate-700/50">
                                        <span className="text-xs text-slate-500">{summary?.count || 0} opportunities</span>
                                    </div>
                                </>
                            )}
                        </button>
                    )
                })}
            </div>

            {/* Chart and Scanner */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* DTE Comparison Chart */}
                <div className="lg:col-span-1 glass-card p-5">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                            <BarChart3 className="w-4 h-4 text-blue-400" />
                        </div>
                        <h3 className="font-semibold text-white">Returns by DTE</h3>
                    </div>
                    <div className="h-64 min-h-[256px] w-full" style={{ minHeight: 256 }}>
                        <ResponsiveContainer width="100%" height={256} minWidth={0} minHeight={0} debounce={50}>
                            <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.5)" />
                                <XAxis dataKey="dte" stroke="#94a3b8" fontSize={12} />
                                <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                        border: '1px solid rgba(51, 65, 85, 0.5)',
                                        borderRadius: '0.75rem'
                                    }}
                                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Avg Return']}
                                />
                                <Bar dataKey="avgReturn" radius={[6, 6, 0, 0]}>
                                    {chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Scanner Controls */}
                <div className="lg:col-span-2 glass-card p-5">
                    <div className="flex items-center justify-between mb-5">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                                <Search className="w-4 h-4 text-purple-400" />
                            </div>
                            <h3 className="font-semibold text-white">Scan for Premiums</h3>
                        </div>
                        <button
                            onClick={() => setShowFilters(!showFilters)}
                            className={`flex items-center gap-2 text-sm px-3 py-1.5 rounded-lg transition-colors ${
                                showFilters ? 'bg-primary/20 text-primary' : 'text-slate-400 hover:text-white hover:bg-white/5'
                            }`}
                        >
                            <Filter className="w-4 h-4" />
                            Filters
                            <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                        </button>
                    </div>

                    {/* Watchlist Dropdown */}
                    <div className="mb-4 relative">
                        <label className="block text-sm text-slate-400 mb-2">Select Watchlist</label>
                        <button
                            onClick={() => setShowWatchlistDropdown(!showWatchlistDropdown)}
                            className="w-full px-4 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl text-left flex items-center justify-between hover:bg-slate-700/60 transition-colors"
                        >
                            <div className="flex items-center gap-3">
                                {selectedWatchlist && (
                                    <span className={sourceColors[selectedWatchlist.source]}>
                                        {sourceIcons[selectedWatchlist.source]}
                                    </span>
                                )}
                                <span className="text-white">
                                    {selectedWatchlist?.name || 'Select a watchlist'}
                                </span>
                                {selectedWatchlist && (
                                    selectedWatchlist.symbols.length === 0 && selectedWatchlist.source !== 'predefined' ? (
                                        <span className="flex items-center gap-1.5 text-xs text-slate-500">
                                            <Loader2 className="w-3 h-3 animate-spin" />
                                            loading symbols...
                                        </span>
                                    ) : (
                                        <span className="text-xs text-slate-500">
                                            ({selectedWatchlist.symbols.length} symbols)
                                        </span>
                                    )
                                )}
                                {watchlistsFetching && !selectedWatchlist?.symbols.length && (
                                    <Loader2 className="w-3.5 h-3.5 text-slate-500 animate-spin" />
                                )}
                            </div>
                            <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${showWatchlistDropdown ? 'rotate-180' : ''}`} />
                        </button>

                        {showWatchlistDropdown && (
                            <div className="absolute z-50 w-full mt-2 bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-h-80 overflow-y-auto">
                                {Object.entries(watchlistsBySource).map(([source, watchlists]) => (
                                    <div key={source}>
                                        <div className="px-4 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wide bg-slate-800/50 flex items-center gap-2">
                                            <span className={sourceColors[source]}>{sourceIcons[source]}</span>
                                            {source === 'predefined' ? 'Preset Watchlists' :
                                             source === 'database' ? 'Database Watchlists' :
                                             source === 'tradingview' ? 'TradingView' :
                                             source === 'robinhood' ? 'Robinhood Portfolio' : source}
                                        </div>
                                        {watchlists.map((wl) => {
                                            const isLoading = wl.symbols.length === 0 && wl.source !== 'predefined'
                                            return (
                                                <button
                                                    key={wl.id}
                                                    onClick={() => {
                                                        if (!isLoading) {
                                                            setSelectedWatchlistId(wl.id)
                                                            setCustomSymbols('')
                                                            setShowWatchlistDropdown(false)
                                                        }
                                                    }}
                                                    disabled={isLoading}
                                                    className={`w-full px-4 py-3 text-left hover:bg-slate-800/60 flex items-center justify-between ${
                                                        selectedWatchlistId === wl.id ? 'bg-primary/10 text-primary' : 'text-white'
                                                    } ${isLoading ? 'opacity-60 cursor-wait' : ''}`}
                                                >
                                                    <span>{wl.name}</span>
                                                    {isLoading ? (
                                                        <span className="flex items-center gap-1.5 text-xs text-slate-500">
                                                            <Loader2 className="w-3 h-3 animate-spin" />
                                                            loading...
                                                        </span>
                                                    ) : (
                                                        <span className="text-xs text-slate-500">
                                                            {wl.symbols.length} symbols
                                                        </span>
                                                    )}
                                                </button>
                                            )
                                        })}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Selected Symbols Preview */}
                    {selectedWatchlist && !customSymbols && (
                        <div className="mb-4 p-3 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            {selectedWatchlist.symbols.length > 0 ? (
                                <div className="flex flex-wrap gap-1.5">
                                    {selectedWatchlist.symbols.slice(0, 15).map((symbol) => (
                                        <span
                                            key={symbol}
                                            className="px-2 py-1 text-xs font-medium bg-slate-700/50 text-slate-300 rounded-lg"
                                        >
                                            {symbol}
                                        </span>
                                    ))}
                                    {selectedWatchlist.symbols.length > 15 && (
                                        <span className="px-2 py-1 text-xs text-slate-500">
                                            +{selectedWatchlist.symbols.length - 15} more
                                        </span>
                                    )}
                                </div>
                            ) : selectedWatchlist.source !== 'predefined' ? (
                                <div className="flex items-center gap-2 text-slate-400">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span className="text-sm">Loading symbols from {selectedWatchlist.source}...</span>
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* Custom Symbols Input */}
                    <div className="mb-4">
                        <label className="block text-sm text-slate-400 mb-2">Or Enter Custom Symbols</label>
                        <input
                            type="text"
                            value={customSymbols}
                            onChange={(e) => setCustomSymbols(e.target.value)}
                            placeholder="AAPL, MSFT, NVDA, TSLA..."
                            className="input-field"
                        />
                    </div>

                    {/* Advanced Filters */}
                    {showFilters && (
                        <div className="grid grid-cols-2 gap-4 mb-5 p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Max Stock Price</label>
                                <div className="relative">
                                    <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                    <input
                                        type="number"
                                        value={maxPrice}
                                        onChange={(e) => setMaxPrice(Number(e.target.value))}
                                        className="input-field pl-9"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Min Premium %</label>
                                <div className="relative">
                                    <Percent className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={minPremium}
                                        onChange={(e) => setMinPremium(Number(e.target.value))}
                                        className="input-field pl-9"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Scan Button with Cancel */}
                    <div className="flex gap-2">
                        <button
                            onClick={handleScan}
                            disabled={isScanning || (!customSymbols && (!selectedWatchlist || selectedWatchlist.symbols.length === 0))}
                            className="btn-primary flex-1 py-3 flex items-center justify-center gap-2 text-base"
                        >
                            {isScanning ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Scanning... {scanProgress?.percent || 0}%
                                </>
                            ) : (
                                <>
                                    <Search className="w-5 h-5" />
                                    Scan {selectedDTE} DTE Premiums
                                </>
                            )}
                        </button>
                        {isScanning && (
                            <button
                                onClick={handleCancelScan}
                                className="px-4 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl border border-red-500/30 flex items-center gap-2 transition-colors"
                                title="Cancel scan"
                            >
                                <XCircle className="w-5 h-5" />
                                Cancel
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* AI-Powered CSP Picks Section */}
            <div className="glass-card overflow-hidden">
                <div className="p-5 border-b border-slate-700/50 bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                                <Brain className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                    AI-Powered CSP Picks
                                    {aiPicksData?.cached && (
                                        <span className="text-xs bg-slate-700/50 text-slate-400 px-2 py-0.5 rounded-full">
                                            cached
                                        </span>
                                    )}
                                </h3>
                                <p className="text-sm text-slate-400">
                                    DeepSeek R1 analyzed {aiPicksData?.total_candidates || 0} opportunities
                                </p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            {aiPicksData?.market_context && (
                                <div className="hidden md:block max-w-xs text-xs text-slate-400 italic">
                                    "{aiPicksData.market_context}"
                                </div>
                            )}
                            <button
                                onClick={() => refetchAIPicks()}
                                disabled={aiPicksFetching}
                                className="btn-secondary flex items-center gap-2"
                            >
                                <RefreshCw className={`w-4 h-4 ${aiPicksFetching ? 'animate-spin' : ''}`} />
                                {aiPicksFetching ? 'Analyzing...' : 'Refresh AI'}
                            </button>
                        </div>
                    </div>
                </div>

                <div className="p-5">
                    {/* Loading State */}
                    {aiPicksLoading && (
                        <div className="flex items-center justify-center py-12">
                            <div className="text-center">
                                <Loader2 className="w-10 h-10 text-violet-400 animate-spin mx-auto mb-3" />
                                <p className="text-slate-400">DeepSeek R1 is analyzing opportunities...</p>
                                <p className="text-xs text-slate-500 mt-1">This may take 10-30 seconds for deep reasoning</p>
                            </div>
                        </div>
                    )}

                    {/* Error State */}
                    {aiPicksError && (
                        <div className="flex items-center justify-center py-8">
                            <div className="text-center">
                                <AlertCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
                                <p className="text-red-400">Failed to load AI recommendations</p>
                                <p className="text-xs text-slate-500 mt-1">
                                    {(aiPicksError as Error).message || 'Please try again'}
                                </p>
                                <button
                                    onClick={() => refetchAIPicks()}
                                    className="mt-4 btn-primary text-sm"
                                >
                                    Retry
                                </button>
                            </div>
                        </div>
                    )}

                    {/* AI Picks Grid */}
                    {!aiPicksLoading && !aiPicksError && aiPicksData?.picks && (
                        <>
                            {aiPicksData.picks.length > 0 ? (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                                    {aiPicksData.picks.map((pick, index) => (
                                        <AIPickCard key={`${pick.symbol}-${pick.strike}-${pick.expiration}`} pick={pick} rank={index + 1} />
                                    ))}
                                </div>
                            ) : (
                                <div className="text-center py-8">
                                    <Star className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                                    <p className="text-slate-400">No AI picks available yet</p>
                                    <p className="text-xs text-slate-500 mt-1">
                                        Run a background scan first to populate the database
                                    </p>
                                </div>
                            )}

                            {/* Metadata Footer */}
                            {aiPicksData.picks.length > 0 && (
                                <div className="mt-4 pt-4 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-500">
                                    <span>
                                        Model: {aiPicksData.model || 'DeepSeek R1'}
                                    </span>
                                    <span>
                                        Generated: {aiPicksData.generated_at ? new Date(aiPicksData.generated_at).toLocaleString() : 'Unknown'}
                                    </span>
                                </div>
                            )}
                        </>
                    )}

                    {/* No Data State */}
                    {!aiPicksLoading && !aiPicksError && !aiPicksData && (
                        <div className="text-center py-8">
                            <Brain className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                            <p className="text-slate-400">AI recommendations will appear here</p>
                            <p className="text-xs text-slate-500 mt-1">
                                Click "Refresh AI" to generate fresh picks
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Results */}
            {scanResults.length > 0 && (
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50 bg-slate-900/30">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                    <Zap className="w-5 h-5 text-amber-400" />
                                    {processedResults.length} Opportunities Found
                                    {showUniqueOnly && <span className="text-xs text-slate-400 ml-2">(unique stocks)</span>}
                                </h3>
                                <p className="text-sm text-slate-400">Click column headers to sort • Drag edges to resize • Click symbol for TradingView</p>
                            </div>
                            <button className="btn-secondary flex items-center gap-2">
                                <Download className="w-4 h-4" />
                                Export
                            </button>
                        </div>

                        {/* Filter Controls */}
                        <div className="flex items-center gap-4 flex-wrap">
                            {/* Max Stock Price Filter */}
                            <div className="flex items-center gap-2">
                                <span className="text-sm text-slate-400">Max Price:</span>
                                <div className="relative">
                                    <DollarSign className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-emerald-400" />
                                    <input
                                        type="number"
                                        value={maxPrice || ''}
                                        onChange={(e) => setMaxPrice(Number(e.target.value) || 0)}
                                        placeholder="Any"
                                        className="w-24 px-2 py-1.5 pl-7 text-sm bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
                                    />
                                    {maxPrice > 0 && (
                                        <button
                                            onClick={() => setMaxPrice(0)}
                                            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                                            title="Clear filter"
                                        >
                                            <XCircle className="w-3.5 h-3.5" />
                                        </button>
                                    )}
                                </div>
                            </div>

                            <div className="flex items-center gap-2">
                                <span className="text-sm text-slate-400">Max DTE:</span>
                                <div className="flex gap-1">
                                    {DTE_FILTER_OPTIONS.map(dte => (
                                        <button
                                            key={dte}
                                            onClick={() => setResultDTEFilter(dte)}
                                            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                                                resultDTEFilter === dte
                                                    ? 'bg-primary text-white'
                                                    : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white'
                                            }`}
                                        >
                                            {dte === 0 ? 'All' : `${dte}d`}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={() => setShowUniqueOnly(!showUniqueOnly)}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all ${
                                    showUniqueOnly
                                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                        : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50'
                                }`}
                            >
                                {showUniqueOnly ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4" />}
                                Unique Stocks Only
                            </button>
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full" style={{ tableLayout: 'fixed' }}>
                            <thead className="bg-slate-800/50 select-none">
                                <tr>
                                    {COLUMNS.map((col) => (
                                        <th
                                            key={col.key}
                                            style={{ width: columnWidths[col.key], minWidth: col.minWidth }}
                                            className="relative group"
                                        >
                                            <button
                                                onClick={() => handleSort(col.key)}
                                                className="w-full px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 hover:text-white transition-colors"
                                            >
                                                <span className="truncate">{col.label}</span>
                                                {getSortIcon(col.key)}
                                            </button>
                                            <div
                                                className="absolute right-0 top-0 h-full w-2 cursor-col-resize hover:bg-primary/30 transition-colors"
                                                onMouseDown={(e) => handleResizeStart(e, col.key)}
                                            />
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-700/50">
                                {processedResults.map((opp: ScanResult, idx: number) => (
                                    <tr
                                        key={`${opp.symbol}-${opp.strike}-${opp.expiration}-${idx}`}
                                        className="hover:bg-slate-800/30 transition-colors"
                                    >
                                        <td style={{ width: columnWidths['symbol'] }} className="px-3 py-3">
                                            <button
                                                onClick={() => openTradingView(opp.symbol)}
                                                className="group flex items-center gap-1.5 font-bold text-primary hover:text-primary/80 transition-colors"
                                                title={`Open ${opp.symbol} on TradingView`}
                                            >
                                                {opp.symbol}
                                                <ExternalLink className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                                            </button>
                                        </td>
                                        <td style={{ width: columnWidths['stock_price'] }} className="px-3 py-3 font-mono text-slate-300 truncate">
                                            ${opp.stock_price}
                                        </td>
                                        <td style={{ width: columnWidths['strike'] }} className="px-3 py-3 font-mono text-white truncate">
                                            ${opp.strike}
                                        </td>
                                        <td style={{ width: columnWidths['expiration'] }} className="px-3 py-3 text-slate-400 text-sm truncate">
                                            {opp.expiration}
                                        </td>
                                        <td style={{ width: columnWidths['dte'] }} className="px-3 py-3">
                                            <span className={`badge-${
                                                opp.dte <= 7 ? 'danger' : opp.dte <= 14 ? 'warning' : 'neutral'
                                            }`}>
                                                {opp.dte}d
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['premium'] }} className="px-3 py-3 font-mono font-medium text-white truncate">
                                            ${opp.premium}
                                        </td>
                                        <td style={{ width: columnWidths['premium_pct'] }} className="px-3 py-3 font-mono text-emerald-400 truncate">
                                            {opp.premium_pct.toFixed(2)}%
                                        </td>
                                        <td style={{ width: columnWidths['monthly_return'] }} className="px-3 py-3">
                                            <span className="font-bold text-emerald-400 bg-emerald-500/15 px-2 py-1 rounded-lg">
                                                {opp.monthly_return.toFixed(2)}%
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['annual_return'] }} className="px-3 py-3 font-mono text-slate-400 truncate">
                                            {opp.annual_return.toFixed(1)}%
                                        </td>
                                        <td style={{ width: columnWidths['iv'] }} className="px-3 py-3 truncate">
                                            <span className={opp.iv >= 50 ? 'text-amber-400' : 'text-slate-400'}>
                                                {opp.iv}%
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['volume'] }} className="px-3 py-3 text-slate-400 truncate">
                                            {opp.volume.toLocaleString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Quick Stats */}
            {comparisonData?.summary?.[selectedDTE] && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <QuickStatCard
                        title="Best Opportunity"
                        value={comparisonData.summary[selectedDTE]?.best_opportunity?.symbol || 'N/A'}
                        subtitle={comparisonData.summary[selectedDTE]?.best_opportunity ?
                            `${comparisonData.summary[selectedDTE].best_opportunity.monthly_return.toFixed(2)}% monthly` : ''}
                        icon={<Zap className="w-5 h-5" />}
                        iconColor="text-amber-400"
                        iconBg="bg-amber-500/20"
                        onSymbolClick={comparisonData.summary[selectedDTE]?.best_opportunity?.symbol ?
                            () => window.open(`https://www.tradingview.com/chart/?symbol=${comparisonData.summary[selectedDTE].best_opportunity.symbol}`, '_blank') : undefined}
                    />
                    <QuickStatCard
                        title="Avg Monthly Return"
                        value={`${comparisonData.summary[selectedDTE]?.avg_monthly_return?.toFixed(2) || 0}%`}
                        subtitle="Across all opportunities"
                        icon={<TrendingUp className="w-5 h-5" />}
                        iconColor="text-emerald-400"
                        iconBg="bg-emerald-500/20"
                    />
                    <QuickStatCard
                        title="Avg IV"
                        value={`${comparisonData.summary[selectedDTE]?.avg_iv?.toFixed(1) || 0}%`}
                        subtitle="Implied Volatility"
                        icon={<Activity className="w-5 h-5" />}
                        iconColor="text-purple-400"
                        iconBg="bg-purple-500/20"
                    />
                    <QuickStatCard
                        title="Opportunities"
                        value={String(comparisonData.summary[selectedDTE]?.count || 0)}
                        subtitle={`At ${selectedDTE} DTE`}
                        icon={<Target className="w-5 h-5" />}
                        iconColor="text-blue-400"
                        iconBg="bg-blue-500/20"
                    />
                </div>
            )}
        </div>
    )
}

interface QuickStatCardProps {
    title: string
    value: string
    subtitle: string
    icon: React.ReactNode
    iconColor?: string
    iconBg?: string
    onSymbolClick?: () => void
}

const QuickStatCard = memo(function QuickStatCard({ title, value, subtitle, icon, iconColor = 'text-primary', iconBg = 'bg-primary/20', onSymbolClick }: QuickStatCardProps) {
    return (
        <div className="stat-card group">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-xs text-slate-400 mb-1 uppercase tracking-wide">{title}</p>
                    {onSymbolClick ? (
                        <button
                            onClick={onSymbolClick}
                            className="text-xl font-bold text-primary hover:text-primary/80 transition-colors flex items-center gap-1"
                        >
                            {value}
                            <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </button>
                    ) : (
                        <p className="text-xl font-bold text-white">{value}</p>
                    )}
                    <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
                </div>
                <div className={`w-10 h-10 rounded-xl ${iconBg} flex items-center justify-center ${iconColor} group-hover:scale-110 transition-transform`}>
                    {icon}
                </div>
            </div>
        </div>
    )
})
