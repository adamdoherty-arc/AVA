import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    AlertCircle, RefreshCw, Search, Filter, TrendingUp,
    Clock, DollarSign, Target, Percent, Activity, ChevronDown,
    Download, Zap, BarChart3, Sparkles, Cloud, Briefcase, List,
    ArrowUpDown, ArrowUp, ArrowDown, ExternalLink, CheckSquare, Square,
    History, Loader2, CheckCircle, XCircle, Database
} from 'lucide-react'
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
    bid: number
    ask: number
    premium: number
    premium_pct: number
    monthly_return: number
    annual_return: number
    iv: number
    volume: number
    open_interest: number
    bid_ask_spread: number
    spread_pct: number
    spread_quality: 'tight' | 'moderate' | 'wide'
    liquidity_score: number
    delta: number
    theta: number
    otm_pct: number
    collateral: number
    // AI-powered fields
    ai_score?: number
    ai_recommendation?: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'CAUTION' | 'AVOID'
    ai_confidence?: number
    fundamental_score?: number
    technical_score?: number
    greeks_score?: number
    risk_score?: number
    sentiment_score?: number
    sector?: string
    industry?: string
    pe_ratio?: number
    market_cap?: number
    beta?: number
}

interface AIInsights {
    summary: string
    top_pick: {
        symbol: string
        strike: number
        expiration: string
        ai_score: number
        recommendation: string
        monthly_return: number
        premium: number
        reason: string
    } | null
    market_conditions: {
        iv_environment: string
        avg_iv: number
        high_iv_opportunities: number
        premium_quality: string
        market_insight: string
    }
    risk_assessment: {
        risk_level: string
        avg_risk_score: number
        avg_otm_distance: number
        risk_insight: string
    }
    sector_analysis: {
        sectors: Record<string, { count: number; avg_score: number; avg_return: number }>
        best_sector: string
        best_sector_score: number
        diversification: string
    }
    recommendations: string[]
}

interface AIStats {
    avg_ai_score: number
    max_ai_score: number
    min_ai_score: number
    avg_monthly_return: number
    max_monthly_return: number
    recommendation_breakdown: Record<string, number>
    strong_buy_count: number
    buy_count: number
    hold_count: number
    top_sectors: Record<string, number>
    unique_symbols: number
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
    type: 'start' | 'progress' | 'batch' | 'complete' | 'error'
    scan_id?: string
    total?: number
    current?: number
    percent?: number
    symbols?: string[]
    count?: number
    total_so_far?: number
    results?: ScanResult[]
    error?: string
    saved?: boolean
}

type SortField = 'symbol' | 'stock_price' | 'strike' | 'expiration' | 'dte' | 'premium' | 'premium_pct' | 'monthly_return' | 'annual_return' | 'iv' | 'volume' | 'liquidity_score' | 'otm_pct' | 'collateral' | 'delta' | 'ai_score'
type SortDirection = 'asc' | 'desc'

const DTE_OPTIONS = [7, 14, 30, 45]
const DTE_FILTER_OPTIONS = [0, 7, 14, 30, 45]
const COLORS = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']

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

const AI_RECOMMENDATION_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    'STRONG_BUY': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
    'BUY': { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' },
    'HOLD': { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
    'CAUTION': { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30' },
    'AVOID': { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
}

const COLUMNS: { key: SortField; label: string; minWidth: number; defaultWidth: number }[] = [
    { key: 'ai_score', label: 'AI', minWidth: 55, defaultWidth: 65 },
    { key: 'symbol', label: 'Symbol', minWidth: 80, defaultWidth: 100 },
    { key: 'stock_price', label: 'Price', minWidth: 70, defaultWidth: 85 },
    { key: 'strike', label: 'Strike', minWidth: 70, defaultWidth: 85 },
    { key: 'otm_pct', label: 'OTM%', minWidth: 60, defaultWidth: 70 },
    { key: 'dte', label: 'DTE', minWidth: 50, defaultWidth: 60 },
    { key: 'premium', label: 'Premium', minWidth: 80, defaultWidth: 95 },
    { key: 'premium_pct', label: 'Prem%', minWidth: 70, defaultWidth: 80 },
    { key: 'monthly_return', label: 'Monthly', minWidth: 80, defaultWidth: 90 },
    { key: 'annual_return', label: 'Annual', minWidth: 70, defaultWidth: 80 },
    { key: 'iv', label: 'IV', minWidth: 50, defaultWidth: 60 },
    { key: 'delta', label: 'Delta', minWidth: 60, defaultWidth: 70 },
    { key: 'liquidity_score', label: 'Liq', minWidth: 50, defaultWidth: 55 },
    { key: 'collateral', label: 'Collateral', minWidth: 90, defaultWidth: 100 },
]

export default function PremiumScanner() {
    const queryClient = useQueryClient()
    const [selectedDTE, setSelectedDTE] = useState(30)
    const [selectedWatchlistId, setSelectedWatchlistId] = useState<string>('popular')
    const [customSymbols, setCustomSymbols] = useState('')
    const [maxPrice, setMaxPrice] = useState(250)  // Increased to include more stocks
    const [minPremium, setMinPremium] = useState(0.5)  // Lowered to show more opportunities
    const [showFilters, setShowFilters] = useState(false)
    const [showWatchlistDropdown, setShowWatchlistDropdown] = useState(false)
    const [showHistoryDropdown, setShowHistoryDropdown] = useState(false)

    // Scan state
    const [isScanning, setIsScanning] = useState(false)
    const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null)
    const [scanResults, setScanResults] = useState<ScanResult[]>([])
    const [scanId, setScanId] = useState<string | null>(null)
    const [scanError, setScanError] = useState<string | null>(null)

    // AI state
    const [aiMode, setAiMode] = useState(true)  // Enable AI by default
    const [aiInsights, setAiInsights] = useState<AIInsights | null>(null)
    const [aiStats, setAiStats] = useState<AIStats | null>(null)
    const [showAiInsights, setShowAiInsights] = useState(true)

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

    // Stock price filter for results
    const [maxStockPriceFilter, setMaxStockPriceFilter] = useState<string>('')

    // Resizing refs
    const resizingRef = useRef<{ column: string; startX: number; startWidth: number } | null>(null)
    const eventSourceRef = useRef<EventSource | null>(null)

    // Fetch watchlists
    const { data: watchlistsData, isLoading: watchlistsLoading } = useQuery<WatchlistsResponse>({
        queryKey: ['scanner-watchlists'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/watchlists')
            return data
        },
        staleTime: 60000,
    })

    // Fetch scan history
    const { data: historyData, refetch: refetchHistory } = useQuery<{ history: ScanHistory[]; count: number }>({
        queryKey: ['scan-history'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/history?limit=10')
            return data
        },
        staleTime: 30000,
    })

    const { data: comparisonData, isLoading: comparisonLoading, refetch: refetchComparison } = useQuery({
        queryKey: ['dte-comparison'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/scanner/dte-comparison')
            return data
        },
        staleTime: 300000,
    })

    // Cleanup EventSource on unmount
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close()
            }
        }
    }, [])

    // Load a previous scan (defined before useEffect that uses it)
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

    // Auto-load most recent scan on mount
    useEffect(() => {
        if (historyData?.history?.length && scanResults.length === 0 && !isScanning) {
            const mostRecent = historyData.history[0]
            loadPreviousScan(mostRecent.scan_id)
        }
    }, [historyData, scanResults.length, isScanning, loadPreviousScan])

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

    // Handle streaming scan with progress
    const handleScan = useCallback(() => {
        const symbols = customSymbols.trim()
            ? customSymbols.split(',').map(s => s.trim().toUpperCase())
            : selectedWatchlist?.symbols || []

        if (symbols.length === 0) return

        // Reset state
        setIsScanning(true)
        setScanProgress(null)
        setScanResults([])
        setScanError(null)
        setScanId(null)
        setAiInsights(null)
        setAiStats(null)

        // Close any existing connection
        if (eventSourceRef.current) {
            eventSourceRef.current.close()
        }

        // Use fetch with streaming for SSE (EventSource doesn't support POST)
        const baseUrl = axiosInstance.defaults.baseURL || ''

        // Use AI endpoint if AI mode is enabled
        const endpoint = aiMode ? '/scanner/scan-ai-stream' : '/scanner/scan-stream'

        fetch(`${baseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbols,
                max_price: maxPrice,
                min_premium_pct: minPremium,
                dte: selectedDTE,
                save_to_db: true,
                ...(aiMode && { min_ai_score: 0 })  // Include AI params when in AI mode
            })
        }).then(async response => {
            const reader = response.body?.getReader()
            const decoder = new TextDecoder()

            if (!reader) {
                throw new Error('No reader available')
            }

            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n\n')
                buffer = lines.pop() || ''

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6))
                            setScanProgress(data as ScanProgress)

                            if (data.type === 'complete') {
                                setScanResults(data.results || [])
                                setScanId(data.scan_id || null)
                                setIsScanning(false)
                                refetchHistory()

                                // Capture AI insights and stats if available
                                if (data.ai_insights) {
                                    setAiInsights(data.ai_insights)
                                }
                                if (data.stats) {
                                    setAiStats(data.stats)
                                }
                            } else if (data.type === 'error') {
                                setScanError(data.error || 'Unknown error')
                                setIsScanning(false)
                            }
                        } catch (e) {
                            console.error('Failed to parse SSE data:', e)
                        }
                    }
                }
            }
        }).catch(error => {
            console.error('Scan error:', error)
            setScanError(error.message || 'Failed to connect')
            setIsScanning(false)
        })
    }, [customSymbols, selectedWatchlist, maxPrice, minPremium, selectedDTE, refetchHistory, aiMode])

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

    // Process results: filter by DTE, stock price, unique, then sort
    const processedResults = useMemo(() => {
        let results = [...scanResults]

        if (resultDTEFilter > 0) {
            results = results.filter((r: ScanResult) => r.dte <= resultDTEFilter)
        }

        // Filter by max stock price
        const maxPriceNum = parseFloat(maxStockPriceFilter)
        if (!isNaN(maxPriceNum) && maxPriceNum > 0) {
            results = results.filter((r: ScanResult) => r.stock_price <= maxPriceNum)
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
    }, [scanResults, resultDTEFilter, maxStockPriceFilter, showUniqueOnly, sortField, sortDirection])

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

            {/* Scan Progress Indicator */}
            {isScanning && scanProgress && (
                <div className="glass-card p-5 border-primary/30 bg-primary/5">
                    <div className="flex items-center gap-4 mb-3">
                        <Loader2 className="w-6 h-6 text-primary animate-spin" />
                        <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-white">
                                    {scanProgress.type === 'start' && 'Starting scan...'}
                                    {scanProgress.type === 'progress' && `Scanning symbols...`}
                                    {scanProgress.type === 'batch' && `Processing batch...`}
                                </span>
                                <span className="text-sm text-primary font-mono">
                                    {scanProgress.percent || 0}%
                                </span>
                            </div>
                            {/* Progress bar */}
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-primary to-violet-500 transition-all duration-300"
                                    style={{ width: `${scanProgress.percent || 0}%` }}
                                />
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">
                            {scanProgress.current || 0} / {scanProgress.total || 0} symbols
                        </span>
                        {scanProgress.symbols && (
                            <span className="text-slate-500">
                                Current: {scanProgress.symbols.join(', ')}
                            </span>
                        )}
                        {scanProgress.total_so_far !== undefined && (
                            <span className="text-emerald-400">
                                {scanProgress.total_so_far} opportunities found
                            </span>
                        )}
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
                    <div className="h-64 min-h-[256px] w-full">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
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
                                    {watchlistsLoading ? 'Loading...' : selectedWatchlist?.name || 'Select a watchlist'}
                                </span>
                                {selectedWatchlist && (
                                    <span className="text-xs text-slate-500">
                                        ({selectedWatchlist.symbols.length} symbols)
                                    </span>
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
                                        {watchlists.map((wl) => (
                                            <button
                                                key={wl.id}
                                                onClick={() => {
                                                    setSelectedWatchlistId(wl.id)
                                                    setCustomSymbols('')
                                                    setShowWatchlistDropdown(false)
                                                }}
                                                className={`w-full px-4 py-3 text-left hover:bg-slate-800/60 flex items-center justify-between ${
                                                    selectedWatchlistId === wl.id ? 'bg-primary/10 text-primary' : 'text-white'
                                                }`}
                                            >
                                                <span>{wl.name}</span>
                                                <span className="text-xs text-slate-500">
                                                    {wl.symbols.length} symbols
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Selected Symbols Preview */}
                    {selectedWatchlist && selectedWatchlist.symbols.length > 0 && !customSymbols && (
                        <div className="mb-4 p-3 bg-slate-800/40 rounded-xl border border-slate-700/50">
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

                    {/* AI Mode Toggle */}
                    <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl border border-slate-700/50 mb-4">
                        <div className="flex items-center gap-3">
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                                aiMode ? 'bg-gradient-to-br from-violet-500 to-purple-600' : 'bg-slate-700'
                            }`}>
                                <Sparkles className={`w-4 h-4 ${aiMode ? 'text-white' : 'text-slate-400'}`} />
                            </div>
                            <div>
                                <div className="text-sm font-medium text-white">AI-Powered Analysis</div>
                                <div className="text-xs text-slate-400">
                                    {aiMode ? 'Multi-criteria scoring & insights enabled' : 'Basic scan mode'}
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={() => setAiMode(!aiMode)}
                            className={`relative w-12 h-6 rounded-full transition-colors ${
                                aiMode ? 'bg-violet-500' : 'bg-slate-600'
                            }`}
                        >
                            <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                                aiMode ? 'translate-x-7' : 'translate-x-1'
                            }`} />
                        </button>
                    </div>

                    {/* Scan Button */}
                    <button
                        onClick={handleScan}
                        disabled={isScanning || (!customSymbols && (!selectedWatchlist || selectedWatchlist.symbols.length === 0))}
                        className={`w-full py-3 flex items-center justify-center gap-2 text-base rounded-xl font-semibold transition-all ${
                            aiMode
                                ? 'bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg shadow-violet-500/25'
                                : 'btn-primary'
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {isScanning ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                {aiMode ? 'AI Analyzing...' : 'Scanning...'} {scanProgress?.percent || 0}%
                            </>
                        ) : (
                            <>
                                {aiMode ? <Sparkles className="w-5 h-5" /> : <Search className="w-5 h-5" />}
                                {aiMode ? `AI Scan ${selectedDTE} DTE` : `Scan ${selectedDTE} DTE Premiums`}
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* AI Insights Panel */}
            {aiInsights && showAiInsights && (
                <div className="glass-card overflow-hidden border-violet-500/30">
                    <div className="p-4 border-b border-slate-700/50 bg-gradient-to-r from-violet-500/10 to-purple-500/10 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                                <Sparkles className="w-4 h-4 text-white" />
                            </div>
                            <div>
                                <h3 className="font-semibold text-white">AI Insights</h3>
                                <p className="text-xs text-slate-400">{aiInsights.summary}</p>
                            </div>
                        </div>
                        <button
                            onClick={() => setShowAiInsights(false)}
                            className="text-slate-400 hover:text-white transition-colors"
                        >
                            <XCircle className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Top Pick */}
                        {aiInsights.top_pick && (
                            <div className="p-4 bg-emerald-500/10 rounded-xl border border-emerald-500/30">
                                <div className="flex items-center gap-2 mb-2">
                                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                                    <span className="text-xs font-semibold text-emerald-400 uppercase">Top Pick</span>
                                </div>
                                <div className="text-lg font-bold text-white">{aiInsights.top_pick.symbol}</div>
                                <div className="text-sm text-slate-300">${aiInsights.top_pick.strike} • {aiInsights.top_pick.monthly_return?.toFixed(1)}%/mo</div>
                                <div className="text-xs text-slate-400 mt-1">{aiInsights.top_pick.reason}</div>
                            </div>
                        )}

                        {/* Market Conditions */}
                        {aiInsights.market_conditions && (
                            <div className="p-4 bg-blue-500/10 rounded-xl border border-blue-500/30">
                                <div className="flex items-center gap-2 mb-2">
                                    <Activity className="w-4 h-4 text-blue-400" />
                                    <span className="text-xs font-semibold text-blue-400 uppercase">IV Environment</span>
                                </div>
                                <div className="text-lg font-bold text-white capitalize">{aiInsights.market_conditions.iv_environment}</div>
                                <div className="text-sm text-slate-300">Avg IV: {aiInsights.market_conditions.avg_iv?.toFixed(1)}%</div>
                                <div className="text-xs text-slate-400 mt-1">{aiInsights.market_conditions.market_insight}</div>
                            </div>
                        )}

                        {/* Risk Assessment */}
                        {aiInsights.risk_assessment && (
                            <div className={`p-4 rounded-xl border ${
                                aiInsights.risk_assessment.risk_level === 'low'
                                    ? 'bg-emerald-500/10 border-emerald-500/30'
                                    : aiInsights.risk_assessment.risk_level === 'moderate'
                                    ? 'bg-amber-500/10 border-amber-500/30'
                                    : 'bg-red-500/10 border-red-500/30'
                            }`}>
                                <div className="flex items-center gap-2 mb-2">
                                    <Target className="w-4 h-4 text-amber-400" />
                                    <span className="text-xs font-semibold text-amber-400 uppercase">Risk Level</span>
                                </div>
                                <div className="text-lg font-bold text-white capitalize">{aiInsights.risk_assessment.risk_level}</div>
                                <div className="text-sm text-slate-300">Avg OTM: {aiInsights.risk_assessment.avg_otm_distance?.toFixed(1)}%</div>
                                <div className="text-xs text-slate-400 mt-1">{aiInsights.risk_assessment.risk_insight}</div>
                            </div>
                        )}

                        {/* Stats Summary */}
                        {aiStats && (
                            <div className="p-4 bg-slate-700/30 rounded-xl border border-slate-600/50">
                                <div className="flex items-center gap-2 mb-2">
                                    <BarChart3 className="w-4 h-4 text-slate-400" />
                                    <span className="text-xs font-semibold text-slate-400 uppercase">AI Scores</span>
                                </div>
                                <div className="space-y-1">
                                    <div className="flex justify-between text-sm">
                                        <span className="text-slate-400">Strong Buy</span>
                                        <span className="text-emerald-400 font-bold">{aiStats.strong_buy_count || 0}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-slate-400">Buy</span>
                                        <span className="text-green-400 font-bold">{aiStats.buy_count || 0}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-slate-400">Hold</span>
                                        <span className="text-amber-400 font-bold">{aiStats.hold_count || 0}</span>
                                    </div>
                                    <div className="flex justify-between text-sm">
                                        <span className="text-slate-400">Avg Score</span>
                                        <span className="text-white font-bold">{aiStats.avg_ai_score || 0}</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* AI Recommendations */}
                    {aiInsights.recommendations && aiInsights.recommendations.length > 0 && (
                        <div className="px-4 pb-4">
                            <div className="p-3 bg-slate-800/50 rounded-xl">
                                <div className="text-xs font-semibold text-slate-400 uppercase mb-2">AI Recommendations</div>
                                <ul className="space-y-1">
                                    {aiInsights.recommendations.slice(0, 3).map((rec, idx) => (
                                        <li key={idx} className="text-sm text-slate-300 flex items-start gap-2">
                                            <span className="text-violet-400 mt-0.5">•</span>
                                            {rec}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Collapsed AI Insights Toggle */}
            {aiInsights && !showAiInsights && (
                <button
                    onClick={() => setShowAiInsights(true)}
                    className="w-full p-3 glass-card border-violet-500/30 hover:border-violet-500/50 flex items-center justify-center gap-2 text-violet-400 hover:text-violet-300 transition-all"
                >
                    <Sparkles className="w-4 h-4" />
                    <span className="text-sm">Show AI Insights</span>
                </button>
            )}

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

                            {/* Stock Price Filter */}
                            <div className="flex items-center gap-2">
                                <span className="text-sm text-slate-400">Max Price:</span>
                                <div className="relative">
                                    <DollarSign className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                                    <input
                                        type="number"
                                        value={maxStockPriceFilter}
                                        onChange={(e) => setMaxStockPriceFilter(e.target.value)}
                                        placeholder="Any"
                                        className="w-24 pl-7 pr-2 py-1.5 text-sm bg-slate-800/60 border border-slate-700/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
                                    />
                                </div>
                                {maxStockPriceFilter && (
                                    <button
                                        onClick={() => setMaxStockPriceFilter('')}
                                        className="text-xs text-slate-500 hover:text-slate-300"
                                    >
                                        Clear
                                    </button>
                                )}
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
                                        {/* AI Score Column */}
                                        <td style={{ width: columnWidths['ai_score'] }} className="px-3 py-3">
                                            {opp.ai_score !== undefined ? (
                                                <div className="flex flex-col items-center gap-0.5">
                                                    <span className={`text-sm font-bold ${
                                                        opp.ai_score >= 80 ? 'text-emerald-400' :
                                                        opp.ai_score >= 60 ? 'text-green-400' :
                                                        opp.ai_score >= 45 ? 'text-amber-400' :
                                                        'text-red-400'
                                                    }`}>
                                                        {opp.ai_score}
                                                    </span>
                                                    {opp.ai_recommendation && (
                                                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                                                            AI_RECOMMENDATION_COLORS[opp.ai_recommendation]?.bg || 'bg-slate-500/20'
                                                        } ${AI_RECOMMENDATION_COLORS[opp.ai_recommendation]?.text || 'text-slate-400'}`}>
                                                            {opp.ai_recommendation === 'STRONG_BUY' ? 'S.BUY' : opp.ai_recommendation}
                                                        </span>
                                                    )}
                                                </div>
                                            ) : (
                                                <span className="text-slate-500">-</span>
                                            )}
                                        </td>
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
                                        <td style={{ width: columnWidths['otm_pct'] }} className="px-3 py-3 font-mono truncate">
                                            <span className={opp.otm_pct > 0 ? 'text-emerald-400' : 'text-amber-400'}>
                                                {opp.otm_pct?.toFixed(1) || 0}%
                                            </span>
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
                                            {opp.premium_pct?.toFixed(2) || 0}%
                                        </td>
                                        <td style={{ width: columnWidths['monthly_return'] }} className="px-3 py-3">
                                            <span className="font-bold text-emerald-400 bg-emerald-500/15 px-2 py-1 rounded-lg">
                                                {opp.monthly_return?.toFixed(2) || 0}%
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['annual_return'] }} className="px-3 py-3 font-mono text-slate-400 truncate">
                                            {opp.annual_return?.toFixed(1) || 0}%
                                        </td>
                                        <td style={{ width: columnWidths['iv'] }} className="px-3 py-3 truncate">
                                            <span className={opp.iv >= 50 ? 'text-amber-400' : 'text-slate-400'}>
                                                {opp.iv || 0}%
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['delta'] }} className="px-3 py-3 font-mono text-slate-400 truncate">
                                            {opp.delta?.toFixed(2) || '-'}
                                        </td>
                                        <td style={{ width: columnWidths['liquidity_score'] }} className="px-3 py-3">
                                            <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                                                (opp.liquidity_score || 0) >= 50 ? 'bg-emerald-500/20 text-emerald-400' :
                                                (opp.liquidity_score || 0) >= 25 ? 'bg-amber-500/20 text-amber-400' :
                                                'bg-red-500/20 text-red-400'
                                            }`}>
                                                {opp.liquidity_score || 0}
                                            </span>
                                        </td>
                                        <td style={{ width: columnWidths['collateral'] }} className="px-3 py-3 font-mono text-slate-300 truncate">
                                            ${opp.collateral?.toLocaleString() || 0}
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

function QuickStatCard({ title, value, subtitle, icon, iconColor = 'text-primary', iconBg = 'bg-primary/20', onSymbolClick }: QuickStatCardProps) {
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
}
