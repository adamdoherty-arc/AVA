import { useState, useEffect, memo, useMemo, useCallback } from 'react'
import { usePositions, useResearch, useRefreshResearch, useSymbolMetadata, useSymbolRecommendation } from '../hooks/useMagnusApi'
import { toast } from 'sonner'
import {
    AlertCircle, RefreshCw, TrendingUp, TrendingDown, DollarSign,
    Briefcase, Target, Clock, Activity, ExternalLink, ChevronDown, ChevronUp,
    PieChart, ArrowUpRight, ArrowDownRight, Brain, Calendar,
    Building2, BarChart3, Lightbulb, AlertTriangle, WifiOff, ServerCrash
} from 'lucide-react'
import { SyncStatusPanel } from '../components/SyncStatusPanel'
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { isTimeoutError, formatErrorMessage, type EnhancedError } from '../lib/axios'

const COLORS = ['#10B981', '#3B82F6', '#8B5CF6']

export default function Positions() {
    const { data: positions, isLoading, error, refetch, isFetching } = usePositions()
    const [activeTab, setActiveTab] = useState<'stocks' | 'options'>('stocks')
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
    const [expandedOptions, setExpandedOptions] = useState<Set<string>>(new Set())

    const { data: researchData, isLoading: researchLoading } = useResearch(selectedSymbol)
    const refreshResearchMutation = useRefreshResearch()
    const { data: metadataData, isLoading: metadataLoading } = useSymbolMetadata(selectedSymbol)
    const { data: recommendationData, isLoading: recommendationLoading } = useSymbolRecommendation(selectedSymbol)

    // Derive data early so hooks can use them consistently
    const summary = positions?.summary || { total_equity: 0, buying_power: 0, total_positions: 0 }
    const stocks = positions?.stocks || []
    const options = positions?.options || []

    // Memoize expensive portfolio calculations to prevent recalculation on every render
    const { totalStockValue, totalOptionValue, totalPL, dailyTheta, totalThetaToCompletion, avgDTE } = useMemo(() => {
        const stockValue = stocks.reduce((sum: number, s: StockPosition) => sum + s.current_value, 0);
        const optionValue = options.reduce((sum: number, o: OptionPosition) => sum + Math.abs(o.current_value), 0);
        const pl = [...stocks, ...options].reduce((sum: number, p: StockPosition | OptionPosition) => sum + p.pl, 0);

        // Daily theta: sum of (theta per contract * quantity) for all options
        // Theta is negative for short options (good for sellers), we show absolute value
        const daily = options.reduce((sum: number, o: OptionPosition) =>
            sum + Math.abs(o.greeks?.theta || 0) * Math.abs(o.quantity), 0);

        // Total theta to completion: sum of (daily theta * DTE) for all options
        // This represents total time value that will decay if options expire worthless
        const toCompletion = options.reduce((sum: number, o: OptionPosition) =>
            sum + Math.abs(o.greeks?.theta || 0) * Math.abs(o.quantity) * (o.dte || 0), 0);

        // Average DTE weighted by position value
        const totalOptionAbsValue = options.reduce((sum: number, o: OptionPosition) => sum + Math.abs(o.current_value), 0);
        const weightedDTE = totalOptionAbsValue > 0
            ? options.reduce((sum: number, o: OptionPosition) => sum + (o.dte || 0) * Math.abs(o.current_value), 0) / totalOptionAbsValue
            : 0;

        return {
            totalStockValue: stockValue,
            totalOptionValue: optionValue,
            totalPL: pl,
            dailyTheta: daily,
            totalThetaToCompletion: toCompletion,
            avgDTE: Math.round(weightedDTE),
        };
    }, [stocks, options])

    // Effect to select first stock
    useEffect(() => {
        if (positions?.stocks?.length > 0 && !selectedSymbol) {
            setSelectedSymbol(positions.stocks[0].symbol)
        }
    }, [positions, selectedSymbol])

    // Memoize allocation data for pie chart - MUST be before early returns
    const allocationData = useMemo(() => [
        { name: 'Stocks', value: totalStockValue },
        { name: 'Options', value: totalOptionValue },
        { name: 'Cash', value: summary.buying_power }
    ].filter(d => d.value > 0), [totalStockValue, totalOptionValue, summary.buying_power])

    // Memoize toggle callback to prevent child re-renders - MUST be before early returns
    const toggleOptionExpanded = useCallback((symbol: string) => {
        setExpandedOptions(prev => {
            const newExpanded = new Set(prev)
            if (newExpanded.has(symbol)) {
                newExpanded.delete(symbol)
            } else {
                newExpanded.add(symbol)
            }
            return newExpanded
        })
    }, [])

    // Early returns AFTER all hooks
    if (isLoading) {
        return (
            <div className="glass-card p-12 text-center">
                <div className="w-12 h-12 mx-auto mb-4 relative">
                    <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                    <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
                </div>
                <p className="text-slate-400">Loading positions from Robinhood...</p>
                <p className="text-xs text-slate-500 mt-2">This may take a moment during market hours</p>
            </div>
        )
    }

    if (error) {
        const enhancedError = error as EnhancedError;
        const isTimeout = isTimeoutError(error);
        const isNetworkError = enhancedError.isNetworkError;
        const isCircuitOpen = enhancedError.isCircuitOpen;

        // Choose appropriate icon based on error type
        const ErrorIcon = isNetworkError ? WifiOff : isCircuitOpen ? ServerCrash : AlertCircle;
        const bgColor = isCircuitOpen ? 'border-amber-500/30 bg-amber-500/5' : 'border-red-500/30 bg-red-500/5';
        const textColor = isCircuitOpen ? 'text-amber-400' : 'text-red-400';

        return (
            <div className={`glass-card p-6 ${bgColor}`}>
                <div className={`flex items-start gap-3 ${textColor}`}>
                    <ErrorIcon className="w-6 h-6 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                        <h3 className="font-semibold">Failed to Load Positions</h3>

                        {isCircuitOpen ? (
                            <>
                                <p className="text-sm text-amber-400/80 mt-1">
                                    Too many failed requests. Service temporarily paused.
                                </p>
                                <p className="text-sm text-slate-400 mt-2">
                                    The system will automatically retry in 30 seconds.
                                </p>
                            </>
                        ) : isTimeout ? (
                            <>
                                <p className="text-sm text-red-400/80 mt-1">
                                    Request timed out. Robinhood API may be slow during market hours.
                                </p>
                                <p className="text-sm text-slate-400 mt-2">
                                    Try again - subsequent requests are often faster due to caching.
                                </p>
                            </>
                        ) : isNetworkError ? (
                            <>
                                <p className="text-sm text-red-400/80 mt-1">
                                    Network error. Please check your internet connection.
                                </p>
                                <p className="text-sm text-slate-400 mt-2">
                                    Make sure the backend server is running on port 8002.
                                </p>
                            </>
                        ) : (
                            <p className="text-sm text-red-400/80 mt-1">
                                {formatErrorMessage(error)}
                            </p>
                        )}

                        <button
                            onClick={() => refetch()}
                            disabled={isFetching}
                            className={`mt-4 flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                isCircuitOpen
                                    ? 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-400'
                                    : 'bg-red-500/20 hover:bg-red-500/30 text-red-400'
                            }`}
                        >
                            <RefreshCw className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
                            {isFetching ? 'Retrying...' : 'Retry'}
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                            <Briefcase className="w-5 h-5 text-white" />
                        </div>
                        Positions
                    </h1>
                    <p className="page-subtitle">Portfolio & Options Management</p>
                </div>
                <SyncStatusPanel variant="expanded" />
            </header>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
                <StatCard
                    title="Total Equity"
                    value={formatCurrency(summary.total_equity)}
                    icon={<Briefcase className="w-5 h-5" />}
                    iconColor="text-blue-400"
                    iconBg="bg-blue-500/20"
                />
                <StatCard
                    title="Buying Power"
                    value={formatCurrency(summary.buying_power)}
                    icon={<DollarSign className="w-5 h-5" />}
                    iconColor="text-emerald-400"
                    iconBg="bg-emerald-500/20"
                />
                <StatCard
                    title="Total P&L"
                    value={formatCurrency(totalPL)}
                    change={totalPL}
                    icon={totalPL >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
                    iconColor={totalPL >= 0 ? 'text-emerald-400' : 'text-red-400'}
                    iconBg={totalPL >= 0 ? 'bg-emerald-500/20' : 'bg-red-500/20'}
                />
                <StatCard
                    title="Daily Theta"
                    value={formatCurrency(dailyTheta)}
                    subtitle="Per day income"
                    icon={<Clock className="w-5 h-5" />}
                    iconColor="text-purple-400"
                    iconBg="bg-purple-500/20"
                />
                <StatCard
                    title="Total to Expiry"
                    value={formatCurrency(totalThetaToCompletion)}
                    subtitle={`~${avgDTE} avg DTE`}
                    icon={<Target className="w-5 h-5" />}
                    iconColor="text-cyan-400"
                    iconBg="bg-cyan-500/20"
                />
                <StatCard
                    title="Positions"
                    value={String(summary.total_positions)}
                    subtitle={`${stocks.length} stocks, ${options.length} options`}
                    icon={<Activity className="w-5 h-5" />}
                    iconColor="text-amber-400"
                    iconBg="bg-amber-500/20"
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Positions Tables */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="glass-card overflow-hidden">
                        {/* Tabs */}
                        <div className="flex border-b border-slate-700/50">
                            <button
                                onClick={() => setActiveTab('stocks')}
                                className={`flex-1 py-4 px-6 font-medium text-sm transition-all flex items-center justify-center gap-2 ${
                                    activeTab === 'stocks'
                                        ? 'text-primary border-b-2 border-primary bg-primary/5'
                                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                                }`}
                            >
                                <TrendingUp className="w-4 h-4" />
                                Stocks ({stocks.length})
                            </button>
                            <button
                                onClick={() => setActiveTab('options')}
                                className={`flex-1 py-4 px-6 font-medium text-sm transition-all flex items-center justify-center gap-2 ${
                                    activeTab === 'options'
                                        ? 'text-primary border-b-2 border-primary bg-primary/5'
                                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                                }`}
                            >
                                <Target className="w-4 h-4" />
                                Options ({options.length})
                            </button>
                        </div>

                        <div className="p-5">
                            {activeTab === 'stocks' ? (
                                <StocksTable
                                    stocks={stocks}
                                    onSelect={(symbol) => setSelectedSymbol(symbol)}
                                    selectedSymbol={selectedSymbol}
                                />
                            ) : (
                                <OptionsTable
                                    options={options}
                                    expanded={expandedOptions}
                                    onToggle={toggleOptionExpanded}
                                    onSelect={(symbol) => setSelectedSymbol(symbol)}
                                    selectedSymbol={selectedSymbol}
                                />
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Sidebar */}
                <div className="space-y-6">
                    {/* Allocation Chart */}
                    <div className="glass-card p-5">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
                                <PieChart className="w-4 h-4 text-primary" />
                            </div>
                            <h3 className="font-semibold text-white">Allocation</h3>
                        </div>
                        <div className="h-44 min-h-[176px] w-full">
                            <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                                <RechartsPie>
                                    <Pie
                                        data={allocationData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={45}
                                        outerRadius={65}
                                        paddingAngle={4}
                                        dataKey="value"
                                    >
                                        {allocationData.map((_, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        formatter={(value: number) => formatCurrency(value)}
                                        contentStyle={{
                                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                            border: '1px solid rgba(51, 65, 85, 0.5)',
                                            borderRadius: '0.5rem'
                                        }}
                                    />
                                </RechartsPie>
                            </ResponsiveContainer>
                        </div>
                        <div className="flex flex-wrap justify-center gap-3 mt-3">
                            {allocationData.map((item, index) => (
                                <div key={item.name} className="flex items-center gap-2 px-2 py-1 bg-slate-800/50 rounded-lg">
                                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[index] }} />
                                    <span className="text-xs text-slate-300">{item.name}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Company Metadata Widget */}
                    {selectedSymbol && (
                        <div className="glass-card p-5">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                                    <Building2 className="w-4 h-4 text-blue-400" />
                                </div>
                                <h3 className="font-semibold text-white">Company Info</h3>
                            </div>

                            <div className="text-center py-2 mb-3">
                                <span className="text-2xl font-bold bg-gradient-to-r from-primary to-blue-400 bg-clip-text text-transparent">
                                    {selectedSymbol}
                                </span>
                                {metadataData?.name && (
                                    <p className="text-xs text-slate-400 mt-1">{metadataData.name}</p>
                                )}
                            </div>

                            {metadataLoading ? (
                                <div className="text-center py-4">
                                    <div className="w-6 h-6 mx-auto relative">
                                        <div className="absolute inset-0 rounded-full border-2 border-slate-700"></div>
                                        <div className="absolute inset-0 rounded-full border-2 border-blue-400 border-t-transparent animate-spin"></div>
                                    </div>
                                </div>
                            ) : metadataData ? (
                                <div className="space-y-2 text-sm">
                                    {metadataData.sector && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Sector</span>
                                            <span className="text-slate-300">{metadataData.sector}</span>
                                        </div>
                                    )}
                                    {metadataData.market_cap_formatted && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Market Cap</span>
                                            <span className="text-slate-300">{metadataData.market_cap_formatted}</span>
                                        </div>
                                    )}
                                    {metadataData.pe_ratio && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">P/E Ratio</span>
                                            <span className="text-slate-300">{metadataData.pe_ratio}</span>
                                        </div>
                                    )}
                                    {metadataData.next_earnings && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Next Earnings</span>
                                            <span className="text-amber-400">{metadataData.next_earnings}</span>
                                        </div>
                                    )}
                                    {metadataData.analyst_rating && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Analyst Rating</span>
                                            <span className={`font-medium ${
                                                metadataData.analyst_rating.toLowerCase().includes('buy') ? 'text-emerald-400' :
                                                metadataData.analyst_rating.toLowerCase().includes('sell') ? 'text-red-400' : 'text-slate-300'
                                            }`}>{metadataData.analyst_rating}</span>
                                        </div>
                                    )}
                                    <div className="pt-2">
                                        <a
                                            href={`https://www.tradingview.com/chart/?symbol=${selectedSymbol}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center justify-center gap-2 w-full py-2 px-3 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg text-sm text-slate-300 transition-colors"
                                        >
                                            <BarChart3 className="w-4 h-4" />
                                            View Chart
                                            <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* AI Recommendation Widget */}
                    {selectedSymbol && (
                        <div className="glass-card p-5">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                                    <Lightbulb className="w-4 h-4 text-purple-400" />
                                </div>
                                <h3 className="font-semibold text-white">AI Recommendation</h3>
                            </div>

                            {recommendationLoading ? (
                                <div className="text-center py-6">
                                    <div className="w-8 h-8 mx-auto mb-3 relative">
                                        <div className="absolute inset-0 rounded-full border-2 border-slate-700"></div>
                                        <div className="absolute inset-0 rounded-full border-2 border-purple-400 border-t-transparent animate-spin"></div>
                                    </div>
                                    <p className="text-sm text-slate-400">Analyzing...</p>
                                </div>
                            ) : recommendationData?.recommendation ? (
                                <div className="space-y-3">
                                    {/* Recommendation Badge */}
                                    <div className="text-center">
                                        <span className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-semibold ${
                                            recommendationData.recommendation.recommendation === 'Hold/Add' ? 'bg-emerald-500/20 text-emerald-400' :
                                            recommendationData.recommendation.recommendation === 'Take Profits' ? 'bg-amber-500/20 text-amber-400' :
                                            recommendationData.recommendation.recommendation === 'Close/Roll' ? 'bg-amber-500/20 text-amber-400' :
                                            recommendationData.recommendation.recommendation === 'Cut Losses' ? 'bg-red-500/20 text-red-400' :
                                            recommendationData.recommendation.recommendation === 'Review/Trim' ? 'bg-red-500/20 text-red-400' :
                                            'bg-slate-500/20 text-slate-300'
                                        }`}>
                                            {recommendationData.recommendation.recommendation === 'Hold/Add' && <TrendingUp className="w-4 h-4" />}
                                            {recommendationData.recommendation.recommendation === 'Take Profits' && <DollarSign className="w-4 h-4" />}
                                            {(recommendationData.recommendation.recommendation === 'Cut Losses' || recommendationData.recommendation.recommendation === 'Review/Trim') && <AlertTriangle className="w-4 h-4" />}
                                            {recommendationData.recommendation.recommendation}
                                        </span>
                                    </div>

                                    {/* Confidence */}
                                    <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
                                        <span className="text-xs text-slate-500">Confidence</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                                                <div
                                                    className="h-full rounded-full bg-purple-400"
                                                    style={{ width: `${recommendationData.recommendation.confidence}%` }}
                                                />
                                            </div>
                                            <span className="text-xs text-slate-300">{recommendationData.recommendation.confidence}%</span>
                                        </div>
                                    </div>

                                    {/* Reasoning */}
                                    {recommendationData.recommendation.reasoning && (
                                        <p className="text-xs text-slate-400 leading-relaxed">
                                            {recommendationData.recommendation.reasoning}
                                        </p>
                                    )}

                                    {/* Action Items */}
                                    {recommendationData.recommendation.actions?.length > 0 && (
                                        <div className="space-y-1">
                                            {recommendationData.recommendation.actions.map((action: string, idx: number) => (
                                                <div key={idx} className="flex items-start gap-2 text-xs">
                                                    <span className="text-purple-400 mt-0.5">-</span>
                                                    <span className="text-slate-300">{action}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* DTE Warning */}
                                    {recommendationData.recommendation.dte_warning && (
                                        <div className="flex items-center gap-2 p-2 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                                            <AlertTriangle className="w-4 h-4 text-amber-400" />
                                            <span className="text-xs text-amber-400">Expiration approaching - monitor closely</span>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-center py-4 text-slate-400">
                                    <p className="text-sm">Select a position to see AI recommendation</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* AI Research Widget */}
                    {selectedSymbol && (
                        <div className="glass-card p-5">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                                        <Brain className="w-4 h-4 text-indigo-400" />
                                    </div>
                                    <h3 className="font-semibold text-white">Deep Research</h3>
                                </div>
                                <button
                                    onClick={() => refreshResearchMutation.mutate(selectedSymbol)}
                                    disabled={refreshResearchMutation.isPending}
                                    className="btn-icon"
                                >
                                    <RefreshCw className={`w-4 h-4 ${refreshResearchMutation.isPending ? 'animate-spin' : ''}`} />
                                </button>
                            </div>

                            {researchLoading || refreshResearchMutation.isPending ? (
                                <div className="text-center py-6">
                                    <div className="w-8 h-8 mx-auto mb-3 relative">
                                        <div className="absolute inset-0 rounded-full border-2 border-slate-700"></div>
                                        <div className="absolute inset-0 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin"></div>
                                    </div>
                                    <p className="text-sm text-slate-400">Running deep analysis...</p>
                                </div>
                            ) : researchData ? (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
                                        <span className="text-xs text-slate-500">AI Score</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full ${
                                                        researchData.overall_score >= 70 ? 'bg-emerald-400' :
                                                        researchData.overall_score >= 40 ? 'bg-amber-400' : 'bg-red-400'
                                                    }`}
                                                    style={{ width: `${researchData.overall_score}%` }}
                                                />
                                            </div>
                                            <span className={`text-xs font-bold ${
                                                researchData.overall_score >= 70 ? 'text-emerald-400' :
                                                researchData.overall_score >= 40 ? 'text-amber-400' : 'text-red-400'
                                            }`}>
                                                {researchData.overall_score}
                                            </span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-slate-400 leading-relaxed">
                                        {researchData.summary?.substring(0, 120)}...
                                    </p>
                                </div>
                            ) : (
                                <div className="text-center py-4 text-slate-400">
                                    <p className="text-xs">Click refresh for multi-agent analysis</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

// Types
interface StockPosition {
    symbol: string
    quantity: number
    avg_buy_price: number
    current_price: number
    cost_basis: number
    current_value: number
    pl: number
    pl_pct: number
}

interface OptionPosition {
    symbol: string
    strategy: string
    type: string
    option_type: string
    strike: number
    expiration: string
    dte: number
    quantity: number
    avg_price: number
    current_price: number
    total_premium: number
    current_value: number
    pl: number
    pl_pct: number
    breakeven: number
    greeks: {
        delta: number
        theta: number
        gamma: number
        vega: number
        iv: number
    }
}

// Helper to get date from DTE
function getDateFromDTE(dte: number): string {
    const date = new Date()
    date.setDate(date.getDate() + dte)
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

// Sub-components
interface StatCardProps {
    title: string
    value: string
    change?: number
    subtitle?: string
    icon?: React.ReactNode
    iconColor?: string
    iconBg?: string
}

const StatCard = memo(function StatCard({ title, value, change, subtitle, icon, iconColor = 'text-primary', iconBg = 'bg-primary/20' }: StatCardProps) {
    return (
        <div className="stat-card group">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-xs text-slate-400 mb-1 uppercase tracking-wide">{title}</p>
                    <p className="text-xl font-bold text-white">{value}</p>
                    {change !== undefined && (
                        <div className={`flex items-center gap-1 mt-1.5 text-sm font-medium ${change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {change >= 0 ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                            {change >= 0 ? 'Profit' : 'Loss'}
                        </div>
                    )}
                    {subtitle && (
                        <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
                    )}
                </div>
                <div className={`w-10 h-10 rounded-xl ${iconBg} flex items-center justify-center ${iconColor} group-hover:scale-110 transition-transform`}>
                    {icon}
                </div>
            </div>
        </div>
    )
})

interface StocksTableProps {
    stocks: StockPosition[]
    onSelect: (symbol: string) => void
    selectedSymbol: string | null
}

function StocksTable({ stocks, onSelect, selectedSymbol }: StocksTableProps) {
    if (stocks.length === 0) {
        return (
            <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/50 flex items-center justify-center">
                    <Briefcase className="w-8 h-8 text-slate-500" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">No Stock Positions</h3>
                <p className="text-slate-400">Sync with Robinhood to load your positions.</p>
            </div>
        )
    }

    return (
        <div className="overflow-x-auto">
            <table className="data-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Qty</th>
                        <th>Avg Price</th>
                        <th>Current</th>
                        <th>Value</th>
                        <th>P&L</th>
                        <th>%</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {stocks.map((stock) => (
                        <tr
                            key={stock.symbol}
                            onClick={() => onSelect(stock.symbol)}
                            className={`cursor-pointer ${selectedSymbol === stock.symbol ? 'bg-primary/10' : ''}`}
                        >
                            <td>
                                <span className="font-semibold text-white">{stock.symbol}</span>
                            </td>
                            <td className="text-slate-300">{stock.quantity}</td>
                            <td className="text-slate-300 font-mono">{formatCurrency(stock.avg_buy_price)}</td>
                            <td className="text-slate-300 font-mono">{formatCurrency(stock.current_price)}</td>
                            <td className="text-white font-mono font-medium">{formatCurrency(stock.current_value)}</td>
                            <td className={`font-mono font-medium ${stock.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {formatCurrency(stock.pl)}
                            </td>
                            <td>
                                <span className={`badge-${(stock.pl_pct ?? 0) >= 0 ? 'success' : 'danger'}`}>
                                    {(stock.pl_pct ?? 0) >= 0 ? '+' : ''}{(stock.pl_pct ?? 0).toFixed(2)}%
                                </span>
                            </td>
                            <td>
                                <a
                                    href={`https://www.tradingview.com/chart/?symbol=${stock.symbol}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    className="btn-icon p-1.5"
                                >
                                    <ExternalLink className="w-4 h-4" />
                                </a>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

interface OptionsTableProps {
    options: OptionPosition[]
    expanded: Set<string>
    onToggle: (symbol: string) => void
    onSelect: (symbol: string) => void
    selectedSymbol: string | null
}

function OptionsTable({ options, expanded, onToggle, onSelect, selectedSymbol }: OptionsTableProps) {
    if (options.length === 0) {
        return (
            <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/50 flex items-center justify-center">
                    <Target className="w-8 h-8 text-slate-500" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">No Option Positions</h3>
                <p className="text-slate-400">Sync with Robinhood to load your positions.</p>
            </div>
        )
    }

    const getDTEBadge = (dte: number) => {
        if (dte <= 7) return 'badge-danger'
        if (dte <= 14) return 'badge-warning'
        return 'badge-neutral'
    }

    const getStrategyBadge = (strategy: string) => {
        switch (strategy) {
            case 'CSP': return 'badge-info'
            case 'CC': return 'badge-purple'
            default: return 'badge-neutral'
        }
    }

    return (
        <div className="space-y-3">
            {options.map((option, idx) => {
                const isExpanded = expanded.has(`${option.symbol}-${idx}`)
                return (
                    <div
                        key={`${option.symbol}-${idx}`}
                        className={`glass-card overflow-hidden transition-all ${
                            selectedSymbol === option.symbol ? 'border-primary/50' : ''
                        }`}
                    >
                        <div
                            className="p-4 flex items-center justify-between cursor-pointer hover:bg-white/[0.02] transition-colors"
                            onClick={() => {
                                onSelect(option.symbol)
                                onToggle(`${option.symbol}-${idx}`)
                            }}
                        >
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 rounded-xl bg-slate-800/80 flex items-center justify-center text-sm font-bold text-slate-400">
                                    {option.symbol.substring(0, 2)}
                                </div>
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-bold text-white">{option.symbol}</span>
                                        <span className={getStrategyBadge(option.strategy)}>{option.strategy}</span>
                                        <span className={getDTEBadge(option.dte)}>{option.dte} DTE</span>
                                    </div>
                                    <div className="text-sm text-slate-400">
                                        ${option.strike} {option.option_type.toUpperCase()} - {option.expiration}
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-6">
                                <div className="text-right">
                                    <div className="font-medium text-white">{option.quantity} contracts</div>
                                    <div className="text-sm text-slate-400 font-mono">{formatCurrency(option.current_value)}</div>
                                </div>
                                <div className="text-right min-w-[100px]">
                                    <div className={`font-bold font-mono ${option.pl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {formatCurrency(option.pl)}
                                    </div>
                                    <div className={`text-sm ${(option.pl_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {(option.pl_pct ?? 0) >= 0 ? '+' : ''}{(option.pl_pct ?? 0).toFixed(1)}%
                                    </div>
                                </div>
                                <a
                                    href={`https://www.tradingview.com/chart/?symbol=${option.symbol}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    className="btn-icon p-1.5"
                                    title={`View ${option.symbol} on TradingView`}
                                >
                                    <ExternalLink className="w-4 h-4" />
                                </a>
                                <div className={`p-2 rounded-lg transition-colors ${isExpanded ? 'bg-primary/20 text-primary' : 'text-slate-400'}`}>
                                    {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                                </div>
                            </div>
                        </div>

                        {isExpanded && (
                            <div className="px-4 pb-4 border-t border-slate-700/50">
                                {/* Greeks Row */}
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
                                    <GreekBadge label="Delta" value={option.greeks.delta} suffix="%" />
                                    <GreekBadge label="Theta" value={option.greeks.theta} prefix="$" positive />
                                    <GreekBadge label="Gamma" value={option.greeks.gamma} />
                                    <GreekBadge label="IV" value={option.greeks.iv} suffix="%" />
                                    <GreekBadge label="Break-even" value={option.breakeven} prefix="$" />
                                </div>

                                {/* Entry/Current/Premium Info */}
                                <div className="flex gap-6 mt-4 text-sm">
                                    <div>
                                        <span className="text-slate-500">Entry: </span>
                                        <span className="font-medium text-white font-mono">{formatCurrency(option.avg_price)}</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-500">Current: </span>
                                        <span className="font-medium text-white font-mono">{formatCurrency(option.current_price)}</span>
                                    </div>
                                    <div>
                                        <span className="text-slate-500">Total Premium: </span>
                                        <span className="font-medium text-emerald-400 font-mono">{formatCurrency(option.total_premium)}</span>
                                    </div>
                                </div>

                                {/* Individual Theta Decay Calendar */}
                                <IndividualThetaDecay option={option} />
                            </div>
                        )}
                    </div>
                )
            })}
        </div>
    )
}

interface GreekBadgeProps {
    label: string
    value: number
    prefix?: string
    suffix?: string
    positive?: boolean
}

function GreekBadge({ label, value, prefix = '', suffix = '', positive }: GreekBadgeProps) {
    const displayValue = prefix === '$' ? formatCurrency(value) : `${prefix}${value.toFixed(2)}${suffix}`
    return (
        <div className="bg-slate-800/60 rounded-xl p-3 text-center border border-slate-700/50">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{label}</div>
            <div className={`font-medium font-mono ${positive && value > 0 ? 'text-emerald-400' : 'text-white'}`}>
                {displayValue}
            </div>
        </div>
    )
}

// Individual Theta Decay Component - shows decay for a single option
interface IndividualThetaDecayProps {
    option: OptionPosition
}

function IndividualThetaDecay({ option }: IndividualThetaDecayProps) {
    const theta = option.greeks?.theta || 0
    const dte = option.dte || 0
    const quantity = Math.abs(option.quantity || 1)

    // Determine if this is a SHORT position (where theta benefits us) or LONG (where theta hurts us)
    // SHORT positions: CSP, CC, Short Put, Short Call, or negative quantity
    // LONG positions: Long Call, Long Put, or positive quantity with no "short" indicator
    const strategy = option.strategy?.toLowerCase() || ''
    const type = option.type?.toLowerCase() || ''

    const isShortPosition =
        strategy === 'csp' ||
        strategy === 'cc' ||
        strategy.includes('short') ||
        type.includes('short') ||
        option.quantity < 0

    // For SHORT positions: theta is POSITIVE (we collect it as income)
    // For LONG positions: theta is NEGATIVE (we lose value to time decay)
    // Theta is already in dollars per contract (e.g., -$0.70 means $0.70/day per contract)
    // DO NOT multiply by 100 - that's already factored into the theta value
    const dailyThetaDollars = Math.abs(theta) * quantity

    // Don't show if no theta or expired
    if (dte <= 0 || dailyThetaDollars === 0) {
        return null
    }

    // Generate decay schedule - show up to 14 days or until expiration
    const daysToShow = Math.min(dte, 14)
    const decaySchedule: { day: number; date: string; theta: number; cumulative: number; percentDecayed: number }[] = []

    let cumulative = 0
    const totalPotentialTheta = dailyThetaDollars * dte

    for (let day = 1; day <= daysToShow; day++) {
        // Theta accelerates as expiration approaches - use simplified acceleration model
        // Theta roughly doubles in the last week
        const dteAtDay = dte - day + 1
        const accelerationFactor = dteAtDay <= 7 ? 1 + (7 - dteAtDay) * 0.15 : 1
        const adjustedDailyTheta = dailyThetaDollars * accelerationFactor

        cumulative += adjustedDailyTheta
        const percentDecayed = totalPotentialTheta > 0 ? (cumulative / totalPotentialTheta) * 100 : 0

        decaySchedule.push({
            day,
            date: getDateFromDTE(day),
            theta: adjustedDailyTheta,
            cumulative,
            percentDecayed: Math.min(percentDecayed, 100)
        })
    }

    // Summary calculations
    const sevenDayTheta = decaySchedule.slice(0, 7).reduce((sum, d) => sum + d.theta, 0)
    const totalToExpiry = dailyThetaDollars * dte

    // Colors and signs based on position type
    const valueColor = isShortPosition ? 'text-emerald-400' : 'text-red-400'
    const bgHighlight = isShortPosition ? 'bg-emerald-500/5' : 'bg-red-500/5'
    const progressGradient = isShortPosition
        ? 'bg-gradient-to-r from-emerald-500 to-emerald-400'
        : 'bg-gradient-to-r from-red-500 to-red-400'
    const borderColor = isShortPosition ? 'border-emerald-500/20' : 'border-red-500/20'
    const headerBg = isShortPosition
        ? 'from-emerald-500/5 to-green-500/5'
        : 'from-red-500/5 to-orange-500/5'
    const sign = isShortPosition ? '+' : '-'
    const label = isShortPosition ? 'Premium Income' : 'Time Decay Cost'

    return (
        <div className={`mt-4 p-4 bg-gradient-to-br ${headerBg} rounded-xl border ${borderColor}`}>
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Calendar className={`w-4 h-4 ${isShortPosition ? 'text-emerald-400' : 'text-red-400'}`} />
                    <span className="text-sm font-semibold text-white">Theta Decay</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${isShortPosition ? 'text-emerald-400 bg-emerald-500/20' : 'text-red-400 bg-red-500/20'}`}>
                        {label}
                    </span>
                </div>
                <span className="text-xs text-slate-400">{dte} DTE</span>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="bg-slate-800/50 rounded-lg p-2.5 text-center">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">Daily</div>
                    <div className={`text-sm font-bold ${valueColor} font-mono`}>{sign}{formatCurrency(dailyThetaDollars)}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-2.5 text-center">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">7-Day</div>
                    <div className={`text-sm font-bold ${valueColor} font-mono`}>{sign}{formatCurrency(sevenDayTheta)}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-2.5 text-center">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wide">To Expiry</div>
                    <div className={`text-sm font-bold ${valueColor} font-mono`}>{sign}{formatCurrency(totalToExpiry)}</div>
                </div>
            </div>

            {/* Decay Table */}
            <div className="overflow-x-auto max-h-[200px] overflow-y-auto">
                <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-900/95">
                        <tr className="text-slate-500 uppercase tracking-wide">
                            <th className="text-left py-1.5 px-2">Day</th>
                            <th className="text-left py-1.5 px-2">Date</th>
                            <th className="text-right py-1.5 px-2">Daily</th>
                            <th className="text-right py-1.5 px-2">Cumulative</th>
                            <th className="text-right py-1.5 px-2">Progress</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/30">
                        {decaySchedule.map((row) => (
                            <tr key={row.day} className={row.day <= 7 ? bgHighlight : ''}>
                                <td className="py-1.5 px-2 text-slate-300">Day {row.day}</td>
                                <td className="py-1.5 px-2 text-slate-400">{row.date}</td>
                                <td className={`py-1.5 px-2 text-right font-mono ${valueColor}`}>{sign}{formatCurrency(row.theta)}</td>
                                <td className={`py-1.5 px-2 text-right font-mono ${valueColor}`}>{sign}{formatCurrency(row.cumulative)}</td>
                                <td className="py-1.5 px-2 text-right">
                                    <div className="flex items-center justify-end gap-1.5">
                                        <div className="w-12 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                                            <div
                                                className={`h-full rounded-full ${progressGradient}`}
                                                style={{ width: `${row.percentDecayed}%` }}
                                            />
                                        </div>
                                        <span className="text-slate-400 w-8 text-right">{row.percentDecayed.toFixed(0)}%</span>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {dte > daysToShow && (
                <div className="mt-2 text-center text-xs text-slate-500">
                    Showing first {daysToShow} of {dte} days
                </div>
            )}
        </div>
    )
}

function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value)
}
