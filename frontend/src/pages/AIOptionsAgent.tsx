import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Bot, RefreshCw, Search, Filter, TrendingUp, Settings,
    Clock, DollarSign, Target, Percent, Activity, ChevronDown,
    Download, Zap, BarChart3, Sparkles, Play, Loader2,
    AlertCircle, CheckCircle, XCircle
} from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

const WATCHLISTS = {
    'All Stocks': [],
    'Top Tech': ['AAPL', 'MSFT', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'TSLA'],
    'Banks': ['BAC', 'WFC', 'C', 'JPM', 'GS', 'MS'],
    'Popular': ['AAPL', 'AMD', 'AMZN', 'BAC', 'F', 'GOOG', 'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'TSLA'],
    'High IV': ['TSLA', 'NVDA', 'AMD', 'PLTR', 'SOFI', 'SNAP', 'GME', 'AMC']
}

const LLM_MODELS = [
    { id: 'qwen2.5:32b', name: 'Qwen 2.5 32B', description: 'High quality, slower' },
    { id: 'qwen2.5:14b', name: 'Qwen 2.5 14B', description: 'Balanced (default)' },
    { id: 'qwen2.5:7b', name: 'Qwen 2.5 7B', description: 'Fast, lower quality' },
    { id: 'llama3.3:70b', name: 'Llama 3.3 70B', description: 'Best for complex analysis' }
]

const SCORE_COLORS = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']

interface AnalysisResult {
    symbol: string
    company_name: string
    current_price: number
    strike: number
    expiration: string
    dte: number
    premium: number
    score: number
    recommendation: 'Strong Buy' | 'Buy' | 'Hold' | 'Avoid'
    reasoning: string
    delta: number
    iv: number
    premium_pct: number
    monthly_return: number
}

export default function AIOptionsAgent() {
    const [selectedWatchlist, setSelectedWatchlist] = useState('Popular')
    const [customSymbols, setCustomSymbols] = useState('')
    const [minDTE, setMinDTE] = useState(20)
    const [maxDTE, setMaxDTE] = useState(40)
    const [minDelta, setMinDelta] = useState(-0.45)
    const [maxDelta, setMaxDelta] = useState(-0.15)
    const [minPremium, setMinPremium] = useState(100)
    const [maxResults, setMaxResults] = useState(200)
    const [minScoreDisplay, setMinScoreDisplay] = useState(50)
    const [useLLMReasoning, setUseLLMReasoning] = useState(false)
    const [selectedModel, setSelectedModel] = useState('qwen2.5:14b')
    const [showSettings, setShowSettings] = useState(false)
    const [activeTab, setActiveTab] = useState<'results' | 'top' | 'performance'>('results')

    // Fetch top recommendations on load
    const { data: topPicks, isLoading: topLoading, refetch: refetchTop } = useQuery({
        queryKey: ['ai-options-top', minScoreDisplay],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/agents/options/top?min_score=${minScoreDisplay}`)
            return data
        },
        staleTime: 300000,
    })

    // Run analysis mutation
    const analysisMutation = useMutation({
        mutationFn: async (params: {
            symbols: string[]
            min_dte: number
            max_dte: number
            min_delta: number
            max_delta: number
            min_premium: number
            max_results: number
            use_llm: boolean
            model: string
        }) => {
            const { data } = await axiosInstance.post('/agents/options/analyze', params)
            return data
        }
    })

    const handleRunAnalysis = () => {
        const symbols = customSymbols.trim()
            ? customSymbols.split(',').map(s => s.trim().toUpperCase())
            : WATCHLISTS[selectedWatchlist as keyof typeof WATCHLISTS] || []

        analysisMutation.mutate({
            symbols,
            min_dte: minDTE,
            max_dte: maxDTE,
            min_delta: minDelta,
            max_delta: maxDelta,
            min_premium: minPremium,
            max_results: maxResults,
            use_llm: useLLMReasoning,
            model: selectedModel
        })
    }

    const results: AnalysisResult[] = analysisMutation.data?.results?.filter(
        (r: AnalysisResult) => r.score >= minScoreDisplay
    ) || []

    const scoreDistribution = results.length > 0 ? [
        { name: '90+', value: results.filter(r => r.score >= 90).length, color: '#10B981' },
        { name: '70-89', value: results.filter(r => r.score >= 70 && r.score < 90).length, color: '#3B82F6' },
        { name: '50-69', value: results.filter(r => r.score >= 50 && r.score < 70).length, color: '#F59E0B' },
        { name: '<50', value: results.filter(r => r.score < 50).length, color: '#EF4444' }
    ] : []

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        AI Options Agent
                    </h1>
                    <p className="page-subtitle">Multi-Criteria Decision Making + LLM-Powered Analysis</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowSettings(!showSettings)}
                        className={`btn-icon ${showSettings ? 'bg-primary/20 text-primary' : ''}`}
                    >
                        <Settings className="w-5 h-5" />
                    </button>
                    <button
                        onClick={() => refetchTop()}
                        className="btn-icon"
                    >
                        <RefreshCw className="w-5 h-5" />
                    </button>
                </div>
            </header>

            {/* LLM Settings Panel */}
            {showSettings && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-purple-400" />
                        LLM Configuration
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {LLM_MODELS.map(model => (
                            <button
                                key={model.id}
                                onClick={() => setSelectedModel(model.id)}
                                className={`p-4 rounded-xl text-left transition-all ${
                                    selectedModel === model.id
                                        ? 'bg-purple-500/20 border-purple-500 border-2'
                                        : 'bg-slate-800/60 border border-slate-700/50 hover:border-slate-600'
                                }`}
                            >
                                <div className="font-semibold text-white">{model.name}</div>
                                <div className="text-sm text-slate-400 mt-1">{model.description}</div>
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Analysis Controls */}
            <div className="glass-card p-5">
                <div className="flex items-center gap-3 mb-5">
                    <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                        <Search className="w-4 h-4 text-purple-400" />
                    </div>
                    <h3 className="font-semibold text-white">Analysis Settings</h3>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Stock Selection */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Stock Selection</label>
                        <div className="flex flex-wrap gap-2 mb-3">
                            {Object.keys(WATCHLISTS).map(name => (
                                <button
                                    key={name}
                                    onClick={() => {
                                        setSelectedWatchlist(name)
                                        setCustomSymbols('')
                                    }}
                                    className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-all ${
                                        selectedWatchlist === name && !customSymbols
                                            ? 'bg-primary text-white'
                                            : 'bg-slate-800/60 text-slate-400 hover:text-white hover:bg-slate-700/60'
                                    }`}
                                >
                                    {name}
                                </button>
                            ))}
                        </div>
                        <input
                            type="text"
                            value={customSymbols}
                            onChange={(e) => setCustomSymbols(e.target.value)}
                            placeholder="Or enter symbols: AAPL, MSFT, NVDA..."
                            className="input-field"
                        />
                    </div>

                    {/* DTE Range */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Days to Expiration</label>
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Min DTE</label>
                                <input
                                    type="number"
                                    value={minDTE}
                                    onChange={(e) => setMinDTE(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Max DTE</label>
                                <input
                                    type="number"
                                    value={maxDTE}
                                    onChange={(e) => setMaxDTE(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3 mt-3">
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Min Delta</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={minDelta}
                                    onChange={(e) => setMinDelta(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Max Delta</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={maxDelta}
                                    onChange={(e) => setMaxDelta(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Other Settings */}
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Analysis Options</label>
                        <div className="space-y-3">
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Min Premium ($)</label>
                                <input
                                    type="number"
                                    value={minPremium}
                                    onChange={(e) => setMinPremium(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-500 mb-1">Max Results</label>
                                <input
                                    type="number"
                                    value={maxResults}
                                    onChange={(e) => setMaxResults(Number(e.target.value))}
                                    className="input-field"
                                />
                            </div>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={useLLMReasoning}
                                    onChange={(e) => setUseLLMReasoning(e.target.checked)}
                                    className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary focus:ring-primary"
                                />
                                <span className="text-sm text-slate-300">Use LLM Reasoning (slower)</span>
                            </label>
                        </div>
                    </div>
                </div>

                {/* Display Filter */}
                <div className="mt-5 pt-5 border-t border-slate-700/50">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <label className="text-sm text-slate-400">Min Score to Display:</label>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={minScoreDisplay}
                                onChange={(e) => setMinScoreDisplay(Number(e.target.value))}
                                className="w-32"
                            />
                            <span className="text-white font-mono">{minScoreDisplay}</span>
                        </div>
                        <button
                            onClick={handleRunAnalysis}
                            disabled={analysisMutation.isPending}
                            className="btn-primary px-6 py-2.5 flex items-center gap-2"
                        >
                            {analysisMutation.isPending ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Play className="w-5 h-5" />
                                    Run Analysis
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2">
                {(['results', 'top', 'performance'] as const).map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${
                            activeTab === tab
                                ? 'bg-primary text-white'
                                : 'bg-slate-800/60 text-slate-400 hover:text-white'
                        }`}
                    >
                        {tab === 'results' && 'Analysis Results'}
                        {tab === 'top' && 'Top Picks'}
                        {tab === 'performance' && 'Performance'}
                    </button>
                ))}
            </div>

            {/* Error */}
            {analysisMutation.error && (
                <div className="glass-card p-5 border-red-500/30 bg-red-500/5">
                    <div className="flex items-center gap-3 text-red-400">
                        <AlertCircle className="w-5 h-5" />
                        <span>Error running analysis. Please try again.</span>
                    </div>
                </div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
                <div className="space-y-6">
                    {/* Summary Stats */}
                    {results.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            <div className="stat-card">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <p className="text-xs text-slate-400 uppercase tracking-wide">Opportunities</p>
                                        <p className="text-2xl font-bold text-white">{results.length}</p>
                                        <p className="text-xs text-slate-500">Score {'>='} {minScoreDisplay}</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center text-blue-400">
                                        <Target className="w-5 h-5" />
                                    </div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <p className="text-xs text-slate-400 uppercase tracking-wide">Strong Buys</p>
                                        <p className="text-2xl font-bold text-emerald-400">
                                            {results.filter(r => r.recommendation === 'Strong Buy').length}
                                        </p>
                                        <p className="text-xs text-slate-500">Score {'>='} 90</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                                        <CheckCircle className="w-5 h-5" />
                                    </div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <p className="text-xs text-slate-400 uppercase tracking-wide">Avg Score</p>
                                        <p className="text-2xl font-bold text-white">
                                            {(results.reduce((a, r) => a + (r.score ?? 0), 0) / (results.length || 1)).toFixed(0)}
                                        </p>
                                        <p className="text-xs text-slate-500">MCDM weighted</p>
                                    </div>
                                    <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400">
                                        <BarChart3 className="w-5 h-5" />
                                    </div>
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="h-20">
                                    <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                                        <PieChart>
                                            <Pie
                                                data={scoreDistribution}
                                                dataKey="value"
                                                nameKey="name"
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={20}
                                                outerRadius={35}
                                            >
                                                {scoreDistribution.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <Tooltip />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Results Table */}
                    {results.length > 0 && (
                        <div className="glass-card overflow-hidden">
                            <div className="p-5 border-b border-slate-700/50 flex items-center justify-between bg-slate-900/30">
                                <div>
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <Zap className="w-5 h-5 text-amber-400" />
                                        {results.length} Analysis Results
                                    </h3>
                                    <p className="text-sm text-slate-400">Sorted by MCDM Score</p>
                                </div>
                                <button className="btn-secondary flex items-center gap-2">
                                    <Download className="w-4 h-4" />
                                    Export
                                </button>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Symbol</th>
                                            <th>Score</th>
                                            <th>Rec</th>
                                            <th>Price</th>
                                            <th>Strike</th>
                                            <th>DTE</th>
                                            <th>Premium</th>
                                            <th>Delta</th>
                                            <th>IV</th>
                                            <th>Monthly %</th>
                                            {useLLMReasoning && <th>Reasoning</th>}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {results.map((result, idx) => (
                                            <tr key={`${result.symbol}-${idx}`}>
                                                <td>
                                                    <div>
                                                        <span className="font-bold text-primary">{result.symbol}</span>
                                                        <span className="block text-xs text-slate-500">{result.company_name}</span>
                                                    </div>
                                                </td>
                                                <td>
                                                    <div className={`inline-flex items-center px-2.5 py-1 rounded-lg font-bold ${
                                                        result.score >= 90 ? 'bg-emerald-500/20 text-emerald-400' :
                                                        result.score >= 70 ? 'bg-blue-500/20 text-blue-400' :
                                                        result.score >= 50 ? 'bg-amber-500/20 text-amber-400' :
                                                        'bg-red-500/20 text-red-400'
                                                    }`}>
                                                        {result.score}
                                                    </div>
                                                </td>
                                                <td>
                                                    <span className={`badge-${
                                                        result.recommendation === 'Strong Buy' ? 'success' :
                                                        result.recommendation === 'Buy' ? 'info' :
                                                        result.recommendation === 'Hold' ? 'warning' : 'danger'
                                                    }`}>
                                                        {result.recommendation}
                                                    </span>
                                                </td>
                                                <td className="font-mono text-slate-300">${result.current_price}</td>
                                                <td className="font-mono text-white">${result.strike}</td>
                                                <td>
                                                    <span className={`badge-${result.dte <= 14 ? 'warning' : 'neutral'}`}>
                                                        {result.dte}d
                                                    </span>
                                                </td>
                                                <td className="font-mono font-medium text-white">${result.premium}</td>
                                                <td className="font-mono text-slate-400">{(result.delta ?? 0).toFixed(2)}</td>
                                                <td className={result.iv >= 50 ? 'text-amber-400' : 'text-slate-400'}>
                                                    {result.iv}%
                                                </td>
                                                <td>
                                                    <span className="font-bold text-emerald-400 bg-emerald-500/15 px-2 py-1 rounded-lg">
                                                        {(result.monthly_return ?? 0).toFixed(2)}%
                                                    </span>
                                                </td>
                                                {useLLMReasoning && (
                                                    <td className="max-w-xs">
                                                        <p className="text-xs text-slate-400 truncate" title={result.reasoning}>
                                                            {result.reasoning}
                                                        </p>
                                                    </td>
                                                )}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Empty State */}
                    {results.length === 0 && !analysisMutation.isPending && (
                        <div className="glass-card p-12 text-center">
                            <Bot className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-white mb-2">No Analysis Results</h3>
                            <p className="text-slate-400 mb-6">
                                Configure your analysis settings above and click "Run Analysis" to find the best CSP opportunities.
                            </p>
                            <button onClick={handleRunAnalysis} className="btn-primary">
                                <Play className="w-4 h-4 mr-2" />
                                Run Analysis
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Top Picks Tab */}
            {activeTab === 'top' && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Zap className="w-5 h-5 text-amber-400" />
                        Top Recommendations (Last 24h)
                    </h3>
                    {topLoading ? (
                        <div className="flex justify-center py-8">
                            <Loader2 className="w-8 h-8 animate-spin text-primary" />
                        </div>
                    ) : topPicks?.recommendations?.length > 0 ? (
                        <div className="space-y-3">
                            {topPicks.recommendations.slice(0, 10).map((pick: AnalysisResult, idx: number) => (
                                <div key={idx} className="flex items-center justify-between p-4 bg-slate-800/40 rounded-xl">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-lg ${
                                            pick.score >= 90 ? 'bg-emerald-500/20 text-emerald-400' :
                                            pick.score >= 70 ? 'bg-blue-500/20 text-blue-400' :
                                            'bg-amber-500/20 text-amber-400'
                                        }`}>
                                            {pick.score}
                                        </div>
                                        <div>
                                            <div className="font-bold text-white">{pick.symbol}</div>
                                            <div className="text-sm text-slate-400">
                                                ${pick.strike} | {pick.dte}d | ${pick.premium}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <span className={`badge-${
                                            pick.recommendation === 'Strong Buy' ? 'success' :
                                            pick.recommendation === 'Buy' ? 'info' : 'neutral'
                                        }`}>
                                            {pick.recommendation}
                                        </span>
                                        <div className="text-sm text-emerald-400 mt-1">
                                            {(pick.monthly_return ?? 0).toFixed(2)}% monthly
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-slate-400">
                            No recent recommendations. Run an analysis first.
                        </div>
                    )}
                </div>
            )}

            {/* Performance Tab */}
            {activeTab === 'performance' && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-emerald-400" />
                        Agent Performance
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm">Total Analyses</div>
                            <div className="text-2xl font-bold text-white mt-1">
                                {analysisMutation.data?.total_analyzed || topPicks?.total_analyzed || 0}
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm">Avg Score</div>
                            <div className="text-2xl font-bold text-emerald-400 mt-1">
                                {topPicks?.avg_score?.toFixed(1) || '-'}
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm">Strong Buy Rate</div>
                            <div className="text-2xl font-bold text-purple-400 mt-1">
                                {topPicks?.strong_buy_rate?.toFixed(1) || '-'}%
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
