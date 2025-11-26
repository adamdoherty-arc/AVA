import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Search, RefreshCw, TrendingUp, TrendingDown, BarChart3,
    Brain, BookOpen, LineChart, Newspaper, Sparkles, Target, Activity
} from 'lucide-react'
import clsx from 'clsx'

export default function Research() {
    const [inputSymbol, setInputSymbol] = useState('')
    const [activeSymbol, setActiveSymbol] = useState<string | null>(null)
    const [activeTab, setActiveTab] = useState<'summary' | 'fundamental' | 'technical' | 'sentiment'>('summary')
    const queryClient = useQueryClient()

    // Fetch research data
    const { data: researchData, isLoading } = useQuery({
        queryKey: ['research', activeSymbol],
        queryFn: async () => {
            if (!activeSymbol) return null
            const { data } = await axiosInstance.get(`/research/${activeSymbol}`)
            return data
        },
        enabled: !!activeSymbol
    })

    // Analyze mutation
    const analyzeMutation = useMutation({
        mutationFn: async (symbol: string) => {
            const { data } = await axiosInstance.get(`/research/${symbol}/refresh`)
            return data
        },
        onSuccess: (data, symbol) => {
            queryClient.setQueryData(['research', symbol], data)
        }
    })

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault()
        if (inputSymbol.trim()) {
            setActiveSymbol(inputSymbol.toUpperCase().trim())
        }
    }

    const handleRefresh = () => {
        if (activeSymbol) {
            analyzeMutation.mutate(activeSymbol)
        }
    }

    // Quick symbols
    const quickSymbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD', 'META', 'GOOG', 'AMZN']

    const getScoreColor = (score: number) => {
        if (score >= 70) return 'text-emerald-400'
        if (score >= 40) return 'text-amber-400'
        return 'text-red-400'
    }

    const getScoreBg = (score: number) => {
        if (score >= 70) return 'from-emerald-500/20 to-emerald-500/5'
        if (score >= 40) return 'from-amber-500/20 to-amber-500/5'
        return 'from-red-500/20 to-red-500/5'
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">AI Research</h1>
                        <p className="text-sm text-slate-400">Multi-agent analysis powered by local LLMs</p>
                    </div>
                </div>
                {activeSymbol && (
                    <button
                        onClick={handleRefresh}
                        disabled={analyzeMutation.isPending}
                        className="btn-primary flex items-center gap-2"
                    >
                        <RefreshCw size={16} className={analyzeMutation.isPending ? 'animate-spin' : ''} />
                        {analyzeMutation.isPending ? 'Analyzing...' : 'Refresh Analysis'}
                    </button>
                )}
            </header>

            {/* Search Card */}
            <div className="glass-card p-6">
                <form onSubmit={handleSearch} className="flex gap-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                        <input
                            type="text"
                            value={inputSymbol}
                            onChange={(e) => setInputSymbol(e.target.value)}
                            placeholder="Enter stock symbol (e.g., AAPL, NVDA, TSLA)"
                            className="input-field pl-12 text-lg"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={!inputSymbol.trim() || isLoading}
                        className="btn-primary px-8 flex items-center gap-2"
                    >
                        <Brain size={20} />
                        Analyze
                    </button>
                </form>

                {/* Quick Symbols */}
                <div className="flex flex-wrap items-center gap-2 mt-4">
                    <span className="text-sm text-slate-400">Quick:</span>
                    {quickSymbols.map(symbol => (
                        <button
                            key={symbol}
                            onClick={() => {
                                setInputSymbol(symbol)
                                setActiveSymbol(symbol)
                            }}
                            className={clsx(
                                "px-3 py-1.5 text-sm rounded-lg font-medium transition-all duration-200",
                                activeSymbol === symbol
                                    ? "bg-primary text-white shadow-lg shadow-primary/20"
                                    : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white border border-slate-700/50"
                            )}
                        >
                            {symbol}
                        </button>
                    ))}
                </div>
            </div>

            {/* Loading State */}
            {(isLoading || analyzeMutation.isPending) && activeSymbol && (
                <div className="glass-card p-12 text-center">
                    <div className="relative w-20 h-20 mx-auto mb-6">
                        <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                        <div className="absolute inset-0 rounded-full border-4 border-t-primary border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                        <div className="absolute inset-3 rounded-full bg-gradient-to-br from-primary/20 to-violet-500/20 flex items-center justify-center">
                            <Brain className="w-8 h-8 text-primary" />
                        </div>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Analyzing {activeSymbol}</h3>
                    <p className="text-slate-400 mb-6">Running multi-agent research pipeline...</p>
                    <div className="flex justify-center gap-3">
                        {['Fundamental', 'Technical', 'Sentiment', 'RAG'].map(agent => (
                            <span key={agent} className="badge-neutral flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                                {agent}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* No Symbol Selected */}
            {!activeSymbol && !isLoading && (
                <div className="glass-card p-16 text-center">
                    <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-violet-500/20 to-purple-500/20 flex items-center justify-center mx-auto mb-6 border border-violet-500/20">
                        <Brain className="w-12 h-12 text-violet-400" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">Enter a Symbol to Analyze</h3>
                    <p className="text-slate-400 max-w-md mx-auto">
                        Our AI research assistant will perform comprehensive fundamental, technical, and sentiment analysis.
                    </p>
                </div>
            )}

            {/* Research Results */}
            {researchData && !isLoading && !analyzeMutation.isPending && (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Score Cards - Left Sidebar */}
                    <div className="lg:col-span-1 space-y-5">
                        {/* Overall Score */}
                        <div className="glass-card p-6 text-center">
                            <div className="flex items-center justify-center gap-2 mb-4">
                                <Target className="w-5 h-5 text-primary" />
                                <h3 className="text-lg font-semibold text-white">{activeSymbol}</h3>
                            </div>
                            <div className={`relative w-32 h-32 mx-auto mb-4`}>
                                <div className={`absolute inset-0 rounded-full bg-gradient-to-br ${getScoreBg(researchData.overall_score || 0)}`}></div>
                                <div className="absolute inset-2 rounded-full bg-slate-900 flex items-center justify-center">
                                    <span className={`text-5xl font-bold ${getScoreColor(researchData.overall_score || 0)}`}>
                                        {researchData.overall_score || 0}
                                    </span>
                                </div>
                            </div>
                            <p className="text-sm text-slate-400">Overall Score</p>
                        </div>

                        {/* Individual Scores */}
                        <div className="glass-card p-5 space-y-4">
                            <ScoreBar label="Fundamental" score={researchData.fundamental?.score || 0} icon={<BarChart3 size={16} />} />
                            <ScoreBar label="Technical" score={researchData.technical?.score || 0} icon={<LineChart size={16} />} />
                            <ScoreBar label="Sentiment" score={researchData.sentiment?.score || 0} icon={<Newspaper size={16} />} />
                        </div>

                        {/* Recommendation */}
                        <div className={clsx(
                            "glass-card p-5 text-center border-2",
                            researchData.recommendation === 'BUY' ? 'border-emerald-500/30 bg-emerald-500/5' :
                            researchData.recommendation === 'SELL' ? 'border-red-500/30 bg-red-500/5' :
                            'border-amber-500/30 bg-amber-500/5'
                        )}>
                            <div className="text-sm text-slate-400 mb-2">AI Recommendation</div>
                            <div className={clsx(
                                "text-3xl font-bold",
                                researchData.recommendation === 'BUY' ? 'text-emerald-400' :
                                researchData.recommendation === 'SELL' ? 'text-red-400' :
                                'text-amber-400'
                            )}>
                                {researchData.recommendation || 'HOLD'}
                            </div>
                        </div>
                    </div>

                    {/* Detailed Analysis - Main Content */}
                    <div className="lg:col-span-3">
                        <div className="glass-card overflow-hidden">
                            {/* Tabs */}
                            <div className="flex border-b border-slate-700/50 bg-slate-900/30">
                                {[
                                    { id: 'summary', label: 'Summary', icon: <BookOpen size={16} /> },
                                    { id: 'fundamental', label: 'Fundamental', icon: <BarChart3 size={16} /> },
                                    { id: 'technical', label: 'Technical', icon: <LineChart size={16} /> },
                                    { id: 'sentiment', label: 'Sentiment', icon: <Newspaper size={16} /> }
                                ].map(tab => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id as typeof activeTab)}
                                        className={clsx(
                                            "flex-1 flex items-center justify-center gap-2 py-4 transition-all duration-200 relative",
                                            activeTab === tab.id
                                                ? "text-primary bg-primary/5"
                                                : "text-slate-400 hover:text-white hover:bg-slate-800/30"
                                        )}
                                    >
                                        {tab.icon}
                                        <span className="font-medium">{tab.label}</span>
                                        {activeTab === tab.id && (
                                            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary to-violet-500"></span>
                                        )}
                                    </button>
                                ))}
                            </div>

                            {/* Tab Content */}
                            <div className="p-6">
                                {activeTab === 'summary' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="flex items-center gap-2 mb-4">
                                            <Sparkles className="w-5 h-5 text-violet-400" />
                                            <h4 className="text-lg font-semibold text-white">Executive Summary</h4>
                                        </div>
                                        <p className="text-slate-300 leading-relaxed">
                                            {researchData.summary || 'No summary available. Click refresh to generate analysis.'}
                                        </p>

                                        {/* Key Points */}
                                        <div className="grid grid-cols-2 gap-5 mt-6">
                                            <div className="glass-card p-5 border border-emerald-500/20 bg-emerald-500/5">
                                                <div className="flex items-center gap-2 text-emerald-400 mb-3">
                                                    <TrendingUp size={18} />
                                                    <span className="font-semibold">Strengths</span>
                                                </div>
                                                <ul className="text-sm text-slate-300 space-y-2">
                                                    {researchData.strengths?.map((s: string, i: number) => (
                                                        <li key={i} className="flex items-start gap-2">
                                                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 flex-shrink-0"></span>
                                                            {s}
                                                        </li>
                                                    )) || <li className="text-slate-500">No data available</li>}
                                                </ul>
                                            </div>
                                            <div className="glass-card p-5 border border-red-500/20 bg-red-500/5">
                                                <div className="flex items-center gap-2 text-red-400 mb-3">
                                                    <TrendingDown size={18} />
                                                    <span className="font-semibold">Risks</span>
                                                </div>
                                                <ul className="text-sm text-slate-300 space-y-2">
                                                    {researchData.risks?.map((r: string, i: number) => (
                                                        <li key={i} className="flex items-start gap-2">
                                                            <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-2 flex-shrink-0"></span>
                                                            {r}
                                                        </li>
                                                    )) || <li className="text-slate-500">No data available</li>}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {activeTab === 'fundamental' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="flex items-center gap-2 mb-4">
                                            <BarChart3 className="w-5 h-5 text-blue-400" />
                                            <h4 className="text-lg font-semibold text-white">Fundamental Analysis</h4>
                                        </div>
                                        <div className="grid grid-cols-3 gap-4 mb-6">
                                            <MetricCard label="P/E Ratio" value={researchData.fundamental?.metrics?.pe_ratio || 'N/A'} />
                                            <MetricCard label="Market Cap" value={researchData.fundamental?.metrics?.market_cap || 'N/A'} />
                                            <MetricCard label="Revenue Growth" value={researchData.fundamental?.metrics?.revenue_growth || 'N/A'} />
                                        </div>
                                        <p className="text-slate-300 leading-relaxed">
                                            {researchData.fundamental?.analysis || 'No fundamental analysis available.'}
                                        </p>
                                    </div>
                                )}

                                {activeTab === 'technical' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="flex items-center gap-2 mb-4">
                                            <Activity className="w-5 h-5 text-cyan-400" />
                                            <h4 className="text-lg font-semibold text-white">Technical Analysis</h4>
                                        </div>
                                        <div className="grid grid-cols-4 gap-4 mb-6">
                                            <MetricCard label="RSI" value={researchData.technical?.indicators?.rsi || 'N/A'} />
                                            <MetricCard label="MACD" value={researchData.technical?.indicators?.macd || 'N/A'} />
                                            <MetricCard label="Trend" value={researchData.technical?.trend || 'N/A'} />
                                            <MetricCard label="Volume" value={researchData.technical?.indicators?.volume || 'N/A'} />
                                        </div>
                                        <p className="text-slate-300 leading-relaxed">
                                            {researchData.technical?.analysis || 'No technical analysis available.'}
                                        </p>
                                    </div>
                                )}

                                {activeTab === 'sentiment' && (
                                    <div className="space-y-6 animate-fade-in">
                                        <div className="flex items-center gap-2 mb-4">
                                            <Newspaper className="w-5 h-5 text-amber-400" />
                                            <h4 className="text-lg font-semibold text-white">Sentiment Analysis</h4>
                                        </div>
                                        <div className="grid grid-cols-3 gap-4 mb-6">
                                            <MetricCard label="Social Score" value={researchData.sentiment?.social_score || 'N/A'} />
                                            <MetricCard label="News Sentiment" value={researchData.sentiment?.news_sentiment || 'N/A'} />
                                            <MetricCard label="Analyst Rating" value={researchData.sentiment?.analyst_rating || 'N/A'} />
                                        </div>
                                        <p className="text-slate-300 leading-relaxed">
                                            {researchData.sentiment?.analysis || 'No sentiment analysis available.'}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

// Sub-components
function ScoreBar({ label, score, icon }: { label: string; score: number; icon: React.ReactNode }) {
    const getBarColor = (s: number) => {
        if (s >= 70) return 'bg-gradient-to-r from-emerald-500 to-emerald-400'
        if (s >= 40) return 'bg-gradient-to-r from-amber-500 to-amber-400'
        return 'bg-gradient-to-r from-red-500 to-red-400'
    }

    return (
        <div>
            <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-slate-400 flex items-center gap-2">{icon}{label}</span>
                <span className="text-sm font-semibold text-white">{score}</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                    className={`h-full ${getBarColor(score)} transition-all duration-700 ease-out rounded-full`}
                    style={{ width: `${score}%` }}
                />
            </div>
        </div>
    )
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
    return (
        <div className="glass-card p-4 text-center hover:border-primary/30 transition-colors">
            <div className="text-xs text-slate-400 mb-1 uppercase tracking-wide">{label}</div>
            <div className="font-bold text-white text-lg">{value}</div>
        </div>
    )
}
