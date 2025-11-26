import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Search, Bot, Sparkles, RefreshCw, Clock, CheckCircle,
    AlertCircle, FileText, TrendingUp, BarChart3, Brain
} from 'lucide-react'
import clsx from 'clsx'

interface ResearchResult {
    query: string
    agents_used: string[]
    findings: {
        agent: string
        title: string
        content: string
        confidence: number
        sources: string[]
    }[]
    synthesis: string
    recommendations: string[]
    processing_time: number
}

const RESEARCH_TEMPLATES = [
    { label: 'Stock Analysis', query: 'Comprehensive analysis of {symbol} including technicals, fundamentals, and sentiment' },
    { label: 'Sector Deep Dive', query: 'Deep dive analysis of the {sector} sector with top picks and risks' },
    { label: 'Market Overview', query: 'Current market conditions, key levels, and trading opportunities' },
    { label: 'Options Strategy', query: 'Best options strategies for {symbol} given current IV and price action' },
    { label: 'Earnings Preview', query: 'Earnings preview for {symbol} with expected move and trading ideas' },
]

export default function MultiAgentResearch() {
    const [query, setQuery] = useState('')
    const [result, setResult] = useState<ResearchResult | null>(null)

    const researchMutation = useMutation({
        mutationFn: async (q: string) => {
            const { data } = await axiosInstance.post('/research/multi-agent', { query: q })
            return data as ResearchResult
        },
        onSuccess: (data) => {
            setResult(data)
        }
    })

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (query.trim()) {
            researchMutation.mutate(query)
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header>
                <h1 className="page-title flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-600 flex items-center justify-center shadow-lg">
                        <Brain className="w-5 h-5 text-white" />
                    </div>
                    Multi-Agent Research
                </h1>
                <p className="page-subtitle">AI-powered deep research using multiple specialized agents</p>
            </header>

            {/* Research Input */}
            <div className="card p-6">
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Research Query</label>
                        <textarea
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            placeholder="Enter your research question... e.g., 'What's the outlook for NVDA given AI chip demand?'"
                            className="w-full h-32 bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 resize-none"
                        />
                    </div>

                    <div>
                        <p className="text-sm text-slate-400 mb-2">Quick Templates</p>
                        <div className="flex flex-wrap gap-2">
                            {RESEARCH_TEMPLATES.map(template => (
                                <button
                                    key={template.label}
                                    type="button"
                                    onClick={() => setQuery(template.query)}
                                    className="px-3 py-1.5 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg text-sm transition-colors"
                                >
                                    {template.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={!query.trim() || researchMutation.isPending}
                        className="w-full bg-primary hover:bg-primary/80 disabled:opacity-50 px-6 py-3 rounded-lg font-semibold flex items-center justify-center gap-2"
                    >
                        {researchMutation.isPending ? (
                            <>
                                <RefreshCw className="w-5 h-5 animate-spin" />
                                Researching with multiple agents...
                            </>
                        ) : (
                            <>
                                <Sparkles className="w-5 h-5" />
                                Start Multi-Agent Research
                            </>
                        )}
                    </button>
                </form>
            </div>

            {/* Processing Status */}
            {researchMutation.isPending && (
                <div className="card p-6">
                    <h3 className="font-semibold mb-4 flex items-center gap-2">
                        <RefreshCw className="w-5 h-5 text-primary animate-spin" />
                        Research in Progress
                    </h3>
                    <div className="space-y-3">
                        {['Technical Agent', 'Fundamental Agent', 'Sentiment Agent', 'News Agent', 'Synthesis Agent'].map((agent, idx) => (
                            <div key={agent} className="flex items-center gap-3">
                                <div className={clsx(
                                    "w-6 h-6 rounded-full flex items-center justify-center",
                                    idx <= 2 ? "bg-emerald-500/20" : "bg-slate-700"
                                )}>
                                    {idx <= 2 ? (
                                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                                    ) : (
                                        <Clock className="w-4 h-4 text-slate-400" />
                                    )}
                                </div>
                                <span className={idx <= 2 ? "text-slate-300" : "text-slate-500"}>{agent}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="space-y-4">
                    {/* Synthesis */}
                    <div className="card p-6 bg-gradient-to-br from-primary/10 to-secondary/10 border-primary/30">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-primary" />
                            AI Synthesis
                        </h3>
                        <p className="text-slate-300 leading-relaxed">{result.synthesis}</p>

                        <div className="mt-4 flex items-center gap-2 text-sm text-slate-400">
                            <Clock className="w-4 h-4" />
                            Processed in {result.processing_time.toFixed(1)}s using {result.agents_used.length} agents
                        </div>
                    </div>

                    {/* Recommendations */}
                    <div className="card p-6">
                        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-emerald-400" />
                            Key Recommendations
                        </h3>
                        <ul className="space-y-2">
                            {result.recommendations.map((rec, idx) => (
                                <li key={idx} className="flex items-start gap-2">
                                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-1 flex-shrink-0" />
                                    <span>{rec}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Agent Findings */}
                    <div className="card p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Bot className="w-5 h-5 text-blue-400" />
                            Agent Findings
                        </h3>
                        <div className="space-y-4">
                            {result.findings.map((finding, idx) => (
                                <div key={idx} className="p-4 bg-slate-800/50 rounded-lg">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-medium flex items-center gap-2">
                                            <Bot className="w-4 h-4 text-primary" />
                                            {finding.agent}
                                        </span>
                                        <span className={clsx(
                                            "px-2 py-0.5 rounded text-xs",
                                            finding.confidence >= 80 ? "bg-emerald-500/20 text-emerald-400" :
                                            finding.confidence >= 60 ? "bg-amber-500/20 text-amber-400" :
                                            "bg-slate-700 text-slate-400"
                                        )}>
                                            {finding.confidence}% confidence
                                        </span>
                                    </div>
                                    <h4 className="font-medium text-slate-300 mb-1">{finding.title}</h4>
                                    <p className="text-sm text-slate-400">{finding.content}</p>
                                    {finding.sources.length > 0 && (
                                        <div className="mt-2 flex flex-wrap gap-1">
                                            {finding.sources.map((source, sIdx) => (
                                                <span key={sIdx} className="text-xs text-slate-500">
                                                    [{source}]
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Info */}
            <div className="card p-4 text-sm text-slate-400">
                <p className="font-medium text-white mb-2">How Multi-Agent Research Works</p>
                <ul className="list-disc list-inside space-y-1">
                    <li><strong>Technical Agent:</strong> Analyzes price action, patterns, and indicators</li>
                    <li><strong>Fundamental Agent:</strong> Reviews financials, valuation, and earnings</li>
                    <li><strong>Sentiment Agent:</strong> Monitors social media and news sentiment</li>
                    <li><strong>News Agent:</strong> Scans recent headlines and events</li>
                    <li><strong>Synthesis Agent:</strong> Combines all findings into actionable insights</li>
                </ul>
            </div>
        </div>
    )
}
