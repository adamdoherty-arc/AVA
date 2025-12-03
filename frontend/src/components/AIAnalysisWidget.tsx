import { useState, useRef, useEffect } from 'react'
import { Brain, Send, X, Loader2, RefreshCw, Sparkles, StopCircle, Maximize2, Minimize2 } from 'lucide-react'
import { useAIStockAnalysis } from '../hooks/useStreamingAI'
import clsx from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'

interface AIAnalysisWidgetProps {
    symbol?: string
    onClose?: () => void
    className?: string
    defaultExpanded?: boolean
}

/**
 * AI-powered stock analysis widget with real-time streaming responses
 * Modern floating widget with collapsible interface
 */
export function AIAnalysisWidget({
    symbol: initialSymbol,
    onClose,
    className,
    defaultExpanded = false
}: AIAnalysisWidgetProps) {
    const [isExpanded, setIsExpanded] = useState(defaultExpanded)
    const [symbol, setSymbol] = useState(initialSymbol || '')
    const [analysisType, setAnalysisType] = useState<'technical' | 'fundamental' | 'options' | 'full'>('full')
    const responseRef = useRef<HTMLDivElement>(null)

    const { analyzeStock, abort, reset, isStreaming, response, error } = useAIStockAnalysis()

    // Auto-scroll response container
    useEffect(() => {
        if (responseRef.current && response) {
            responseRef.current.scrollTop = responseRef.current.scrollHeight
        }
    }, [response])

    // Analyze when symbol changes from props
    useEffect(() => {
        if (initialSymbol && initialSymbol !== symbol) {
            setSymbol(initialSymbol)
        }
    }, [initialSymbol])

    const handleAnalyze = () => {
        if (!symbol.trim()) return
        reset()
        analyzeStock(symbol.trim().toUpperCase(), analysisType)
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleAnalyze()
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className={clsx(
                'fixed bottom-4 right-4 z-50',
                isExpanded ? 'w-[480px]' : 'w-auto',
                className
            )}
        >
            <div className={clsx(
                'glass-card overflow-hidden shadow-2xl shadow-purple-500/10 border border-purple-500/20',
                isExpanded && 'h-[600px] flex flex-col'
            )}>
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-slate-700/50 bg-gradient-to-r from-purple-500/10 to-pink-500/10">
                    <button
                        onClick={() => setIsExpanded(!isExpanded)}
                        className="flex items-center gap-3"
                    >
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center shadow-lg">
                            <Brain className="w-5 h-5 text-white" />
                        </div>
                        <div className="text-left">
                            <h3 className="font-semibold text-white flex items-center gap-2">
                                AI Analysis
                                <Sparkles className="w-4 h-4 text-amber-400" />
                            </h3>
                            {!isExpanded && symbol && (
                                <p className="text-xs text-slate-400">Analyzing {symbol}</p>
                            )}
                        </div>
                    </button>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setIsExpanded(!isExpanded)}
                            className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50"
                        >
                            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                        </button>
                        {onClose && (
                            <button
                                onClick={onClose}
                                className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>

                <AnimatePresence>
                    {isExpanded && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="flex-1 flex flex-col overflow-hidden"
                        >
                            {/* Input Section */}
                            <div className="p-4 border-b border-slate-700/50 space-y-3">
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={symbol}
                                        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                                        onKeyDown={handleKeyDown}
                                        placeholder="Enter symbol (e.g., AAPL)"
                                        className="flex-1 px-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all"
                                    />
                                    {isStreaming ? (
                                        <button
                                            onClick={abort}
                                            className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors flex items-center gap-2"
                                        >
                                            <StopCircle className="w-4 h-4" />
                                            Stop
                                        </button>
                                    ) : (
                                        <button
                                            onClick={handleAnalyze}
                                            disabled={!symbol.trim()}
                                            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                                        >
                                            <Send className="w-4 h-4" />
                                            Analyze
                                        </button>
                                    )}
                                </div>

                                {/* Analysis Type Selector */}
                                <div className="flex gap-2">
                                    {(['technical', 'fundamental', 'options', 'full'] as const).map((type) => (
                                        <button
                                            key={type}
                                            onClick={() => setAnalysisType(type)}
                                            className={clsx(
                                                'px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                                                analysisType === type
                                                    ? 'bg-purple-500/20 text-purple-400 ring-1 ring-purple-500/50'
                                                    : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                                            )}
                                        >
                                            {type.charAt(0).toUpperCase() + type.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Response Section */}
                            <div
                                ref={responseRef}
                                className="flex-1 overflow-y-auto p-4 space-y-4"
                            >
                                {error && (
                                    <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                                        <p className="text-red-400 text-sm">{error}</p>
                                        <button
                                            onClick={handleAnalyze}
                                            className="mt-2 text-xs text-red-400 hover:text-red-300 flex items-center gap-1"
                                        >
                                            <RefreshCw className="w-3 h-3" />
                                            Try again
                                        </button>
                                    </div>
                                )}

                                {response && (
                                    <div className="prose prose-invert prose-sm max-w-none">
                                        <div className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                                            {response}
                                        </div>
                                    </div>
                                )}

                                {isStreaming && (
                                    <div className="flex items-center gap-2 text-purple-400">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="text-sm">Analyzing...</span>
                                    </div>
                                )}

                                {!response && !error && !isStreaming && (
                                    <div className="text-center py-8">
                                        <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                                        <p className="text-slate-400 text-sm">
                                            Enter a symbol and click Analyze to get AI-powered insights
                                        </p>
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="p-3 border-t border-slate-700/50 bg-slate-900/50">
                                <p className="text-[10px] text-slate-500 text-center">
                                    AI-powered analysis using real-time market data. Not financial advice.
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.div>
    )
}

/**
 * Quick analysis button that opens the widget
 */
export function QuickAnalysisButton({ symbol, onClick }: { symbol: string; onClick?: () => void }) {
    return (
        <button
            onClick={onClick}
            className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors text-sm"
        >
            <Brain className="w-4 h-4" />
            AI Analysis
        </button>
    )
}

export default AIAnalysisWidget
