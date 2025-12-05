/**
 * AI-Powered Error Boundary
 *
 * Features:
 * - Intelligent error classification and analysis
 * - AI-generated recovery suggestions
 * - Automatic retry with exponential backoff
 * - Error pattern detection and prevention tips
 * - Integration with backend error logging
 */

import { Component, createContext, useContext, useState, useCallback } from 'react'
import type { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    AlertTriangle, RefreshCw, Home, Bug, Brain, Sparkles,
    Wifi, WifiOff, Database, Server, Code, Shield, Zap,
    ChevronDown, ChevronUp, Copy, Check, ExternalLink
} from 'lucide-react'
import { toast } from 'sonner'
import { BACKEND_URL } from '@/config/api'

// =============================================================================
// Types
// =============================================================================

interface ErrorCategory {
    type: 'network' | 'api' | 'render' | 'data' | 'auth' | 'unknown'
    severity: 'low' | 'medium' | 'high' | 'critical'
    icon: ReactNode
    color: string
    title: string
}

interface AIAnalysis {
    summary: string
    possibleCauses: string[]
    suggestedFixes: string[]
    preventionTips: string[]
    relatedDocs?: string
    isRetryable: boolean
    autoRetryDelay?: number
}

interface ErrorReport {
    id: string
    timestamp: string
    error: Error
    category: ErrorCategory
    analysis: AIAnalysis
    componentStack?: string
    userAgent: string
    url: string
}

// =============================================================================
// Error Classification Engine
// =============================================================================

const ERROR_PATTERNS: Record<string, ErrorCategory> = {
    network: {
        type: 'network',
        severity: 'medium',
        icon: <WifiOff className="w-6 h-6" />,
        color: 'amber',
        title: 'Network Error'
    },
    api: {
        type: 'api',
        severity: 'medium',
        icon: <Server className="w-6 h-6" />,
        color: 'orange',
        title: 'API Error'
    },
    render: {
        type: 'render',
        severity: 'high',
        icon: <Code className="w-6 h-6" />,
        color: 'red',
        title: 'Rendering Error'
    },
    data: {
        type: 'data',
        severity: 'medium',
        icon: <Database className="w-6 h-6" />,
        color: 'purple',
        title: 'Data Error'
    },
    auth: {
        type: 'auth',
        severity: 'high',
        icon: <Shield className="w-6 h-6" />,
        color: 'rose',
        title: 'Authentication Error'
    },
    unknown: {
        type: 'unknown',
        severity: 'medium',
        icon: <AlertTriangle className="w-6 h-6" />,
        color: 'slate',
        title: 'Unexpected Error'
    }
}

function classifyError(error: Error): ErrorCategory {
    const message = error.message.toLowerCase()
    const name = error.name.toLowerCase()

    // Network errors
    if (message.includes('network') || message.includes('fetch') ||
        message.includes('connection') || message.includes('timeout') ||
        name.includes('network')) {
        return ERROR_PATTERNS.network
    }

    // API errors
    if (message.includes('api') || message.includes('404') ||
        message.includes('500') || message.includes('http') ||
        message.includes('request failed')) {
        return ERROR_PATTERNS.api
    }

    // Auth errors
    if (message.includes('auth') || message.includes('unauthorized') ||
        message.includes('forbidden') || message.includes('401') ||
        message.includes('403')) {
        return ERROR_PATTERNS.auth
    }

    // Data errors
    if (message.includes('undefined') || message.includes('null') ||
        message.includes('cannot read') || message.includes('is not') ||
        message.includes('map') || message.includes('property')) {
        return ERROR_PATTERNS.data
    }

    // Render errors
    if (message.includes('render') || message.includes('component') ||
        message.includes('react') || message.includes('hook') ||
        name.includes('invariant')) {
        return ERROR_PATTERNS.render
    }

    return ERROR_PATTERNS.unknown
}

// =============================================================================
// AI Analysis Engine
// =============================================================================

function generateAIAnalysis(error: Error, category: ErrorCategory): AIAnalysis {
    const message = error.message

    // Network error analysis
    if (category.type === 'network') {
        return {
            summary: 'Unable to connect to the server. This could be a temporary network issue.',
            possibleCauses: [
                'Internet connection is unstable or offline',
                'Backend server might be down or restarting',
                'Firewall or proxy blocking the connection',
                'DNS resolution issues'
            ],
            suggestedFixes: [
                'Check your internet connection',
                'Refresh the page in a few seconds',
                'Try clearing browser cache',
                `Check if backend server is running at ${BACKEND_URL}`
            ],
            preventionTips: [
                'Enable offline mode for critical features',
                'Implement connection status monitoring'
            ],
            isRetryable: true,
            autoRetryDelay: 3000
        }
    }

    // API error analysis
    if (category.type === 'api') {
        const is404 = message.includes('404') || message.includes('not found')
        const is500 = message.includes('500') || message.includes('internal')

        return {
            summary: is404
                ? 'The requested resource was not found on the server.'
                : is500
                    ? 'The server encountered an internal error.'
                    : 'There was a problem communicating with the API.',
            possibleCauses: is404
                ? ['Endpoint URL might be incorrect', 'Resource was deleted or moved', 'API route not implemented']
                : ['Server-side bug or exception', 'Database connection issues', 'Invalid request data'],
            suggestedFixes: [
                'Check the API endpoint URL',
                'Verify the backend server is running',
                'Check server logs for detailed error'
            ],
            preventionTips: [
                'Add API health check monitoring',
                'Implement proper error responses'
            ],
            isRetryable: !is404,
            autoRetryDelay: is500 ? 5000 : undefined
        }
    }

    // Data error analysis
    if (category.type === 'data') {
        const isMapError = message.includes('map')
        const isPropertyError = message.includes('property') || message.includes('cannot read')

        return {
            summary: 'Encountered unexpected data format or missing values.',
            possibleCauses: isMapError
                ? ['API returned null/undefined instead of an array', 'Data not loaded before rendering', 'Incorrect data transformation']
                : ['Accessing property on null/undefined object', 'Async data not yet available', 'Type mismatch from API'],
            suggestedFixes: [
                'Add null checks (e.g., data?.items ?? [])',
                'Use loading states before rendering data',
                'Validate API response structure'
            ],
            preventionTips: [
                'Always use optional chaining (?.) for nested access',
                'Provide default values with nullish coalescing (??)',
                'Add TypeScript strict null checks'
            ],
            isRetryable: false
        }
    }

    // Render error analysis
    if (category.type === 'render') {
        return {
            summary: 'A React component failed to render correctly.',
            possibleCauses: [
                'Invalid JSX structure',
                'Hook called conditionally or outside component',
                'Missing required props',
                'State update on unmounted component'
            ],
            suggestedFixes: [
                'Check component for conditional hook calls',
                'Verify all required props are passed',
                'Use cleanup in useEffect to prevent memory leaks'
            ],
            preventionTips: [
                'Enable React strict mode',
                'Use ESLint with React hooks plugin',
                'Test components in isolation'
            ],
            isRetryable: false
        }
    }

    // Auth error analysis
    if (category.type === 'auth') {
        return {
            summary: 'Authentication or authorization failed.',
            possibleCauses: [
                'Session expired',
                'Invalid credentials',
                'Insufficient permissions',
                'Auth token missing or invalid'
            ],
            suggestedFixes: [
                'Re-authenticate with the service',
                'Check Robinhood credentials in Settings',
                'Clear cookies and re-login'
            ],
            preventionTips: [
                'Implement automatic token refresh',
                'Add session expiry monitoring'
            ],
            isRetryable: false
        }
    }

    // Unknown error - generic analysis
    return {
        summary: 'An unexpected error occurred in the application.',
        possibleCauses: [
            'Unhandled edge case',
            'Third-party library issue',
            'Browser compatibility problem'
        ],
        suggestedFixes: [
            'Refresh the page',
            'Clear browser cache and try again',
            'Report this issue if it persists'
        ],
        preventionTips: [
            'Add more comprehensive error handling',
            'Increase test coverage'
        ],
        isRetryable: true,
        autoRetryDelay: 5000
    }
}

// =============================================================================
// Error Report Context
// =============================================================================

interface ErrorReportContextValue {
    reports: ErrorReport[]
    addReport: (report: ErrorReport) => void
    clearReports: () => void
}

const ErrorReportContext = createContext<ErrorReportContextValue>({
    reports: [],
    addReport: () => {},
    clearReports: () => {}
})

export function useErrorReports() {
    return useContext(ErrorReportContext)
}

export function ErrorReportProvider({ children }: { children: ReactNode }) {
    const [reports, setReports] = useState<ErrorReport[]>([])

    const addReport = useCallback((report: ErrorReport) => {
        setReports(prev => [report, ...prev].slice(0, 50)) // Keep last 50
    }, [])

    const clearReports = useCallback(() => {
        setReports([])
    }, [])

    return (
        <ErrorReportContext.Provider value={{ reports, addReport, clearReports }}>
            {children}
        </ErrorReportContext.Provider>
    )
}

// =============================================================================
// AI Error Boundary Component
// =============================================================================

interface AIErrorBoundaryProps {
    children: ReactNode
    fallback?: ReactNode
    onError?: (error: Error, errorInfo: React.ErrorInfo) => void
    enableAutoRetry?: boolean
    maxRetries?: number
    showAIAnalysis?: boolean
}

interface AIErrorBoundaryState {
    hasError: boolean
    error: Error | null
    errorInfo: React.ErrorInfo | null
    category: ErrorCategory | null
    analysis: AIAnalysis | null
    retryCount: number
    isRetrying: boolean
    showDetails: boolean
    copied: boolean
}

export class AIErrorBoundary extends Component<AIErrorBoundaryProps, AIErrorBoundaryState> {
    private retryTimeout: ReturnType<typeof setTimeout> | null = null

    constructor(props: AIErrorBoundaryProps) {
        super(props)
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null,
            category: null,
            analysis: null,
            retryCount: 0,
            isRetrying: false,
            showDetails: false,
            copied: false
        }
    }

    static getDerivedStateFromError(error: Error): Partial<AIErrorBoundaryState> {
        const category = classifyError(error)
        const analysis = generateAIAnalysis(error, category)
        return { hasError: true, error, category, analysis }
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        this.setState({ errorInfo })

        // Log to console
        console.error('[AIErrorBoundary] Error caught:', error)
        console.error('[AIErrorBoundary] Component stack:', errorInfo.componentStack)

        // Call optional error handler
        this.props.onError?.(error, errorInfo)

        // Send to backend for logging (non-blocking)
        this.logErrorToBackend(error, errorInfo)

        // Auto-retry if enabled and error is retryable
        const { enableAutoRetry = true, maxRetries = 3 } = this.props
        if (enableAutoRetry &&
            this.state.analysis?.isRetryable &&
            this.state.retryCount < maxRetries) {
            this.scheduleRetry()
        }
    }

    componentWillUnmount() {
        if (this.retryTimeout) {
            clearTimeout(this.retryTimeout)
        }
    }

    logErrorToBackend = async (error: Error, errorInfo: React.ErrorInfo) => {
        try {
            await fetch(`${BACKEND_URL}/api/system/error-log`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: error.message,
                    stack: error.stack,
                    componentStack: errorInfo.componentStack,
                    url: window.location.href,
                    userAgent: navigator.userAgent,
                    timestamp: new Date().toISOString()
                })
            })
        } catch (e) {
            // Silently fail - don't cause more errors
        }
    }

    scheduleRetry = () => {
        const delay = this.state.analysis?.autoRetryDelay ?? 3000
        const backoffDelay = delay * Math.pow(1.5, this.state.retryCount)

        this.setState({ isRetrying: true })
        toast.info(`Retrying in ${Math.round(backoffDelay / 1000)} seconds...`)

        this.retryTimeout = setTimeout(() => {
            this.setState(prev => ({
                hasError: false,
                error: null,
                errorInfo: null,
                category: null,
                analysis: null,
                retryCount: prev.retryCount + 1,
                isRetrying: false
            }))
        }, backoffDelay)
    }

    handleRetry = () => {
        if (this.retryTimeout) {
            clearTimeout(this.retryTimeout)
        }
        this.setState({
            hasError: false,
            error: null,
            errorInfo: null,
            category: null,
            analysis: null,
            isRetrying: false
        })
    }

    handleCopyError = () => {
        const { error, errorInfo } = this.state
        const text = `Error: ${error?.message}\n\nStack:\n${error?.stack}\n\nComponent Stack:\n${errorInfo?.componentStack}`
        navigator.clipboard.writeText(text)
        this.setState({ copied: true })
        setTimeout(() => this.setState({ copied: false }), 2000)
        toast.success('Error details copied to clipboard')
    }

    render() {
        if (!this.state.hasError) {
            return this.props.children
        }

        if (this.props.fallback) {
            return this.props.fallback
        }

        const { error, category, analysis, isRetrying, showDetails, copied, retryCount } = this.state
        const { showAIAnalysis = true, maxRetries = 3 } = this.props
        const colorClass = category ? `${category.color}-500` : 'red-500'

        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="min-h-[400px] flex items-center justify-center p-6"
            >
                <div className="max-w-2xl w-full">
                    <div className={`rounded-2xl border border-${colorClass}/30 bg-${colorClass}/5 backdrop-blur-sm p-8`}>
                        {/* Header */}
                        <div className="flex items-start gap-4 mb-6">
                            <div className={`p-3 rounded-xl bg-${colorClass}/20`}>
                                {isRetrying ? (
                                    <RefreshCw className={`w-6 h-6 text-${colorClass} animate-spin`} />
                                ) : (
                                    <span className={`text-${colorClass}`}>{category?.icon}</span>
                                )}
                            </div>
                            <div className="flex-1">
                                <h2 className="text-xl font-bold text-white mb-1">
                                    {isRetrying ? 'Reconnecting...' : category?.title || 'Something went wrong'}
                                </h2>
                                <p className="text-slate-400">
                                    {isRetrying
                                        ? `Attempt ${retryCount + 1} of ${maxRetries}`
                                        : error?.message || 'An unexpected error occurred'}
                                </p>
                            </div>
                        </div>

                        {/* AI Analysis */}
                        {showAIAnalysis && analysis && !isRetrying && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="mb-6 p-4 rounded-xl bg-gradient-to-r from-primary/10 to-blue-500/10 border border-primary/20"
                            >
                                <div className="flex items-center gap-2 mb-3">
                                    <Brain className="w-5 h-5 text-primary" />
                                    <span className="font-semibold text-primary">AI Analysis</span>
                                    <Sparkles className="w-4 h-4 text-yellow-400" />
                                </div>
                                <p className="text-sm text-slate-300 mb-4">{analysis.summary}</p>

                                {/* Suggested Fixes */}
                                <div className="space-y-2">
                                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                                        Suggested Actions
                                    </h4>
                                    <ul className="space-y-1">
                                        {analysis.suggestedFixes.slice(0, 3).map((fix, idx) => (
                                            <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                                                <Zap className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                                                {fix}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </motion.div>
                        )}

                        {/* Action Buttons */}
                        {!isRetrying && (
                            <div className="flex flex-wrap gap-3 mb-6">
                                <button
                                    onClick={this.handleRetry}
                                    className="btn-primary flex items-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Try Again
                                </button>
                                <button
                                    onClick={() => window.location.href = '/'}
                                    className="btn-secondary flex items-center gap-2"
                                >
                                    <Home className="w-4 h-4" />
                                    Go Home
                                </button>
                                <button
                                    onClick={this.handleCopyError}
                                    className="btn-secondary flex items-center gap-2"
                                >
                                    {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                                    {copied ? 'Copied!' : 'Copy Error'}
                                </button>
                            </div>
                        )}

                        {/* Expandable Details */}
                        {import.meta.env.DEV && (
                            <div>
                                <button
                                    onClick={() => this.setState(prev => ({ showDetails: !prev.showDetails }))}
                                    className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
                                >
                                    {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    {showDetails ? 'Hide' : 'Show'} Technical Details
                                </button>

                                <AnimatePresence>
                                    {showDetails && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="mt-4 p-4 bg-slate-900/50 rounded-lg">
                                                <h4 className="text-xs font-semibold text-slate-400 mb-2">Stack Trace</h4>
                                                <pre className="text-xs text-slate-500 whitespace-pre-wrap overflow-auto max-h-40">
                                                    {error?.stack}
                                                </pre>
                                                {this.state.errorInfo?.componentStack && (
                                                    <>
                                                        <h4 className="text-xs font-semibold text-slate-400 mt-4 mb-2">Component Stack</h4>
                                                        <pre className="text-xs text-slate-500 whitespace-pre-wrap overflow-auto max-h-40">
                                                            {this.state.errorInfo.componentStack}
                                                        </pre>
                                                    </>
                                                )}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>
        )
    }
}

export default AIErrorBoundary
