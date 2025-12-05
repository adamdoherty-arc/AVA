import { Component } from 'react'
import type { ReactNode } from 'react'
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react'

interface Props {
    children: ReactNode
    fallback?: ReactNode
    onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

interface State {
    hasError: boolean
    error: Error | null
    errorInfo: React.ErrorInfo | null
}

/**
 * Modern Error Boundary with crash protection and recovery options
 * Catches JavaScript errors anywhere in the child component tree
 */
export class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props)
        this.state = { hasError: false, error: null, errorInfo: null }
    }

    static getDerivedStateFromError(error: Error): Partial<State> {
        return { hasError: true, error }
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        this.setState({ errorInfo })

        // Log error to console in development
        console.error('ErrorBoundary caught an error:', error, errorInfo)

        // Call optional error handler
        this.props.onError?.(error, errorInfo)

        // In production, you could send to error tracking service
        if (import.meta.env.PROD) {
            // Example: sendToErrorTracking(error, errorInfo)
        }
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null, errorInfo: null })
    }

    handleGoHome = () => {
        window.location.href = '/'
    }

    handleReportBug = () => {
        const { error, errorInfo } = this.state
        const errorDetails = encodeURIComponent(
            `Error: ${error?.message}\n\nStack: ${error?.stack}\n\nComponent Stack: ${errorInfo?.componentStack}`
        )
        window.open(`https://github.com/anthropics/claude-code/issues/new?body=${errorDetails}`, '_blank')
    }

    render() {
        if (this.state.hasError) {
            // Custom fallback if provided
            if (this.props.fallback) {
                return this.props.fallback
            }

            // Default error UI
            return (
                <div className="min-h-[400px] flex items-center justify-center p-8">
                    <div className="max-w-lg w-full">
                        <div className="glass-card p-8 text-center">
                            {/* Icon */}
                            <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center mx-auto mb-6">
                                <AlertTriangle className="w-8 h-8 text-red-400" />
                            </div>

                            {/* Title */}
                            <h2 className="text-xl font-bold text-white mb-2">
                                Something went wrong
                            </h2>

                            {/* Description */}
                            <p className="text-slate-400 mb-6">
                                An unexpected error occurred. Don't worry, your data is safe.
                            </p>

                            {/* Error Details (Development only) */}
                            {import.meta.env.DEV && this.state.error && (
                                <div className="mb-6 p-4 bg-slate-800/50 rounded-lg text-left overflow-auto max-h-40">
                                    <p className="text-red-400 text-sm font-mono break-words">
                                        {this.state.error.message}
                                    </p>
                                    {this.state.error.stack && (
                                        <pre className="text-xs text-slate-500 mt-2 whitespace-pre-wrap">
                                            {this.state.error.stack.slice(0, 500)}...
                                        </pre>
                                    )}
                                </div>
                            )}

                            {/* Actions */}
                            <div className="flex flex-col sm:flex-row gap-3 justify-center">
                                <button
                                    onClick={this.handleRetry}
                                    className="btn-primary flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Try Again
                                </button>
                                <button
                                    onClick={this.handleGoHome}
                                    className="btn-secondary flex items-center justify-center gap-2"
                                >
                                    <Home className="w-4 h-4" />
                                    Go Home
                                </button>
                                {import.meta.env.DEV && (
                                    <button
                                        onClick={this.handleReportBug}
                                        className="btn-secondary flex items-center justify-center gap-2"
                                    >
                                        <Bug className="w-4 h-4" />
                                        Report Bug
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}

/**
 * Hook-friendly wrapper for functional components
 */
export function withErrorBoundary<P extends object>(
    Component: React.ComponentType<P>,
    fallback?: ReactNode
) {
    return function WithErrorBoundary(props: P) {
        return (
            <ErrorBoundary fallback={fallback}>
                <Component {...props} />
            </ErrorBoundary>
        )
    }
}

export default ErrorBoundary
