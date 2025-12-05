import * as React from "react"
import { cn } from "@/lib/utils"
import { Button } from "./button"

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
  onReset?: () => void
  className?: string
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
  errorInfo: React.ErrorInfo | null
}

/**
 * Error Boundary Component
 *
 * Catches JavaScript errors in child component tree and displays a fallback UI.
 * Especially useful for:
 * - Streaming/SSE connection errors
 * - React Query failures
 * - Lazy loading failures
 * - Runtime rendering errors
 */
export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ errorInfo })

    // Log to console for debugging
    console.error("[ErrorBoundary] Caught error:", error)
    console.error("[ErrorBoundary] Component stack:", errorInfo.componentStack)

    // Call optional error handler
    this.props.onError?.(error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default fallback UI
      return (
        <div
          className={cn(
            "flex flex-col items-center justify-center min-h-[200px] p-6 rounded-lg border border-destructive/50 bg-destructive/10",
            this.props.className
          )}
        >
          <div className="flex items-center gap-2 mb-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 text-destructive"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <h3 className="text-lg font-semibold text-destructive">
              Something went wrong
            </h3>
          </div>

          <p className="text-sm text-muted-foreground mb-4 text-center max-w-md">
            {this.state.error?.message || "An unexpected error occurred"}
          </p>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={this.handleReset}
            >
              Try Again
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.location.reload()}
            >
              Reload Page
            </Button>
          </div>

          {import.meta.env.DEV && this.state.errorInfo && (
            <details className="mt-4 p-4 bg-muted rounded-md w-full max-w-2xl overflow-auto">
              <summary className="text-sm font-medium cursor-pointer">
                Error Details (Development Only)
              </summary>
              <pre className="mt-2 text-xs whitespace-pre-wrap">
                {this.state.error?.stack}
              </pre>
              <pre className="mt-2 text-xs whitespace-pre-wrap text-muted-foreground">
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Streaming Error Boundary - Specialized for SSE/WebSocket errors
 *
 * Automatically retries on connection errors and provides
 * real-time status feedback.
 */
interface StreamingErrorBoundaryProps extends ErrorBoundaryProps {
  maxRetries?: number
  retryDelay?: number
}

interface StreamingErrorBoundaryState extends ErrorBoundaryState {
  retryCount: number
  isRetrying: boolean
}

export class StreamingErrorBoundary extends React.Component<
  StreamingErrorBoundaryProps,
  StreamingErrorBoundaryState
> {
  private retryTimeout: ReturnType<typeof setTimeout> | null = null

  constructor(props: StreamingErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRetrying: false,
    }
  }

  static getDerivedStateFromError(error: Error): Partial<StreamingErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ errorInfo })

    console.error("[StreamingErrorBoundary] Caught error:", error)
    this.props.onError?.(error, errorInfo)

    // Auto-retry for connection errors
    const maxRetries = this.props.maxRetries ?? 3
    if (this.state.retryCount < maxRetries && this.isRetryableError(error)) {
      this.scheduleRetry()
    }
  }

  componentWillUnmount() {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout)
    }
  }

  isRetryableError(error: Error): boolean {
    const retryablePatterns = [
      /network/i,
      /connection/i,
      /timeout/i,
      /abort/i,
      /stream/i,
      /sse/i,
      /websocket/i,
      /fetch/i,
    ]
    return retryablePatterns.some((pattern) =>
      pattern.test(error.message) || pattern.test(error.name)
    )
  }

  scheduleRetry = () => {
    const delay = this.props.retryDelay ?? 2000
    this.setState({ isRetrying: true })

    this.retryTimeout = setTimeout(() => {
      this.setState((prev) => ({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: prev.retryCount + 1,
        isRetrying: false,
      }))
    }, delay * (this.state.retryCount + 1)) // Exponential backoff
  }

  handleReset = () => {
    if (this.retryTimeout) {
      clearTimeout(this.retryTimeout)
    }
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      isRetrying: false,
    })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      const maxRetries = this.props.maxRetries ?? 3
      const canRetry = this.state.retryCount < maxRetries

      return (
        <div
          className={cn(
            "flex flex-col items-center justify-center min-h-[200px] p-6 rounded-lg border",
            this.state.isRetrying
              ? "border-yellow-500/50 bg-yellow-500/10"
              : "border-destructive/50 bg-destructive/10",
            this.props.className
          )}
        >
          <div className="flex items-center gap-2 mb-4">
            {this.state.isRetrying ? (
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-yellow-500 border-t-transparent" />
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6 text-destructive"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M18.364 5.636a9 9 0 010 12.728m-3.536-3.536a4 4 0 010-5.656m-8.486 8.486a9 9 0 010-12.728m3.536 3.536a4 4 0 010 5.656"
                />
              </svg>
            )}
            <h3 className="text-lg font-semibold">
              {this.state.isRetrying ? "Reconnecting..." : "Connection Error"}
            </h3>
          </div>

          <p className="text-sm text-muted-foreground mb-2 text-center">
            {this.state.isRetrying
              ? `Attempting to reconnect (${this.state.retryCount + 1}/${maxRetries})...`
              : this.state.error?.message || "Lost connection to streaming data"}
          </p>

          {!this.state.isRetrying && (
            <>
              {canRetry && (
                <p className="text-xs text-muted-foreground mb-4">
                  Retried {this.state.retryCount}/{maxRetries} times
                </p>
              )}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={this.handleReset}
                >
                  Retry Now
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => window.location.reload()}
                >
                  Reload Page
                </Button>
              </div>
            </>
          )}
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * withErrorBoundary HOC - Wrap any component with error boundary
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, "children">
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  )

  WrappedComponent.displayName = `withErrorBoundary(${
    Component.displayName || Component.name || "Component"
  })`

  return WrappedComponent
}

/**
 * useErrorHandler hook - For throwing errors to nearest ErrorBoundary
 */
export function useErrorHandler(): (error: Error) => void {
  const [, setError] = React.useState<Error | null>(null)

  return React.useCallback((error: Error) => {
    setError(() => {
      throw error
    })
  }, [])
}
