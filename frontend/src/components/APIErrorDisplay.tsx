/**
 * APIErrorDisplay - Reusable API Error Component
 * ===============================================
 *
 * Provides consistent error display across all pages with:
 * - Timeout error handling
 * - Network error detection
 * - Circuit breaker status display
 * - Retry functionality
 *
 * @author AVA Trading Platform
 * @updated 2025-12-05
 */

import { RefreshCw, AlertCircle, WifiOff, ServerCrash, Clock } from 'lucide-react';
import { isTimeoutError, formatErrorMessage, type EnhancedError } from '../lib/axios';

interface APIErrorDisplayProps {
    error: Error | null;
    onRetry?: () => void;
    isRetrying?: boolean;
    title?: string;
    className?: string;
    compact?: boolean;
}

/**
 * Reusable error display component for API failures
 *
 * @example
 * ```tsx
 * <APIErrorDisplay
 *   error={error}
 *   onRetry={() => refetch()}
 *   isRetrying={isFetching}
 *   title="Failed to Load Data"
 * />
 * ```
 */
export function APIErrorDisplay({
    error,
    onRetry,
    isRetrying = false,
    title = 'Request Failed',
    className = '',
    compact = false,
}: APIErrorDisplayProps) {
    if (!error) return null;

    const enhancedError = error as EnhancedError;
    const isTimeout = isTimeoutError(error);
    const isNetworkError = enhancedError.isNetworkError === true;
    const isCircuitOpen = enhancedError.isCircuitOpen === true;

    // Determine error type and styling
    const getErrorConfig = () => {
        if (isCircuitOpen) {
            return {
                Icon: ServerCrash,
                bgColor: 'border-amber-500/30 bg-amber-500/5',
                textColor: 'text-amber-400',
                buttonColor: 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-400',
                title: 'Service Temporarily Unavailable',
                message: 'Too many failed requests. The system will automatically recover.',
                suggestion: 'Please wait 30 seconds before retrying.',
            };
        }

        if (isTimeout) {
            return {
                Icon: Clock,
                bgColor: 'border-orange-500/30 bg-orange-500/5',
                textColor: 'text-orange-400',
                buttonColor: 'bg-orange-500/20 hover:bg-orange-500/30 text-orange-400',
                title: 'Request Timed Out',
                message: 'The server is taking too long to respond.',
                suggestion: 'Try again - subsequent requests are often faster due to caching.',
            };
        }

        if (isNetworkError) {
            return {
                Icon: WifiOff,
                bgColor: 'border-red-500/30 bg-red-500/5',
                textColor: 'text-red-400',
                buttonColor: 'bg-red-500/20 hover:bg-red-500/30 text-red-400',
                title: 'Network Error',
                message: 'Unable to connect to the server.',
                suggestion: 'Check your internet connection and make sure the backend is running.',
            };
        }

        // Generic error
        return {
            Icon: AlertCircle,
            bgColor: 'border-red-500/30 bg-red-500/5',
            textColor: 'text-red-400',
            buttonColor: 'bg-red-500/20 hover:bg-red-500/30 text-red-400',
            title: title,
            message: formatErrorMessage(error),
            suggestion: null,
        };
    };

    const config = getErrorConfig();
    const { Icon, bgColor, textColor, buttonColor, message, suggestion } = config;

    if (compact) {
        return (
            <div className={`flex items-center gap-3 p-3 rounded-lg ${bgColor} ${textColor} ${className}`}>
                <Icon className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm flex-1">{message}</span>
                {onRetry && (
                    <button
                        onClick={onRetry}
                        disabled={isRetrying}
                        className={`p-1.5 rounded-md ${buttonColor} transition-colors`}
                    >
                        <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                    </button>
                )}
            </div>
        );
    }

    return (
        <div className={`glass-card p-6 ${bgColor} ${className}`}>
            <div className={`flex items-start gap-3 ${textColor}`}>
                <Icon className="w-6 h-6 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                    <h3 className="font-semibold">{config.title}</h3>
                    <p className={`text-sm mt-1 opacity-80`}>{message}</p>
                    {suggestion && (
                        <p className="text-sm text-slate-400 mt-2">{suggestion}</p>
                    )}
                    {onRetry && (
                        <button
                            onClick={onRetry}
                            disabled={isRetrying}
                            className={`mt-4 flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${buttonColor}`}
                        >
                            <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                            {isRetrying ? 'Retrying...' : 'Retry'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}

/**
 * Inline error message for smaller contexts
 */
export function InlineAPIError({
    error,
    onRetry,
    isRetrying = false,
}: {
    error: Error | null;
    onRetry?: () => void;
    isRetrying?: boolean;
}) {
    if (!error) return null;

    return (
        <APIErrorDisplay
            error={error}
            onRetry={onRetry}
            isRetrying={isRetrying}
            compact
        />
    );
}

export default APIErrorDisplay;
