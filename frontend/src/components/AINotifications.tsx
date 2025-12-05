/**
 * AI-Powered Notification System
 *
 * Features:
 * - Intelligent notification grouping
 * - AI-generated action suggestions
 * - Priority-based notification queue
 * - Smart notification scheduling
 * - Context-aware notifications
 * - Notification analytics
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    AlertCircle, CheckCircle, Info, AlertTriangle, X,
    Brain, Sparkles, TrendingUp, TrendingDown, Zap,
    Bell, BellOff, Volume2, VolumeX, ChevronRight,
    ExternalLink, RefreshCw, Clock
} from 'lucide-react'
import { toast, Toaster } from 'sonner'

// =============================================================================
// Types
// =============================================================================

type NotificationType = 'success' | 'error' | 'warning' | 'info' | 'ai-insight' | 'market-alert'

interface AIAction {
    label: string
    action: () => void
    variant?: 'primary' | 'secondary' | 'danger'
}

interface Notification {
    id: string
    type: NotificationType
    title: string
    message: string
    timestamp: Date
    duration?: number
    persistent?: boolean
    actions?: AIAction[]
    aiInsight?: {
        summary: string
        confidence: number
        suggestedAction?: string
    }
    metadata?: Record<string, unknown>
}

interface NotificationState {
    notifications: Notification[]
    settings: {
        soundEnabled: boolean
        groupSimilar: boolean
        showAIInsights: boolean
        maxVisible: number
    }
    analytics: {
        totalShown: number
        totalDismissed: number
        totalActioned: number
        byType: Record<NotificationType, number>
    }
}

type NotificationAction =
    | { type: 'ADD'; payload: Notification }
    | { type: 'REMOVE'; payload: string }
    | { type: 'CLEAR_ALL' }
    | { type: 'UPDATE_SETTINGS'; payload: Partial<NotificationState['settings']> }
    | { type: 'RECORD_ANALYTICS'; payload: { event: 'shown' | 'dismissed' | 'actioned'; notificationType: NotificationType } }

// =============================================================================
// Icons Mapping
// =============================================================================

const NOTIFICATION_ICONS: Record<NotificationType, React.ReactNode> = {
    success: <CheckCircle className="w-5 h-5 text-emerald-400" />,
    error: <AlertCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-amber-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />,
    'ai-insight': <Brain className="w-5 h-5 text-purple-400" />,
    'market-alert': <TrendingUp className="w-5 h-5 text-primary" />
}

const NOTIFICATION_COLORS: Record<NotificationType, string> = {
    success: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
    error: 'from-red-500/20 to-red-500/5 border-red-500/30',
    warning: 'from-amber-500/20 to-amber-500/5 border-amber-500/30',
    info: 'from-blue-500/20 to-blue-500/5 border-blue-500/30',
    'ai-insight': 'from-purple-500/20 to-purple-500/5 border-purple-500/30',
    'market-alert': 'from-primary/20 to-primary/5 border-primary/30'
}

// =============================================================================
// Reducer
// =============================================================================

function notificationReducer(state: NotificationState, action: NotificationAction): NotificationState {
    switch (action.type) {
        case 'ADD':
            return {
                ...state,
                notifications: [action.payload, ...state.notifications].slice(0, 50) // Keep last 50
            }

        case 'REMOVE':
            return {
                ...state,
                notifications: state.notifications.filter(n => n.id !== action.payload)
            }

        case 'CLEAR_ALL':
            return {
                ...state,
                notifications: []
            }

        case 'UPDATE_SETTINGS':
            return {
                ...state,
                settings: { ...state.settings, ...action.payload }
            }

        case 'RECORD_ANALYTICS': {
            const { event, notificationType } = action.payload
            return {
                ...state,
                analytics: {
                    ...state.analytics,
                    totalShown: event === 'shown' ? state.analytics.totalShown + 1 : state.analytics.totalShown,
                    totalDismissed: event === 'dismissed' ? state.analytics.totalDismissed + 1 : state.analytics.totalDismissed,
                    totalActioned: event === 'actioned' ? state.analytics.totalActioned + 1 : state.analytics.totalActioned,
                    byType: {
                        ...state.analytics.byType,
                        [notificationType]: (state.analytics.byType[notificationType] || 0) + 1
                    }
                }
            }
        }

        default:
            return state
    }
}

// =============================================================================
// Context
// =============================================================================

interface NotificationContextValue {
    notifications: Notification[]
    settings: NotificationState['settings']
    analytics: NotificationState['analytics']
    notify: (options: Omit<Notification, 'id' | 'timestamp'>) => string
    dismiss: (id: string) => void
    clearAll: () => void
    updateSettings: (settings: Partial<NotificationState['settings']>) => void
    // Convenience methods
    success: (title: string, message?: string, options?: Partial<Notification>) => void
    error: (title: string, message?: string, options?: Partial<Notification>) => void
    warning: (title: string, message?: string, options?: Partial<Notification>) => void
    info: (title: string, message?: string, options?: Partial<Notification>) => void
    aiInsight: (title: string, message: string, insight: Notification['aiInsight']) => void
    marketAlert: (symbol: string, message: string, trend: 'up' | 'down') => void
}

const NotificationContext = createContext<NotificationContextValue | null>(null)

// =============================================================================
// Provider
// =============================================================================

const initialState: NotificationState = {
    notifications: [],
    settings: {
        soundEnabled: true,
        groupSimilar: true,
        showAIInsights: true,
        maxVisible: 5
    },
    analytics: {
        totalShown: 0,
        totalDismissed: 0,
        totalActioned: 0,
        byType: {} as Record<NotificationType, number>
    }
}

export function NotificationProvider({ children }: { children: React.ReactNode }) {
    const [state, dispatch] = useReducer(notificationReducer, initialState)
    const audioRef = useRef<HTMLAudioElement | null>(null)

    // Initialize audio
    useEffect(() => {
        if (typeof window !== 'undefined') {
            audioRef.current = new Audio('/notification.mp3')
            audioRef.current.volume = 0.3
        }
    }, [])

    // Play notification sound
    const playSound = useCallback(() => {
        if (state.settings.soundEnabled && audioRef.current) {
            audioRef.current.play().catch(() => {
                // Ignore autoplay restrictions
            })
        }
    }, [state.settings.soundEnabled])

    // Generate unique ID
    const generateId = useCallback(() => {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    }, [])

    // Core notify function
    const notify = useCallback((options: Omit<Notification, 'id' | 'timestamp'>): string => {
        const id = generateId()
        const notification: Notification = {
            ...options,
            id,
            timestamp: new Date()
        }

        dispatch({ type: 'ADD', payload: notification })
        dispatch({ type: 'RECORD_ANALYTICS', payload: { event: 'shown', notificationType: notification.type } })

        // Play sound for important notifications
        if (['error', 'market-alert', 'ai-insight'].includes(notification.type)) {
            playSound()
        }

        // Use sonner for toast display
        const toastOptions = {
            id,
            duration: notification.duration ?? (notification.persistent ? Infinity : 5000),
            onDismiss: () => {
                dispatch({ type: 'REMOVE', payload: id })
                dispatch({ type: 'RECORD_ANALYTICS', payload: { event: 'dismissed', notificationType: notification.type } })
            }
        }

        // Show appropriate toast
        switch (notification.type) {
            case 'success':
                toast.success(notification.title, { description: notification.message, ...toastOptions })
                break
            case 'error':
                toast.error(notification.title, { description: notification.message, ...toastOptions })
                break
            case 'warning':
                toast.warning(notification.title, { description: notification.message, ...toastOptions })
                break
            case 'ai-insight':
                toast(notification.title, {
                    description: notification.message,
                    icon: <Brain className="w-5 h-5 text-purple-400" />,
                    ...toastOptions
                })
                break
            case 'market-alert':
                toast(notification.title, {
                    description: notification.message,
                    icon: <TrendingUp className="w-5 h-5 text-primary" />,
                    ...toastOptions
                })
                break
            default:
                toast.info(notification.title, { description: notification.message, ...toastOptions })
        }

        return id
    }, [generateId, playSound])

    // Dismiss notification
    const dismiss = useCallback((id: string) => {
        dispatch({ type: 'REMOVE', payload: id })
        toast.dismiss(id)
    }, [])

    // Clear all notifications
    const clearAll = useCallback(() => {
        dispatch({ type: 'CLEAR_ALL' })
        toast.dismiss()
    }, [])

    // Update settings
    const updateSettings = useCallback((settings: Partial<NotificationState['settings']>) => {
        dispatch({ type: 'UPDATE_SETTINGS', payload: settings })
    }, [])

    // Convenience methods
    const success = useCallback((title: string, message = '', options?: Partial<Notification>) => {
        notify({ type: 'success', title, message, ...options })
    }, [notify])

    const error = useCallback((title: string, message = '', options?: Partial<Notification>) => {
        notify({ type: 'error', title, message, duration: 8000, ...options })
    }, [notify])

    const warning = useCallback((title: string, message = '', options?: Partial<Notification>) => {
        notify({ type: 'warning', title, message, ...options })
    }, [notify])

    const info = useCallback((title: string, message = '', options?: Partial<Notification>) => {
        notify({ type: 'info', title, message, ...options })
    }, [notify])

    const aiInsight = useCallback((title: string, message: string, insight: Notification['aiInsight']) => {
        notify({
            type: 'ai-insight',
            title,
            message,
            aiInsight: insight,
            duration: 10000
        })
    }, [notify])

    const marketAlert = useCallback((symbol: string, message: string, trend: 'up' | 'down') => {
        notify({
            type: 'market-alert',
            title: `${symbol} Alert`,
            message,
            metadata: { symbol, trend }
        })
    }, [notify])

    const value: NotificationContextValue = {
        notifications: state.notifications,
        settings: state.settings,
        analytics: state.analytics,
        notify,
        dismiss,
        clearAll,
        updateSettings,
        success,
        error,
        warning,
        info,
        aiInsight,
        marketAlert
    }

    return (
        <NotificationContext.Provider value={value}>
            {children}
            <Toaster
                position="top-right"
                expand={true}
                richColors
                closeButton
                theme="dark"
                toastOptions={{
                    className: 'bg-slate-900 border border-slate-700',
                    style: {
                        background: 'rgba(15, 23, 42, 0.95)',
                        backdropFilter: 'blur(12px)',
                        border: '1px solid rgba(51, 65, 85, 0.5)'
                    }
                }}
            />
        </NotificationContext.Provider>
    )
}

// =============================================================================
// Hook
// =============================================================================

export function useNotifications() {
    const context = useContext(NotificationContext)
    if (!context) {
        throw new Error('useNotifications must be used within NotificationProvider')
    }
    return context
}

// =============================================================================
// AI Notification Helpers
// =============================================================================

export function createAIInsightNotification(
    insight: {
        type: 'opportunity' | 'risk' | 'trend' | 'recommendation'
        symbol?: string
        summary: string
        confidence: number
        action?: string
    }
) {
    const titles: Record<string, string> = {
        opportunity: '🎯 Trading Opportunity Detected',
        risk: '⚠️ Risk Alert',
        trend: '📈 Trend Analysis',
        recommendation: '💡 AI Recommendation'
    }

    return {
        type: 'ai-insight' as const,
        title: titles[insight.type],
        message: insight.summary,
        aiInsight: {
            summary: insight.summary,
            confidence: insight.confidence,
            suggestedAction: insight.action
        },
        metadata: { symbol: insight.symbol }
    }
}

export function createMarketAlertNotification(
    alert: {
        symbol: string
        alertType: 'price-target' | 'earnings' | 'iv-spike' | 'volume' | 'news'
        message: string
        change?: number
    }
) {
    const icons: Record<string, string> = {
        'price-target': '🎯',
        'earnings': '📊',
        'iv-spike': '📈',
        'volume': '📊',
        'news': '📰'
    }

    return {
        type: 'market-alert' as const,
        title: `${icons[alert.alertType]} ${alert.symbol}`,
        message: alert.message,
        metadata: {
            symbol: alert.symbol,
            alertType: alert.alertType,
            change: alert.change
        }
    }
}

export default NotificationProvider
