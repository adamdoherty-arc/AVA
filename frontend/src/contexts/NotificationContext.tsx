import { createContext, useContext, useState, useCallback } from 'react'
import type { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, AlertCircle, Info, AlertTriangle, X, Bell } from 'lucide-react'
import clsx from 'clsx'

// Types
type NotificationType = 'success' | 'error' | 'warning' | 'info'

interface Notification {
    id: string
    type: NotificationType
    title: string
    message?: string
    duration?: number
    action?: {
        label: string
        onClick: () => void
    }
}

interface NotificationContextValue {
    notifications: Notification[]
    addNotification: (notification: Omit<Notification, 'id'>) => string
    removeNotification: (id: string) => void
    clearAll: () => void
    // Convenience methods
    success: (title: string, message?: string) => string
    error: (title: string, message?: string) => string
    warning: (title: string, message?: string) => string
    info: (title: string, message?: string) => string
}

const NotificationContext = createContext<NotificationContextValue | undefined>(undefined)

// Generate unique ID
const generateId = () => `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

// Provider
export function NotificationProvider({ children }: { children: ReactNode }) {
    const [notifications, setNotifications] = useState<Notification[]>([])

    const addNotification = useCallback((notification: Omit<Notification, 'id'>): string => {
        const id = generateId()
        const newNotification: Notification = {
            ...notification,
            id,
            duration: notification.duration ?? 5000
        }

        setNotifications(prev => [...prev, newNotification])

        // Auto-remove after duration (unless duration is 0)
        if (newNotification.duration && newNotification.duration > 0) {
            setTimeout(() => {
                setNotifications(prev => prev.filter(n => n.id !== id))
            }, newNotification.duration)
        }

        return id
    }, [])

    const removeNotification = useCallback((id: string) => {
        setNotifications(prev => prev.filter(n => n.id !== id))
    }, [])

    const clearAll = useCallback(() => {
        setNotifications([])
    }, [])

    // Convenience methods
    const success = useCallback((title: string, message?: string) => {
        return addNotification({ type: 'success', title, message })
    }, [addNotification])

    const error = useCallback((title: string, message?: string) => {
        return addNotification({ type: 'error', title, message, duration: 8000 })
    }, [addNotification])

    const warning = useCallback((title: string, message?: string) => {
        return addNotification({ type: 'warning', title, message })
    }, [addNotification])

    const info = useCallback((title: string, message?: string) => {
        return addNotification({ type: 'info', title, message })
    }, [addNotification])

    const value: NotificationContextValue = {
        notifications,
        addNotification,
        removeNotification,
        clearAll,
        success,
        error,
        warning,
        info
    }

    return (
        <NotificationContext.Provider value={value}>
            {children}
            <NotificationContainer
                notifications={notifications}
                onRemove={removeNotification}
            />
        </NotificationContext.Provider>
    )
}

// Hook
export function useNotification() {
    const context = useContext(NotificationContext)
    if (!context) {
        throw new Error('useNotification must be used within a NotificationProvider')
    }
    return context
}

// Notification Container Component
function NotificationContainer({
    notifications,
    onRemove
}: {
    notifications: Notification[]
    onRemove: (id: string) => void
}) {
    return (
        <div className="fixed top-4 right-4 z-[100] flex flex-col gap-3 pointer-events-none max-w-sm w-full">
            <AnimatePresence mode="popLayout">
                {notifications.map(notification => (
                    <NotificationItem
                        key={notification.id}
                        notification={notification}
                        onRemove={() => onRemove(notification.id)}
                    />
                ))}
            </AnimatePresence>
        </div>
    )
}

// Individual Notification Component
function NotificationItem({
    notification,
    onRemove
}: {
    notification: Notification
    onRemove: () => void
}) {
    const icons = {
        success: <CheckCircle className="w-5 h-5 text-emerald-400" />,
        error: <AlertCircle className="w-5 h-5 text-red-400" />,
        warning: <AlertTriangle className="w-5 h-5 text-amber-400" />,
        info: <Info className="w-5 h-5 text-blue-400" />
    }

    const colors = {
        success: 'border-emerald-500/30 bg-emerald-500/10',
        error: 'border-red-500/30 bg-red-500/10',
        warning: 'border-amber-500/30 bg-amber-500/10',
        info: 'border-blue-500/30 bg-blue-500/10'
    }

    return (
        <motion.div
            initial={{ opacity: 0, x: 50, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 50, scale: 0.95 }}
            transition={{ type: 'spring', duration: 0.3 }}
            className={clsx(
                'pointer-events-auto glass-card p-4 shadow-xl border',
                colors[notification.type]
            )}
        >
            <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                    {icons[notification.type]}
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-white">
                        {notification.title}
                    </p>
                    {notification.message && (
                        <p className="mt-1 text-sm text-slate-400">
                            {notification.message}
                        </p>
                    )}
                    {notification.action && (
                        <button
                            onClick={notification.action.onClick}
                            className="mt-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
                        >
                            {notification.action.label}
                        </button>
                    )}
                </div>
                <button
                    onClick={onRemove}
                    className="flex-shrink-0 p-1 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50"
                >
                    <X className="w-4 h-4" />
                </button>
            </div>
        </motion.div>
    )
}

/**
 * Alert banner component for inline notifications
 */
export function Alert({
    type,
    title,
    message,
    onDismiss,
    className
}: {
    type: NotificationType
    title: string
    message?: string
    onDismiss?: () => void
    className?: string
}) {
    const icons = {
        success: <CheckCircle className="w-5 h-5" />,
        error: <AlertCircle className="w-5 h-5" />,
        warning: <AlertTriangle className="w-5 h-5" />,
        info: <Info className="w-5 h-5" />
    }

    const colors = {
        success: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400',
        error: 'border-red-500/30 bg-red-500/10 text-red-400',
        warning: 'border-amber-500/30 bg-amber-500/10 text-amber-400',
        info: 'border-blue-500/30 bg-blue-500/10 text-blue-400'
    }

    return (
        <div className={clsx('glass-card p-4 border', colors[type], className)}>
            <div className="flex items-start gap-3">
                <div className="flex-shrink-0">
                    {icons[type]}
                </div>
                <div className="flex-1">
                    <p className="font-semibold">{title}</p>
                    {message && (
                        <p className="mt-1 text-sm opacity-80">{message}</p>
                    )}
                </div>
                {onDismiss && (
                    <button
                        onClick={onDismiss}
                        className="flex-shrink-0 p-1 hover:bg-slate-800/50 rounded-lg transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                )}
            </div>
        </div>
    )
}

/**
 * Notification bell with badge for unread count
 */
export function NotificationBell({ count = 0, onClick }: { count?: number; onClick?: () => void }) {
    return (
        <button
            onClick={onClick}
            className="relative p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-800/50"
        >
            <Bell className="w-5 h-5" />
            {count > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 flex items-center justify-center text-[10px] font-bold bg-red-500 text-white rounded-full">
                    {count > 99 ? '99+' : count}
                </span>
            )}
        </button>
    )
}

export default NotificationContext
