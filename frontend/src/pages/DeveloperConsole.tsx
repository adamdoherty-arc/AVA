import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import { motion, AnimatePresence } from 'framer-motion'
import {
    LineChart, Line, ResponsiveContainer, AreaChart, Area
} from 'recharts'
import {
    Terminal, RefreshCw, Search, Play, Clock, Activity,
    CheckCircle, XCircle, AlertCircle, ChevronDown,
    Zap, Database, Bell, Settings, TrendingUp, BarChart3,
    Calendar, Power, PowerOff, Filter, Timer, Cpu, Sparkles,
    Brain, Send, AlertTriangle, Shield, ArrowUpRight,
    ArrowDownRight, Minus, X, Loader2, Bot, ChevronUp
} from 'lucide-react'
import clsx from 'clsx'

// ============================================================================
// Types
// ============================================================================

interface Automation {
    id: number
    name: string
    display_name: string
    automation_type: string
    celery_task_name: string | null
    schedule_type: string | null
    schedule_config: Record<string, string> | null
    schedule_display: string | null
    queue: string | null
    category: string
    description: string | null
    is_enabled: boolean
    timeout_seconds: number | null
    last_run_status: string | null
    last_run_at: string | null
    last_duration_seconds: number | null
    last_error: string | null
    success_rate_24h: number | null
    total_runs_24h: number | null
    failed_runs_24h: number | null
    avg_duration_24h: number | null
}

interface DashboardStats {
    automations: { total: number; enabled: number; disabled: number }
    executions: {
        total_executions: number
        successful: number
        failed: number
        skipped: number
        running: number
        success_rate: number | null
        avg_duration: number | null
    }
    recent_failures: Array<{
        name: string
        display_name: string
        started_at: string
        error_message: string | null
    }>
    categories: Array<{ category: string; count: number; enabled_count: number }>
}

interface HealthPrediction {
    status: string
    metrics: {
        total_executions: number
        successful: number
        failed: number
        success_rate: number
        trend: 'improving' | 'stable' | 'declining'
        risk_score: number
    }
    prediction: {
        health_status: 'healthy' | 'warning' | 'critical'
        next_24h_failure_risk: 'low' | 'medium' | 'high'
    }
    ai_insights?: {
        summary: string
        recommendations: string[]
        priority_action: string
    }
}

interface Toast {
    id: string
    type: 'success' | 'error' | 'info' | 'warning'
    message: string
}

// ============================================================================
// Constants
// ============================================================================

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
    'market_data': <TrendingUp className="w-4 h-4" />,
    'predictions': <Zap className="w-4 h-4" />,
    'notifications': <Bell className="w-4 h-4" />,
    'maintenance': <Settings className="w-4 h-4" />,
    'rag': <Database className="w-4 h-4" />,
}

const CATEGORY_LABELS: Record<string, string> = {
    'market_data': 'Market Data',
    'predictions': 'Predictions',
    'notifications': 'Notifications',
    'maintenance': 'Maintenance',
    'rag': 'RAG Knowledge Base',
}

const CATEGORY_COLORS: Record<string, string> = {
    'market_data': 'from-blue-500 to-cyan-500',
    'predictions': 'from-purple-500 to-pink-500',
    'notifications': 'from-amber-500 to-orange-500',
    'maintenance': 'from-slate-500 to-slate-600',
    'rag': 'from-emerald-500 to-teal-500',
}

// ============================================================================
// Helper Functions
// ============================================================================

const formatDuration = (seconds: number | null) => {
    if (!seconds) return '-'
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

const formatRelativeTime = (dateStr: string | null) => {
    if (!dateStr) return 'Never'
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    return `${diffDays}d ago`
}

// Generate mock sparkline data for demo
const generateSparklineData = (successRate: number | null, runs: number | null) => {
    const points = 12
    const base = successRate || 85
    return Array.from({ length: points }, (_, i) => ({
        value: Math.max(0, Math.min(100, base + (Math.random() - 0.5) * 20)),
        runs: Math.floor((runs || 10) / points)
    }))
}

// ============================================================================
// Toast Notification Component
// ============================================================================

function ToastContainer({ toasts, removeToast }: { toasts: Toast[], removeToast: (id: string) => void }) {
    return (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
            <AnimatePresence>
                {toasts.map(toast => (
                    <motion.div
                        key={toast.id}
                        initial={{ opacity: 0, x: 100, scale: 0.8 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: 100, scale: 0.8 }}
                        className={clsx(
                            "flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl backdrop-blur-xl border",
                            toast.type === 'success' && "bg-emerald-500/20 border-emerald-500/30 text-emerald-300",
                            toast.type === 'error' && "bg-red-500/20 border-red-500/30 text-red-300",
                            toast.type === 'warning' && "bg-amber-500/20 border-amber-500/30 text-amber-300",
                            toast.type === 'info' && "bg-blue-500/20 border-blue-500/30 text-blue-300"
                        )}
                    >
                        {toast.type === 'success' && <CheckCircle className="w-5 h-5" />}
                        {toast.type === 'error' && <XCircle className="w-5 h-5" />}
                        {toast.type === 'warning' && <AlertTriangle className="w-5 h-5" />}
                        {toast.type === 'info' && <AlertCircle className="w-5 h-5" />}
                        <span className="text-sm font-medium">{toast.message}</span>
                        <button onClick={() => removeToast(toast.id)} className="ml-2 opacity-60 hover:opacity-100">
                            <X className="w-4 h-4" />
                        </button>
                    </motion.div>
                ))}
            </AnimatePresence>
        </div>
    )
}

// ============================================================================
// AI Chat Panel Component
// ============================================================================

function AIChatPanel({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    const [question, setQuestion] = useState('')
    const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([])
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const askMutation = useMutation({
        mutationFn: async (q: string) => {
            const { data } = await axiosInstance.post('/automations/ai/ask', { question: q })
            return data
        },
        onSuccess: (data) => {
            setMessages(prev => [...prev, { role: 'assistant', content: data.answer || 'No response' }])
        },
        onError: () => {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
        }
    })

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (!question.trim()) return
        setMessages(prev => [...prev, { role: 'user', content: question }])
        askMutation.mutate(question)
        setQuestion('')
    }

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 20, scale: 0.95 }}
                    className="fixed bottom-24 right-4 w-96 h-[500px] z-50 rounded-2xl overflow-hidden shadow-2xl border border-slate-700/50 bg-slate-900/95 backdrop-blur-xl flex flex-col"
                >
                    {/* Header */}
                    <div className="p-4 border-b border-slate-700/50 bg-gradient-to-r from-purple-500/20 to-pink-500/20">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                                    <Brain className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">AVA AI Assistant</h3>
                                    <p className="text-xs text-slate-400">Ask about automations</p>
                                </div>
                            </div>
                            <button onClick={onClose} className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
                                <X className="w-5 h-5 text-slate-400" />
                            </button>
                        </div>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {messages.length === 0 && (
                            <div className="text-center text-slate-500 mt-8">
                                <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                <p className="text-sm">Ask me anything about your automations!</p>
                                <div className="mt-4 space-y-2">
                                    {["What's the system health?", "Which tasks are failing?", "Show me slow automations"].map(q => (
                                        <button
                                            key={q}
                                            onClick={() => setQuestion(q)}
                                            className="block w-full text-left px-3 py-2 text-xs bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors text-slate-400"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={clsx(
                                    "max-w-[85%] p-3 rounded-xl text-sm",
                                    msg.role === 'user'
                                        ? "ml-auto bg-primary text-white"
                                        : "bg-slate-800 text-slate-200"
                                )}
                            >
                                {msg.content}
                            </motion.div>
                        ))}
                        {askMutation.isPending && (
                            <div className="flex items-center gap-2 text-slate-400">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span className="text-sm">Thinking...</span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700/50">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                placeholder="Ask about automations..."
                                className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-primary/50"
                            />
                            <button
                                type="submit"
                                disabled={!question.trim() || askMutation.isPending}
                                className="p-2 bg-primary rounded-xl text-white disabled:opacity-50 hover:bg-primary/80 transition-colors"
                            >
                                <Send className="w-5 h-5" />
                            </button>
                        </div>
                    </form>
                </motion.div>
            )}
        </AnimatePresence>
    )
}

// ============================================================================
// Health Indicator Component
// ============================================================================

function HealthIndicator({ prediction }: { prediction: HealthPrediction | null }) {
    if (!prediction) return null

    const { health_status, next_24h_failure_risk } = prediction.prediction || {}
    const { trend, risk_score } = prediction.metrics || {}

    const statusConfig = {
        healthy: { color: 'emerald', icon: Shield, label: 'Healthy' },
        warning: { color: 'amber', icon: AlertTriangle, label: 'Warning' },
        critical: { color: 'red', icon: XCircle, label: 'Critical' }
    }

    const config = statusConfig[health_status as keyof typeof statusConfig] || statusConfig.healthy
    const Icon = config.icon

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={clsx(
                "p-4 rounded-xl border backdrop-blur-sm",
                health_status === 'healthy' && "bg-emerald-500/10 border-emerald-500/30",
                health_status === 'warning' && "bg-amber-500/10 border-amber-500/30",
                health_status === 'critical' && "bg-red-500/10 border-red-500/30"
            )}
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                    <div className={clsx(
                        "w-10 h-10 rounded-xl flex items-center justify-center",
                        health_status === 'healthy' && "bg-emerald-500/20 text-emerald-400",
                        health_status === 'warning' && "bg-amber-500/20 text-amber-400",
                        health_status === 'critical' && "bg-red-500/20 text-red-400"
                    )}>
                        <Icon className="w-5 h-5" />
                    </div>
                    <div>
                        <h4 className="font-semibold text-white">{config.label}</h4>
                        <p className="text-xs text-slate-400">System Health Status</p>
                    </div>
                </div>
                <div className="flex items-center gap-1 text-sm">
                    {trend === 'improving' && <ArrowUpRight className="w-4 h-4 text-emerald-400" />}
                    {trend === 'declining' && <ArrowDownRight className="w-4 h-4 text-red-400" />}
                    {trend === 'stable' && <Minus className="w-4 h-4 text-slate-400" />}
                    <span className={clsx(
                        trend === 'improving' && "text-emerald-400",
                        trend === 'declining' && "text-red-400",
                        trend === 'stable' && "text-slate-400"
                    )}>
                        {trend}
                    </span>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <p className="text-xs text-slate-500 mb-1">Risk Score</p>
                    <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${risk_score || 0}%` }}
                            transition={{ duration: 1 }}
                            className={clsx(
                                "h-full rounded-full",
                                (risk_score || 0) < 30 && "bg-emerald-500",
                                (risk_score || 0) >= 30 && (risk_score || 0) < 60 && "bg-amber-500",
                                (risk_score || 0) >= 60 && "bg-red-500"
                            )}
                        />
                    </div>
                    <p className="text-xs text-right mt-1 text-slate-400">{risk_score || 0}/100</p>
                </div>
                <div>
                    <p className="text-xs text-slate-500 mb-1">24h Failure Risk</p>
                    <p className={clsx(
                        "text-lg font-bold capitalize",
                        next_24h_failure_risk === 'low' && "text-emerald-400",
                        next_24h_failure_risk === 'medium' && "text-amber-400",
                        next_24h_failure_risk === 'high' && "text-red-400"
                    )}>
                        {next_24h_failure_risk || 'Unknown'}
                    </p>
                </div>
            </div>

            {prediction.ai_insights && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 pt-4 border-t border-slate-700/50"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <Sparkles className="w-4 h-4 text-purple-400" />
                        <span className="text-xs font-medium text-purple-400">AI Insight</span>
                    </div>
                    <p className="text-sm text-slate-300">{prediction.ai_insights.summary}</p>
                    {prediction.ai_insights.priority_action && (
                        <p className="mt-2 text-xs text-amber-400/80 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            {prediction.ai_insights.priority_action}
                        </p>
                    )}
                </motion.div>
            )}
        </motion.div>
    )
}

// ============================================================================
// Automation Card Component
// ============================================================================

function AutomationCard({
    automation,
    isExpanded,
    onToggleExpand,
    onToggleEnabled,
    onTrigger,
    isToggling,
    isTriggering
}: {
    automation: Automation
    isExpanded: boolean
    onToggleExpand: () => void
    onToggleEnabled: (enabled: boolean) => void
    onTrigger: () => void
    isToggling: boolean
    isTriggering: boolean
}) {
    const sparklineData = generateSparklineData(automation.success_rate_24h, automation.total_runs_24h)

    const getStatusColor = (status: string | null) => {
        switch (status) {
            case 'success': return 'text-emerald-400 bg-emerald-400/20'
            case 'running': return 'text-blue-400 bg-blue-400/20'
            case 'failed': return 'text-red-400 bg-red-400/20'
            case 'skipped': return 'text-amber-400 bg-amber-400/20'
            default: return 'text-slate-400 bg-slate-400/20'
        }
    }

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={clsx(
                "rounded-xl border transition-all overflow-hidden",
                automation.is_enabled
                    ? "bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50"
                    : "bg-slate-900/30 border-slate-800/30 opacity-60"
            )}
        >
            {/* Main Row */}
            <div className="flex items-center justify-between p-4">
                <div className="flex items-center gap-4 flex-1 min-w-0">
                    {/* Status Indicator */}
                    <motion.span
                        animate={{
                            scale: automation.last_run_status === 'running' ? [1, 1.2, 1] : 1,
                            opacity: automation.last_run_status === 'running' ? [1, 0.5, 1] : 1
                        }}
                        transition={{ repeat: automation.last_run_status === 'running' ? Infinity : 0, duration: 1.5 }}
                        className={clsx(
                            "w-2.5 h-2.5 rounded-full flex-shrink-0",
                            automation.is_enabled
                                ? automation.last_run_status === 'running'
                                    ? "bg-blue-400"
                                    : automation.last_run_status === 'failed'
                                        ? "bg-red-400"
                                        : "bg-emerald-400"
                                : "bg-slate-500"
                        )}
                    />

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                            <h4 className="font-medium text-white truncate">
                                {automation.display_name}
                            </h4>
                            {automation.last_run_status && (
                                <span className={clsx(
                                    "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs",
                                    getStatusColor(automation.last_run_status)
                                )}>
                                    {automation.last_run_status === 'success' && <CheckCircle className="w-3 h-3" />}
                                    {automation.last_run_status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
                                    {automation.last_run_status === 'failed' && <XCircle className="w-3 h-3" />}
                                    {automation.last_run_status}
                                </span>
                            )}
                        </div>
                        <div className="flex items-center gap-4 mt-1 text-xs text-slate-400">
                            <span className="flex items-center gap-1">
                                <Calendar className="w-3 h-3" />
                                {automation.schedule_display || 'Manual'}
                            </span>
                            <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {formatRelativeTime(automation.last_run_at)}
                            </span>
                            {automation.success_rate_24h !== null && (
                                <span className={clsx(
                                    "flex items-center gap-1",
                                    automation.success_rate_24h >= 90 ? "text-emerald-400" :
                                    automation.success_rate_24h >= 70 ? "text-amber-400" : "text-red-400"
                                )}>
                                    <BarChart3 className="w-3 h-3" />
                                    {automation.success_rate_24h}%
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Sparkline Chart */}
                    <div className="hidden md:block w-24 h-10">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={sparklineData}>
                                <defs>
                                    <linearGradient id={`sparkline-${automation.id}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={(automation.success_rate_24h || 0) >= 90 ? "#10b981" : "#f59e0b"} stopOpacity={0.3} />
                                        <stop offset="100%" stopColor={(automation.success_rate_24h || 0) >= 90 ? "#10b981" : "#f59e0b"} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke={(automation.success_rate_24h || 0) >= 90 ? "#10b981" : "#f59e0b"}
                                    fill={`url(#sparkline-${automation.id})`}
                                    strokeWidth={1.5}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-3 flex-shrink-0">
                    {/* Run Now Button */}
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={onTrigger}
                        disabled={!automation.is_enabled || isTriggering}
                        className={clsx(
                            "p-2 rounded-lg transition-colors",
                            automation.is_enabled
                                ? "bg-primary/20 text-primary hover:bg-primary/30"
                                : "bg-slate-700/30 text-slate-500 cursor-not-allowed"
                        )}
                        title="Run Now"
                    >
                        {isTriggering ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    </motion.button>

                    {/* Toggle Switch */}
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={automation.is_enabled}
                            onChange={(e) => onToggleEnabled(e.target.checked)}
                            disabled={isToggling}
                        />
                        <motion.div
                            className={clsx(
                                "w-11 h-6 rounded-full transition-colors",
                                automation.is_enabled ? "bg-primary" : "bg-slate-700"
                            )}
                        >
                            <motion.div
                                animate={{ x: automation.is_enabled ? 20 : 2 }}
                                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                className="absolute top-[2px] w-5 h-5 bg-white rounded-full shadow-lg"
                            />
                        </motion.div>
                    </label>

                    {/* Expand Button */}
                    <button
                        onClick={onToggleExpand}
                        className="p-2 rounded-lg bg-slate-700/30 hover:bg-slate-700/50 transition-colors"
                    >
                        <motion.div
                            animate={{ rotate: isExpanded ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                        >
                            <ChevronDown className="w-4 h-4" />
                        </motion.div>
                    </button>
                </div>
            </div>

            {/* Expanded Details */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="border-t border-slate-700/50"
                    >
                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            <div>
                                <h5 className="text-slate-400 mb-1">Description</h5>
                                <p className="text-slate-300">
                                    {automation.description || 'No description available'}
                                </p>
                            </div>
                            <div>
                                <h5 className="text-slate-400 mb-1">Technical Details</h5>
                                <div className="space-y-1 text-slate-300 font-mono text-xs">
                                    <p><span className="text-slate-500">Task:</span> {automation.celery_task_name}</p>
                                    <p><span className="text-slate-500">Queue:</span> {automation.queue || 'default'}</p>
                                    <p><span className="text-slate-500">Timeout:</span> {automation.timeout_seconds || 300}s</p>
                                </div>
                            </div>
                        </div>

                        {automation.last_error && (
                            <div className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                                <div className="flex items-center gap-2 text-red-400 text-sm mb-1">
                                    <AlertTriangle className="w-4 h-4" />
                                    <span className="font-medium">Last Error</span>
                                </div>
                                <p className="text-xs text-red-300/80 font-mono">{automation.last_error}</p>
                            </div>
                        )}

                        {automation.total_runs_24h !== null && automation.total_runs_24h > 0 && (
                            <div className="mx-4 mb-4 p-3 bg-slate-800/50 rounded-lg">
                                <h5 className="text-slate-400 mb-2 text-xs font-medium">Last 24 Hours Performance</h5>
                                <div className="grid grid-cols-4 gap-4 text-center">
                                    <div>
                                        <p className="text-lg font-bold text-white">{automation.total_runs_24h}</p>
                                        <p className="text-xs text-slate-500">Total Runs</p>
                                    </div>
                                    <div>
                                        <p className="text-lg font-bold text-emerald-400">{automation.total_runs_24h - (automation.failed_runs_24h || 0)}</p>
                                        <p className="text-xs text-slate-500">Successful</p>
                                    </div>
                                    <div>
                                        <p className="text-lg font-bold text-red-400">{automation.failed_runs_24h || 0}</p>
                                        <p className="text-xs text-slate-500">Failed</p>
                                    </div>
                                    <div>
                                        <p className="text-lg font-bold text-primary">{formatDuration(automation.avg_duration_24h || null)}</p>
                                        <p className="text-xs text-slate-500">Avg Duration</p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    )
}

// ============================================================================
// Main Component
// ============================================================================

export default function DeveloperConsole() {
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
    const [expandedAutomation, setExpandedAutomation] = useState<string | null>(null)
    const [showDangerZone, setShowDangerZone] = useState(false)
    const [showAIChat, setShowAIChat] = useState(false)
    const [toasts, setToasts] = useState<Toast[]>([])
    const queryClient = useQueryClient()

    // Toast helpers
    const addToast = (type: Toast['type'], message: string) => {
        const id = Math.random().toString(36).slice(2)
        setToasts(prev => [...prev, { id, type, message }])
        setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
    }
    const removeToast = (id: string) => setToasts(prev => prev.filter(t => t.id !== id))

    // Fetch dashboard stats
    const { data: dashboard, isLoading: dashboardLoading } = useQuery<DashboardStats>({
        queryKey: ['automations-dashboard'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/automations/dashboard')
            return data
        },
        refetchInterval: 10000,
    })

    // Fetch automations list
    const { data: automationsData, isLoading: automationsLoading, refetch } = useQuery<{
        automations: Automation[]
        grouped: Record<string, Automation[]>
    }>({
        queryKey: ['automations', { category: selectedCategory, search: searchQuery }],
        queryFn: async () => {
            const params = new URLSearchParams()
            if (selectedCategory) params.set('category', selectedCategory)
            if (searchQuery) params.set('search', searchQuery)
            const { data } = await axiosInstance.get(`/automations?${params}`)
            return data
        },
        refetchInterval: 30000,
    })

    // Fetch categories
    const { data: categoriesData } = useQuery<{ categories: string[] }>({
        queryKey: ['automation-categories'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/automations/categories')
            return data
        },
    })

    // Fetch health prediction
    const { data: healthPrediction } = useQuery<HealthPrediction>({
        queryKey: ['automations-health'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/automations/ai/health-prediction')
            return data
        },
        refetchInterval: 60000,
    })

    // Toggle automation mutation
    const toggleMutation = useMutation({
        mutationFn: async ({ name, enabled }: { name: string; enabled: boolean }) => {
            const { data } = await axiosInstance.patch(`/automations/${name}`, { enabled })
            return { name, enabled, data }
        },
        onSuccess: ({ name, enabled }) => {
            queryClient.invalidateQueries({ queryKey: ['automations'] })
            queryClient.invalidateQueries({ queryKey: ['automations-dashboard'] })
            addToast('success', `${name} ${enabled ? 'enabled' : 'disabled'}`)
        },
        onError: (_, { name }) => {
            addToast('error', `Failed to toggle ${name}`)
        }
    })

    // Trigger automation mutation
    const triggerMutation = useMutation({
        mutationFn: async (name: string) => {
            const { data } = await axiosInstance.post(`/automations/${name}/run`)
            return { name, data }
        },
        onSuccess: ({ name }) => {
            queryClient.invalidateQueries({ queryKey: ['automations'] })
            addToast('success', `${name} triggered successfully`)
        },
        onError: (_, name) => {
            addToast('error', `Failed to trigger ${name}`)
        }
    })

    // Bulk mutations
    const bulkEnableMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/automations/bulk/enable-all')
            return data
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['automations'] })
            queryClient.invalidateQueries({ queryKey: ['automations-dashboard'] })
            addToast('success', `Enabled ${data.updated?.length || 0} automations`)
        }
    })

    const bulkDisableMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/automations/bulk/disable-all')
            return data
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['automations'] })
            queryClient.invalidateQueries({ queryKey: ['automations-dashboard'] })
            addToast('warning', `Disabled ${data.updated?.length || 0} automations`)
        }
    })

    const automations = automationsData?.automations || []
    const categories = categoriesData?.categories || []
    const isLoading = dashboardLoading || automationsLoading

    // Group automations by category
    const groupedAutomations = automations.reduce((acc, auto) => {
        const cat = auto.category
        if (!acc[cat]) acc[cat] = []
        acc[cat].push(auto)
        return acc
    }, {} as Record<string, Automation[]>)

    return (
        <div className="space-y-6 relative">
            {/* Toast Notifications */}
            <ToastContainer toasts={toasts} removeToast={removeToast} />

            {/* AI Chat Panel */}
            <AIChatPanel isOpen={showAIChat} onClose={() => setShowAIChat(false)} />

            {/* AI Chat FAB */}
            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowAIChat(!showAIChat)}
                className="fixed bottom-4 right-4 z-40 w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 shadow-2xl shadow-purple-500/30 flex items-center justify-center text-white"
            >
                {showAIChat ? <ChevronDown className="w-6 h-6" /> : <Brain className="w-6 h-6" />}
            </motion.button>

            {/* Header */}
            <header className="flex items-center justify-between">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                >
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-orange-500/30">
                            <Terminal className="w-6 h-6 text-white" />
                        </div>
                        <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                            Automations Console
                        </span>
                    </h1>
                    <p className="page-subtitle ml-15">AI-powered task management and monitoring</p>
                </motion.div>
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => refetch()}
                    disabled={isLoading}
                    className="btn-icon"
                >
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </motion.button>
            </header>

            {/* Health Prediction Banner */}
            {healthPrediction && (
                <HealthIndicator prediction={healthPrediction} />
            )}

            {/* Dashboard Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                    {
                        icon: Activity,
                        label: 'Total Automations',
                        value: dashboard?.automations?.total || 0,
                        subtitle: `${dashboard?.automations?.enabled || 0} enabled`,
                        color: 'blue'
                    },
                    {
                        icon: CheckCircle,
                        label: 'Success Rate',
                        value: `${dashboard?.executions?.success_rate?.toFixed(1) || '-'}%`,
                        subtitle: `${dashboard?.executions?.successful || 0} / ${dashboard?.executions?.total_executions || 0}`,
                        color: (dashboard?.executions?.success_rate || 0) >= 90 ? 'emerald' : 'amber'
                    },
                    {
                        icon: XCircle,
                        label: 'Failed (24h)',
                        value: dashboard?.executions?.failed || 0,
                        subtitle: 'Recent failures',
                        color: (dashboard?.executions?.failed || 0) > 0 ? 'red' : 'emerald'
                    },
                    {
                        icon: Timer,
                        label: 'Avg Duration',
                        value: formatDuration(dashboard?.executions?.avg_duration || null),
                        subtitle: `${dashboard?.executions?.running || 0} running`,
                        color: 'purple'
                    }
                ].map((stat, i) => (
                    <motion.div
                        key={stat.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="card p-4 hover:border-slate-600/50 transition-colors"
                    >
                        <div className="flex items-center gap-2 text-slate-400 mb-1">
                            <stat.icon className={clsx(
                                "w-4 h-4",
                                stat.color === 'emerald' && "text-emerald-400",
                                stat.color === 'amber' && "text-amber-400",
                                stat.color === 'red' && "text-red-400",
                                stat.color === 'blue' && "text-blue-400",
                                stat.color === 'purple' && "text-purple-400"
                            )} />
                            <span className="text-sm">{stat.label}</span>
                        </div>
                        <p className={clsx(
                            "text-2xl font-bold",
                            stat.color === 'emerald' && "text-emerald-400",
                            stat.color === 'amber' && "text-amber-400",
                            stat.color === 'red' && "text-red-400",
                            stat.color === 'blue' && "text-white",
                            stat.color === 'purple' && "text-purple-400"
                        )}>
                            {stat.value}
                        </p>
                        <p className="text-xs text-slate-500 mt-1">{stat.subtitle}</p>
                    </motion.div>
                ))}
            </div>

            {/* Search and Filter Bar */}
            <div className="flex flex-wrap items-center gap-4">
                <div className="relative flex-1 min-w-[200px]">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search automations..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="input-field pl-10 w-full"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-slate-400" />
                    <select
                        value={selectedCategory || ''}
                        onChange={(e) => setSelectedCategory(e.target.value || null)}
                        className="input-field"
                    >
                        <option value="">All Categories</option>
                        {categories.map(cat => (
                            <option key={cat} value={cat}>
                                {CATEGORY_LABELS[cat] || cat}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Automations by Category */}
            <AnimatePresence mode="popLayout">
                {Object.entries(groupedAutomations).map(([category, categoryAutomations], catIdx) => (
                    <motion.div
                        key={category}
                        layout
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ delay: catIdx * 0.05 }}
                        className="card p-6"
                    >
                        <h3 className="flex items-center gap-2 text-lg font-semibold text-white mb-4">
                            <span className={clsx(
                                "w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br",
                                CATEGORY_COLORS[category] || "from-slate-500 to-slate-600"
                            )}>
                                {CATEGORY_ICONS[category] || <Cpu className="w-4 h-4 text-white" />}
                            </span>
                            {CATEGORY_LABELS[category] || category}
                            <span className="ml-2 text-sm font-normal text-slate-400">
                                ({categoryAutomations.length})
                            </span>
                        </h3>

                        <div className="space-y-3">
                            {categoryAutomations.map(automation => (
                                <AutomationCard
                                    key={automation.name}
                                    automation={automation}
                                    isExpanded={expandedAutomation === automation.name}
                                    onToggleExpand={() => setExpandedAutomation(
                                        expandedAutomation === automation.name ? null : automation.name
                                    )}
                                    onToggleEnabled={(enabled) => toggleMutation.mutate({
                                        name: automation.name,
                                        enabled
                                    })}
                                    onTrigger={() => triggerMutation.mutate(automation.name)}
                                    isToggling={toggleMutation.isPending}
                                    isTriggering={triggerMutation.isPending}
                                />
                            ))}
                        </div>
                    </motion.div>
                ))}
            </AnimatePresence>

            {/* Danger Zone */}
            <motion.div
                layout
                className="card border-red-500/30 bg-red-950/10 overflow-hidden"
            >
                <button
                    onClick={() => setShowDangerZone(!showDangerZone)}
                    className="w-full flex items-center justify-between p-4 text-left"
                >
                    <div className="flex items-center gap-3">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                        <span className="font-semibold text-red-400">Danger Zone</span>
                    </div>
                    <motion.div animate={{ rotate: showDangerZone ? 180 : 0 }}>
                        <ChevronUp className="w-4 h-4 text-red-400" />
                    </motion.div>
                </button>

                <AnimatePresence>
                    {showDangerZone && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="px-4 pb-4 space-y-4 border-t border-red-500/20"
                        >
                            <p className="text-sm text-slate-400 mt-4">
                                Bulk operations that affect all automations. Use with caution.
                            </p>

                            <div className="flex flex-wrap gap-3">
                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => bulkEnableMutation.mutate()}
                                    disabled={bulkEnableMutation.isPending}
                                    className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-lg hover:bg-emerald-500/30 transition-colors"
                                >
                                    {bulkEnableMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Power className="w-4 h-4" />}
                                    Enable All
                                </motion.button>

                                <motion.button
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => bulkDisableMutation.mutate()}
                                    disabled={bulkDisableMutation.isPending}
                                    className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                                >
                                    {bulkDisableMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <PowerOff className="w-4 h-4" />}
                                    Disable All
                                </motion.button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </div>
    )
}
