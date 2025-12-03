import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Shield, RefreshCw, Activity, CheckCircle, XCircle, AlertTriangle,
    Zap, TrendingUp, TrendingDown, Clock, FileCode, Bug, Wrench,
    Play, Target, Brain, BarChart3, FileWarning, Sparkles, Database,
    List, History, Eye, EyeOff, Search, Filter, ChevronDown, ChevronRight,
    X, ExternalLink, GitBranch, Layers, AlertCircle, Command, Keyboard,
    CheckSquare, Square, Trash2, Archive, Settings, Download, Upload,
    Bot, Lightbulb, Wand2, MoreVertical, Copy, MessageSquare, Send
} from 'lucide-react'
import { useState, useMemo, useEffect, useCallback, useRef } from 'react'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, AreaChart, Area
} from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'
import { toast, Toaster } from 'sonner'
import { Command as CommandPrimitive } from 'cmdk'

// ============ Type Definitions ============

interface QAStatus {
    is_running: boolean
    last_run_time: string | null
    next_run_time: string | null
    health_score: number
    total_cycles: number
    total_issues_found: number
    total_issues_fixed: number
    last_run_duration_seconds: number
    critical_failures: number
}

interface Accomplishment {
    timestamp: string
    category: string
    module: string
    message: string
    severity: string
    files_affected: string[]
}

interface HotSpot {
    file: string
    file_path?: string
    issue_count: number
    severity_score: number
    patterns: string[]
    last_issue_at?: string
}

interface PatternStats {
    total_patterns: number
    total_occurrences: number
    by_severity: Record<string, number>
    by_category: Record<string, number>
}

interface QASummary {
    status: QAStatus
    recent_accomplishments: Accomplishment[]
    accomplishments_by_category: Record<string, number>
    pattern_statistics: PatternStats
    top_hot_spots: HotSpot[]
}

interface DBIssue {
    id: number
    issue_hash: string
    module_name: string
    check_name: string
    title: string
    description: string | null
    severity: 'critical' | 'high' | 'medium' | 'low'
    status: 'open' | 'fixing' | 'fixed' | 'ignored' | 'wont_fix'
    occurrence_count: number
    first_seen_at: string
    last_seen_at: string
    affected_files: string[]
    suggested_fix: string | null
    auto_fixable: boolean
    tags: string[]
}

interface DBIssueDetail extends DBIssue {
    occurrences: {
        id: number
        run_id: number
        detected_at: string
        context: string | null
    }[]
    fixes: {
        id: number
        attempted_at: string
        success: boolean
        fix_type: string
        description: string | null
        commit_hash: string | null
    }[]
}

interface DBRun {
    id: number
    run_id: string
    started_at: string
    completed_at: string | null
    status: string
    triggered_by: string
    health_score_before: number | null
    health_score_after: number | null
    total_checks: number
    passed_checks: number
    failed_checks: number
    issues_found: number
    issues_fixed: number
    duration_seconds: number | null
}

interface DBRunDetail extends DBRun {
    check_results: {
        id: number
        module_name: string
        check_name: string
        status: string
        message: string | null
        duration_ms: number | null
    }[]
}

interface DBDashboardSummary {
    database_available: boolean
    total_issues: number
    open_issues: number
    fixed_issues: number
    by_severity: Record<string, number>
    by_status: Record<string, number>
    recent_runs: DBRun[]
    top_issues: DBIssue[]
    health_trend: { timestamp: string; score: number }[]
}

interface DBHealthTrend {
    timestamp: string
    hour?: string
    avg_score: number
    min_score: number
    max_score: number
}

interface AIAnalysis {
    summary: string
    root_cause: string
    suggested_fixes: string[]
    priority: string
    estimated_effort: string
    related_issues: number[]
}

// ============ Constants ============

const COLORS = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
const SEVERITY_COLORS = {
    critical: '#EF4444',
    high: '#F97316',
    medium: '#EAB308',
    low: '#3B82F6'
}
const STATUS_COLORS = {
    open: '#EF4444',
    fixing: '#F97316',
    fixed: '#10B981',
    ignored: '#6B7280',
    wont_fix: '#6B7280'
}

type TabType = 'overview' | 'issues' | 'runs' | 'hotspots' | 'ai-assistant'

// ============ Animation Variants ============

const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
}

const staggerContainer = {
    animate: {
        transition: {
            staggerChildren: 0.1
        }
    }
}

const scaleIn = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 }
}

// ============ Skeleton Components ============

const Skeleton = ({ className = '' }: { className?: string }) => (
    <div className={`animate-pulse bg-slate-700/50 rounded ${className}`} />
)

const CardSkeleton = () => (
    <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-5 w-5 rounded-full" />
        </div>
        <Skeleton className="h-10 w-20 mb-2" />
        <Skeleton className="h-3 w-32" />
    </div>
)

const TableRowSkeleton = () => (
    <tr className="border-t border-slate-700/50">
        <td className="p-4"><Skeleton className="h-6 w-16" /></td>
        <td className="p-4"><Skeleton className="h-6 w-16" /></td>
        <td className="p-4"><Skeleton className="h-6 w-48" /></td>
        <td className="p-4"><Skeleton className="h-6 w-24" /></td>
        <td className="p-4"><Skeleton className="h-6 w-12" /></td>
        <td className="p-4"><Skeleton className="h-6 w-20" /></td>
        <td className="p-4"><Skeleton className="h-6 w-16" /></td>
    </tr>
)

// ============ Command Palette Component ============

const CommandPalette = ({
    isOpen,
    onClose,
    onRunQA,
    onSetTab,
    onRefresh
}: {
    isOpen: boolean
    onClose: () => void
    onRunQA: () => void
    onSetTab: (tab: TabType) => void
    onRefresh: () => void
}) => {
    const [search, setSearch] = useState('')

    useEffect(() => {
        if (isOpen) setSearch('')
    }, [isOpen])

    const commands = [
        { id: 'run-qa', label: 'Run QA Now', icon: Play, action: () => { onRunQA(); onClose() } },
        { id: 'overview', label: 'Go to Overview', icon: BarChart3, action: () => { onSetTab('overview'); onClose() } },
        { id: 'issues', label: 'Go to Issues', icon: Bug, action: () => { onSetTab('issues'); onClose() } },
        { id: 'runs', label: 'Go to Run History', icon: History, action: () => { onSetTab('runs'); onClose() } },
        { id: 'hotspots', label: 'Go to Hot Spots', icon: FileWarning, action: () => { onSetTab('hotspots'); onClose() } },
        { id: 'ai', label: 'Open AI Assistant', icon: Bot, action: () => { onSetTab('ai-assistant'); onClose() } },
        { id: 'refresh', label: 'Refresh Data', icon: RefreshCw, action: () => { onRefresh(); onClose() } },
    ]

    if (!isOpen) return null

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-start justify-center pt-[20vh]"
                onClick={onClose}
            >
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -20 }}
                    className="bg-slate-900 border border-slate-700/50 rounded-xl shadow-2xl w-full max-w-lg overflow-hidden"
                    onClick={(e) => e.stopPropagation()}
                >
                    <CommandPrimitive className="flex flex-col">
                        <div className="flex items-center border-b border-slate-700/50 px-4">
                            <Search className="w-5 h-5 text-slate-400" />
                            <CommandPrimitive.Input
                                value={search}
                                onValueChange={setSearch}
                                placeholder="Type a command or search..."
                                className="flex-1 bg-transparent border-0 outline-none px-4 py-4 text-white placeholder:text-slate-500"
                            />
                            <kbd className="px-2 py-1 text-xs bg-slate-800 text-slate-400 rounded">ESC</kbd>
                        </div>
                        <CommandPrimitive.List className="max-h-[300px] overflow-y-auto p-2">
                            <CommandPrimitive.Empty className="py-6 text-center text-slate-500">
                                No commands found.
                            </CommandPrimitive.Empty>
                            <CommandPrimitive.Group heading="Actions" className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:text-slate-500 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                                {commands.filter(c =>
                                    c.label.toLowerCase().includes(search.toLowerCase())
                                ).map((command) => (
                                    <CommandPrimitive.Item
                                        key={command.id}
                                        value={command.label}
                                        onSelect={command.action}
                                        className="flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer text-slate-300 hover:bg-slate-800 aria-selected:bg-slate-800"
                                    >
                                        <command.icon className="w-4 h-4 text-slate-400" />
                                        <span>{command.label}</span>
                                    </CommandPrimitive.Item>
                                ))}
                            </CommandPrimitive.Group>
                        </CommandPrimitive.List>
                        <div className="border-t border-slate-700/50 px-4 py-2 flex items-center justify-between text-xs text-slate-500">
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-slate-800 rounded">↑↓</kbd>
                                <span>Navigate</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <kbd className="px-1.5 py-0.5 bg-slate-800 rounded">↵</kbd>
                                <span>Select</span>
                            </div>
                        </div>
                    </CommandPrimitive>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    )
}

// ============ AI Status Interface ============

interface AIStatus {
    available: boolean
    providers: string[]
    total_cost: number
    cache_enabled: boolean
    intelligent_routing: boolean
    error?: string
    message?: string
}

interface AIMessage {
    role: 'user' | 'assistant'
    content: string
    ai_powered?: boolean
    provider?: string
    model?: string
    timestamp?: Date
}

// ============ Markdown Renderer Component ============

const MarkdownRenderer = ({ content }: { content: string }) => {
    // Parse markdown-like syntax to React elements
    const renderContent = (text: string) => {
        const lines = text.split('\n')
        const elements: React.ReactNode[] = []
        let inCodeBlock = false
        let codeContent: string[] = []
        let codeLanguage = ''

        lines.forEach((line, idx) => {
            // Code block handling
            if (line.startsWith('```')) {
                if (!inCodeBlock) {
                    inCodeBlock = true
                    codeLanguage = line.slice(3).trim()
                    codeContent = []
                } else {
                    elements.push(
                        <pre key={`code-${idx}`} className="bg-slate-900 rounded-lg p-3 my-2 overflow-x-auto text-xs">
                            <code className="text-green-400">{codeContent.join('\n')}</code>
                        </pre>
                    )
                    inCodeBlock = false
                    codeContent = []
                }
                return
            }

            if (inCodeBlock) {
                codeContent.push(line)
                return
            }

            // Headers
            if (line.startsWith('### ')) {
                elements.push(<h4 key={idx} className="text-md font-semibold text-white mt-3 mb-1">{line.slice(4)}</h4>)
                return
            }
            if (line.startsWith('## ')) {
                elements.push(<h3 key={idx} className="text-lg font-semibold text-white mt-3 mb-1">{line.slice(3)}</h3>)
                return
            }
            if (line.startsWith('# ')) {
                elements.push(<h2 key={idx} className="text-xl font-bold text-white mt-3 mb-2">{line.slice(2)}</h2>)
                return
            }

            // Bold headers (like **Title:**)
            if (line.match(/^\*\*[^*]+:\*\*/)) {
                const text = line.replace(/\*\*/g, '')
                elements.push(<p key={idx} className="font-semibold text-blue-300 mt-2">{text}</p>)
                return
            }

            // List items
            if (line.match(/^[-*]\s/)) {
                const content = line.slice(2)
                elements.push(
                    <li key={idx} className="ml-4 text-slate-300 flex items-start gap-2">
                        <span className="text-purple-400 mt-1">•</span>
                        <span dangerouslySetInnerHTML={{ __html: formatInline(content) }} />
                    </li>
                )
                return
            }

            // Numbered lists
            if (line.match(/^\d+\.\s/)) {
                const content = line.replace(/^\d+\.\s/, '')
                const num = line.match(/^(\d+)\./)?.[1]
                elements.push(
                    <li key={idx} className="ml-4 text-slate-300 flex items-start gap-2">
                        <span className="text-blue-400 font-mono min-w-[20px]">{num}.</span>
                        <span dangerouslySetInnerHTML={{ __html: formatInline(content) }} />
                    </li>
                )
                return
            }

            // Empty lines
            if (!line.trim()) {
                elements.push(<div key={idx} className="h-2" />)
                return
            }

            // Regular paragraph
            elements.push(
                <p key={idx} className="text-slate-300" dangerouslySetInnerHTML={{ __html: formatInline(line) }} />
            )
        })

        return elements
    }

    const formatInline = (text: string): string => {
        // Bold **text**
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
        // Italic *text* or _text_
        text = text.replace(/\*([^*]+)\*/g, '<em class="text-slate-200">$1</em>')
        text = text.replace(/_([^_]+)_/g, '<em class="text-slate-200">$1</em>')
        // Code `text`
        text = text.replace(/`([^`]+)`/g, '<code class="bg-slate-900 px-1 py-0.5 rounded text-green-400 text-xs">$1</code>')
        // Links [text](url) - simplified, don't actually link in this context
        text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '<span class="text-blue-400 underline">$1</span>')
        return text
    }

    return <div className="space-y-1">{renderContent(content)}</div>
}

// ============ AI Chat Component ============

const AIAssistantTab = ({ issues }: { issues: DBIssue[] }) => {
    const [messages, setMessages] = useState<AIMessage[]>([
        {
            role: 'assistant',
            content: "Hello! I'm your AI QA Assistant. I can help you analyze issues, suggest fixes, and prioritize your work. What would you like to know?",
            timestamp: new Date()
        }
    ])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    // Query AI status
    const { data: aiStatus } = useQuery<AIStatus>({
        queryKey: ['qa-ai-status'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/ai/status')
            return data
        },
        staleTime: 60000,
        refetchInterval: 120000
    })

    // Query AI recommendations
    const { data: recommendations } = useQuery<{ recommendations: string[]; priority: string; stats?: { open_issues: number; critical: number; high: number } }>({
        queryKey: ['qa-ai-recommendations'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/ai/recommendations')
            return data
        },
        staleTime: 30000
    })

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const analyzeWithAI = useMutation({
        mutationFn: async (prompt: string) => {
            const { data } = await axiosInstance.post('/qa/ai/analyze', {
                prompt,
                use_real_ai: true,
                context: {
                    open_issues: issues.filter(i => i.status === 'open').length,
                    critical_issues: issues.filter(i => i.severity === 'critical').length,
                    issues_summary: issues.slice(0, 10).map(i => ({
                        title: i.title,
                        severity: i.severity,
                        status: i.status,
                        module: i.module_name
                    }))
                }
            })
            return data
        }
    })

    const generateLocalResponse = (query: string, issues: DBIssue[]) => {
        const lowerQuery = query.toLowerCase()
        const openIssues = issues.filter(i => i.status === 'open' || i.status === 'fixing')
        const criticalIssues = issues.filter(i => i.severity === 'critical')

        if (lowerQuery.includes('summary') || lowerQuery.includes('overview')) {
            return `**QA Summary:**\n\n- Total Issues: ${issues.length}\n- Open Issues: ${openIssues.length}\n- Critical Issues: ${criticalIssues.length}\n\n${criticalIssues.length > 0 ? `**Critical Issues to Address:**\n${criticalIssues.slice(0, 3).map(i => `- ${i.title}`).join('\n')}` : 'No critical issues! Great work!'}`
        }

        if (lowerQuery.includes('critical') || lowerQuery.includes('urgent')) {
            return criticalIssues.length > 0
                ? `**Critical Issues (${criticalIssues.length}):**\n\n${criticalIssues.map(i => `- **${i.title}** (${i.module_name})\n  ${i.description || 'No description'}`).join('\n\n')}`
                : "Great news! No critical issues found."
        }

        if (lowerQuery.includes('prioritize') || lowerQuery.includes('priority')) {
            const prioritized = [...openIssues].sort((a, b) => {
                const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 }
                return severityOrder[a.severity] - severityOrder[b.severity] || b.occurrence_count - a.occurrence_count
            })
            return `**Prioritized Issues:**\n\n${prioritized.slice(0, 5).map((i, idx) => `${idx + 1}. **[${i.severity.toUpperCase()}]** ${i.title}\n   Module: ${i.module_name} | Occurrences: ${i.occurrence_count}`).join('\n\n')}`
        }

        if (lowerQuery.includes('fix') || lowerQuery.includes('suggest')) {
            const issueToFix = openIssues[0]
            if (issueToFix) {
                return `**Suggested Fix for: ${issueToFix.title}**\n\n${issueToFix.suggested_fix || 'Based on the issue type, I recommend:\n1. Review the affected file(s)\n2. Check recent changes that might have caused this\n3. Add proper error handling\n4. Write tests to prevent regression'}`
            }
            return "No open issues to fix!"
        }

        return `I can help you with:\n- **"Give me a summary"** - Overview of all issues\n- **"Show critical issues"** - List urgent problems\n- **"Prioritize issues"** - Ranked list by importance\n- **"Suggest a fix"** - Get fix recommendations\n\nWhat would you like to know?`
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim()) return

        const userMessage = input.trim()
        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date() }])
        setIsLoading(true)

        try {
            const result = await analyzeWithAI.mutateAsync(userMessage)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: result.response || generateLocalResponse(userMessage, issues),
                ai_powered: result.ai_powered,
                provider: result.provider,
                model: result.model,
                timestamp: new Date()
            }])
        } catch {
            const localResponse = generateLocalResponse(userMessage, issues)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: localResponse,
                ai_powered: false,
                timestamp: new Date()
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const clearChat = () => {
        setMessages([{
            role: 'assistant',
            content: "Chat cleared. How can I help you with QA issues?",
            timestamp: new Date()
        }])
    }

    const copyMessage = (content: string) => {
        navigator.clipboard.writeText(content)
        toast.success('Copied to clipboard')
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-[700px]">
            {/* Main Chat Area */}
            <motion.div
                variants={fadeInUp}
                initial="initial"
                animate="animate"
                className="card lg:col-span-3 flex flex-col"
            >
                {/* Header with AI Status */}
                <div className="p-4 border-b border-slate-700/50">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="relative">
                                <Bot className="w-6 h-6 text-purple-400" />
                                {aiStatus?.available && (
                                    <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-slate-900" />
                                )}
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                    AI QA Assistant
                                    {aiStatus?.available ? (
                                        <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full flex items-center gap-1">
                                            <Sparkles className="w-3 h-3" />
                                            LLM Active
                                        </span>
                                    ) : (
                                        <span className="px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded-full">
                                            Rule-Based
                                        </span>
                                    )}
                                </h3>
                                {aiStatus?.available && aiStatus.providers && (
                                    <p className="text-xs text-slate-500">
                                        Providers: {aiStatus.providers.join(', ')}
                                        {aiStatus.intelligent_routing && ' • Smart Routing'}
                                        {aiStatus.cache_enabled && ' • Cached'}
                                    </p>
                                )}
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={clearChat}
                                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                                title="Clear chat"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`max-w-[85%] rounded-xl px-4 py-3 ${
                                msg.role === 'user'
                                    ? 'bg-blue-500/20 text-blue-100'
                                    : 'bg-slate-800/50 text-slate-200'
                            }`}>
                                {msg.role === 'assistant' && (
                                    <div className="flex items-center justify-between gap-2 mb-2 pb-2 border-b border-slate-700/50">
                                        <div className="flex items-center gap-2">
                                            <Bot className="w-4 h-4 text-purple-400" />
                                            <span className="text-xs font-medium text-purple-400">AI Assistant</span>
                                            {msg.ai_powered && (
                                                <span className="px-1.5 py-0.5 text-[10px] bg-green-500/20 text-green-400 rounded flex items-center gap-1">
                                                    <Sparkles className="w-2.5 h-2.5" />
                                                    AI
                                                </span>
                                            )}
                                            {msg.provider && msg.model && (
                                                <span className="text-[10px] text-slate-500">
                                                    via {msg.provider}/{msg.model}
                                                </span>
                                            )}
                                        </div>
                                        <button
                                            onClick={() => copyMessage(msg.content)}
                                            className="p-1 text-slate-500 hover:text-slate-300 transition-colors"
                                            title="Copy message"
                                        >
                                            <Copy className="w-3 h-3" />
                                        </button>
                                    </div>
                                )}
                                <div className="text-sm">
                                    {msg.role === 'assistant' ? (
                                        <MarkdownRenderer content={msg.content} />
                                    ) : (
                                        <span>{msg.content}</span>
                                    )}
                                </div>
                                {msg.timestamp && (
                                    <div className="text-[10px] text-slate-500 mt-2 text-right">
                                        {msg.timestamp.toLocaleTimeString()}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex justify-start"
                        >
                            <div className="bg-slate-800/50 rounded-xl px-4 py-3">
                                <div className="flex items-center gap-2 text-purple-400">
                                    <Bot className="w-4 h-4 animate-pulse" />
                                    <span className="text-sm">Analyzing with {aiStatus?.available ? 'AI' : 'rules'}...</span>
                                    <div className="flex gap-1">
                                        <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:0ms]" />
                                        <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:150ms]" />
                                        <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:300ms]" />
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Quick Actions */}
                <div className="px-4 py-2 border-t border-slate-700/50 flex gap-2 flex-wrap">
                    {[
                        { label: 'Summary', icon: BarChart3 },
                        { label: 'Critical issues', icon: AlertCircle },
                        { label: 'Prioritize', icon: Target },
                        { label: 'Suggest fix', icon: Lightbulb },
                        { label: 'Patterns', icon: Layers },
                        { label: 'Hot spots', icon: FileWarning }
                    ].map(({ label, icon: Icon }) => (
                        <button
                            key={label}
                            onClick={() => setInput(label)}
                            className="px-3 py-1.5 text-xs bg-slate-800 text-slate-400 rounded-full hover:bg-slate-700 hover:text-white transition-colors flex items-center gap-1.5"
                        >
                            <Icon className="w-3 h-3" />
                            {label}
                        </button>
                    ))}
                </div>

                {/* Input */}
                <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700/50">
                    <div className="flex items-center gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask me about QA issues, priorities, or fixes..."
                            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder:text-slate-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/50"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="p-2.5 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </div>
                </form>
            </motion.div>

            {/* Side Panel - AI Status & Recommendations */}
            <motion.div
                variants={fadeInUp}
                initial="initial"
                animate="animate"
                className="space-y-4"
            >
                {/* AI Status Card */}
                <div className="card p-4">
                    <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-purple-400" />
                        AI Status
                    </h4>
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-slate-400">Status</span>
                            {aiStatus?.available ? (
                                <span className="flex items-center gap-1 text-xs text-green-400">
                                    <CheckCircle className="w-3 h-3" />
                                    Online
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-xs text-yellow-400">
                                    <AlertTriangle className="w-3 h-3" />
                                    Fallback
                                </span>
                            )}
                        </div>
                        {aiStatus?.providers && aiStatus.providers.length > 0 && (
                            <div>
                                <span className="text-xs text-slate-400 block mb-1">Available Providers</span>
                                <div className="flex flex-wrap gap-1">
                                    {aiStatus.providers.map(p => (
                                        <span key={p} className="px-2 py-0.5 text-[10px] bg-slate-800 text-slate-300 rounded">
                                            {p}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                        {aiStatus?.intelligent_routing && (
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                                <Wand2 className="w-3 h-3 text-purple-400" />
                                Intelligent routing enabled
                            </div>
                        )}
                        {aiStatus?.cache_enabled && (
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                                <Database className="w-3 h-3 text-blue-400" />
                                Response caching active
                            </div>
                        )}
                    </div>
                </div>

                {/* Recommendations Card */}
                {recommendations && (
                    <div className="card p-4">
                        <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                            <Lightbulb className="w-4 h-4 text-yellow-400" />
                            AI Recommendations
                            {recommendations.priority && (
                                <span className={`px-2 py-0.5 text-[10px] rounded ${
                                    recommendations.priority === 'critical' ? 'bg-red-500/20 text-red-400' :
                                    recommendations.priority === 'high' ? 'bg-orange-500/20 text-orange-400' :
                                    recommendations.priority === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                    'bg-green-500/20 text-green-400'
                                }`}>
                                    {recommendations.priority}
                                </span>
                            )}
                        </h4>
                        <ul className="space-y-2">
                            {recommendations.recommendations.map((rec, idx) => (
                                <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                                    <span className="text-purple-400 mt-0.5">•</span>
                                    {rec}
                                </li>
                            ))}
                        </ul>
                        {recommendations.stats && (
                            <div className="mt-3 pt-3 border-t border-slate-700/50 grid grid-cols-3 gap-2">
                                <div className="text-center">
                                    <div className="text-lg font-bold text-white">{recommendations.stats.open_issues}</div>
                                    <div className="text-[10px] text-slate-500">Open</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-lg font-bold text-red-400">{recommendations.stats.critical}</div>
                                    <div className="text-[10px] text-slate-500">Critical</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-lg font-bold text-orange-400">{recommendations.stats.high}</div>
                                    <div className="text-[10px] text-slate-500">High</div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Keyboard Shortcuts */}
                <div className="card p-4">
                    <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                        <Keyboard className="w-4 h-4 text-slate-400" />
                        Shortcuts
                    </h4>
                    <div className="space-y-2 text-xs">
                        <div className="flex items-center justify-between text-slate-400">
                            <span>Command Palette</span>
                            <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">⌘K</kbd>
                        </div>
                        <div className="flex items-center justify-between text-slate-400">
                            <span>AI Assistant</span>
                            <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">Alt+5</kbd>
                        </div>
                        <div className="flex items-center justify-between text-slate-400">
                            <span>Refresh</span>
                            <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300">⌘R</kbd>
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    )
}

// ============ WebSocket Hook ============

interface WebSocketMessage {
    type: 'connected' | 'qa_update' | 'new_issue' | 'issue_updated' | 'run_started' | 'run_completed' | 'pong' | 'subscribed' | 'status_response'
    timestamp: string
    status?: QAStatus
    summary?: DBDashboardSummary
    data?: Record<string, unknown>
    message?: string
    active_connections?: number
}

interface UseWebSocketReturn {
    isConnected: boolean
    lastMessage: WebSocketMessage | null
    activeConnections: number
    send: (message: Record<string, unknown>) => void
    reconnect: () => void
}

const useQAWebSocket = (enabled: boolean = true): UseWebSocketReturn => {
    const [isConnected, setIsConnected] = useState(false)
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
    const [activeConnections, setActiveConnections] = useState(0)
    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const queryClient = useQueryClient()

    const connect = useCallback(() => {
        if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) return

        try {
            // Get WebSocket URL from API base URL
            const wsUrl = `ws://localhost:8002/api/qa/ws`
            wsRef.current = new WebSocket(wsUrl)

            wsRef.current.onopen = () => {
                setIsConnected(true)
                toast.success('Real-time updates connected', { duration: 2000 })
            }

            wsRef.current.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data)
                    setLastMessage(message)

                    if (message.active_connections) {
                        setActiveConnections(message.active_connections)
                    }

                    // Handle different message types
                    switch (message.type) {
                        case 'qa_update':
                            // Invalidate queries to refresh data
                            queryClient.invalidateQueries({ queryKey: ['qa-summary'] })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-dashboard'] })
                            break

                        case 'new_issue':
                            toast.warning('New QA issue detected', { duration: 5000 })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-issues'] })
                            break

                        case 'issue_updated':
                            queryClient.invalidateQueries({ queryKey: ['qa-db-issues'] })
                            break

                        case 'run_started':
                            toast.info('QA cycle started', { duration: 3000 })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-runs'] })
                            break

                        case 'run_completed':
                            toast.success('QA cycle completed', { duration: 3000 })
                            queryClient.invalidateQueries({ queryKey: ['qa-summary'] })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-dashboard'] })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-runs'] })
                            queryClient.invalidateQueries({ queryKey: ['qa-db-issues'] })
                            break
                    }
                } catch {
                    // Ignore parse errors
                }
            }

            wsRef.current.onclose = () => {
                setIsConnected(false)
                wsRef.current = null

                // Attempt reconnect after 5 seconds
                if (enabled) {
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect()
                    }, 5000)
                }
            }

            wsRef.current.onerror = () => {
                setIsConnected(false)
            }
        } catch {
            setIsConnected(false)
        }
    }, [enabled, queryClient])

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
            reconnectTimeoutRef.current = null
        }
        if (wsRef.current) {
            wsRef.current.close()
            wsRef.current = null
        }
        setIsConnected(false)
    }, [])

    const send = useCallback((message: Record<string, unknown>) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message))
        }
    }, [])

    const reconnect = useCallback(() => {
        disconnect()
        setTimeout(connect, 100)
    }, [connect, disconnect])

    useEffect(() => {
        if (enabled) {
            connect()
        } else {
            disconnect()
        }

        return () => {
            disconnect()
        }
    }, [enabled, connect, disconnect])

    // Ping every 30 seconds to keep connection alive
    useEffect(() => {
        if (!isConnected) return

        const pingInterval = setInterval(() => {
            send({ type: 'ping' })
        }, 30000)

        return () => clearInterval(pingInterval)
    }, [isConnected, send])

    return { isConnected, lastMessage, activeConnections, send, reconnect }
}

// ============ Main Component ============

export default function QADashboard() {
    const [activeTab, setActiveTab] = useState<TabType>('overview')
    const [autoRefresh, setAutoRefresh] = useState(true)
    const [selectedIssue, setSelectedIssue] = useState<number | null>(null)
    const [selectedRun, setSelectedRun] = useState<number | null>(null)
    const [selectedIssues, setSelectedIssues] = useState<Set<number>>(new Set())
    const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
    const [useWebSocket, setUseWebSocket] = useState(true)
    const [issueFilters, setIssueFilters] = useState({
        status: 'all',
        severity: 'all',
        search: ''
    })
    const queryClient = useQueryClient()

    // WebSocket connection for real-time updates
    const { isConnected: wsConnected, activeConnections, reconnect: wsReconnect } = useQAWebSocket(useWebSocket)

    // ============ Keyboard Shortcuts ============

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Cmd/Ctrl + K for command palette
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault()
                setCommandPaletteOpen(true)
            }
            // Escape to close modals
            if (e.key === 'Escape') {
                setCommandPaletteOpen(false)
                setSelectedIssue(null)
            }
            // Number keys for quick tab switching
            if (e.altKey && !e.ctrlKey && !e.metaKey) {
                const tabMap: Record<string, TabType> = {
                    '1': 'overview',
                    '2': 'issues',
                    '3': 'runs',
                    '4': 'hotspots',
                    '5': 'ai-assistant'
                }
                if (tabMap[e.key]) {
                    e.preventDefault()
                    setActiveTab(tabMap[e.key])
                }
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [])

    // ============ Queries ============

    const { data: summary, isLoading, refetch: refetchSummary } = useQuery<QASummary>({
        queryKey: ['qa-summary'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/summary')
            return data
        },
        refetchInterval: autoRefresh ? 30000 : false,
        staleTime: 10000
    })

    const { data: dbStatus } = useQuery<{ available: boolean; message: string }>({
        queryKey: ['qa-db-status'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/db/status')
            return data
        },
        staleTime: 60000
    })

    const { data: dbDashboard, refetch: refetchDbDashboard } = useQuery<DBDashboardSummary>({
        queryKey: ['qa-db-dashboard'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/db/dashboard')
            return data
        },
        enabled: dbStatus?.available ?? false,
        refetchInterval: autoRefresh ? 30000 : false
    })

    const { data: dbIssues, isLoading: isLoadingIssues } = useQuery<{ issues: DBIssue[]; total: number }>({
        queryKey: ['qa-db-issues', issueFilters],
        queryFn: async () => {
            const params = new URLSearchParams()
            if (issueFilters.status !== 'all') params.append('status', issueFilters.status)
            if (issueFilters.severity !== 'all') params.append('severity', issueFilters.severity)
            params.append('limit', '100')
            const { data } = await axiosInstance.get(`/qa/db/issues?${params}`)
            return data
        },
        enabled: dbStatus?.available ?? false,
        refetchInterval: autoRefresh ? 30000 : false
    })

    const { data: dbRuns } = useQuery<{ runs: DBRun[] }>({
        queryKey: ['qa-db-runs'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/db/runs?limit=50')
            return data
        },
        enabled: dbStatus?.available ?? false,
        refetchInterval: autoRefresh ? 30000 : false
    })

    const { data: issueDetail, isLoading: isLoadingDetail } = useQuery<DBIssueDetail>({
        queryKey: ['qa-db-issue', selectedIssue],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/qa/db/issues/${selectedIssue}`)
            return data
        },
        enabled: selectedIssue !== null
    })

    const { data: runDetail } = useQuery<DBRunDetail>({
        queryKey: ['qa-db-run', selectedRun],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/qa/db/runs/${selectedRun}`)
            return data
        },
        enabled: selectedRun !== null
    })

    const { data: dbHealthTrend } = useQuery<{ trend: DBHealthTrend[] }>({
        queryKey: ['qa-db-health-trend'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/db/health-trend?hours=48')
            return data
        },
        enabled: dbStatus?.available ?? false,
        refetchInterval: autoRefresh ? 60000 : false
    })

    const { data: dbHotSpots } = useQuery<{ hot_spots: HotSpot[] }>({
        queryKey: ['qa-db-hot-spots'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/db/hot-spots?limit=20')
            return data
        },
        enabled: dbStatus?.available ?? false,
        refetchInterval: autoRefresh ? 60000 : false
    })

    const { data: healthHistory } = useQuery<{ history: { timestamp: string; score: number }[] }>({
        queryKey: ['qa-health-history'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/health-history?days=7')
            return data
        },
        refetchInterval: autoRefresh ? 60000 : false
    })

    const { data: accomplishments } = useQuery<{ accomplishments: Accomplishment[]; by_category: Record<string, number> }>({
        queryKey: ['qa-accomplishments'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/qa/accomplishments?limit=20')
            return data
        },
        refetchInterval: autoRefresh ? 30000 : false
    })

    // ============ Mutations with Optimistic Updates ============

    const runQAMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/qa/run-once')
            return data
        },
        onMutate: () => {
            toast.loading('Starting QA cycle...', { id: 'qa-run' })
        },
        onSuccess: () => {
            toast.success('QA cycle started successfully!', { id: 'qa-run' })
            queryClient.invalidateQueries({ queryKey: ['qa-summary'] })
            queryClient.invalidateQueries({ queryKey: ['qa-db-dashboard'] })
            queryClient.invalidateQueries({ queryKey: ['qa-db-runs'] })
        },
        onError: (error: Error) => {
            toast.error(`Failed to start QA: ${error.message}`, { id: 'qa-run' })
        }
    })

    const updateIssueStatusMutation = useMutation({
        mutationFn: async ({ issueId, status, notes }: { issueId: number; status: string; notes?: string }) => {
            const { data } = await axiosInstance.patch(`/qa/db/issues/${issueId}/status`, { status, notes })
            return data
        },
        onMutate: async ({ issueId, status }) => {
            // Cancel any outgoing refetches
            await queryClient.cancelQueries({ queryKey: ['qa-db-issues'] })

            // Snapshot the previous value
            const previousIssues = queryClient.getQueryData(['qa-db-issues', issueFilters])

            // Optimistically update
            queryClient.setQueryData(['qa-db-issues', issueFilters], (old: { issues: DBIssue[]; total: number } | undefined) => {
                if (!old) return old
                return {
                    ...old,
                    issues: old.issues.map(issue =>
                        issue.id === issueId ? { ...issue, status: status as DBIssue['status'] } : issue
                    )
                }
            })

            toast.loading('Updating issue...', { id: `issue-${issueId}` })

            return { previousIssues }
        },
        onSuccess: (_, { issueId, status }) => {
            toast.success(`Issue marked as ${status}`, { id: `issue-${issueId}` })
            queryClient.invalidateQueries({ queryKey: ['qa-db-dashboard'] })
            if (selectedIssue) {
                queryClient.invalidateQueries({ queryKey: ['qa-db-issue', selectedIssue] })
            }
        },
        onError: (err, { issueId }, context) => {
            // Rollback on error
            if (context?.previousIssues) {
                queryClient.setQueryData(['qa-db-issues', issueFilters], context.previousIssues)
            }
            toast.error('Failed to update issue', { id: `issue-${issueId}` })
        }
    })

    // Bulk update mutation
    const bulkUpdateMutation = useMutation({
        mutationFn: async ({ issueIds, status }: { issueIds: number[]; status: string }) => {
            const promises = issueIds.map(id =>
                axiosInstance.patch(`/qa/db/issues/${id}/status`, { status })
            )
            return Promise.all(promises)
        },
        onMutate: () => {
            toast.loading(`Updating ${selectedIssues.size} issues...`, { id: 'bulk-update' })
        },
        onSuccess: (_, { status }) => {
            toast.success(`${selectedIssues.size} issues marked as ${status}`, { id: 'bulk-update' })
            setSelectedIssues(new Set())
            queryClient.invalidateQueries({ queryKey: ['qa-db-issues'] })
            queryClient.invalidateQueries({ queryKey: ['qa-db-dashboard'] })
        },
        onError: () => {
            toast.error('Failed to update some issues', { id: 'bulk-update' })
        }
    })

    // ============ Filtered Issues ============

    const filteredIssues = useMemo(() => {
        if (!dbIssues?.issues) return []
        let filtered = dbIssues.issues
        if (issueFilters.search) {
            const searchLower = issueFilters.search.toLowerCase()
            filtered = filtered.filter(issue =>
                issue.title.toLowerCase().includes(searchLower) ||
                issue.module_name.toLowerCase().includes(searchLower) ||
                issue.check_name.toLowerCase().includes(searchLower)
            )
        }
        return filtered
    }, [dbIssues?.issues, issueFilters.search])

    // ============ Bulk Selection Handlers ============

    const toggleIssueSelection = (issueId: number) => {
        setSelectedIssues(prev => {
            const next = new Set(prev)
            if (next.has(issueId)) {
                next.delete(issueId)
            } else {
                next.add(issueId)
            }
            return next
        })
    }

    const toggleAllIssues = () => {
        if (selectedIssues.size === filteredIssues.length) {
            setSelectedIssues(new Set())
        } else {
            setSelectedIssues(new Set(filteredIssues.map(i => i.id)))
        }
    }

    // ============ Helper Functions ============

    const getHealthColor = (score: number) => {
        if (score >= 80) return 'text-emerald-400'
        if (score >= 60) return 'text-yellow-400'
        return 'text-red-400'
    }

    const getHealthBgColor = (score: number) => {
        if (score >= 80) return 'from-emerald-500 to-green-600'
        if (score >= 60) return 'from-yellow-500 to-amber-600'
        return 'from-red-500 to-rose-600'
    }

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'critical': return 'text-red-400 bg-red-500/20'
            case 'high': return 'text-orange-400 bg-orange-500/20'
            case 'medium': return 'text-yellow-400 bg-yellow-500/20'
            case 'low': return 'text-blue-400 bg-blue-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'open': return 'text-red-400 bg-red-500/20 border-red-500/30'
            case 'fixing': return 'text-orange-400 bg-orange-500/20 border-orange-500/30'
            case 'fixed': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30'
            case 'ignored': return 'text-slate-400 bg-slate-500/20 border-slate-500/30'
            case 'wont_fix': return 'text-slate-400 bg-slate-500/20 border-slate-500/30'
            default: return 'text-slate-400 bg-slate-500/20 border-slate-500/30'
        }
    }

    const getCategoryIcon = (category: string) => {
        switch (category) {
            case 'auto_fix': return <Wrench className="w-4 h-4" />
            case 'enhancement': return <Sparkles className="w-4 h-4" />
            case 'issue_found': return <Bug className="w-4 h-4" />
            case 'learning': return <Brain className="w-4 h-4" />
            default: return <Activity className="w-4 h-4" />
        }
    }

    const formatTime = (isoString: string | null) => {
        if (!isoString) return 'Never'
        const date = new Date(isoString)
        return date.toLocaleString()
    }

    const formatDuration = (seconds: number | null) => {
        if (!seconds) return '-'
        if (seconds < 60) return `${seconds.toFixed(1)}s`
        return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`
    }

    const refreshAll = useCallback(() => {
        refetchSummary()
        refetchDbDashboard()
        queryClient.invalidateQueries({ queryKey: ['qa-db-issues'] })
        queryClient.invalidateQueries({ queryKey: ['qa-db-runs'] })
        toast.success('Data refreshed')
    }, [refetchSummary, refetchDbDashboard, queryClient])

    const categoryData = summary?.accomplishments_by_category
        ? Object.entries(summary.accomplishments_by_category).map(([name, value]) => ({
            name: name.replace('_', ' '),
            value
        }))
        : []

    const severityChartData = dbDashboard?.by_severity
        ? Object.entries(dbDashboard.by_severity).map(([name, value]) => ({
            name,
            value,
            fill: SEVERITY_COLORS[name as keyof typeof SEVERITY_COLORS] || '#6B7280'
        }))
        : []

    const statusChartData = dbDashboard?.by_status
        ? Object.entries(dbDashboard.by_status).map(([name, value]) => ({
            name: name.replace('_', ' '),
            value,
            fill: STATUS_COLORS[name as keyof typeof STATUS_COLORS] || '#6B7280'
        }))
        : []

    // ============ Render Functions ============

    const renderTabs = () => (
        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-xl mb-6">
            {[
                { id: 'overview', label: 'Overview', icon: BarChart3, shortcut: '1' },
                { id: 'issues', label: 'Issues', icon: Bug, count: dbDashboard?.open_issues, shortcut: '2' },
                { id: 'runs', label: 'Run History', icon: History, shortcut: '3' },
                { id: 'hotspots', label: 'Hot Spots', icon: FileWarning, shortcut: '4' },
                { id: 'ai-assistant', label: 'AI Assistant', icon: Bot, shortcut: '5' }
            ].map(tab => (
                <motion.button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as TabType)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all relative ${
                        activeTab === tab.id
                            ? 'bg-blue-500/20 text-blue-400'
                            : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                    {tab.count !== undefined && tab.count > 0 && (
                        <span className="px-2 py-0.5 text-xs rounded-full bg-red-500/20 text-red-400">
                            {tab.count}
                        </span>
                    )}
                    <kbd className="hidden md:inline-block ml-2 px-1.5 py-0.5 text-[10px] bg-slate-700/50 text-slate-500 rounded">
                        Alt+{tab.shortcut}
                    </kbd>
                </motion.button>
            ))}
        </div>
    )

    const renderOverviewTab = () => (
        <motion.div variants={staggerContainer} initial="initial" animate="animate">
            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {isLoading ? (
                    <>
                        <CardSkeleton />
                        <CardSkeleton />
                        <CardSkeleton />
                        <CardSkeleton />
                    </>
                ) : (
                    <>
                        {/* Health Score */}
                        <motion.div variants={fadeInUp} className="card p-5">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-slate-400 text-sm">Health Score</span>
                                <Target className={`w-5 h-5 ${getHealthColor(summary?.status?.health_score || 0)}`} />
                            </div>
                            <div className={`text-4xl font-bold ${getHealthColor(summary?.status?.health_score || 0)}`}>
                                {summary?.status?.health_score?.toFixed(1) || '0'}
                            </div>
                            <div className="text-slate-500 text-sm mt-1">out of 100</div>
                            <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${summary?.status?.health_score || 0}%` }}
                                    transition={{ duration: 1, ease: 'easeOut' }}
                                    className={`h-full rounded-full bg-gradient-to-r ${getHealthBgColor(summary?.status?.health_score || 0)}`}
                                />
                            </div>
                        </motion.div>

                        {/* Open Issues */}
                        <motion.div variants={fadeInUp} className="card p-5">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-slate-400 text-sm">Open Issues</span>
                                <AlertCircle className={`w-5 h-5 ${(dbDashboard?.open_issues || 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`} />
                            </div>
                            <div className={`text-4xl font-bold ${(dbDashboard?.open_issues || 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                {dbDashboard?.open_issues || 0}
                            </div>
                            <div className="text-slate-500 text-sm mt-1">
                                Total: {dbDashboard?.total_issues || 0}
                            </div>
                        </motion.div>

                        {/* Issues Fixed */}
                        <motion.div variants={fadeInUp} className="card p-5">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-slate-400 text-sm">Issues Fixed</span>
                                <CheckCircle className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div className="text-4xl font-bold text-emerald-400">
                                {dbDashboard?.fixed_issues || summary?.status?.total_issues_fixed || 0}
                            </div>
                            <div className="text-slate-500 text-sm mt-1">
                                Found: {summary?.status?.total_issues_found || 0}
                            </div>
                        </motion.div>

                        {/* QA Cycles */}
                        <motion.div variants={fadeInUp} className="card p-5">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-slate-400 text-sm">Total QA Cycles</span>
                                <Activity className="w-5 h-5 text-blue-400" />
                            </div>
                            <div className="text-4xl font-bold text-white">
                                {summary?.status?.total_cycles || 0}
                            </div>
                            <div className="text-slate-500 text-sm mt-1">
                                Last: {formatTime(summary?.status?.last_run_time)}
                            </div>
                        </motion.div>
                    </>
                )}
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {/* Health History Chart */}
                <motion.div variants={fadeInUp} className="card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-blue-400" />
                        Health Score Trend
                    </h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height={240} minWidth={200}>
                            <AreaChart data={dbHealthTrend?.trend || healthHistory?.history || []}>
                                <defs>
                                    <linearGradient id="healthGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                                        <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis
                                    dataKey="timestamp"
                                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                                    tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                />
                                <YAxis
                                    domain={[0, 100]}
                                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '8px'
                                    }}
                                    formatter={(value: number) => [value.toFixed(1), 'Score']}
                                    labelFormatter={(label) => new Date(label).toLocaleString()}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="avg_score"
                                    stroke="#10B981"
                                    fill="url(#healthGradient)"
                                    strokeWidth={2}
                                    name="Avg"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="score"
                                    stroke="#10B981"
                                    fill="url(#healthGradient)"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Issues by Severity/Status */}
                <motion.div variants={fadeInUp} className="card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-purple-400" />
                        Issues by Severity & Status
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <div className="text-sm text-slate-400 mb-2">By Severity</div>
                            {severityChartData.length > 0 ? (
                                <div className="space-y-2">
                                    {severityChartData.map(item => (
                                        <motion.div
                                            key={item.name}
                                            className="flex items-center gap-2"
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                        >
                                            <div
                                                className="w-3 h-3 rounded-full"
                                                style={{ backgroundColor: item.fill }}
                                            />
                                            <span className="text-slate-300 capitalize">{item.name}</span>
                                            <span className="ml-auto text-white font-semibold">{item.value}</span>
                                        </motion.div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-slate-500">No data</div>
                            )}
                        </div>
                        <div>
                            <div className="text-sm text-slate-400 mb-2">By Status</div>
                            {statusChartData.length > 0 ? (
                                <div className="space-y-2">
                                    {statusChartData.map(item => (
                                        <motion.div
                                            key={item.name}
                                            className="flex items-center gap-2"
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                        >
                                            <div
                                                className="w-3 h-3 rounded-full"
                                                style={{ backgroundColor: item.fill }}
                                            />
                                            <span className="text-slate-300 capitalize">{item.name}</span>
                                            <span className="ml-auto text-white font-semibold">{item.value}</span>
                                        </motion.div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-slate-500">No data</div>
                            )}
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Recent Activity & Top Issues */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Recent Accomplishments */}
                <motion.div variants={fadeInUp} className="card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-amber-400" />
                        Recent Accomplishments
                    </h3>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        <AnimatePresence>
                            {accomplishments?.accomplishments?.slice(0, 10).map((acc, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg"
                                >
                                    <div className={`p-2 rounded-lg ${getSeverityColor(acc.severity)}`}>
                                        {getCategoryIcon(acc.category)}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-white text-sm font-medium truncate">
                                            {acc.message}
                                        </div>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-xs text-slate-500">{acc.module}</span>
                                            <span className="text-xs text-slate-600">•</span>
                                            <span className="text-xs text-slate-500">
                                                {new Date(acc.timestamp).toLocaleTimeString()}
                                            </span>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {(!accomplishments?.accomplishments || accomplishments.accomplishments.length === 0) && (
                            <div className="text-center text-slate-500 py-8">
                                No accomplishments yet. Run QA to start!
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Top Open Issues */}
                <motion.div variants={fadeInUp} className="card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                        Top Open Issues
                    </h3>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        <AnimatePresence>
                            {dbDashboard?.top_issues?.slice(0, 10).map((issue, index) => (
                                <motion.div
                                    key={issue.id}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-700/50 transition-colors"
                                    onClick={() => setSelectedIssue(issue.id)}
                                    whileHover={{ scale: 1.01 }}
                                >
                                    <div className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                                        {issue.severity.toUpperCase()}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-white text-sm font-medium truncate">
                                            {issue.title}
                                        </div>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-xs text-slate-500">{issue.module_name}</span>
                                            <span className="text-xs text-slate-600">•</span>
                                            <span className="text-xs text-slate-500">
                                                {issue.occurrence_count}x
                                            </span>
                                        </div>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-slate-500" />
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {(!dbDashboard?.top_issues || dbDashboard.top_issues.length === 0) && (
                            <div className="text-center text-slate-500 py-8">
                                No open issues. Great job!
                            </div>
                        )}
                    </div>
                </motion.div>
            </div>
        </motion.div>
    )

    const renderIssuesTab = () => (
        <motion.div variants={fadeInUp} initial="initial" animate="animate">
            {/* Filters & Bulk Actions */}
            <div className="card p-4 mb-6">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Search className="w-4 h-4 text-slate-400" />
                        <input
                            type="text"
                            placeholder="Search issues..."
                            value={issueFilters.search}
                            onChange={(e) => setIssueFilters({ ...issueFilters, search: e.target.value })}
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                        />
                    </div>
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <select
                            value={issueFilters.status}
                            onChange={(e) => setIssueFilters({ ...issueFilters, status: e.target.value })}
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                        >
                            <option value="all">All Status</option>
                            <option value="open">Open</option>
                            <option value="fixing">Fixing</option>
                            <option value="fixed">Fixed</option>
                            <option value="ignored">Ignored</option>
                            <option value="wont_fix">Won't Fix</option>
                        </select>
                        <select
                            value={issueFilters.severity}
                            onChange={(e) => setIssueFilters({ ...issueFilters, severity: e.target.value })}
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
                        >
                            <option value="all">All Severity</option>
                            <option value="critical">Critical</option>
                            <option value="high">High</option>
                            <option value="medium">Medium</option>
                            <option value="low">Low</option>
                        </select>
                    </div>

                    {/* Bulk Actions */}
                    <AnimatePresence>
                        {selectedIssues.size > 0 && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="flex items-center gap-2 ml-auto"
                            >
                                <span className="text-sm text-slate-400">
                                    {selectedIssues.size} selected
                                </span>
                                <button
                                    onClick={() => bulkUpdateMutation.mutate({
                                        issueIds: Array.from(selectedIssues),
                                        status: 'fixed'
                                    })}
                                    className="px-3 py-1.5 bg-emerald-500/20 text-emerald-400 rounded-lg hover:bg-emerald-500/30 text-sm flex items-center gap-1"
                                >
                                    <CheckCircle className="w-4 h-4" />
                                    Mark Fixed
                                </button>
                                <button
                                    onClick={() => bulkUpdateMutation.mutate({
                                        issueIds: Array.from(selectedIssues),
                                        status: 'ignored'
                                    })}
                                    className="px-3 py-1.5 bg-slate-500/20 text-slate-400 rounded-lg hover:bg-slate-500/30 text-sm flex items-center gap-1"
                                >
                                    <EyeOff className="w-4 h-4" />
                                    Ignore
                                </button>
                                <button
                                    onClick={() => setSelectedIssues(new Set())}
                                    className="p-1.5 text-slate-400 hover:text-white"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    <div className="ml-auto text-sm text-slate-400">
                        {filteredIssues.length} issues found
                    </div>
                </div>
            </div>

            {/* Issues List */}
            <div className="card overflow-hidden">
                <table className="w-full">
                    <thead className="bg-slate-800/50">
                        <tr>
                            <th className="text-left p-4 text-slate-400 font-medium w-12">
                                <button
                                    onClick={toggleAllIssues}
                                    className="p-1 hover:bg-slate-700/50 rounded"
                                >
                                    {selectedIssues.size === filteredIssues.length && filteredIssues.length > 0 ? (
                                        <CheckSquare className="w-4 h-4 text-blue-400" />
                                    ) : (
                                        <Square className="w-4 h-4" />
                                    )}
                                </button>
                            </th>
                            <th className="text-left p-4 text-slate-400 font-medium">Severity</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Status</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Issue</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Module</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Count</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Last Seen</th>
                            <th className="text-left p-4 text-slate-400 font-medium">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {isLoadingIssues ? (
                            <>
                                <TableRowSkeleton />
                                <TableRowSkeleton />
                                <TableRowSkeleton />
                                <TableRowSkeleton />
                                <TableRowSkeleton />
                            </>
                        ) : (
                            <AnimatePresence>
                                {filteredIssues.map((issue, index) => (
                                    <motion.tr
                                        key={issue.id}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -10 }}
                                        transition={{ delay: index * 0.02 }}
                                        className={`border-t border-slate-700/50 hover:bg-slate-800/30 cursor-pointer ${
                                            selectedIssues.has(issue.id) ? 'bg-blue-500/10' : ''
                                        }`}
                                        onClick={() => setSelectedIssue(issue.id)}
                                    >
                                        <td className="p-4" onClick={(e) => e.stopPropagation()}>
                                            <button
                                                onClick={() => toggleIssueSelection(issue.id)}
                                                className="p-1 hover:bg-slate-700/50 rounded"
                                            >
                                                {selectedIssues.has(issue.id) ? (
                                                    <CheckSquare className="w-4 h-4 text-blue-400" />
                                                ) : (
                                                    <Square className="w-4 h-4 text-slate-500" />
                                                )}
                                            </button>
                                        </td>
                                        <td className="p-4">
                                            <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                                                {issue.severity.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <span className={`px-2 py-1 rounded border text-xs font-medium ${getStatusColor(issue.status)}`}>
                                                {issue.status.replace('_', ' ')}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="text-white text-sm font-medium max-w-md truncate">
                                                {issue.title}
                                            </div>
                                            <div className="text-xs text-slate-500 mt-1">{issue.check_name}</div>
                                        </td>
                                        <td className="p-4 text-slate-300">{issue.module_name}</td>
                                        <td className="p-4 text-white font-mono">{issue.occurrence_count}</td>
                                        <td className="p-4 text-slate-400 text-sm">
                                            {new Date(issue.last_seen_at).toLocaleDateString()}
                                        </td>
                                        <td className="p-4">
                                            <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                                                {issue.status === 'open' && (
                                                    <>
                                                        <button
                                                            onClick={() => updateIssueStatusMutation.mutate({ issueId: issue.id, status: 'fixed' })}
                                                            className="p-1 rounded hover:bg-emerald-500/20 text-emerald-400"
                                                            title="Mark as Fixed"
                                                        >
                                                            <CheckCircle className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => updateIssueStatusMutation.mutate({ issueId: issue.id, status: 'ignored' })}
                                                            className="p-1 rounded hover:bg-slate-500/20 text-slate-400"
                                                            title="Ignore"
                                                        >
                                                            <EyeOff className="w-4 h-4" />
                                                        </button>
                                                    </>
                                                )}
                                                {issue.status === 'fixed' && (
                                                    <button
                                                        onClick={() => updateIssueStatusMutation.mutate({ issueId: issue.id, status: 'open' })}
                                                        className="p-1 rounded hover:bg-red-500/20 text-red-400"
                                                        title="Reopen"
                                                    >
                                                        <AlertCircle className="w-4 h-4" />
                                                    </button>
                                                )}
                                                <button
                                                    onClick={() => setSelectedIssue(issue.id)}
                                                    className="p-1 rounded hover:bg-blue-500/20 text-blue-400"
                                                    title="View Details"
                                                >
                                                    <Eye className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </motion.tr>
                                ))}
                            </AnimatePresence>
                        )}
                    </tbody>
                </table>
                {filteredIssues.length === 0 && !isLoadingIssues && (
                    <div className="text-center text-slate-500 py-12">
                        No issues found matching your filters
                    </div>
                )}
            </div>
        </motion.div>
    )

    const renderRunsTab = () => (
        <motion.div
            variants={fadeInUp}
            initial="initial"
            animate="animate"
            className="grid grid-cols-1 lg:grid-cols-3 gap-6"
        >
            {/* Runs List */}
            <div className="lg:col-span-2 card overflow-hidden">
                <div className="p-4 border-b border-slate-700/50">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <History className="w-5 h-5 text-blue-400" />
                        QA Run History
                    </h3>
                </div>
                <div className="max-h-[600px] overflow-y-auto">
                    <AnimatePresence>
                        {dbRuns?.runs?.map((run, index) => (
                            <motion.div
                                key={run.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.03 }}
                                className={`p-4 border-b border-slate-700/30 hover:bg-slate-800/30 cursor-pointer transition-colors ${
                                    selectedRun === run.id ? 'bg-blue-500/10 border-l-2 border-l-blue-500' : ''
                                }`}
                                onClick={() => setSelectedRun(run.id)}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2 h-2 rounded-full ${
                                            run.status === 'completed' ? 'bg-emerald-500' :
                                            run.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                                        }`} />
                                        <span className="text-white font-medium">Run #{run.run_id.slice(0, 8)}</span>
                                        <span className="text-xs text-slate-500">{run.triggered_by}</span>
                                    </div>
                                    <span className="text-sm text-slate-400">
                                        {formatDuration(run.duration_seconds)}
                                    </span>
                                </div>
                                <div className="grid grid-cols-4 gap-4 text-sm">
                                    <div>
                                        <div className="text-slate-500">Checks</div>
                                        <div className="text-white">
                                            <span className="text-emerald-400">{run.passed_checks}</span>
                                            <span className="text-slate-500">/</span>
                                            <span>{run.total_checks}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-slate-500">Issues</div>
                                        <div className="text-white">
                                            <span className="text-red-400">{run.issues_found}</span>
                                            <span className="text-slate-500"> found</span>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-slate-500">Fixed</div>
                                        <div className="text-emerald-400">{run.issues_fixed}</div>
                                    </div>
                                    <div>
                                        <div className="text-slate-500">Score</div>
                                        <div className={getHealthColor(run.health_score_after || 0)}>
                                            {run.health_score_after?.toFixed(1) || '-'}
                                        </div>
                                    </div>
                                </div>
                                <div className="mt-2 text-xs text-slate-500">
                                    {new Date(run.started_at).toLocaleString()}
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {(!dbRuns?.runs || dbRuns.runs.length === 0) && (
                        <div className="text-center text-slate-500 py-12">
                            No QA runs recorded yet
                        </div>
                    )}
                </div>
            </div>

            {/* Run Details */}
            <div className="card p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <FileCode className="w-5 h-5 text-purple-400" />
                    Run Details
                </h3>
                {runDetail ? (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="space-y-4"
                    >
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <div className="text-slate-400 text-xs">Status</div>
                                <div className={`font-medium ${
                                    runDetail.status === 'completed' ? 'text-emerald-400' :
                                    runDetail.status === 'failed' ? 'text-red-400' : 'text-yellow-400'
                                }`}>
                                    {runDetail.status}
                                </div>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <div className="text-slate-400 text-xs">Duration</div>
                                <div className="text-white font-medium">
                                    {formatDuration(runDetail.duration_seconds)}
                                </div>
                            </div>
                        </div>

                        <div>
                            <div className="text-slate-400 text-sm mb-2">Check Results</div>
                            <div className="space-y-2 max-h-80 overflow-y-auto">
                                {runDetail.check_results?.map((check, index) => (
                                    <motion.div
                                        key={check.id}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.02 }}
                                        className="flex items-center gap-2 p-2 bg-slate-800/30 rounded"
                                    >
                                        {check.status === 'passed' ? (
                                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                                        ) : check.status === 'failed' ? (
                                            <XCircle className="w-4 h-4 text-red-400" />
                                        ) : (
                                            <AlertTriangle className="w-4 h-4 text-yellow-400" />
                                        )}
                                        <div className="flex-1 min-w-0">
                                            <div className="text-white text-sm truncate">{check.check_name}</div>
                                            <div className="text-xs text-slate-500">{check.module_name}</div>
                                        </div>
                                        {check.duration_ms && (
                                            <span className="text-xs text-slate-500">{check.duration_ms}ms</span>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <div className="text-center text-slate-500 py-8">
                        Select a run to view details
                    </div>
                )}
            </div>
        </motion.div>
    )

    const renderHotspotsTab = () => (
        <motion.div
            variants={fadeInUp}
            initial="initial"
            animate="animate"
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
            {/* Hot Spots List */}
            <div className="card p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <FileWarning className="w-5 h-5 text-red-400" />
                    Problem Files (Hot Spots)
                </h3>
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                    <AnimatePresence>
                        {(dbHotSpots?.hot_spots || summary?.top_hot_spots)?.map((spot, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.05 }}
                                className="p-4 bg-slate-800/50 rounded-lg"
                            >
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="text-2xl font-bold text-slate-600 w-8">
                                        #{index + 1}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-white font-medium truncate">
                                            {spot.file_path || spot.file}
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4 text-sm">
                                    <div className="flex items-center gap-1">
                                        <Bug className="w-4 h-4 text-red-400" />
                                        <span className="text-slate-400">{spot.issue_count} issues</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <AlertTriangle className="w-4 h-4 text-orange-400" />
                                        <span className="text-slate-400">Score: {spot.severity_score}</span>
                                    </div>
                                </div>
                                {spot.patterns?.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-3">
                                        {spot.patterns.slice(0, 5).map((pattern, i) => (
                                            <span
                                                key={i}
                                                className="px-2 py-0.5 text-xs bg-slate-700 text-slate-400 rounded"
                                            >
                                                {pattern}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {(!dbHotSpots?.hot_spots && !summary?.top_hot_spots) && (
                        <div className="text-center text-slate-500 py-8">
                            No hot spots detected. Great job!
                        </div>
                    )}
                </div>
            </div>

            {/* Pattern Statistics */}
            <div className="card p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Brain className="w-5 h-5 text-violet-400" />
                    Pattern Learning Statistics
                </h3>
                <div className="grid grid-cols-2 gap-4 mb-6">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-slate-800/50 rounded-lg p-4"
                    >
                        <div className="text-slate-400 text-sm">Total Patterns</div>
                        <div className="text-2xl font-bold text-white mt-1">
                            {summary?.pattern_statistics?.total_patterns || 0}
                        </div>
                    </motion.div>
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.1 }}
                        className="bg-slate-800/50 rounded-lg p-4"
                    >
                        <div className="text-slate-400 text-sm">Total Occurrences</div>
                        <div className="text-2xl font-bold text-white mt-1">
                            {summary?.pattern_statistics?.total_occurrences || 0}
                        </div>
                    </motion.div>
                </div>

                {/* Activity by Category */}
                <h4 className="text-slate-400 text-sm mb-3">Activity by Category</h4>
                <div className="h-64">
                    {categoryData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={240} minWidth={200}>
                            <PieChart>
                                <Pie
                                    data={categoryData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                    label={({ name, value }) => `${name}: ${value}`}
                                >
                                    {categoryData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">
                            No data yet
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    )

    const renderIssueModal = () => {
        if (!selectedIssue) return null

        return (
            <AnimatePresence>
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedIssue(null)}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="bg-slate-900 rounded-xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-slate-700/50"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {isLoadingDetail ? (
                            <div className="p-6 flex items-center justify-center">
                                <RefreshCw className="w-8 h-8 animate-spin text-blue-400" />
                            </div>
                        ) : issueDetail ? (
                            <>
                                {/* Header */}
                                <div className="p-6 border-b border-slate-700/50">
                                    <div className="flex items-start justify-between">
                                        <div>
                                            <div className="flex items-center gap-3 mb-2">
                                                <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(issueDetail.severity)}`}>
                                                    {issueDetail.severity.toUpperCase()}
                                                </span>
                                                <span className={`px-2 py-1 rounded border text-xs font-medium ${getStatusColor(issueDetail.status)}`}>
                                                    {issueDetail.status.replace('_', ' ')}
                                                </span>
                                            </div>
                                            <h2 className="text-xl font-semibold text-white">{issueDetail.title}</h2>
                                            <div className="text-sm text-slate-400 mt-1">
                                                {issueDetail.module_name} → {issueDetail.check_name}
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => setSelectedIssue(null)}
                                            className="p-2 hover:bg-slate-700/50 rounded-lg"
                                        >
                                            <X className="w-5 h-5 text-slate-400" />
                                        </button>
                                    </div>
                                </div>

                                {/* Content */}
                                <div className="p-6 max-h-[calc(90vh-200px)] overflow-y-auto">
                                    {/* Description */}
                                    {issueDetail.description && (
                                        <div className="mb-6">
                                            <h3 className="text-sm font-medium text-slate-400 mb-2">Description</h3>
                                            <p className="text-white bg-slate-800/50 p-4 rounded-lg">
                                                {issueDetail.description}
                                            </p>
                                        </div>
                                    )}

                                    {/* Stats */}
                                    <div className="grid grid-cols-4 gap-4 mb-6">
                                        <div className="bg-slate-800/50 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs">Occurrences</div>
                                            <div className="text-white font-bold">{issueDetail.occurrence_count}</div>
                                        </div>
                                        <div className="bg-slate-800/50 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs">First Seen</div>
                                            <div className="text-white text-sm">
                                                {new Date(issueDetail.first_seen_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                        <div className="bg-slate-800/50 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs">Last Seen</div>
                                            <div className="text-white text-sm">
                                                {new Date(issueDetail.last_seen_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                        <div className="bg-slate-800/50 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs">Auto-fixable</div>
                                            <div className={`font-bold ${issueDetail.auto_fixable ? 'text-emerald-400' : 'text-slate-400'}`}>
                                                {issueDetail.auto_fixable ? 'Yes' : 'No'}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Affected Files */}
                                    {issueDetail.affected_files?.length > 0 && (
                                        <div className="mb-6">
                                            <h3 className="text-sm font-medium text-slate-400 mb-2">Affected Files</h3>
                                            <div className="bg-slate-800/50 rounded-lg p-3 space-y-1">
                                                {issueDetail.affected_files.map((file, i) => (
                                                    <div key={i} className="text-sm text-white font-mono truncate">
                                                        {file}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Suggested Fix */}
                                    {issueDetail.suggested_fix && (
                                        <div className="mb-6">
                                            <h3 className="text-sm font-medium text-slate-400 mb-2 flex items-center gap-2">
                                                <Lightbulb className="w-4 h-4 text-yellow-400" />
                                                Suggested Fix
                                            </h3>
                                            <div className="bg-slate-800/50 rounded-lg p-4 text-white">
                                                {issueDetail.suggested_fix}
                                            </div>
                                        </div>
                                    )}

                                    {/* Fix History */}
                                    {issueDetail.fixes?.length > 0 && (
                                        <div className="mb-6">
                                            <h3 className="text-sm font-medium text-slate-400 mb-2">Fix History</h3>
                                            <div className="space-y-2">
                                                {issueDetail.fixes.map((fix) => (
                                                    <div key={fix.id} className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                                                        {fix.success ? (
                                                            <CheckCircle className="w-5 h-5 text-emerald-400 mt-0.5" />
                                                        ) : (
                                                            <XCircle className="w-5 h-5 text-red-400 mt-0.5" />
                                                        )}
                                                        <div>
                                                            <div className="text-white font-medium">{fix.fix_type}</div>
                                                            {fix.description && (
                                                                <div className="text-slate-400 text-sm">{fix.description}</div>
                                                            )}
                                                            <div className="text-slate-500 text-xs mt-1">
                                                                {new Date(fix.attempted_at).toLocaleString()}
                                                                {fix.commit_hash && (
                                                                    <span className="ml-2 font-mono">
                                                                        {fix.commit_hash.slice(0, 7)}
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Tags */}
                                    {issueDetail.tags?.length > 0 && (
                                        <div>
                                            <h3 className="text-sm font-medium text-slate-400 mb-2">Tags</h3>
                                            <div className="flex flex-wrap gap-2">
                                                {issueDetail.tags.map((tag, i) => (
                                                    <span
                                                        key={i}
                                                        className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-sm"
                                                    >
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Actions Footer */}
                                <div className="p-6 border-t border-slate-700/50 flex items-center justify-end gap-3">
                                    {issueDetail.status === 'open' && (
                                        <>
                                            <button
                                                onClick={() => {
                                                    updateIssueStatusMutation.mutate({ issueId: issueDetail.id, status: 'fixing' })
                                                }}
                                                className="px-4 py-2 bg-orange-500/20 text-orange-400 rounded-lg hover:bg-orange-500/30 transition-colors"
                                            >
                                                Mark as Fixing
                                            </button>
                                            <button
                                                onClick={() => {
                                                    updateIssueStatusMutation.mutate({ issueId: issueDetail.id, status: 'fixed' })
                                                    setSelectedIssue(null)
                                                }}
                                                className="px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-lg hover:bg-emerald-500/30 transition-colors"
                                            >
                                                Mark as Fixed
                                            </button>
                                            <button
                                                onClick={() => {
                                                    updateIssueStatusMutation.mutate({ issueId: issueDetail.id, status: 'ignored' })
                                                    setSelectedIssue(null)
                                                }}
                                                className="px-4 py-2 bg-slate-500/20 text-slate-400 rounded-lg hover:bg-slate-500/30 transition-colors"
                                            >
                                                Ignore
                                            </button>
                                        </>
                                    )}
                                    {issueDetail.status === 'fixing' && (
                                        <button
                                            onClick={() => {
                                                updateIssueStatusMutation.mutate({ issueId: issueDetail.id, status: 'fixed' })
                                                setSelectedIssue(null)
                                            }}
                                            className="px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-lg hover:bg-emerald-500/30 transition-colors"
                                        >
                                            Mark as Fixed
                                        </button>
                                    )}
                                    {(issueDetail.status === 'fixed' || issueDetail.status === 'ignored') && (
                                        <button
                                            onClick={() => {
                                                updateIssueStatusMutation.mutate({ issueId: issueDetail.id, status: 'open' })
                                            }}
                                            className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                                        >
                                            Reopen Issue
                                        </button>
                                    )}
                                    <button
                                        onClick={() => setSelectedIssue(null)}
                                        className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
                                    >
                                        Close
                                    </button>
                                </div>
                            </>
                        ) : (
                            <div className="p-6 text-center text-slate-500">
                                Issue not found
                            </div>
                        )}
                    </motion.div>
                </motion.div>
            </AnimatePresence>
        )
    }

    // ============ Main Render ============

    return (
        <div className="space-y-6">
            {/* Toast Notifications */}
            <Toaster
                position="top-right"
                toastOptions={{
                    style: {
                        background: '#1e293b',
                        border: '1px solid #334155',
                        color: '#f1f5f9'
                    }
                }}
            />

            {/* Command Palette */}
            <CommandPalette
                isOpen={commandPaletteOpen}
                onClose={() => setCommandPaletteOpen(false)}
                onRunQA={() => runQAMutation.mutate()}
                onSetTab={setActiveTab}
                onRefresh={refreshAll}
            />

            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <motion.div
                            className={`w-10 h-10 rounded-xl bg-gradient-to-br ${getHealthBgColor(summary?.status?.health_score || 0)} flex items-center justify-center shadow-lg`}
                            whileHover={{ scale: 1.05 }}
                        >
                            <Shield className="w-5 h-5 text-white" />
                        </motion.div>
                        QA Dashboard
                    </h1>
                    <p className="page-subtitle flex items-center gap-2">
                        Continuous Quality Assurance & Enhancement System
                        {dbStatus?.available && (
                            <span className="flex items-center gap-1 text-emerald-400 text-xs">
                                <Database className="w-3 h-3" />
                                DB Connected
                            </span>
                        )}
                        {wsConnected ? (
                            <span className="flex items-center gap-1 text-purple-400 text-xs">
                                <Activity className="w-3 h-3" />
                                Live {activeConnections > 1 ? `(${activeConnections})` : ''}
                            </span>
                        ) : useWebSocket ? (
                            <span className="flex items-center gap-1 text-yellow-400 text-xs cursor-pointer" onClick={wsReconnect}>
                                <AlertTriangle className="w-3 h-3" />
                                Reconnecting...
                            </span>
                        ) : null}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {/* WebSocket Toggle */}
                    <button
                        onClick={() => setUseWebSocket(!useWebSocket)}
                        className={`btn-secondary flex items-center gap-2 ${wsConnected ? 'bg-purple-500/20 text-purple-400' : ''}`}
                        title={useWebSocket ? 'Disable real-time updates' : 'Enable real-time updates'}
                    >
                        <Activity className={`w-4 h-4 ${wsConnected ? 'animate-pulse' : ''}`} />
                        {wsConnected ? 'Live' : 'Offline'}
                    </button>
                    <button
                        onClick={() => setCommandPaletteOpen(true)}
                        className="btn-secondary flex items-center gap-2"
                    >
                        <Command className="w-4 h-4" />
                        <kbd className="px-1.5 py-0.5 text-xs bg-slate-700 rounded">⌘K</kbd>
                    </button>
                    <button
                        onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`btn-secondary flex items-center gap-2 ${autoRefresh ? 'bg-emerald-500/20 text-emerald-400' : ''}`}
                    >
                        <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
                        {autoRefresh ? 'Auto' : 'Manual'}
                    </button>
                    <motion.button
                        onClick={() => runQAMutation.mutate()}
                        disabled={runQAMutation.isPending}
                        className="btn-primary flex items-center gap-2"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        <Play className="w-4 h-4" />
                        {runQAMutation.isPending ? 'Running...' : 'Run QA Now'}
                    </motion.button>
                </div>
            </header>

            {isLoading ? (
                <div className="flex items-center justify-center h-64">
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    >
                        <RefreshCw className="w-8 h-8 text-blue-400" />
                    </motion.div>
                </div>
            ) : (
                <>
                    {renderTabs()}

                    <AnimatePresence mode="wait">
                        {activeTab === 'overview' && renderOverviewTab()}
                        {activeTab === 'issues' && renderIssuesTab()}
                        {activeTab === 'runs' && renderRunsTab()}
                        {activeTab === 'hotspots' && renderHotspotsTab()}
                        {activeTab === 'ai-assistant' && <AIAssistantTab issues={dbIssues?.issues || []} />}
                    </AnimatePresence>
                </>
            )}

            {/* Issue Detail Modal */}
            {renderIssueModal()}

            {/* Keyboard Shortcuts Help */}
            <div className="fixed bottom-4 right-4 text-xs text-slate-500 flex items-center gap-2">
                <Keyboard className="w-3 h-3" />
                <span>Press</span>
                <kbd className="px-1.5 py-0.5 bg-slate-800 rounded">⌘K</kbd>
                <span>for commands</span>
            </div>
        </div>
    )
}
