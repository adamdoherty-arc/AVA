import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Bot, RefreshCw, Search, Play, CheckCircle, XCircle,
    AlertCircle, Activity, Clock, Zap, ChevronDown, ChevronRight,
    Brain, Target, TrendingUp, Shield, MessageSquare, Settings,
    Database, BarChart3, Users
} from 'lucide-react'
import clsx from 'clsx'

interface Agent {
    name: string
    display_name: string
    description: string
    category: string
    status: 'active' | 'idle' | 'error'
    capabilities: string[]
    last_invoked: string | null
    invocation_count: number
    avg_response_time_ms: number
    success_rate: number
}

interface AgentCategory {
    name: string
    agents: Agent[]
    icon: string
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
    'Trading': <TrendingUp className="w-4 h-4" />,
    'Analysis': <Brain className="w-4 h-4" />,
    'Sports': <Target className="w-4 h-4" />,
    'Monitoring': <Activity className="w-4 h-4" />,
    'Research': <Database className="w-4 h-4" />,
    'System': <Settings className="w-4 h-4" />,
}

export default function AgentManagement() {
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
    const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
    const [testInput, setTestInput] = useState('')

    const { data: agents, isLoading, refetch, error } = useQuery<Agent[]>({
        queryKey: ['agents'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/agents')
            return data.agents || data
        },
        staleTime: 60000,
    })

    const { data: categories } = useQuery<string[]>({
        queryKey: ['agent-categories'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/agents/categories')
            return data.categories || data
        },
    })

    const invokeMutation = useMutation({
        mutationFn: async ({ agentName, input }: { agentName: string; input: string }) => {
            const { data } = await axiosInstance.post(`/agents/${agentName}/invoke`, { query: input })
            return data
        }
    })

    const filteredAgents = agents?.filter(agent => {
        const matchesSearch = searchQuery === '' ||
            agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            agent.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            agent.description.toLowerCase().includes(searchQuery.toLowerCase())
        const matchesCategory = selectedCategory === null || agent.category === selectedCategory
        return matchesSearch && matchesCategory
    }) || []

    const agentsByCategory = filteredAgents.reduce((acc, agent) => {
        if (!acc[agent.category]) acc[agent.category] = []
        acc[agent.category].push(agent)
        return acc
    }, {} as Record<string, Agent[]>)

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active': return 'text-emerald-400 bg-emerald-400/20'
            case 'idle': return 'text-amber-400 bg-amber-400/20'
            case 'error': return 'text-red-400 bg-red-400/20'
            default: return 'text-slate-400 bg-slate-400/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'active': return <CheckCircle className="w-3 h-3" />
            case 'idle': return <Clock className="w-3 h-3" />
            case 'error': return <XCircle className="w-3 h-3" />
            default: return <AlertCircle className="w-3 h-3" />
        }
    }

    const totalAgents = agents?.length || 0
    const activeAgents = agents?.filter(a => a.status === 'active').length || 0
    const avgSuccessRate = (agents?.reduce((acc, a) => acc + a.success_rate, 0) ?? 0) / (totalAgents || 1)

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        Agent Management
                    </h1>
                    <p className="page-subtitle">Monitor and manage AI agents</p>
                </div>
                <button
                    onClick={() => refetch()}
                    disabled={isLoading}
                    className="btn-icon"
                >
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Users className="w-4 h-4" />
                        <span className="text-sm">Total Agents</span>
                    </div>
                    <p className="text-2xl font-bold">{totalAgents}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Zap className="w-4 h-4" />
                        <span className="text-sm">Active Now</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{activeAgents}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-sm">Avg Success Rate</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">{avgSuccessRate.toFixed(1)}%</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Database className="w-4 h-4" />
                        <span className="text-sm">Categories</span>
                    </div>
                    <p className="text-2xl font-bold">{categories?.length || 0}</p>
                </div>
            </div>

            {/* Search and Filters */}
            <div className="card p-4">
                <div className="flex flex-col md:flex-row gap-4">
                    {/* Search */}
                    <div className="flex-1 relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                        <input
                            type="text"
                            placeholder="Search agents..."
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                    </div>

                    {/* Category Filter */}
                    <div className="flex gap-2 flex-wrap">
                        <button
                            onClick={() => setSelectedCategory(null)}
                            className={clsx(
                                "px-3 py-2 rounded-lg text-sm font-medium transition-all",
                                selectedCategory === null
                                    ? "bg-primary text-white"
                                    : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50"
                            )}
                        >
                            All
                        </button>
                        {categories?.map(category => (
                            <button
                                key={category}
                                onClick={() => setSelectedCategory(category)}
                                className={clsx(
                                    "px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1",
                                    selectedCategory === category
                                        ? "bg-primary text-white"
                                        : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50"
                                )}
                            >
                                {CATEGORY_ICONS[category] || <Bot className="w-4 h-4" />}
                                {category}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Agent List */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading agents...</span>
                </div>
            ) : error ? (
                <div className="card p-8 flex items-center justify-center text-red-400">
                    <AlertCircle className="w-6 h-6 mr-2" />
                    Failed to load agents
                </div>
            ) : filteredAgents.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No agents found matching your search</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {Object.entries(agentsByCategory).map(([category, categoryAgents]) => (
                        <section key={category}>
                            <div className="flex items-center gap-2 mb-3">
                                {CATEGORY_ICONS[category] || <Bot className="w-5 h-5" />}
                                <h2 className="text-lg font-semibold">{category}</h2>
                                <span className="px-2 py-0.5 rounded-full bg-slate-700 text-slate-400 text-xs">
                                    {categoryAgents.length}
                                </span>
                            </div>
                            <div className="grid gap-3">
                                {categoryAgents.map(agent => (
                                    <AgentCard
                                        key={agent.name}
                                        agent={agent}
                                        isExpanded={expandedAgent === agent.name}
                                        onToggle={() => setExpandedAgent(
                                            expandedAgent === agent.name ? null : agent.name
                                        )}
                                        getStatusColor={getStatusColor}
                                        getStatusIcon={getStatusIcon}
                                        onInvoke={(input) => {
                                            invokeMutation.mutate({ agentName: agent.name, input })
                                        }}
                                        isInvoking={invokeMutation.isPending}
                                        invokeResult={invokeMutation.data}
                                    />
                                ))}
                            </div>
                        </section>
                    ))}
                </div>
            )}
        </div>
    )
}

function AgentCard({
    agent,
    isExpanded,
    onToggle,
    getStatusColor,
    getStatusIcon,
    onInvoke,
    isInvoking,
    invokeResult
}: {
    agent: Agent
    isExpanded: boolean
    onToggle: () => void
    getStatusColor: (status: string) => string
    getStatusIcon: (status: string) => React.ReactNode
    onInvoke: (input: string) => void
    isInvoking: boolean
    invokeResult: any
}) {
    const [testInput, setTestInput] = useState('')

    return (
        <div className="card overflow-hidden">
            <button
                onClick={onToggle}
                className="w-full p-4 flex items-center gap-4 hover:bg-slate-800/50 transition-colors"
            >
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-700 to-slate-600 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-primary" />
                </div>

                <div className="flex-1 text-left">
                    <div className="flex items-center gap-2">
                        <span className="font-semibold">{agent.display_name}</span>
                        <span className={clsx(
                            "px-2 py-0.5 rounded-full text-xs flex items-center gap-1",
                            getStatusColor(agent.status)
                        )}>
                            {getStatusIcon(agent.status)}
                            {agent.status}
                        </span>
                    </div>
                    <p className="text-sm text-slate-400 line-clamp-1">{agent.description}</p>
                </div>

                <div className="flex items-center gap-6 text-sm">
                    <div className="text-center">
                        <p className="text-slate-400">Calls</p>
                        <p className="font-mono font-medium">{agent.invocation_count}</p>
                    </div>
                    <div className="text-center">
                        <p className="text-slate-400">Avg Time</p>
                        <p className="font-mono font-medium">{agent.avg_response_time_ms}ms</p>
                    </div>
                    <div className="text-center">
                        <p className="text-slate-400">Success</p>
                        <p className={clsx(
                            "font-mono font-medium",
                            agent.success_rate >= 90 ? "text-emerald-400" :
                            agent.success_rate >= 70 ? "text-amber-400" : "text-red-400"
                        )}>
                            {agent.success_rate.toFixed(0)}%
                        </p>
                    </div>
                    <ChevronDown className={clsx(
                        "w-5 h-5 text-slate-400 transition-transform",
                        isExpanded && "rotate-180"
                    )} />
                </div>
            </button>

            {isExpanded && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700/50 space-y-4">
                    {/* Capabilities */}
                    {agent.capabilities?.length > 0 && (
                        <div>
                            <p className="text-xs text-slate-400 mb-2">Capabilities</p>
                            <div className="flex flex-wrap gap-1">
                                {agent.capabilities.map(cap => (
                                    <span
                                        key={cap}
                                        className="px-2 py-1 rounded bg-slate-700/50 text-xs text-slate-300"
                                    >
                                        {cap}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Test Agent */}
                    <div>
                        <p className="text-xs text-slate-400 mb-2">Test Agent</p>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={testInput}
                                onChange={e => setTestInput(e.target.value)}
                                placeholder="Enter test query..."
                                className="flex-1 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm"
                            />
                            <button
                                onClick={() => onInvoke(testInput)}
                                disabled={isInvoking || !testInput}
                                className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-50"
                            >
                                {isInvoking ? (
                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Play className="w-4 h-4" />
                                )}
                                Test
                            </button>
                        </div>
                    </div>

                    {/* Last Invoked */}
                    {agent.last_invoked && (
                        <p className="text-xs text-slate-400">
                            Last invoked: {new Date(agent.last_invoked).toLocaleString()}
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}
