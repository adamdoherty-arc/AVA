import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Sparkles, RefreshCw, Plus, CheckCircle, Clock, AlertCircle,
    Play, Pause, Trash2, Bot, TestTube, ArrowRight, Filter,
    Calendar, Tag, User
} from 'lucide-react'

interface Enhancement {
    id: string
    title: string
    description: string
    status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'testing'
    priority: 'low' | 'medium' | 'high' | 'critical'
    category: string
    created_at: string
    updated_at: string
    assigned_to: string
    progress: number
    estimated_hours: number
    actual_hours: number
    tags: string[]
}

interface EnhancementStats {
    total: number
    pending: number
    in_progress: number
    completed: number
    failed: number
}

export default function EnhancementManager() {
    const queryClient = useQueryClient()
    const [filter, setFilter] = useState<string>('all')
    const [priorityFilter, setPriorityFilter] = useState<string>('all')

    const { data: stats } = useQuery<EnhancementStats>({
        queryKey: ['enhancement-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/enhancements/stats')
            return data
        }
    })

    const { data: enhancements, isLoading, refetch } = useQuery<Enhancement[]>({
        queryKey: ['enhancements', filter, priorityFilter],
        queryFn: async () => {
            const params = new URLSearchParams()
            if (filter !== 'all') params.append('status', filter)
            if (priorityFilter !== 'all') params.append('priority', priorityFilter)
            const { data } = await axiosInstance.get(`/enhancements?${params}`)
            return data?.enhancements || []
        }
    })

    const updateStatusMutation = useMutation({
        mutationFn: async ({ id, status }: { id: string; status: string }) => {
            const { data } = await axiosInstance.patch(`/enhancements/${id}`, { status })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['enhancements'] })
            queryClient.invalidateQueries({ queryKey: ['enhancement-stats'] })
        }
    })

    const deleteMutation = useMutation({
        mutationFn: async (id: string) => {
            const { data } = await axiosInstance.delete(`/enhancements/${id}`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['enhancements'] })
            queryClient.invalidateQueries({ queryKey: ['enhancement-stats'] })
        }
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'text-emerald-400 bg-emerald-500/20'
            case 'in_progress': return 'text-blue-400 bg-blue-500/20'
            case 'pending': return 'text-slate-400 bg-slate-500/20'
            case 'failed': return 'text-red-400 bg-red-500/20'
            case 'testing': return 'text-purple-400 bg-purple-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed': return <CheckCircle className="w-4 h-4" />
            case 'in_progress': return <Play className="w-4 h-4" />
            case 'pending': return <Clock className="w-4 h-4" />
            case 'failed': return <AlertCircle className="w-4 h-4" />
            case 'testing': return <TestTube className="w-4 h-4" />
            default: return <Clock className="w-4 h-4" />
        }
    }

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'critical': return 'text-red-400 bg-red-500/20 border-red-500/30'
            case 'high': return 'text-amber-400 bg-amber-500/20 border-amber-500/30'
            case 'medium': return 'text-blue-400 bg-blue-500/20 border-blue-500/30'
            case 'low': return 'text-slate-400 bg-slate-500/20 border-slate-500/30'
            default: return 'text-slate-400 bg-slate-500/20 border-slate-500/30'
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center shadow-lg">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        Enhancement Manager
                    </h1>
                    <p className="page-subtitle">Track and manage platform enhancements and features</p>
                </div>
                <div className="flex items-center gap-2">
                    <Link to="/enhancements/agent" className="btn-secondary flex items-center gap-2">
                        <Bot className="w-4 h-4" />
                        AI Agent
                    </Link>
                    <Link to="/enhancements/qa" className="btn-secondary flex items-center gap-2">
                        <TestTube className="w-4 h-4" />
                        QA Testing
                    </Link>
                    <button className="btn-primary flex items-center gap-2">
                        <Plus className="w-4 h-4" />
                        New Enhancement
                    </button>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Total</div>
                    <div className="text-2xl font-bold text-white">{stats?.total ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Pending</div>
                    <div className="text-2xl font-bold text-slate-400">{stats?.pending ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">In Progress</div>
                    <div className="text-2xl font-bold text-blue-400">{stats?.in_progress ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Completed</div>
                    <div className="text-2xl font-bold text-emerald-400">{stats?.completed ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Failed</div>
                    <div className="text-2xl font-bold text-red-400">{stats?.failed ?? 0}</div>
                </div>
            </div>

            {/* Filters */}
            <div className="glass-card p-5">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Status:</span>
                        {['all', 'pending', 'in_progress', 'completed', 'testing', 'failed'].map(status => (
                            <button
                                key={status}
                                onClick={() => setFilter(status)}
                                className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                                    filter === status
                                        ? 'bg-primary text-white'
                                        : 'bg-slate-800/60 text-slate-400 hover:text-white'
                                }`}
                            >
                                {status === 'all' ? 'All' : status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </button>
                        ))}
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-400">Priority:</span>
                        {['all', 'critical', 'high', 'medium', 'low'].map(priority => (
                            <button
                                key={priority}
                                onClick={() => setPriorityFilter(priority)}
                                className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                                    priorityFilter === priority
                                        ? 'bg-primary text-white'
                                        : 'bg-slate-800/60 text-slate-400 hover:text-white'
                                }`}
                            >
                                {priority.charAt(0).toUpperCase() + priority.slice(1)}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Enhancements List */}
            <div className="glass-card overflow-hidden">
                <div className="p-5 border-b border-slate-700/50">
                    <h3 className="text-lg font-semibold text-white">
                        Enhancements ({enhancements?.length ?? 0})
                    </h3>
                </div>
                <div className="divide-y divide-slate-700/50 max-h-[600px] overflow-y-auto">
                    {enhancements && enhancements.length > 0 ? (
                        enhancements.map((enhancement) => (
                            <div key={enhancement.id} className="p-4 hover:bg-slate-800/30 transition-colors">
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <h4 className="font-semibold text-white">{enhancement.title}</h4>
                                            <span className={`px-2 py-0.5 rounded border text-xs font-medium ${getPriorityColor(enhancement.priority)}`}>
                                                {enhancement.priority}
                                            </span>
                                            <span className={`px-2 py-0.5 rounded-lg text-xs font-medium flex items-center gap-1 ${getStatusColor(enhancement.status)}`}>
                                                {getStatusIcon(enhancement.status)}
                                                {enhancement.status.replace('_', ' ')}
                                            </span>
                                        </div>
                                        <p className="text-sm text-slate-400 mb-3">{enhancement.description}</p>

                                        {enhancement.status === 'in_progress' && (
                                            <div className="mb-3">
                                                <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                    <span>Progress</span>
                                                    <span>{enhancement.progress}%</span>
                                                </div>
                                                <div className="h-1.5 bg-slate-700 rounded-full">
                                                    <div
                                                        className="h-full bg-blue-500 rounded-full transition-all"
                                                        style={{ width: `${enhancement.progress}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}

                                        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
                                            <span className="flex items-center gap-1">
                                                <Tag className="w-3 h-3" />
                                                {enhancement.category}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <User className="w-3 h-3" />
                                                {enhancement.assigned_to}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Calendar className="w-3 h-3" />
                                                {new Date(enhancement.created_at).toLocaleDateString()}
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                Est: {enhancement.estimated_hours}h
                                                {enhancement.actual_hours > 0 && ` / Actual: ${enhancement.actual_hours}h`}
                                            </span>
                                        </div>

                                        {enhancement.tags.length > 0 && (
                                            <div className="flex flex-wrap gap-1 mt-2">
                                                {enhancement.tags.map((tag, idx) => (
                                                    <span key={idx} className="px-2 py-0.5 bg-slate-800/60 text-slate-400 rounded text-xs">
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    <div className="flex items-center gap-2">
                                        {enhancement.status === 'pending' && (
                                            <button
                                                onClick={() => updateStatusMutation.mutate({ id: enhancement.id, status: 'in_progress' })}
                                                className="p-2 text-blue-400 hover:bg-blue-500/10 rounded-lg transition-colors"
                                                title="Start"
                                            >
                                                <Play className="w-4 h-4" />
                                            </button>
                                        )}
                                        {enhancement.status === 'in_progress' && (
                                            <>
                                                <button
                                                    onClick={() => updateStatusMutation.mutate({ id: enhancement.id, status: 'testing' })}
                                                    className="p-2 text-purple-400 hover:bg-purple-500/10 rounded-lg transition-colors"
                                                    title="Send to QA"
                                                >
                                                    <TestTube className="w-4 h-4" />
                                                </button>
                                                <button
                                                    onClick={() => updateStatusMutation.mutate({ id: enhancement.id, status: 'completed' })}
                                                    className="p-2 text-emerald-400 hover:bg-emerald-500/10 rounded-lg transition-colors"
                                                    title="Mark Complete"
                                                >
                                                    <CheckCircle className="w-4 h-4" />
                                                </button>
                                            </>
                                        )}
                                        <button
                                            onClick={() => deleteMutation.mutate(enhancement.id)}
                                            className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                                            title="Delete"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            {isLoading ? 'Loading enhancements...' : 'No enhancements found'}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Link
                    to="/enhancements/agent"
                    className="glass-card p-5 flex items-center gap-4 hover:border-purple-500/50 transition-colors"
                >
                    <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                        <Bot className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                        <div className="font-semibold text-white">AI Enhancement Agent</div>
                        <div className="text-sm text-slate-400">Automate enhancement implementation</div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-600 ml-auto" />
                </Link>
                <Link
                    to="/enhancements/qa"
                    className="glass-card p-5 flex items-center gap-4 hover:border-emerald-500/50 transition-colors"
                >
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                        <TestTube className="w-6 h-6 text-emerald-400" />
                    </div>
                    <div>
                        <div className="font-semibold text-white">QA Testing</div>
                        <div className="text-sm text-slate-400">Review and test enhancements</div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-600 ml-auto" />
                </Link>
                <div className="glass-card p-5 flex items-center gap-4 hover:border-blue-500/50 transition-colors cursor-pointer">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                        <Calendar className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                        <div className="font-semibold text-white">Sprint Planning</div>
                        <div className="text-sm text-slate-400">Plan enhancement sprints</div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-600 ml-auto" />
                </div>
            </div>
        </div>
    )
}
