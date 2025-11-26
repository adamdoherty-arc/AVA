import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Bot, RefreshCw, Play, Pause, CheckCircle, XCircle,
    AlertCircle, Clock, Zap, MessageSquare, Code, FileText,
    Terminal, Loader2, ArrowLeft
} from 'lucide-react'
import { Link } from 'react-router-dom'

interface AgentTask {
    id: string
    enhancement_id: string
    enhancement_title: string
    status: 'queued' | 'running' | 'completed' | 'failed'
    steps: {
        step: string
        status: 'pending' | 'running' | 'completed' | 'failed'
        output?: string
        duration_ms?: number
    }[]
    started_at: string
    completed_at?: string
    total_duration_ms: number
    logs: string[]
    result?: {
        files_modified: string[]
        tests_passed: boolean
        code_quality_score: number
    }
}

interface AgentConfig {
    model: string
    max_iterations: number
    auto_test: boolean
    auto_commit: boolean
    review_before_commit: boolean
}

export default function EnhancementAgent() {
    const queryClient = useQueryClient()
    const [selectedTask, setSelectedTask] = useState<string | null>(null)

    const { data: tasks, isLoading, refetch } = useQuery<AgentTask[]>({
        queryKey: ['agent-tasks'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/enhancements/agent/tasks')
            return data?.tasks || []
        },
        refetchInterval: 5000
    })

    const { data: config } = useQuery<AgentConfig>({
        queryKey: ['agent-config'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/enhancements/agent/config')
            return data
        }
    })

    const runAgentMutation = useMutation({
        mutationFn: async (enhancementId: string) => {
            const { data } = await axiosInstance.post('/enhancements/agent/run', { enhancement_id: enhancementId })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agent-tasks'] })
        }
    })

    const cancelTaskMutation = useMutation({
        mutationFn: async (taskId: string) => {
            const { data } = await axiosInstance.post(`/enhancements/agent/tasks/${taskId}/cancel`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agent-tasks'] })
        }
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'text-emerald-400 bg-emerald-500/20'
            case 'running': return 'text-blue-400 bg-blue-500/20'
            case 'queued': return 'text-slate-400 bg-slate-500/20'
            case 'failed': return 'text-red-400 bg-red-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed': return <CheckCircle className="w-4 h-4" />
            case 'running': return <Loader2 className="w-4 h-4 animate-spin" />
            case 'queued': return <Clock className="w-4 h-4" />
            case 'failed': return <XCircle className="w-4 h-4" />
            default: return <Clock className="w-4 h-4" />
        }
    }

    const selectedTaskData = tasks?.find(t => t.id === selectedTask)

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link to="/enhancements" className="btn-icon">
                        <ArrowLeft className="w-5 h-5" />
                    </Link>
                    <div>
                        <h1 className="page-title flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg">
                                <Bot className="w-5 h-5 text-white" />
                            </div>
                            Enhancement Agent
                        </h1>
                        <p className="page-subtitle">AI-powered enhancement implementation</p>
                    </div>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Agent Config */}
            {config && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Zap className="w-5 h-5 text-amber-400" />
                        Agent Configuration
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Model</div>
                            <div className="text-white font-medium">{config.model}</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Max Iterations</div>
                            <div className="text-white font-medium">{config.max_iterations}</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Auto Test</div>
                            <div className={`font-medium ${config.auto_test ? 'text-emerald-400' : 'text-slate-400'}`}>
                                {config.auto_test ? 'Enabled' : 'Disabled'}
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Auto Commit</div>
                            <div className={`font-medium ${config.auto_commit ? 'text-emerald-400' : 'text-slate-400'}`}>
                                {config.auto_commit ? 'Enabled' : 'Disabled'}
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Review Required</div>
                            <div className={`font-medium ${config.review_before_commit ? 'text-amber-400' : 'text-slate-400'}`}>
                                {config.review_before_commit ? 'Yes' : 'No'}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Tasks List */}
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50">
                        <h3 className="text-lg font-semibold text-white">Agent Tasks</h3>
                    </div>
                    <div className="divide-y divide-slate-700/50 max-h-[500px] overflow-y-auto">
                        {tasks && tasks.length > 0 ? (
                            tasks.map((task) => (
                                <div
                                    key={task.id}
                                    className={`p-4 hover:bg-slate-800/30 transition-colors cursor-pointer ${
                                        selectedTask === task.id ? 'bg-slate-800/50' : ''
                                    }`}
                                    onClick={() => setSelectedTask(task.id)}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <span className={`p-2 rounded-lg ${getStatusColor(task.status)}`}>
                                                {getStatusIcon(task.status)}
                                            </span>
                                            <div>
                                                <div className="font-medium text-white">{task.enhancement_title}</div>
                                                <div className="text-sm text-slate-400">
                                                    {task.status === 'running'
                                                        ? `Step ${task.steps.filter(s => s.status === 'completed').length + 1}/${task.steps.length}`
                                                        : new Date(task.started_at).toLocaleString()
                                                    }
                                                </div>
                                            </div>
                                        </div>
                                        {task.status === 'running' && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    cancelTaskMutation.mutate(task.id)
                                                }}
                                                className="p-2 text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                                            >
                                                <Pause className="w-4 h-4" />
                                            </button>
                                        )}
                                    </div>
                                    {task.status === 'running' && (
                                        <div className="mt-3">
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>Progress</span>
                                                <span>{Math.round((task.steps.filter(s => s.status === 'completed').length / task.steps.length) * 100)}%</span>
                                            </div>
                                            <div className="h-1.5 bg-slate-700 rounded-full">
                                                <div
                                                    className="h-full bg-blue-500 rounded-full transition-all"
                                                    style={{ width: `${(task.steps.filter(s => s.status === 'completed').length / task.steps.length) * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))
                        ) : (
                            <div className="p-12 text-center text-slate-400">
                                {isLoading ? 'Loading tasks...' : 'No agent tasks'}
                            </div>
                        )}
                    </div>
                </div>

                {/* Task Details */}
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50">
                        <h3 className="text-lg font-semibold text-white">Task Details</h3>
                    </div>
                    {selectedTaskData ? (
                        <div className="p-5 space-y-4">
                            {/* Steps */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Execution Steps</h4>
                                <div className="space-y-2">
                                    {selectedTaskData.steps.map((step, idx) => (
                                        <div key={idx} className="flex items-center gap-3 p-3 bg-slate-800/40 rounded-lg">
                                            <span className={`p-1.5 rounded-lg ${getStatusColor(step.status)}`}>
                                                {getStatusIcon(step.status)}
                                            </span>
                                            <div className="flex-1">
                                                <div className="text-white text-sm">{step.step}</div>
                                                {step.duration_ms && (
                                                    <div className="text-xs text-slate-500">{step.duration_ms}ms</div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Result */}
                            {selectedTaskData.result && (
                                <div>
                                    <h4 className="text-sm font-medium text-slate-400 mb-3">Result</h4>
                                    <div className="grid grid-cols-3 gap-3">
                                        <div className="bg-slate-800/40 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs mb-1">Files Modified</div>
                                            <div className="text-white font-medium">{selectedTaskData.result.files_modified.length}</div>
                                        </div>
                                        <div className="bg-slate-800/40 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs mb-1">Tests</div>
                                            <div className={`font-medium ${selectedTaskData.result.tests_passed ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {selectedTaskData.result.tests_passed ? 'Passed' : 'Failed'}
                                            </div>
                                        </div>
                                        <div className="bg-slate-800/40 rounded-lg p-3">
                                            <div className="text-slate-400 text-xs mb-1">Code Quality</div>
                                            <div className={`font-medium ${
                                                selectedTaskData.result.code_quality_score >= 80 ? 'text-emerald-400' :
                                                selectedTaskData.result.code_quality_score >= 60 ? 'text-amber-400' : 'text-red-400'
                                            }`}>
                                                {selectedTaskData.result.code_quality_score}%
                                            </div>
                                        </div>
                                    </div>
                                    {selectedTaskData.result.files_modified.length > 0 && (
                                        <div className="mt-3">
                                            <div className="text-xs text-slate-400 mb-2">Modified Files:</div>
                                            <div className="space-y-1 max-h-32 overflow-y-auto">
                                                {selectedTaskData.result.files_modified.map((file, idx) => (
                                                    <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                                                        <Code className="w-3 h-3" />
                                                        {file}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Logs */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
                                    <Terminal className="w-4 h-4" />
                                    Logs
                                </h4>
                                <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-xs max-h-48 overflow-y-auto">
                                    {selectedTaskData.logs.map((log, idx) => (
                                        <div key={idx} className="text-slate-400">{log}</div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            <Bot className="w-12 h-12 mx-auto mb-4 text-slate-600" />
                            <p>Select a task to view details</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Tips */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">How the Agent Works</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400 flex-shrink-0">
                            1
                        </div>
                        <div>
                            <div className="font-medium text-white">Analyze</div>
                            <p className="text-sm text-slate-400">Reads enhancement requirements and codebase</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400 flex-shrink-0">
                            2
                        </div>
                        <div>
                            <div className="font-medium text-white">Plan</div>
                            <p className="text-sm text-slate-400">Creates implementation steps</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400 flex-shrink-0">
                            3
                        </div>
                        <div>
                            <div className="font-medium text-white">Implement</div>
                            <p className="text-sm text-slate-400">Writes and modifies code</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400 flex-shrink-0">
                            4
                        </div>
                        <div>
                            <div className="font-medium text-white">Verify</div>
                            <p className="text-sm text-slate-400">Runs tests and quality checks</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
