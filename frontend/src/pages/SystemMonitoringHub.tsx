import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Monitor, RefreshCw, Activity, Server, Cpu, MemoryStick,
    HardDrive, Clock, CheckCircle, XCircle, AlertTriangle,
    Zap, Database, Bot, Globe, Play, Pause, RotateCcw
} from 'lucide-react'
import { useState } from 'react'
import {
    LineChart, ResponsiveContainer, AreaChart, Area
} from 'recharts'

interface SystemMetrics {
    cpu: {
        current: number
        history: { time: string; value: number }[]
    }
    memory: {
        current: number
        total_gb: number
        available_gb: number
        history: { time: string; value: number }[]
    }
    disk: {
        percent: number
        total_gb: number
        free_gb: number
    }
    network: {
        bytes_sent_mb: number
        bytes_recv_mb: number
        packets_sent: number
        packets_recv: number
    }
}

interface BackgroundTask {
    id: string
    name: string
    status: 'running' | 'completed' | 'failed' | 'pending'
    progress: number
    started_at: string
    duration?: string
    error?: string
}

export default function SystemMonitoringHub() {
    const [autoRefresh, setAutoRefresh] = useState(true)

    const { data: metrics, isLoading, refetch } = useQuery<SystemMetrics>({
        queryKey: ['system-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/system/metrics')
            return data
        },
        refetchInterval: autoRefresh ? 5000 : false,
        staleTime: 3000
    })

    const { data: tasks } = useQuery<BackgroundTask[]>({
        queryKey: ['background-tasks'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/system/tasks')
            return data?.tasks || []
        },
        refetchInterval: autoRefresh ? 5000 : false
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'running': return 'text-blue-400 bg-blue-500/20'
            case 'completed': return 'text-emerald-400 bg-emerald-500/20'
            case 'failed': return 'text-red-400 bg-red-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'running': return <Play className="w-4 h-4" />
            case 'completed': return <CheckCircle className="w-4 h-4" />
            case 'failed': return <XCircle className="w-4 h-4" />
            default: return <Clock className="w-4 h-4" />
        }
    }

    const getUsageColor = (percent: number) => {
        if (percent < 50) return '#10B981'
        if (percent < 80) return '#F59E0B'
        return '#EF4444'
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg">
                            <Monitor className="w-5 h-5 text-white" />
                        </div>
                        System Monitoring
                    </h1>
                    <p className="page-subtitle">Real-time performance metrics and background tasks</p>
                </div>
                <div className="flex items-center gap-3">
                    <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary"
                        />
                        Auto-refresh (5s)
                    </label>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Resource Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* CPU */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                <Cpu className="w-5 h-5 text-blue-400" />
                            </div>
                            <span className="font-semibold text-white">CPU</span>
                        </div>
                        <span className={`text-2xl font-bold ${
                            (metrics?.cpu?.current ?? 0) < 50 ? 'text-emerald-400' :
                            (metrics?.cpu?.current ?? 0) < 80 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                            {metrics?.cpu?.current ?? 0}%
                        </span>
                    </div>
                    <div className="h-20">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <AreaChart data={metrics?.cpu?.history || []}>
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#3B82F6"
                                    fill="rgba(59, 130, 246, 0.2)"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Memory */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                                <MemoryStick className="w-5 h-5 text-purple-400" />
                            </div>
                            <span className="font-semibold text-white">Memory</span>
                        </div>
                        <span className={`text-2xl font-bold ${
                            (metrics?.memory?.current ?? 0) < 50 ? 'text-emerald-400' :
                            (metrics?.memory?.current ?? 0) < 80 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                            {metrics?.memory?.current ?? 0}%
                        </span>
                    </div>
                    <div className="h-20">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <AreaChart data={metrics?.memory?.history || []}>
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#A855F7"
                                    fill="rgba(168, 85, 247, 0.2)"
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-slate-400 mt-2">
                        {metrics?.memory?.available_gb?.toFixed(1) ?? 0} GB / {metrics?.memory?.total_gb?.toFixed(1) ?? 0} GB available
                    </div>
                </div>

                {/* Disk */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                                <HardDrive className="w-5 h-5 text-amber-400" />
                            </div>
                            <span className="font-semibold text-white">Disk</span>
                        </div>
                        <span className={`text-2xl font-bold ${
                            (metrics?.disk?.percent ?? 0) < 50 ? 'text-emerald-400' :
                            (metrics?.disk?.percent ?? 0) < 80 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                            {metrics?.disk?.percent ?? 0}%
                        </span>
                    </div>
                    <div className="h-20 flex items-center justify-center">
                        <div className="relative w-20 h-20">
                            <svg className="w-full h-full -rotate-90">
                                <circle
                                    cx="40"
                                    cy="40"
                                    r="36"
                                    fill="none"
                                    stroke="rgba(51, 65, 85, 0.5)"
                                    strokeWidth="8"
                                />
                                <circle
                                    cx="40"
                                    cy="40"
                                    r="36"
                                    fill="none"
                                    stroke={getUsageColor(metrics?.disk?.percent ?? 0)}
                                    strokeWidth="8"
                                    strokeLinecap="round"
                                    strokeDasharray={`${(metrics?.disk?.percent ?? 0) * 2.26} 226`}
                                />
                            </svg>
                        </div>
                    </div>
                    <div className="text-xs text-slate-400 mt-2">
                        {metrics?.disk?.free_gb?.toFixed(0) ?? 0} GB / {metrics?.disk?.total_gb?.toFixed(0) ?? 0} GB free
                    </div>
                </div>

                {/* Network */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                <Globe className="w-5 h-5 text-emerald-400" />
                            </div>
                            <span className="font-semibold text-white">Network</span>
                        </div>
                        <Activity className="w-5 h-5 text-emerald-400 animate-pulse" />
                    </div>
                    <div className="space-y-2 mt-4">
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Sent</span>
                            <span className="text-white font-mono">{metrics?.network?.bytes_sent_mb?.toFixed(1) ?? 0} MB</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Received</span>
                            <span className="text-white font-mono">{metrics?.network?.bytes_recv_mb?.toFixed(1) ?? 0} MB</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Packets</span>
                            <span className="text-white font-mono">
                                {((metrics?.network?.packets_sent ?? 0) + (metrics?.network?.packets_recv ?? 0)).toLocaleString()}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Background Tasks */}
            <div className="glass-card p-5">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Zap className="w-5 h-5 text-amber-400" />
                        Background Tasks
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                        <span>{tasks?.filter(t => t.status === 'running').length || 0} running</span>
                    </div>
                </div>

                {tasks && tasks.length > 0 ? (
                    <div className="space-y-3">
                        {tasks.map((task) => (
                            <div key={task.id} className="flex items-center gap-4 p-4 bg-slate-800/40 rounded-xl">
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getStatusColor(task.status)}`}>
                                    {getStatusIcon(task.status)}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                        <span className="font-medium text-white">{task.name}</span>
                                        <span className={`px-2 py-0.5 rounded-lg text-xs font-medium ${getStatusColor(task.status)}`}>
                                            {task.status}
                                        </span>
                                    </div>
                                    {task.status === 'running' && (
                                        <div className="mt-2">
                                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                                <span>Progress</span>
                                                <span>{task.progress}%</span>
                                            </div>
                                            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-blue-500 transition-all"
                                                    style={{ width: `${task.progress}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                    <div className="flex items-center gap-4 mt-1 text-xs text-slate-500">
                                        <span>Started: {task.started_at}</span>
                                        {task.duration && <span>Duration: {task.duration}</span>}
                                    </div>
                                    {task.error && (
                                        <div className="text-xs text-red-400 mt-1">{task.error}</div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-8 text-slate-400">
                        No background tasks running
                    </div>
                )}
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button className="glass-card p-4 flex items-center gap-4 hover:border-blue-500/50 transition-colors">
                    <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                        <RotateCcw className="w-5 h-5 text-blue-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Restart Services</div>
                        <div className="text-sm text-slate-400">Restart background services</div>
                    </div>
                </button>
                <button className="glass-card p-4 flex items-center gap-4 hover:border-amber-500/50 transition-colors">
                    <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                        <Database className="w-5 h-5 text-amber-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Clear Cache</div>
                        <div className="text-sm text-slate-400">Clear application cache</div>
                    </div>
                </button>
                <button className="glass-card p-4 flex items-center gap-4 hover:border-emerald-500/50 transition-colors">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                        <Server className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Run Health Check</div>
                        <div className="text-sm text-slate-400">Full system diagnostics</div>
                    </div>
                </button>
            </div>
        </div>
    )
}
