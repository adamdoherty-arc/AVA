import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Heart, RefreshCw, Database, Server, Cpu, MemoryStick,
    HardDrive, Wifi, Activity, CheckCircle, XCircle, AlertTriangle,
    Clock, Zap, Bot, Globe
} from 'lucide-react'

interface ServiceStatus {
    healthy: boolean
    status: string
    last_check?: string
    response_time?: number
}

interface HealthData {
    database: {
        healthy: boolean
        active_connections: number
        max_connections: number
        available: number
        errors: number
    }
    llm: {
        healthy: boolean
        model: string
        status: string
        response_time: string
    }
    apis: {
        [key: string]: ServiceStatus
    }
    system: {
        cpu_percent: number
        memory_percent: number
        memory_total_gb: number
        memory_available_gb: number
        disk_percent: number
        disk_total_gb: number
        disk_free_gb: number
    }
    errors: {
        total: number
        by_type: { [key: string]: number }
        last_error?: string
    }
}

export default function HealthDashboard() {
    const { data: health, isLoading, refetch, dataUpdatedAt } = useQuery<HealthData>({
        queryKey: ['system-health'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/system/health/detailed')
            return data
        },
        refetchInterval: 30000, // Auto-refresh every 30 seconds
        staleTime: 10000
    })

    const getStatusIcon = (healthy: boolean) => {
        return healthy ? (
            <CheckCircle className="w-5 h-5 text-emerald-400" />
        ) : (
            <XCircle className="w-5 h-5 text-red-400" />
        )
    }

    const getStatusColor = (healthy: boolean) => {
        return healthy ? 'text-emerald-400' : 'text-red-400'
    }

    const getUsageColor = (percent: number) => {
        if (percent < 50) return 'bg-emerald-500'
        if (percent < 80) return 'bg-amber-500'
        return 'bg-red-500'
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
                            <Heart className="w-5 h-5 text-white" />
                        </div>
                        Health Dashboard
                    </h1>
                    <p className="page-subtitle">Real-time system health monitoring and diagnostics</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="text-sm text-slate-400">
                        Last updated: {new Date(dataUpdatedAt).toLocaleTimeString()}
                    </div>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Overall Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Database */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                <Database className="w-5 h-5 text-blue-400" />
                            </div>
                            <span className="font-semibold text-white">Database</span>
                        </div>
                        {isLoading ? (
                            <div className="w-5 h-5 rounded-full bg-slate-600 animate-pulse" />
                        ) : (
                            getStatusIcon(health?.database?.healthy ?? false)
                        )}
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {health?.database?.active_connections ?? 0}/{health?.database?.max_connections ?? 20}
                    </div>
                    <div className="text-sm text-slate-400">Active connections</div>
                </div>

                {/* LLM Service */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                                <Bot className="w-5 h-5 text-purple-400" />
                            </div>
                            <span className="font-semibold text-white">Local LLM</span>
                        </div>
                        {isLoading ? (
                            <div className="w-5 h-5 rounded-full bg-slate-600 animate-pulse" />
                        ) : (
                            getStatusIcon(health?.llm?.healthy ?? false)
                        )}
                    </div>
                    <div className="text-lg font-bold text-white truncate">
                        {health?.llm?.model ?? 'Unknown'}
                    </div>
                    <div className="text-sm text-slate-400">{health?.llm?.status ?? 'Checking...'}</div>
                </div>

                {/* External APIs */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                                <Globe className="w-5 h-5 text-amber-400" />
                            </div>
                            <span className="font-semibold text-white">APIs</span>
                        </div>
                        {isLoading ? (
                            <div className="w-5 h-5 rounded-full bg-slate-600 animate-pulse" />
                        ) : (
                            (() => {
                                const apis = health?.apis ?? {}
                                const total = Object.keys(apis).length
                                const healthy = Object.values(apis).filter(a => a.healthy).length
                                return healthy === total ? (
                                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                                ) : healthy > 0 ? (
                                    <AlertTriangle className="w-5 h-5 text-amber-400" />
                                ) : (
                                    <XCircle className="w-5 h-5 text-red-400" />
                                )
                            })()
                        )}
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {Object.values(health?.apis ?? {}).filter(a => a.healthy).length}/
                        {Object.keys(health?.apis ?? {}).length}
                    </div>
                    <div className="text-sm text-slate-400">Services online</div>
                </div>

                {/* Errors */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                                <AlertTriangle className="w-5 h-5 text-red-400" />
                            </div>
                            <span className="font-semibold text-white">Errors</span>
                        </div>
                        {isLoading ? (
                            <div className="w-5 h-5 rounded-full bg-slate-600 animate-pulse" />
                        ) : (
                            getStatusIcon((health?.errors?.total ?? 0) === 0)
                        )}
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {health?.errors?.total ?? 0}
                    </div>
                    <div className="text-sm text-slate-400">Total errors</div>
                </div>
            </div>

            {/* Detailed Sections */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Database Details */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Database className="w-5 h-5 text-blue-400" />
                        Database Connection Pool
                    </h3>

                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">Pool Usage</span>
                                <span className="text-white">
                                    {((health?.database?.active_connections ?? 0) / (health?.database?.max_connections ?? 20) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${getUsageColor((health?.database?.active_connections ?? 0) / (health?.database?.max_connections ?? 20) * 100)} transition-all`}
                                    style={{ width: `${(health?.database?.active_connections ?? 0) / (health?.database?.max_connections ?? 20) * 100}%` }}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Active</div>
                                <div className="text-xl font-bold text-white">{health?.database?.active_connections ?? 0}</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Available</div>
                                <div className="text-xl font-bold text-white">{health?.database?.available ?? 0}</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Max</div>
                                <div className="text-xl font-bold text-white">{health?.database?.max_connections ?? 20}</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Errors</div>
                                <div className={`text-xl font-bold ${(health?.database?.errors ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                    {health?.database?.errors ?? 0}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* API Status */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Globe className="w-5 h-5 text-amber-400" />
                        External API Status
                    </h3>

                    <div className="space-y-3">
                        {Object.entries(health?.apis ?? {}).map(([name, status]) => (
                            <div key={name} className="flex items-center justify-between p-3 bg-slate-800/40 rounded-xl">
                                <div className="flex items-center gap-3">
                                    {getStatusIcon(status.healthy)}
                                    <span className="font-medium text-white">{name}</span>
                                </div>
                                <span className={`text-sm ${status.healthy ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {status.status}
                                </span>
                            </div>
                        ))}
                        {Object.keys(health?.apis ?? {}).length === 0 && (
                            <div className="text-center py-4 text-slate-400">No API status available</div>
                        )}
                    </div>
                </div>

                {/* System Resources */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Server className="w-5 h-5 text-emerald-400" />
                        System Resources
                    </h3>

                    <div className="space-y-4">
                        {/* CPU */}
                        <div>
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2 text-slate-400">
                                    <Cpu className="w-4 h-4" />
                                    <span className="text-sm">CPU</span>
                                </div>
                                <span className="text-white font-mono">{health?.system?.cpu_percent ?? 0}%</span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${getUsageColor(health?.system?.cpu_percent ?? 0)} transition-all`}
                                    style={{ width: `${health?.system?.cpu_percent ?? 0}%` }}
                                />
                            </div>
                        </div>

                        {/* Memory */}
                        <div>
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2 text-slate-400">
                                    <MemoryStick className="w-4 h-4" />
                                    <span className="text-sm">Memory</span>
                                </div>
                                <span className="text-white font-mono">
                                    {health?.system?.memory_percent ?? 0}%
                                    <span className="text-slate-500 text-xs ml-2">
                                        ({health?.system?.memory_available_gb?.toFixed(1) ?? 0} GB free)
                                    </span>
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${getUsageColor(health?.system?.memory_percent ?? 0)} transition-all`}
                                    style={{ width: `${health?.system?.memory_percent ?? 0}%` }}
                                />
                            </div>
                        </div>

                        {/* Disk */}
                        <div>
                            <div className="flex items-center justify-between mb-1">
                                <div className="flex items-center gap-2 text-slate-400">
                                    <HardDrive className="w-4 h-4" />
                                    <span className="text-sm">Disk</span>
                                </div>
                                <span className="text-white font-mono">
                                    {health?.system?.disk_percent ?? 0}%
                                    <span className="text-slate-500 text-xs ml-2">
                                        ({health?.system?.disk_free_gb?.toFixed(1) ?? 0} GB free)
                                    </span>
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${getUsageColor(health?.system?.disk_percent ?? 0)} transition-all`}
                                    style={{ width: `${health?.system?.disk_percent ?? 0}%` }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* LLM Service */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Bot className="w-5 h-5 text-purple-400" />
                        Local LLM Service
                    </h3>

                    <div className="space-y-4">
                        <div className="flex items-center justify-between p-3 bg-slate-800/40 rounded-xl">
                            <div>
                                <div className="text-slate-400 text-sm">Status</div>
                                <div className={`font-semibold ${health?.llm?.healthy ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {health?.llm?.status ?? 'Unknown'}
                                </div>
                            </div>
                            {getStatusIcon(health?.llm?.healthy ?? false)}
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Model</div>
                                <div className="text-white font-medium">{health?.llm?.model ?? 'N/A'}</div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-3">
                                <div className="text-slate-400 text-sm">Response Time</div>
                                <div className="text-white font-medium">{health?.llm?.response_time ?? 'N/A'}</div>
                            </div>
                        </div>

                        <div className="text-sm text-slate-400">
                            <strong>Available Models:</strong>
                            <ul className="mt-2 space-y-1">
                                <li>Qwen 2.5 32B - High quality, slower</li>
                                <li>Qwen 2.5 14B - Balanced (default)</li>
                                <li>Qwen 2.5 7B - Fast, lower quality</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Summary */}
            {(health?.errors?.total ?? 0) > 0 && (
                <div className="glass-card p-5 border-red-500/30">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        Error Summary
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-red-500/10 rounded-xl p-4">
                            <div className="text-red-400 text-sm">Total Errors</div>
                            <div className="text-2xl font-bold text-white">{health?.errors?.total ?? 0}</div>
                        </div>
                        <div className="bg-red-500/10 rounded-xl p-4">
                            <div className="text-red-400 text-sm">Unique Types</div>
                            <div className="text-2xl font-bold text-white">
                                {Object.keys(health?.errors?.by_type ?? {}).length}
                            </div>
                        </div>
                        <div className="bg-red-500/10 rounded-xl p-4">
                            <div className="text-red-400 text-sm">Last Error</div>
                            <div className="text-white truncate">{health?.errors?.last_error ?? 'None'}</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
