import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Settings, RefreshCw, Server, Database, Bot, Globe,
    Play, Pause, RotateCcw, Trash2, CheckCircle, XCircle,
    AlertTriangle, Clock, Zap, HardDrive, Activity
} from 'lucide-react'

interface ServiceStatus {
    name: string
    status: 'running' | 'stopped' | 'error'
    uptime: string
    memory_mb: number
    cpu_percent: number
    last_restart: string
    health_check: boolean
}

interface SystemConfig {
    cache_enabled: boolean
    cache_ttl_seconds: number
    log_level: string
    max_connections: number
    rate_limit_per_minute: number
    auto_restart_enabled: boolean
}

export default function SystemManagementHub() {
    const queryClient = useQueryClient()
    const [selectedService, setSelectedService] = useState<string | null>(null)

    const { data: services, isLoading, refetch } = useQuery<ServiceStatus[]>({
        queryKey: ['system-services'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/system/services')
            return data?.services || []
        },
        refetchInterval: 10000
    })

    const { data: config } = useQuery<SystemConfig>({
        queryKey: ['system-config'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/system/config')
            return data
        }
    })

    const restartServiceMutation = useMutation({
        mutationFn: async (serviceName: string) => {
            const { data } = await axiosInstance.post(`/system/services/${serviceName}/restart`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['system-services'] })
        }
    })

    const stopServiceMutation = useMutation({
        mutationFn: async (serviceName: string) => {
            const { data } = await axiosInstance.post(`/system/services/${serviceName}/stop`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['system-services'] })
        }
    })

    const clearLogsMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/system/logs/clear')
            return data
        }
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'running': return 'text-emerald-400 bg-emerald-500/20'
            case 'stopped': return 'text-slate-400 bg-slate-500/20'
            case 'error': return 'text-red-400 bg-red-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'running': return <Play className="w-4 h-4" />
            case 'stopped': return <Pause className="w-4 h-4" />
            case 'error': return <XCircle className="w-4 h-4" />
            default: return <Clock className="w-4 h-4" />
        }
    }

    const getServiceIcon = (name: string) => {
        if (name.includes('database') || name.includes('db')) return Database
        if (name.includes('llm') || name.includes('ai')) return Bot
        if (name.includes('api') || name.includes('web')) return Globe
        if (name.includes('cache')) return HardDrive
        return Server
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-500 to-slate-700 flex items-center justify-center shadow-lg">
                            <Settings className="w-5 h-5 text-white" />
                        </div>
                        System Management
                    </h1>
                    <p className="page-subtitle">Manage services, configuration, and system operations</p>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-slate-400">Services Running</p>
                            <p className="text-2xl font-bold text-emerald-400">
                                {services?.filter(s => s.status === 'running').length || 0}
                            </p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                            <CheckCircle className="w-5 h-5 text-emerald-400" />
                        </div>
                    </div>
                </div>
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-slate-400">Services Stopped</p>
                            <p className="text-2xl font-bold text-slate-400">
                                {services?.filter(s => s.status === 'stopped').length || 0}
                            </p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-slate-500/20 flex items-center justify-center">
                            <Pause className="w-5 h-5 text-slate-400" />
                        </div>
                    </div>
                </div>
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-slate-400">Errors</p>
                            <p className="text-2xl font-bold text-red-400">
                                {services?.filter(s => s.status === 'error').length || 0}
                            </p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                            <AlertTriangle className="w-5 h-5 text-red-400" />
                        </div>
                    </div>
                </div>
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm text-slate-400">Total Memory</p>
                            <p className="text-2xl font-bold text-blue-400">
                                {services?.reduce((sum, s) => sum + s.memory_mb, 0).toFixed(0) || 0} MB
                            </p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                            <Activity className="w-5 h-5 text-blue-400" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Services List */}
            <div className="glass-card overflow-hidden">
                <div className="p-5 border-b border-slate-700/50">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Server className="w-5 h-5 text-blue-400" />
                        Services
                    </h3>
                </div>
                <div className="divide-y divide-slate-700/50">
                    {services && services.length > 0 ? (
                        services.map((service) => {
                            const ServiceIcon = getServiceIcon(service.name)
                            return (
                                <div
                                    key={service.name}
                                    className={`p-4 hover:bg-slate-800/30 transition-colors cursor-pointer ${
                                        selectedService === service.name ? 'bg-slate-800/50' : ''
                                    }`}
                                    onClick={() => setSelectedService(selectedService === service.name ? null : service.name)}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-4">
                                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${getStatusColor(service.status)}`}>
                                                <ServiceIcon className="w-5 h-5" />
                                            </div>
                                            <div>
                                                <div className="font-semibold text-white">{service.name}</div>
                                                <div className="text-sm text-slate-400">Uptime: {service.uptime}</div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <div className="text-right">
                                                <div className="text-sm text-slate-400">{service.memory_mb} MB</div>
                                                <div className="text-sm text-slate-400">{service.cpu_percent}% CPU</div>
                                            </div>
                                            <span className={`px-3 py-1 rounded-lg text-sm font-medium flex items-center gap-1.5 ${getStatusColor(service.status)}`}>
                                                {getStatusIcon(service.status)}
                                                {service.status}
                                            </span>
                                        </div>
                                    </div>

                                    {selectedService === service.name && (
                                        <div className="mt-4 pt-4 border-t border-slate-700/50">
                                            <div className="flex items-center justify-between">
                                                <div className="grid grid-cols-3 gap-4 text-sm">
                                                    <div>
                                                        <span className="text-slate-400">Last Restart:</span>
                                                        <span className="ml-2 text-white">{service.last_restart}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-400">Health Check:</span>
                                                        <span className={`ml-2 ${service.health_check ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            {service.health_check ? 'Passing' : 'Failing'}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation()
                                                            restartServiceMutation.mutate(service.name)
                                                        }}
                                                        disabled={restartServiceMutation.isPending}
                                                        className="btn-secondary flex items-center gap-2 text-sm"
                                                    >
                                                        <RotateCcw className="w-4 h-4" />
                                                        Restart
                                                    </button>
                                                    {service.status === 'running' && (
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation()
                                                                stopServiceMutation.mutate(service.name)
                                                            }}
                                                            disabled={stopServiceMutation.isPending}
                                                            className="btn-secondary flex items-center gap-2 text-sm text-red-400 hover:text-red-300"
                                                        >
                                                            <Pause className="w-4 h-4" />
                                                            Stop
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )
                        })
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            No services found
                        </div>
                    )}
                </div>
            </div>

            {/* Configuration */}
            {config && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Settings className="w-5 h-5 text-amber-400" />
                        System Configuration
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Cache</div>
                            <div className={`text-lg font-medium ${config.cache_enabled ? 'text-emerald-400' : 'text-slate-400'}`}>
                                {config.cache_enabled ? 'Enabled' : 'Disabled'}
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Cache TTL</div>
                            <div className="text-lg font-medium text-white">{config.cache_ttl_seconds}s</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Log Level</div>
                            <div className="text-lg font-medium text-white">{config.log_level}</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Max Connections</div>
                            <div className="text-lg font-medium text-white">{config.max_connections}</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Rate Limit</div>
                            <div className="text-lg font-medium text-white">{config.rate_limit_per_minute}/min</div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="text-slate-400 text-sm mb-1">Auto Restart</div>
                            <div className={`text-lg font-medium ${config.auto_restart_enabled ? 'text-emerald-400' : 'text-slate-400'}`}>
                                {config.auto_restart_enabled ? 'Enabled' : 'Disabled'}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                    onClick={() => {
                        services?.forEach(s => restartServiceMutation.mutate(s.name))
                    }}
                    className="glass-card p-4 flex items-center gap-4 hover:border-blue-500/50 transition-colors"
                >
                    <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                        <RotateCcw className="w-5 h-5 text-blue-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Restart All Services</div>
                        <div className="text-sm text-slate-400">Restart all running services</div>
                    </div>
                </button>
                <button
                    onClick={() => clearLogsMutation.mutate()}
                    className="glass-card p-4 flex items-center gap-4 hover:border-amber-500/50 transition-colors"
                >
                    <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                        <Trash2 className="w-5 h-5 text-amber-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Clear Logs</div>
                        <div className="text-sm text-slate-400">Clear all application logs</div>
                    </div>
                </button>
                <button className="glass-card p-4 flex items-center gap-4 hover:border-emerald-500/50 transition-colors">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                        <Zap className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div className="text-left">
                        <div className="font-medium text-white">Run Diagnostics</div>
                        <div className="text-sm text-slate-400">Full system health check</div>
                    </div>
                </button>
            </div>
        </div>
    )
}
