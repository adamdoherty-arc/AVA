import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Cpu, RefreshCw, Database, Trash2, Clock, Zap,
    TrendingUp, TrendingDown, AlertCircle, CheckCircle
} from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

interface CacheMetrics {
    total_size_mb: number
    max_size_mb: number
    item_count: number
    hit_count: number
    miss_count: number
    hit_rate: number
    evictions: number
    last_cleared: string
    caches: {
        name: string
        size_mb: number
        items: number
        hits: number
        misses: number
        ttl_seconds: number
    }[]
}

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#A855F7', '#06B6D4']

export default function CacheMetrics() {
    const queryClient = useQueryClient()

    const { data: metrics, isLoading, refetch } = useQuery<CacheMetrics>({
        queryKey: ['cache-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/cache/metrics')
            return data
        },
        refetchInterval: 10000
    })

    const clearCacheMutation = useMutation({
        mutationFn: async (cacheName?: string) => {
            const { data } = await axiosInstance.post('/cache/clear', { cache_name: cacheName })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['cache-metrics'] })
        }
    })

    const usagePercent = metrics ? (metrics.total_size_mb / metrics.max_size_mb) * 100 : 0

    const pieData = metrics?.caches.map(cache => ({
        name: cache.name,
        value: cache.size_mb
    })) || []

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg">
                            <Cpu className="w-5 h-5 text-white" />
                        </div>
                        Cache Metrics
                    </h1>
                    <p className="page-subtitle">Monitor and manage application cache performance</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => clearCacheMutation.mutate(undefined)}
                        disabled={clearCacheMutation.isPending}
                        className="btn-secondary flex items-center gap-2"
                    >
                        <Trash2 className="w-4 h-4" />
                        Clear All
                    </button>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Memory Usage */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                <Database className="w-5 h-5 text-blue-400" />
                            </div>
                            <span className="font-semibold text-white">Memory</span>
                        </div>
                    </div>
                    <div className="text-2xl font-bold text-white mb-2">
                        {metrics?.total_size_mb.toFixed(1) ?? 0} MB
                    </div>
                    <div className="flex items-center justify-between text-sm text-slate-400 mb-2">
                        <span>of {metrics?.max_size_mb ?? 0} MB</span>
                        <span>{usagePercent.toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                            className={`h-full transition-all ${
                                usagePercent < 50 ? 'bg-emerald-500' :
                                usagePercent < 80 ? 'bg-amber-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${usagePercent}%` }}
                        />
                    </div>
                </div>

                {/* Hit Rate */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                <TrendingUp className="w-5 h-5 text-emerald-400" />
                            </div>
                            <span className="font-semibold text-white">Hit Rate</span>
                        </div>
                    </div>
                    <div className={`text-2xl font-bold ${
                        (metrics?.hit_rate ?? 0) >= 80 ? 'text-emerald-400' :
                        (metrics?.hit_rate ?? 0) >= 50 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                        {metrics?.hit_rate.toFixed(1) ?? 0}%
                    </div>
                    <div className="text-sm text-slate-400 mt-2">
                        {metrics?.hit_count.toLocaleString() ?? 0} hits / {metrics?.miss_count.toLocaleString() ?? 0} misses
                    </div>
                </div>

                {/* Items */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                                <Zap className="w-5 h-5 text-purple-400" />
                            </div>
                            <span className="font-semibold text-white">Cached Items</span>
                        </div>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {metrics?.item_count.toLocaleString() ?? 0}
                    </div>
                    <div className="text-sm text-slate-400 mt-2">
                        {metrics?.evictions.toLocaleString() ?? 0} evictions
                    </div>
                </div>

                {/* Last Cleared */}
                <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                                <Clock className="w-5 h-5 text-amber-400" />
                            </div>
                            <span className="font-semibold text-white">Last Cleared</span>
                        </div>
                    </div>
                    <div className="text-lg font-bold text-white">
                        {metrics?.last_cleared ? new Date(metrics.last_cleared).toLocaleString() : 'Never'}
                    </div>
                </div>
            </div>

            {/* Cache Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Pie Chart */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">Cache Distribution</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    dataKey="value"
                                    nameKey="name"
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    label={({ name, value }) => `${name}: ${value.toFixed(1)}MB`}
                                    labelLine={false}
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Cache List */}
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">Individual Caches</h3>
                    <div className="space-y-3">
                        {metrics?.caches.map((cache, idx) => (
                            <div key={cache.name} className="bg-slate-800/40 rounded-xl p-4">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-3">
                                        <div
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                                        />
                                        <span className="font-medium text-white">{cache.name}</span>
                                    </div>
                                    <button
                                        onClick={() => clearCacheMutation.mutate(cache.name)}
                                        className="text-sm text-slate-400 hover:text-red-400 transition-colors"
                                    >
                                        Clear
                                    </button>
                                </div>
                                <div className="grid grid-cols-4 gap-4 text-sm">
                                    <div>
                                        <span className="text-slate-400">Size</span>
                                        <div className="text-white font-mono">{cache.size_mb.toFixed(2)} MB</div>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Items</span>
                                        <div className="text-white font-mono">{cache.items.toLocaleString()}</div>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">Hit Rate</span>
                                        <div className={`font-mono ${
                                            cache.hits / (cache.hits + cache.misses) >= 0.8 ? 'text-emerald-400' : 'text-amber-400'
                                        }`}>
                                            {((cache.hits / (cache.hits + cache.misses || 1)) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div>
                                        <span className="text-slate-400">TTL</span>
                                        <div className="text-white font-mono">{cache.ttl_seconds}s</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Status */}
            {clearCacheMutation.isSuccess && (
                <div className="glass-card p-4 border-emerald-500/30 bg-emerald-500/5">
                    <div className="flex items-center gap-3 text-emerald-400">
                        <CheckCircle className="w-5 h-5" />
                        <span>Cache cleared successfully</span>
                    </div>
                </div>
            )}

            {clearCacheMutation.isError && (
                <div className="glass-card p-4 border-red-500/30 bg-red-500/5">
                    <div className="flex items-center gap-3 text-red-400">
                        <AlertCircle className="w-5 h-5" />
                        <span>Failed to clear cache</span>
                    </div>
                </div>
            )}
        </div>
    )
}
