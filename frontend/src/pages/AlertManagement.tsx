import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Bell, RefreshCw, Plus, Trash2, Edit2, CheckCircle, XCircle,
    TrendingUp, TrendingDown, DollarSign, Percent, Clock, Zap
} from 'lucide-react'
import clsx from 'clsx'

interface Alert {
    id: string
    name: string
    symbol: string
    condition: 'price_above' | 'price_below' | 'percent_change' | 'volume_spike' | 'iv_above' | 'iv_below'
    value: number
    status: 'active' | 'triggered' | 'disabled'
    created_at: string
    triggered_at?: string
    notification_channels: ('email' | 'telegram' | 'push')[]
}

const CONDITIONS = [
    { id: 'price_above', label: 'Price Above', icon: TrendingUp },
    { id: 'price_below', label: 'Price Below', icon: TrendingDown },
    { id: 'percent_change', label: 'Percent Change', icon: Percent },
    { id: 'volume_spike', label: 'Volume Spike', icon: Zap },
    { id: 'iv_above', label: 'IV Above', icon: TrendingUp },
    { id: 'iv_below', label: 'IV Below', icon: TrendingDown },
]

export default function AlertManagement() {
    const queryClient = useQueryClient()
    const [showForm, setShowForm] = useState(false)
    const [editingAlert, setEditingAlert] = useState<Alert | null>(null)

    // Form state
    const [name, setName] = useState('')
    const [symbol, setSymbol] = useState('')
    const [condition, setCondition] = useState<Alert['condition']>('price_above')
    const [value, setValue] = useState(0)
    const [channels, setChannels] = useState<Alert['notification_channels']>(['push'])

    const { data, isLoading, refetch } = useQuery({
        queryKey: ['alerts'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/alerts')
            return data
        },
        staleTime: 30000,
    })

    const alerts: Alert[] = data?.alerts || []
    const activeCount = alerts.filter(a => a.status === 'active').length
    const triggeredCount = alerts.filter(a => a.status === 'triggered').length

    const createMutation = useMutation({
        mutationFn: async (alert: Partial<Alert>) => {
            const { data } = await axiosInstance.post('/portfolio/alerts', alert)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['alerts'] })
            resetForm()
        }
    })

    const deleteMutation = useMutation({
        mutationFn: async (id: string) => {
            await axiosInstance.delete(`/portfolio/alerts/${id}`)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['alerts'] })
        }
    })

    const toggleMutation = useMutation({
        mutationFn: async ({ id, status }: { id: string; status: string }) => {
            const { data } = await axiosInstance.patch(`/portfolio/alerts/${id}`, { status })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['alerts'] })
        }
    })

    const resetForm = () => {
        setShowForm(false)
        setEditingAlert(null)
        setName('')
        setSymbol('')
        setCondition('price_above')
        setValue(0)
        setChannels(['push'])
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        createMutation.mutate({
            name,
            symbol: symbol.toUpperCase(),
            condition,
            value,
            notification_channels: channels
        })
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                            <Bell className="w-5 h-5 text-white" />
                        </div>
                        Alert Management
                    </h1>
                    <p className="page-subtitle">Create and manage price alerts and notifications</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowForm(!showForm)}
                        className="bg-primary hover:bg-primary/80 px-4 py-2 rounded-lg flex items-center gap-2"
                    >
                        <Plus className="w-4 h-4" />
                        New Alert
                    </button>
                    <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </header>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Bell className="w-4 h-4" />
                        <span className="text-sm">Total Alerts</span>
                    </div>
                    <p className="text-2xl font-bold">{alerts.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <CheckCircle className="w-4 h-4" />
                        <span className="text-sm">Active</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">{activeCount}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Zap className="w-4 h-4" />
                        <span className="text-sm">Triggered</span>
                    </div>
                    <p className="text-2xl font-bold text-amber-400">{triggeredCount}</p>
                </div>
            </div>

            {/* Create Alert Form */}
            {showForm && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold mb-4">Create New Alert</h3>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Alert Name</label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={e => setName(e.target.value)}
                                    placeholder="e.g., AAPL Breakout"
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Symbol</label>
                                <input
                                    type="text"
                                    value={symbol}
                                    onChange={e => setSymbol(e.target.value)}
                                    placeholder="AAPL"
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    required
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Condition</label>
                                <select
                                    value={condition}
                                    onChange={e => setCondition(e.target.value as Alert['condition'])}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                >
                                    {CONDITIONS.map(c => (
                                        <option key={c.id} value={c.id}>{c.label}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Value</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={value}
                                    onChange={e => setValue(Number(e.target.value))}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Notification Channels</label>
                            <div className="flex gap-4">
                                {(['email', 'telegram', 'push'] as const).map(ch => (
                                    <label key={ch} className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={channels.includes(ch)}
                                            onChange={e => {
                                                if (e.target.checked) {
                                                    setChannels([...channels, ch])
                                                } else {
                                                    setChannels(channels.filter(c => c !== ch))
                                                }
                                            }}
                                            className="w-4 h-4 rounded"
                                        />
                                        <span className="capitalize">{ch}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="flex gap-2">
                            <button
                                type="submit"
                                disabled={createMutation.isPending}
                                className="bg-primary hover:bg-primary/80 disabled:opacity-50 px-6 py-2 rounded-lg font-medium"
                            >
                                {createMutation.isPending ? 'Creating...' : 'Create Alert'}
                            </button>
                            <button
                                type="button"
                                onClick={resetForm}
                                className="bg-slate-700 hover:bg-slate-600 px-6 py-2 rounded-lg"
                            >
                                Cancel
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {/* Alerts List */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Loading alerts...</span>
                </div>
            ) : alerts.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No alerts created yet</p>
                    <p className="text-sm mt-1">Create your first alert to get started</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {alerts.map(alert => {
                        const conditionInfo = CONDITIONS.find(c => c.id === alert.condition)
                        const ConditionIcon = conditionInfo?.icon || Bell

                        return (
                            <div key={alert.id} className="card p-4 hover:border-primary/50 transition-colors">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className={clsx(
                                            "w-10 h-10 rounded-lg flex items-center justify-center",
                                            alert.status === 'active' ? "bg-emerald-500/20" :
                                            alert.status === 'triggered' ? "bg-amber-500/20" :
                                            "bg-slate-700"
                                        )}>
                                            <ConditionIcon className={clsx(
                                                "w-5 h-5",
                                                alert.status === 'active' ? "text-emerald-400" :
                                                alert.status === 'triggered' ? "text-amber-400" :
                                                "text-slate-500"
                                            )} />
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-semibold">{alert.name}</span>
                                                <span className="font-mono text-primary">{alert.symbol}</span>
                                            </div>
                                            <p className="text-sm text-slate-400">
                                                {conditionInfo?.label}: {alert.value}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <span className={clsx(
                                            "px-2 py-1 rounded text-xs font-medium",
                                            alert.status === 'active' ? "bg-emerald-500/20 text-emerald-400" :
                                            alert.status === 'triggered' ? "bg-amber-500/20 text-amber-400" :
                                            "bg-slate-700 text-slate-400"
                                        )}>
                                            {alert.status}
                                        </span>

                                        <button
                                            onClick={() => toggleMutation.mutate({
                                                id: alert.id,
                                                status: alert.status === 'active' ? 'disabled' : 'active'
                                            })}
                                            className="btn-icon text-sm"
                                            title={alert.status === 'active' ? 'Disable' : 'Enable'}
                                        >
                                            {alert.status === 'active' ? (
                                                <XCircle className="w-4 h-4 text-slate-400" />
                                            ) : (
                                                <CheckCircle className="w-4 h-4 text-emerald-400" />
                                            )}
                                        </button>

                                        <button
                                            onClick={() => deleteMutation.mutate(alert.id)}
                                            className="btn-icon text-sm text-rose-400"
                                            title="Delete"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>

                                <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
                                    <span className="flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        Created: {alert.created_at}
                                    </span>
                                    {alert.triggered_at && (
                                        <span className="flex items-center gap-1 text-amber-400">
                                            <Zap className="w-3 h-3" />
                                            Triggered: {alert.triggered_at}
                                        </span>
                                    )}
                                    <span>
                                        Channels: {alert.notification_channels.join(', ')}
                                    </span>
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
