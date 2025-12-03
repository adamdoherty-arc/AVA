import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    CreditCard, RefreshCw, CheckCircle, XCircle, AlertCircle,
    Crown, Zap, Clock, Calendar, DollarSign, Star, Shield,
    ArrowRight, Sparkles
} from 'lucide-react'

interface Subscription {
    id: string
    plan: 'free' | 'basic' | 'pro' | 'enterprise'
    status: 'active' | 'canceled' | 'past_due' | 'trialing'
    current_period_start: string
    current_period_end: string
    cancel_at_period_end: boolean
    amount: number
    currency: string
    features: string[]
}

interface UsageStats {
    api_calls: { used: number; limit: number }
    ai_queries: { used: number; limit: number }
    storage_mb: { used: number; limit: number }
    exports: { used: number; limit: number }
}

interface Plan {
    id: string
    name: string
    price: number
    interval: 'month' | 'year'
    features: string[]
    popular: boolean
    current: boolean
}

export default function SubscriptionManagement() {
    const queryClient = useQueryClient()

    const { data: subscription, isLoading } = useQuery<Subscription>({
        queryKey: ['subscription'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/subscriptions')
            return data
        }
    })

    const { data: usage } = useQuery<UsageStats>({
        queryKey: ['subscription-usage'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/subscriptions/usage')
            return data
        }
    })

    const { data: plans } = useQuery<Plan[]>({
        queryKey: ['subscription-plans'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/subscriptions/plans')
            return data?.plans || []
        }
    })

    const cancelMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/subscriptions/cancel')
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['subscription'] })
        }
    })

    const reactivateMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/subscriptions/reactivate')
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['subscription'] })
        }
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'active': return 'text-emerald-400 bg-emerald-500/20'
            case 'trialing': return 'text-blue-400 bg-blue-500/20'
            case 'past_due': return 'text-amber-400 bg-amber-500/20'
            case 'canceled': return 'text-red-400 bg-red-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getPlanIcon = (plan: string) => {
        switch (plan) {
            case 'free': return Star
            case 'basic': return Zap
            case 'pro': return Crown
            case 'enterprise': return Shield
            default: return Star
        }
    }

    const getUsagePercent = (used: number, limit: number) => {
        if (limit === 0) return 0
        return Math.min((used / limit) * 100, 100)
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
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <CreditCard className="w-5 h-5 text-white" />
                        </div>
                        Subscription Management
                    </h1>
                    <p className="page-subtitle">Manage your plan, billing, and usage</p>
                </div>
            </header>

            {/* Current Subscription */}
            {subscription && (
                <div className="glass-card p-6">
                    <div className="flex items-start justify-between">
                        <div className="flex items-center gap-4">
                            <div className={`w-16 h-16 rounded-xl flex items-center justify-center ${
                                subscription.plan === 'enterprise' ? 'bg-purple-500/20' :
                                subscription.plan === 'pro' ? 'bg-amber-500/20' :
                                subscription.plan === 'basic' ? 'bg-blue-500/20' : 'bg-slate-500/20'
                            }`}>
                                {(() => {
                                    const Icon = getPlanIcon(subscription.plan)
                                    return <Icon className={`w-8 h-8 ${
                                        subscription.plan === 'enterprise' ? 'text-purple-400' :
                                        subscription.plan === 'pro' ? 'text-amber-400' :
                                        subscription.plan === 'basic' ? 'text-blue-400' : 'text-slate-400'
                                    }`} />
                                })()}
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-white capitalize">{subscription.plan} Plan</h2>
                                <span className={`px-3 py-1 rounded-lg text-sm font-medium ${getStatusColor(subscription.status)}`}>
                                    {subscription.status === 'active' ? 'Active' :
                                     subscription.status === 'trialing' ? 'Trial' :
                                     subscription.status === 'past_due' ? 'Past Due' : 'Canceled'}
                                </span>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-3xl font-bold text-white">
                                ${subscription.amount}
                                <span className="text-lg text-slate-400">/mo</span>
                            </div>
                            <div className="text-sm text-slate-400">
                                Renews {new Date(subscription.current_period_end).toLocaleDateString()}
                            </div>
                        </div>
                    </div>

                    {subscription.cancel_at_period_end && (
                        <div className="mt-4 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-amber-400">
                                    <AlertCircle className="w-5 h-5" />
                                    <span>Your subscription will cancel on {new Date(subscription.current_period_end).toLocaleDateString()}</span>
                                </div>
                                <button
                                    onClick={() => reactivateMutation.mutate()}
                                    className="btn-primary text-sm"
                                >
                                    Reactivate
                                </button>
                            </div>
                        </div>
                    )}

                    <div className="mt-6 flex items-center gap-4">
                        {!subscription.cancel_at_period_end && subscription.plan !== 'free' && (
                            <button
                                onClick={() => cancelMutation.mutate()}
                                className="btn-secondary text-red-400 hover:text-red-300"
                            >
                                Cancel Subscription
                            </button>
                        )}
                        <button className="btn-secondary">
                            Update Payment Method
                        </button>
                        <button className="btn-secondary">
                            View Invoices
                        </button>
                    </div>
                </div>
            )}

            {/* Usage Stats */}
            {usage && (
                <div className="glass-card p-5">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Zap className="w-5 h-5 text-amber-400" />
                        Current Usage
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-slate-400 text-sm">API Calls</span>
                                <span className="text-white text-sm">
                                    {(usage.api_calls?.used ?? 0).toLocaleString()} / {(usage.api_calls?.limit ?? 0).toLocaleString()}
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full">
                                <div
                                    className={`h-full rounded-full transition-all ${getUsageColor(getUsagePercent(usage.api_calls?.used ?? 0, usage.api_calls?.limit ?? 1))}`}
                                    style={{ width: `${getUsagePercent(usage.api_calls?.used ?? 0, usage.api_calls?.limit ?? 1)}%` }}
                                />
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-slate-400 text-sm">AI Queries</span>
                                <span className="text-white text-sm">
                                    {(usage.ai_queries?.used ?? 0).toLocaleString()} / {(usage.ai_queries?.limit ?? 0).toLocaleString()}
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full">
                                <div
                                    className={`h-full rounded-full transition-all ${getUsageColor(getUsagePercent(usage.ai_queries?.used ?? 0, usage.ai_queries?.limit ?? 1))}`}
                                    style={{ width: `${getUsagePercent(usage.ai_queries?.used ?? 0, usage.ai_queries?.limit ?? 1)}%` }}
                                />
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-slate-400 text-sm">Storage</span>
                                <span className="text-white text-sm">
                                    {usage.storage_mb?.used ?? 0} MB / {usage.storage_mb?.limit ?? 0} MB
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full">
                                <div
                                    className={`h-full rounded-full transition-all ${getUsageColor(getUsagePercent(usage.storage_mb?.used ?? 0, usage.storage_mb?.limit ?? 1))}`}
                                    style={{ width: `${getUsagePercent(usage.storage_mb?.used ?? 0, usage.storage_mb?.limit ?? 1)}%` }}
                                />
                            </div>
                        </div>
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-slate-400 text-sm">Exports</span>
                                <span className="text-white text-sm">
                                    {usage.exports?.used ?? 0} / {usage.exports?.limit ?? 0}
                                </span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full">
                                <div
                                    className={`h-full rounded-full transition-all ${getUsageColor(getUsagePercent(usage.exports?.used ?? 0, usage.exports?.limit ?? 1))}`}
                                    style={{ width: `${getUsagePercent(usage.exports?.used ?? 0, usage.exports?.limit ?? 1)}%` }}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Available Plans */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Available Plans</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {plans?.map((plan) => {
                        const Icon = getPlanIcon(plan.id)
                        return (
                            <div
                                key={plan.id}
                                className={`relative rounded-xl p-5 border transition-colors ${
                                    plan.current
                                        ? 'bg-primary/10 border-primary'
                                        : plan.popular
                                        ? 'bg-amber-500/5 border-amber-500/30 hover:border-amber-500/50'
                                        : 'bg-slate-800/40 border-slate-700/50 hover:border-slate-600/50'
                                }`}
                            >
                                {plan.popular && (
                                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-amber-500 text-black text-xs font-bold rounded-full">
                                        POPULAR
                                    </div>
                                )}
                                {plan.current && (
                                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-primary text-white text-xs font-bold rounded-full">
                                        CURRENT
                                    </div>
                                )}
                                <div className="flex items-center gap-3 mb-4">
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                                        plan.id === 'enterprise' ? 'bg-purple-500/20' :
                                        plan.id === 'pro' ? 'bg-amber-500/20' :
                                        plan.id === 'basic' ? 'bg-blue-500/20' : 'bg-slate-500/20'
                                    }`}>
                                        <Icon className={`w-5 h-5 ${
                                            plan.id === 'enterprise' ? 'text-purple-400' :
                                            plan.id === 'pro' ? 'text-amber-400' :
                                            plan.id === 'basic' ? 'text-blue-400' : 'text-slate-400'
                                        }`} />
                                    </div>
                                    <div>
                                        <h4 className="font-semibold text-white">{plan.name}</h4>
                                        <div className="text-xl font-bold text-white">
                                            ${plan.price}
                                            <span className="text-sm text-slate-400">/{plan.interval}</span>
                                        </div>
                                    </div>
                                </div>
                                <ul className="space-y-2 mb-4">
                                    {plan.features.slice(0, 5).map((feature, idx) => (
                                        <li key={idx} className="flex items-center gap-2 text-sm text-slate-300">
                                            <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                            {feature}
                                        </li>
                                    ))}
                                </ul>
                                {!plan.current && (
                                    <button className={`w-full py-2 rounded-lg font-medium transition-colors ${
                                        plan.popular
                                            ? 'bg-amber-500 text-black hover:bg-amber-400'
                                            : 'bg-slate-700 text-white hover:bg-slate-600'
                                    }`}>
                                        {plan.price > (subscription?.amount ?? 0) ? 'Upgrade' : 'Downgrade'}
                                    </button>
                                )}
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Features Comparison */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Current Plan Features</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {subscription?.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-2 p-3 bg-slate-800/40 rounded-lg">
                            <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                            <span className="text-sm text-white">{feature}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
