import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Shield, RefreshCw, AlertTriangle, TrendingDown, DollarSign,
    Percent, PieChart, BarChart3, Target, Activity, Zap
} from 'lucide-react'
import clsx from 'clsx'

interface RiskMetrics {
    portfolio_value: number
    total_risk: number
    risk_pct: number
    max_drawdown: number
    var_95: number
    var_99: number
    beta: number
    sharpe_ratio: number
    sortino_ratio: number
    correlation_spy: number
    positions_at_risk: {
        symbol: string
        risk_amount: number
        risk_pct: number
        delta: number
        position_size: number
    }[]
    sector_exposure: { sector: string; value: number; pct: number }[]
    greeks: {
        total_delta: number
        total_gamma: number
        total_theta: number
        total_vega: number
    }
}

export default function RiskDashboard() {
    const { data, isLoading, refetch } = useQuery<RiskMetrics>({
        queryKey: ['risk-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/risk')
            return data
        },
        staleTime: 60000,
    })

    const getRiskLevel = (pct: number) => {
        if (pct <= 5) return { label: 'Low', color: 'text-emerald-400', bg: 'bg-emerald-500/20' }
        if (pct <= 15) return { label: 'Moderate', color: 'text-amber-400', bg: 'bg-amber-500/20' }
        return { label: 'High', color: 'text-rose-400', bg: 'bg-rose-500/20' }
    }

    const riskLevel = data ? getRiskLevel(data.risk_pct) : { label: 'Unknown', color: 'text-slate-400', bg: 'bg-slate-700' }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Shield className="w-5 h-5 text-white" />
                        </div>
                        Risk Dashboard
                    </h1>
                    <p className="page-subtitle">Portfolio risk analysis and exposure monitoring</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Calculating risk metrics...</span>
                </div>
            ) : data ? (
                <>
                    {/* Risk Overview */}
                    <div className="card p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold">Overall Risk Level</h3>
                            <span className={clsx("px-3 py-1 rounded-full text-sm font-bold", riskLevel.bg, riskLevel.color)}>
                                {riskLevel.label} Risk
                            </span>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-slate-400 mb-1">
                                    <DollarSign className="w-4 h-4" />
                                    <span className="text-sm">Portfolio Value</span>
                                </div>
                                <p className="text-2xl font-bold">${data.portfolio_value.toLocaleString()}</p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-slate-400 mb-1">
                                    <AlertTriangle className="w-4 h-4" />
                                    <span className="text-sm">Total Risk</span>
                                </div>
                                <p className={clsx("text-2xl font-bold", riskLevel.color)}>
                                    ${data.total_risk.toLocaleString()}
                                </p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-slate-400 mb-1">
                                    <Percent className="w-4 h-4" />
                                    <span className="text-sm">Risk %</span>
                                </div>
                                <p className={clsx("text-2xl font-bold", riskLevel.color)}>
                                    {data.risk_pct.toFixed(1)}%
                                </p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-4">
                                <div className="flex items-center gap-2 text-slate-400 mb-1">
                                    <TrendingDown className="w-4 h-4" />
                                    <span className="text-sm">Max Drawdown</span>
                                </div>
                                <p className="text-2xl font-bold text-rose-400">
                                    {data.max_drawdown.toFixed(1)}%
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* VaR & Ratios */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="card p-4">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <BarChart3 className="w-5 h-5 text-blue-400" />
                                Value at Risk (VaR)
                            </h3>
                            <div className="space-y-4">
                                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                                    <span className="text-slate-400">VaR 95%</span>
                                    <span className="font-mono font-bold text-amber-400">
                                        ${data.var_95.toLocaleString()}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                                    <span className="text-slate-400">VaR 99%</span>
                                    <span className="font-mono font-bold text-rose-400">
                                        ${data.var_99.toLocaleString()}
                                    </span>
                                </div>
                                <p className="text-xs text-slate-500">
                                    95% VaR: 5% chance of losing more than this amount in one day
                                </p>
                            </div>
                        </div>

                        <div className="card p-4">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Target className="w-5 h-5 text-purple-400" />
                                Risk-Adjusted Metrics
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <p className="text-xs text-slate-400">Beta</p>
                                    <p className={clsx(
                                        "text-xl font-bold",
                                        data.beta <= 1 ? "text-emerald-400" : "text-amber-400"
                                    )}>
                                        {data.beta.toFixed(2)}
                                    </p>
                                </div>
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <p className="text-xs text-slate-400">SPY Correlation</p>
                                    <p className="text-xl font-bold">{(data.correlation_spy * 100).toFixed(0)}%</p>
                                </div>
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <p className="text-xs text-slate-400">Sharpe Ratio</p>
                                    <p className={clsx(
                                        "text-xl font-bold",
                                        data.sharpe_ratio >= 1 ? "text-emerald-400" : "text-amber-400"
                                    )}>
                                        {data.sharpe_ratio.toFixed(2)}
                                    </p>
                                </div>
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <p className="text-xs text-slate-400">Sortino Ratio</p>
                                    <p className={clsx(
                                        "text-xl font-bold",
                                        data.sortino_ratio >= 1.5 ? "text-emerald-400" : "text-amber-400"
                                    )}>
                                        {data.sortino_ratio.toFixed(2)}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Greeks Summary */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Zap className="w-5 h-5 text-amber-400" />
                            Portfolio Greeks
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-sm text-slate-400">Delta</p>
                                <p className={clsx(
                                    "text-2xl font-bold",
                                    data.greeks.total_delta >= 0 ? "text-emerald-400" : "text-rose-400"
                                )}>
                                    {data.greeks.total_delta.toFixed(0)}
                                </p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-sm text-slate-400">Gamma</p>
                                <p className="text-2xl font-bold text-blue-400">
                                    {data.greeks.total_gamma.toFixed(1)}
                                </p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-sm text-slate-400">Theta</p>
                                <p className={clsx(
                                    "text-2xl font-bold",
                                    data.greeks.total_theta >= 0 ? "text-emerald-400" : "text-rose-400"
                                )}>
                                    ${data.greeks.total_theta.toFixed(0)}
                                </p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                                <p className="text-sm text-slate-400">Vega</p>
                                <p className="text-2xl font-bold text-purple-400">
                                    ${data.greeks.total_vega.toFixed(0)}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Positions at Risk */}
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5 text-rose-400" />
                            Positions at Risk
                        </h3>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-slate-700/50">
                                        <th className="text-left p-3 text-sm text-slate-400">Symbol</th>
                                        <th className="text-right p-3 text-sm text-slate-400">Position</th>
                                        <th className="text-right p-3 text-sm text-slate-400">Risk $</th>
                                        <th className="text-right p-3 text-sm text-slate-400">Risk %</th>
                                        <th className="text-right p-3 text-sm text-slate-400">Delta</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.positions_at_risk.map((pos, idx) => (
                                        <tr key={idx} className={idx % 2 === 0 ? "bg-slate-800/20" : ""}>
                                            <td className="p-3 font-mono font-bold text-primary">{pos.symbol}</td>
                                            <td className="p-3 text-right font-mono">${pos.position_size.toLocaleString()}</td>
                                            <td className="p-3 text-right font-mono text-rose-400">
                                                ${pos.risk_amount.toLocaleString()}
                                            </td>
                                            <td className="p-3 text-right">
                                                <span className={clsx(
                                                    "px-2 py-0.5 rounded text-xs",
                                                    pos.risk_pct <= 2 ? "bg-emerald-500/20 text-emerald-400" :
                                                    pos.risk_pct <= 5 ? "bg-amber-500/20 text-amber-400" :
                                                    "bg-rose-500/20 text-rose-400"
                                                )}>
                                                    {pos.risk_pct.toFixed(1)}%
                                                </span>
                                            </td>
                                            <td className="p-3 text-right font-mono">{pos.delta.toFixed(2)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            ) : null}
        </div>
    )
}
