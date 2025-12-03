import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Layers, RefreshCw, Search, TrendingUp, TrendingDown,
    Target, AlertCircle, ArrowUpRight, ArrowDownRight, Loader2
} from 'lucide-react'
import {
    ResponsiveContainer, ComposedChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, ReferenceArea, ReferenceLine
} from 'recharts'

interface Zone {
    type: 'supply' | 'demand'
    price_high: number
    price_low: number
    strength: number
    touches: number
    created_at: string
    last_tested: string
    status: 'fresh' | 'tested' | 'broken'
}

interface ZoneAnalysis {
    symbol: string
    current_price: number
    change_pct: number
    trend: 'bullish' | 'bearish' | 'neutral'
    zones: Zone[]
    nearest_supply: Zone | null
    nearest_demand: Zone | null
    price_history: { date: string; price: number }[]
    support_levels: number[]
    resistance_levels: number[]
}

export default function SupplyDemandZones() {
    const [symbol, setSymbol] = useState('SPY')
    const [searchInput, setSearchInput] = useState('SPY')
    const [timeframe, setTimeframe] = useState('1D')

    const { data: analysis, isLoading, refetch } = useQuery<ZoneAnalysis>({
        queryKey: ['supply-demand', symbol, timeframe],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/technicals/supply-demand/${symbol}?timeframe=${timeframe}`)
            return data
        },
        staleTime: 60000
    })

    const handleSearch = () => {
        if (searchInput.trim()) {
            setSymbol(searchInput.trim().toUpperCase())
        }
    }

    const getZoneColor = (zone: Zone) => {
        if (zone.type === 'supply') {
            return zone.status === 'fresh' ? 'rgba(239, 68, 68, 0.3)' :
                   zone.status === 'tested' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(239, 68, 68, 0.05)'
        }
        return zone.status === 'fresh' ? 'rgba(16, 185, 129, 0.3)' :
               zone.status === 'tested' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.05)'
    }

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'fresh': return 'bg-emerald-500/20 text-emerald-400'
            case 'tested': return 'bg-amber-500/20 text-amber-400'
            case 'broken': return 'bg-slate-500/20 text-slate-400'
            default: return 'bg-slate-500/20 text-slate-400'
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Layers className="w-5 h-5 text-white" />
                        </div>
                        Supply & Demand Zones
                    </h1>
                    <p className="page-subtitle">Identify key price zones and support/resistance levels</p>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Search & Filters */}
            <div className="glass-card p-5">
                <div className="flex flex-wrap items-center gap-4">
                    <div className="relative flex-1 max-w-md">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            value={searchInput}
                            onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="Enter symbol..."
                            className="input-field pl-10"
                        />
                    </div>
                    <select
                        value={timeframe}
                        onChange={(e) => setTimeframe(e.target.value)}
                        className="input-field w-32"
                    >
                        <option value="1H">1 Hour</option>
                        <option value="4H">4 Hour</option>
                        <option value="1D">Daily</option>
                        <option value="1W">Weekly</option>
                    </select>
                    <button onClick={handleSearch} className="btn-primary px-6">
                        Analyze
                    </button>
                </div>
            </div>

            {isLoading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
            ) : analysis ? (
                <>
                    {/* Symbol Overview */}
                    <div className="glass-card p-5">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-6">
                                <div>
                                    <div className="text-3xl font-bold text-white">{analysis.symbol}</div>
                                    <div className="text-sm text-slate-400">Supply & Demand Analysis</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-mono text-white">${(analysis.current_price ?? 0).toFixed(2)}</div>
                                    <div className={`flex items-center gap-1 text-sm ${
                                        (analysis.change_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                        {(analysis.change_pct ?? 0) >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                                        {Math.abs(analysis.change_pct ?? 0).toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                            <div className={`px-4 py-2 rounded-xl text-lg font-medium ${
                                analysis.trend === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' :
                                analysis.trend === 'bearish' ? 'bg-red-500/20 text-red-400' :
                                'bg-slate-500/20 text-slate-400'
                            }`}>
                                {analysis.trend.charAt(0).toUpperCase() + analysis.trend.slice(1)} Trend
                            </div>
                        </div>
                    </div>

                    {/* Nearest Zones */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Nearest Supply */}
                        <div className="glass-card p-5 border-red-500/30">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                                    <TrendingDown className="w-5 h-5 text-red-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">Nearest Supply Zone</h3>
                                    <p className="text-sm text-slate-400">Resistance above price</p>
                                </div>
                            </div>
                            {analysis.nearest_supply ? (
                                <div className="space-y-3">
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Zone Range</span>
                                        <span className="text-red-400 font-mono">
                                            ${(analysis.nearest_supply.price_low ?? 0).toFixed(2)} - ${(analysis.nearest_supply.price_high ?? 0).toFixed(2)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Distance</span>
                                        <span className="text-white font-mono">
                                            {((((analysis.nearest_supply?.price_low ?? 0) - (analysis.current_price ?? 0)) / (analysis.current_price || 1)) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Strength</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-20 h-2 bg-slate-700 rounded-full">
                                                <div className="h-full bg-red-500 rounded-full" style={{ width: `${analysis.nearest_supply.strength}%` }} />
                                            </div>
                                            <span className="text-white text-sm">{analysis.nearest_supply.strength}%</span>
                                        </div>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Status</span>
                                        <span className={`px-2 py-0.5 rounded-lg text-xs ${getStatusBadge(analysis.nearest_supply.status)}`}>
                                            {analysis.nearest_supply.status}
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-slate-400 py-4">No supply zone found above</div>
                            )}
                        </div>

                        {/* Nearest Demand */}
                        <div className="glass-card p-5 border-emerald-500/30">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                    <TrendingUp className="w-5 h-5 text-emerald-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">Nearest Demand Zone</h3>
                                    <p className="text-sm text-slate-400">Support below price</p>
                                </div>
                            </div>
                            {analysis.nearest_demand ? (
                                <div className="space-y-3">
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Zone Range</span>
                                        <span className="text-emerald-400 font-mono">
                                            ${(analysis.nearest_demand.price_low ?? 0).toFixed(2)} - ${(analysis.nearest_demand.price_high ?? 0).toFixed(2)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Distance</span>
                                        <span className="text-white font-mono">
                                            {((((analysis.current_price ?? 0) - (analysis.nearest_demand?.price_high ?? 0)) / (analysis.current_price || 1)) * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Strength</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-20 h-2 bg-slate-700 rounded-full">
                                                <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${analysis.nearest_demand.strength}%` }} />
                                            </div>
                                            <span className="text-white text-sm">{analysis.nearest_demand.strength}%</span>
                                        </div>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Status</span>
                                        <span className={`px-2 py-0.5 rounded-lg text-xs ${getStatusBadge(analysis.nearest_demand.status)}`}>
                                            {analysis.nearest_demand.status}
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-slate-400 py-4">No demand zone found below</div>
                            )}
                        </div>
                    </div>

                    {/* Price Chart with Zones */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4">Price Chart with Zones</h3>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                                <ComposedChart data={analysis.price_history}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                                    <YAxis stroke="#94a3b8" fontSize={12} domain={['auto', 'auto']} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1e293b',
                                            border: '1px solid #334155',
                                            borderRadius: '0.5rem'
                                        }}
                                    />

                                    {/* Supply Zones */}
                                    {analysis.zones.filter(z => z.type === 'supply').map((zone, idx) => (
                                        <ReferenceArea
                                            key={`supply-${idx}`}
                                            y1={zone.price_low}
                                            y2={zone.price_high}
                                            fill={getZoneColor(zone)}
                                            stroke="rgba(239, 68, 68, 0.5)"
                                        />
                                    ))}

                                    {/* Demand Zones */}
                                    {analysis.zones.filter(z => z.type === 'demand').map((zone, idx) => (
                                        <ReferenceArea
                                            key={`demand-${idx}`}
                                            y1={zone.price_low}
                                            y2={zone.price_high}
                                            fill={getZoneColor(zone)}
                                            stroke="rgba(16, 185, 129, 0.5)"
                                        />
                                    ))}

                                    {/* Support/Resistance Lines */}
                                    {analysis.support_levels.map((level, idx) => (
                                        <ReferenceLine
                                            key={`support-${idx}`}
                                            y={level}
                                            stroke="#10B981"
                                            strokeDasharray="5 5"
                                        />
                                    ))}
                                    {analysis.resistance_levels.map((level, idx) => (
                                        <ReferenceLine
                                            key={`resistance-${idx}`}
                                            y={level}
                                            stroke="#EF4444"
                                            strokeDasharray="5 5"
                                        />
                                    ))}

                                    {/* Current Price */}
                                    <ReferenceLine y={analysis.current_price} stroke="#3B82F6" strokeWidth={2} />

                                    <Line
                                        type="monotone"
                                        dataKey="price"
                                        stroke="#fff"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* All Zones */}
                    <div className="glass-card overflow-hidden">
                        <div className="p-5 border-b border-slate-700/50">
                            <h3 className="text-lg font-semibold text-white">All Identified Zones</h3>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th>Type</th>
                                        <th>Range</th>
                                        <th>Strength</th>
                                        <th>Touches</th>
                                        <th>Status</th>
                                        <th>Last Tested</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {analysis.zones.map((zone, idx) => (
                                        <tr key={idx}>
                                            <td>
                                                <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                                                    zone.type === 'supply' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'
                                                }`}>
                                                    {zone.type.toUpperCase()}
                                                </span>
                                            </td>
                                            <td className="font-mono text-white">
                                                ${(zone.price_low ?? 0).toFixed(2)} - ${(zone.price_high ?? 0).toFixed(2)}
                                            </td>
                                            <td>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-16 h-2 bg-slate-700 rounded-full">
                                                        <div
                                                            className={`h-full rounded-full ${zone.type === 'supply' ? 'bg-red-500' : 'bg-emerald-500'}`}
                                                            style={{ width: `${zone.strength}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm text-slate-400">{zone.strength}%</span>
                                                </div>
                                            </td>
                                            <td className="text-slate-300">{zone.touches}</td>
                                            <td>
                                                <span className={`px-2 py-0.5 rounded-lg text-xs ${getStatusBadge(zone.status)}`}>
                                                    {zone.status}
                                                </span>
                                            </td>
                                            <td className="text-slate-400 text-sm">{zone.last_tested}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Key Levels */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Target className="w-5 h-5 text-red-400" />
                                Resistance Levels
                            </h3>
                            <div className="space-y-2">
                                {analysis.resistance_levels.map((level, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-red-500/10 rounded-lg">
                                        <span className="text-sm text-slate-400">R{idx + 1}</span>
                                        <span className="font-mono text-red-400">${(level ?? 0).toFixed(2)}</span>
                                        <span className="text-xs text-slate-500">
                                            +{((((level ?? 0) - (analysis.current_price ?? 0)) / (analysis.current_price || 1)) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="glass-card p-5">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Target className="w-5 h-5 text-emerald-400" />
                                Support Levels
                            </h3>
                            <div className="space-y-2">
                                {analysis.support_levels.map((level, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-emerald-500/10 rounded-lg">
                                        <span className="text-sm text-slate-400">S{idx + 1}</span>
                                        <span className="font-mono text-emerald-400">${(level ?? 0).toFixed(2)}</span>
                                        <span className="text-xs text-slate-500">
                                            {((((level ?? 0) - (analysis.current_price ?? 0)) / (analysis.current_price || 1)) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="glass-card p-12 text-center">
                    <Layers className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Enter a Symbol</h3>
                    <p className="text-slate-400">Search for a stock to analyze supply and demand zones</p>
                </div>
            )}
        </div>
    )
}
