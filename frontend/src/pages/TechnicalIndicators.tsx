import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    LineChart, RefreshCw, Search, TrendingUp, TrendingDown,
    Activity, BarChart3, Zap, ArrowUpRight, ArrowDownRight,
    Loader2
} from 'lucide-react'
import {
    ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ReferenceLine
} from 'recharts'

const TIMEFRAMES = ['1D', '1W', '1M', '3M', '6M', '1Y']

interface TechnicalData {
    symbol: string
    price: number
    change_pct: number
    indicators: {
        rsi: number
        macd: { value: number; signal: number; histogram: number }
        stochastic: { k: number; d: number }
        bollinger: { upper: number; middle: number; lower: number }
        atr: number
        iv_rank: number
        volume_ratio: number
        adx: number
        trend: 'Bullish' | 'Bearish' | 'Neutral'
    }
    signals: {
        indicator: string
        signal: 'Buy' | 'Sell' | 'Neutral'
        strength: number
        description: string
    }[]
    chart_data: { date: string; close: number; volume: number; rsi: number }[]
}

export default function TechnicalIndicators() {
    const [symbol, setSymbol] = useState('AAPL')
    const [searchInput, setSearchInput] = useState('AAPL')
    const [timeframe, setTimeframe] = useState('1M')

    const { data: technicals, isLoading, refetch } = useQuery<TechnicalData>({
        queryKey: ['technicals', symbol, timeframe],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/technicals/${symbol}?timeframe=${timeframe}`)
            return data
        },
        staleTime: 60000
    })

    const handleSearch = () => {
        if (searchInput.trim()) {
            setSymbol(searchInput.trim().toUpperCase())
        }
    }

    const getSignalColor = (signal: string) => {
        switch (signal) {
            case 'Buy': return 'text-emerald-400 bg-emerald-500/20'
            case 'Sell': return 'text-red-400 bg-red-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getRSIColor = (rsi: number) => {
        if (rsi >= 70) return 'text-red-400'
        if (rsi <= 30) return 'text-emerald-400'
        return 'text-slate-300'
    }

    const getTrendIcon = (trend: string) => {
        switch (trend) {
            case 'Bullish': return <TrendingUp className="w-5 h-5 text-emerald-400" />
            case 'Bearish': return <TrendingDown className="w-5 h-5 text-red-400" />
            default: return <Activity className="w-5 h-5 text-slate-400" />
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-lg">
                            <LineChart className="w-5 h-5 text-white" />
                        </div>
                        Technical Indicators
                    </h1>
                    <p className="page-subtitle">Comprehensive technical analysis dashboard</p>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Search and Controls */}
            <div className="glass-card p-5">
                <div className="flex flex-wrap items-center gap-4">
                    {/* Symbol Search */}
                    <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                        <div className="relative flex-1">
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
                        <button onClick={handleSearch} className="btn-primary px-4">
                            Analyze
                        </button>
                    </div>

                    {/* Timeframe */}
                    <div className="flex items-center gap-2">
                        {TIMEFRAMES.map(tf => (
                            <button
                                key={tf}
                                onClick={() => setTimeframe(tf)}
                                className={`px-3 py-1.5 text-sm rounded-lg font-medium transition-all ${
                                    timeframe === tf
                                        ? 'bg-primary text-white'
                                        : 'bg-slate-800/60 text-slate-400 hover:text-white'
                                }`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {isLoading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
            ) : technicals ? (
                <>
                    {/* Symbol Header */}
                    <div className="glass-card p-5">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className="text-3xl font-bold text-white">{technicals.symbol}</div>
                                <div className="text-2xl font-mono text-white">${technicals.price.toFixed(2)}</div>
                                <div className={`flex items-center gap-1 ${
                                    technicals.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                                }`}>
                                    {technicals.change_pct >= 0 ? (
                                        <ArrowUpRight className="w-5 h-5" />
                                    ) : (
                                        <ArrowDownRight className="w-5 h-5" />
                                    )}
                                    <span className="font-semibold">{Math.abs(technicals.change_pct).toFixed(2)}%</span>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                {getTrendIcon(technicals.indicators.trend)}
                                <span className={`font-semibold ${
                                    technicals.indicators.trend === 'Bullish' ? 'text-emerald-400' :
                                    technicals.indicators.trend === 'Bearish' ? 'text-red-400' : 'text-slate-400'
                                }`}>
                                    {technicals.indicators.trend}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Indicator Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                        {/* RSI */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">RSI (14)</div>
                            <div className={`text-2xl font-bold ${getRSIColor(technicals.indicators.rsi)}`}>
                                {technicals.indicators.rsi.toFixed(1)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                {technicals.indicators.rsi >= 70 ? 'Overbought' :
                                 technicals.indicators.rsi <= 30 ? 'Oversold' : 'Neutral'}
                            </div>
                        </div>

                        {/* MACD */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">MACD</div>
                            <div className={`text-2xl font-bold ${
                                technicals.indicators.macd.histogram > 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                                {technicals.indicators.macd.value.toFixed(2)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                Signal: {technicals.indicators.macd.signal.toFixed(2)}
                            </div>
                        </div>

                        {/* Stochastic */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">Stochastic</div>
                            <div className="text-2xl font-bold text-white">
                                {technicals.indicators.stochastic.k.toFixed(0)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                %K: {technicals.indicators.stochastic.k.toFixed(0)} / %D: {technicals.indicators.stochastic.d.toFixed(0)}
                            </div>
                        </div>

                        {/* ATR */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">ATR (14)</div>
                            <div className="text-2xl font-bold text-white">
                                ${technicals.indicators.atr.toFixed(2)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                Avg True Range
                            </div>
                        </div>

                        {/* IV Rank */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">IV Rank</div>
                            <div className={`text-2xl font-bold ${
                                technicals.indicators.iv_rank >= 50 ? 'text-amber-400' : 'text-slate-300'
                            }`}>
                                {technicals.indicators.iv_rank.toFixed(0)}%
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                {technicals.indicators.iv_rank >= 50 ? 'High IV' : 'Low IV'}
                            </div>
                        </div>

                        {/* ADX */}
                        <div className="glass-card p-4">
                            <div className="text-sm text-slate-400 mb-1">ADX</div>
                            <div className={`text-2xl font-bold ${
                                technicals.indicators.adx >= 25 ? 'text-purple-400' : 'text-slate-300'
                            }`}>
                                {technicals.indicators.adx.toFixed(1)}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">
                                {technicals.indicators.adx >= 25 ? 'Strong Trend' : 'Weak Trend'}
                            </div>
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-blue-400" />
                            Price & RSI Chart
                        </h3>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                                <ComposedChart data={technicals.chart_data}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.5)" />
                                    <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                                    <YAxis yAxisId="price" stroke="#94a3b8" fontSize={12} domain={['auto', 'auto']} />
                                    <YAxis yAxisId="rsi" orientation="right" stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                            border: '1px solid rgba(51, 65, 85, 0.5)',
                                            borderRadius: '0.75rem'
                                        }}
                                    />
                                    <Legend />
                                    <Bar yAxisId="price" dataKey="volume" fill="rgba(59, 130, 246, 0.3)" name="Volume" />
                                    <Line yAxisId="price" type="monotone" dataKey="close" stroke="#10B981" strokeWidth={2} dot={false} name="Price" />
                                    <Line yAxisId="rsi" type="monotone" dataKey="rsi" stroke="#F59E0B" strokeWidth={1.5} dot={false} name="RSI" />
                                    <ReferenceLine yAxisId="rsi" y={70} stroke="#EF4444" strokeDasharray="3 3" />
                                    <ReferenceLine yAxisId="rsi" y={30} stroke="#10B981" strokeDasharray="3 3" />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Signals */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Zap className="w-5 h-5 text-amber-400" />
                            Trading Signals
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {technicals.signals.map((signal, idx) => (
                                <div key={idx} className="bg-slate-800/40 rounded-xl p-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-semibold text-white">{signal.indicator}</span>
                                        <span className={`px-2.5 py-1 rounded-lg text-sm font-medium ${getSignalColor(signal.signal)}`}>
                                            {signal.signal}
                                        </span>
                                    </div>
                                    <div className="text-sm text-slate-400 mb-2">{signal.description}</div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-slate-500">Strength:</span>
                                        <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full ${
                                                    signal.signal === 'Buy' ? 'bg-emerald-500' :
                                                    signal.signal === 'Sell' ? 'bg-red-500' : 'bg-slate-500'
                                                }`}
                                                style={{ width: `${signal.strength}%` }}
                                            />
                                        </div>
                                        <span className="text-xs text-slate-400">{signal.strength}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Bollinger Bands */}
                    <div className="glass-card p-5">
                        <h3 className="text-lg font-semibold text-white mb-4">Bollinger Bands</h3>
                        <div className="grid grid-cols-3 gap-4">
                            <div className="bg-slate-800/40 rounded-xl p-4 text-center">
                                <div className="text-slate-400 text-sm mb-1">Upper Band</div>
                                <div className="text-xl font-bold text-red-400">
                                    ${technicals.indicators.bollinger.upper.toFixed(2)}
                                </div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-4 text-center">
                                <div className="text-slate-400 text-sm mb-1">Middle (SMA)</div>
                                <div className="text-xl font-bold text-white">
                                    ${technicals.indicators.bollinger.middle.toFixed(2)}
                                </div>
                            </div>
                            <div className="bg-slate-800/40 rounded-xl p-4 text-center">
                                <div className="text-slate-400 text-sm mb-1">Lower Band</div>
                                <div className="text-xl font-bold text-emerald-400">
                                    ${technicals.indicators.bollinger.lower.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="glass-card p-12 text-center">
                    <LineChart className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">Enter a Symbol</h3>
                    <p className="text-slate-400">
                        Search for a stock symbol above to view technical analysis
                    </p>
                </div>
            )}
        </div>
    )
}
