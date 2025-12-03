import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  Cloud, TrendingUp, TrendingDown, Search, RefreshCw, ArrowUpRight, ArrowDownRight
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { API_HOST } from '@/config/api'

interface IchimokuData {
  symbol: string
  current_price: number
  period: string
  ichimoku: {
    tenkan: number
    kijun: number
    senkou_a: number
    senkou_b: number
    cloud_top: number
    cloud_bottom: number
    cloud_color: string
  }
  signal: {
    overall: string
    strength: string
    cloud_position: string
    cloud_bias: string
    tk_bullish_cross: boolean
    tk_bearish_cross: boolean
    recommendation: string
  }
  interpretation: {
    trend: string
    tk_relationship: string
    cloud_future: string
  }
  chart_data: Array<{
    date: string
    close: number
    tenkan: number | null
    kijun: number | null
    senkou_a: number | null
    senkou_b: number | null
  }>
}

const API_BASE = API_HOST

export default function IchimokuCloud() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')
  const [period, setPeriod] = useState('3M')

  const { data, isLoading, error, refetch } = useQuery<IchimokuData>({
    queryKey: ['ichimoku', symbol, period],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/advanced-technicals/ichimoku/${symbol}?period=${period}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase())
    }
  }

  const getSignalColor = (signal: string) => {
    if (signal.includes('BULLISH') || signal === 'BUY') return 'text-green-500'
    if (signal.includes('BEARISH') || signal === 'SELL') return 'text-red-500'
    return 'text-slate-400'
  }

  const getSignalBg = (signal: string) => {
    if (signal.includes('BULLISH')) return 'bg-green-500/10 border-green-500/30'
    if (signal.includes('BEARISH')) return 'bg-red-500/10 border-red-500/30'
    return 'bg-slate-800/50 border-slate-700'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Cloud className="w-7 h-7 text-cyan-500" />
            Ichimoku Cloud Analysis
          </h1>
          <p className="text-slate-400 mt-1">
            Full Ichimoku Cloud with Tenkan-sen, Kijun-sen, Senkou Spans, and Trading Signals
          </p>
        </div>
      </div>

      {/* Search Bar */}
      <Card className="bg-slate-900/50 border-slate-800">
        <CardContent className="p-4">
          <div className="flex gap-4 items-center">
            <div className="flex-1 flex gap-2">
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Enter symbol..."
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              />
              <Button onClick={handleSearch} className="bg-cyan-600 hover:bg-cyan-700">
                <Search className="w-4 h-4 mr-2" />
                Analyze
              </Button>
            </div>
            <div className="flex gap-2">
              {['1M', '3M', '6M', '1Y'].map((p) => (
                <Button
                  key={p}
                  variant={period === p ? 'default' : 'outline'}
                  onClick={() => setPeriod(p)}
                  className={period === p ? 'bg-cyan-600' : 'border-slate-700'}
                >
                  {p}
                </Button>
              ))}
            </div>
            <Button variant="outline" onClick={() => refetch()} className="border-slate-700">
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading Ichimoku analysis...</div>
      ) : error ? (
        <div className="text-center py-12 text-red-400">Error loading data. Please try again.</div>
      ) : data ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Current Price</span>
                <p className="text-xl font-bold text-white">${(data.current_price ?? 0).toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Tenkan-sen</span>
                <p className="text-xl font-bold text-blue-400">${(data.ichimoku?.tenkan ?? 0).toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Kijun-sen</span>
                <p className="text-xl font-bold text-red-400">${(data.ichimoku?.kijun ?? 0).toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Senkou A</span>
                <p className="text-xl font-bold text-green-400">${(data.ichimoku?.senkou_a ?? 0).toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Senkou B</span>
                <p className="text-xl font-bold text-purple-400">${(data.ichimoku?.senkou_b ?? 0).toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className={`border ${data.ichimoku.cloud_color === 'green' ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Cloud Color</span>
                <p className={`text-xl font-bold ${data.ichimoku.cloud_color === 'green' ? 'text-green-500' : 'text-red-500'}`}>
                  {data.ichimoku.cloud_color.toUpperCase()}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Signal Overview */}
          <Card className={`border ${getSignalBg(data.signal.overall)}`}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {data.signal.overall.includes('BULLISH') ? (
                    <ArrowUpRight className="w-8 h-8 text-green-500" />
                  ) : data.signal.overall.includes('BEARISH') ? (
                    <ArrowDownRight className="w-8 h-8 text-red-500" />
                  ) : (
                    <Cloud className="w-8 h-8 text-slate-400" />
                  )}
                  <div>
                    <span className={`text-2xl font-bold ${getSignalColor(data.signal.overall)}`}>
                      {data.signal.overall.replace(/_/g, ' ')}
                    </span>
                    <p className="text-slate-400">Strength: {data.signal.strength}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-white">{data.signal.recommendation}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Chart */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white">Ichimoku Cloud Chart</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-96 w-full">
                <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                  <ComposedChart data={data.chart_data}>
                    <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748b" domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Legend />

                    {/* Cloud (Senkou Span A and B) */}
                    <Area
                      type="monotone"
                      dataKey="senkou_a"
                      stroke="#22c55e"
                      fill={data.ichimoku.cloud_color === 'green' ? '#22c55e20' : '#ef444420'}
                      name="Senkou A"
                    />
                    <Area
                      type="monotone"
                      dataKey="senkou_b"
                      stroke="#a855f7"
                      fill={data.ichimoku.cloud_color === 'green' ? '#22c55e20' : '#ef444420'}
                      name="Senkou B"
                    />

                    {/* Price Line */}
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="#ffffff"
                      strokeWidth={2}
                      dot={false}
                      name="Price"
                    />

                    {/* Tenkan-sen */}
                    <Line
                      type="monotone"
                      dataKey="tenkan"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      dot={false}
                      name="Tenkan-sen"
                    />

                    {/* Kijun-sen */}
                    <Line
                      type="monotone"
                      dataKey="kijun"
                      stroke="#ef4444"
                      strokeWidth={1.5}
                      dot={false}
                      name="Kijun-sen"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Detailed Analysis */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Signal Details */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Signal Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Cloud Position</span>
                    <span className={getSignalColor(data.signal.cloud_position)}>
                      {data.signal.cloud_position.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Cloud Bias</span>
                    <span className={getSignalColor(data.signal.cloud_bias)}>
                      {data.signal.cloud_bias}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">TK Bullish Cross</span>
                    <span className={data.signal.tk_bullish_cross ? 'text-green-500' : 'text-slate-500'}>
                      {data.signal.tk_bullish_cross ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">TK Bearish Cross</span>
                    <span className={data.signal.tk_bearish_cross ? 'text-red-500' : 'text-slate-500'}>
                      {data.signal.tk_bearish_cross ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Interpretation */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Market Interpretation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Current Trend</span>
                    <span className={getSignalColor(data.interpretation.trend)}>
                      {data.interpretation.trend}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">TK Relationship</span>
                    <span className={getSignalColor(data.interpretation.tk_relationship)}>
                      {data.interpretation.tk_relationship}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Future Cloud</span>
                    <span className={getSignalColor(data.interpretation.cloud_future)}>
                      {data.interpretation.cloud_future}
                    </span>
                  </div>
                </div>

                {/* Cloud Levels */}
                <div className="mt-4 p-4 bg-slate-800/50 rounded-lg">
                  <h4 className="text-white font-medium mb-3">Cloud Support/Resistance</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-slate-400 text-sm">Cloud Top</span>
                      <p className="text-lg font-bold text-green-400">${(data.ichimoku?.cloud_top ?? 0).toFixed(2)}</p>
                    </div>
                    <div>
                      <span className="text-slate-400 text-sm">Cloud Bottom</span>
                      <p className="text-lg font-bold text-red-400">${(data.ichimoku?.cloud_bottom ?? 0).toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Ichimoku Legend */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white">Ichimoku Components Guide</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-4 h-1 bg-blue-500 rounded"></div>
                    <span className="text-blue-400 font-medium">Tenkan-sen</span>
                  </div>
                  <p className="text-xs text-slate-400">9-period midpoint. Fast conversion line for short-term momentum.</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-4 h-1 bg-red-500 rounded"></div>
                    <span className="text-red-400 font-medium">Kijun-sen</span>
                  </div>
                  <p className="text-xs text-slate-400">26-period midpoint. Base line for medium-term trend.</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-4 h-1 bg-green-500 rounded"></div>
                    <span className="text-green-400 font-medium">Senkou A</span>
                  </div>
                  <p className="text-xs text-slate-400">Leading Span A. Midpoint of Tenkan/Kijun, plotted 26 periods ahead.</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-4 h-1 bg-purple-500 rounded"></div>
                    <span className="text-purple-400 font-medium">Senkou B</span>
                  </div>
                  <p className="text-xs text-slate-400">Leading Span B. 52-period midpoint, plotted 26 periods ahead.</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-4 h-4 bg-gradient-to-r from-green-500/30 to-red-500/30 rounded"></div>
                    <span className="text-slate-400 font-medium">Kumo (Cloud)</span>
                  </div>
                  <p className="text-xs text-slate-400">Area between Senkou A & B. Support when above, resistance when below.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  )
}
