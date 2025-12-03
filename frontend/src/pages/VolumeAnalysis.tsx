import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  BarChart3, TrendingUp, TrendingDown, Search, RefreshCw, Target, ArrowUpRight, ArrowDownRight
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts'
import { API_HOST } from '@/config/api'

interface VolumeProfileData {
  symbol: string
  current_price: number
  period: string
  volume_profile: {
    poc: { price: number; volume: number; pct_of_total: number }
    vah: number
    val: number
    value_area_width: number
    value_area_width_pct: number
  }
  high_volume_nodes: Array<{ price: number; distance_pct: number; type: string }>
  low_volume_nodes: Array<{ price: number; distance_pct: number; note: string }>
  signals: {
    position: string
    bias: string
    setup_quality: string
    recommendation: string
    distance_from_poc_pct: number
    near_hvn: boolean
    near_lvn: boolean
  }
  distribution: {
    price_levels: number[]
    volume_at_price: number[]
  }
}

interface CVDData {
  symbol: string
  current_price: number
  period: string
  cvd: {
    current: number
    change_5d: number
    change_10d: number
    trend_5d: string
    trend_10d: string
  }
  interpretation: string
  description: string
  divergences: Array<{ type: string; price: number; signal: string; strength: number }>
  chart_data: Array<{ date: string; price: number; cvd: number; volume: number }>
}

const API_BASE = API_HOST

export default function VolumeAnalysis() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')
  const [period, setPeriod] = useState('3M')

  const { data: vpData, isLoading: vpLoading, refetch: vpRefetch } = useQuery<VolumeProfileData>({
    queryKey: ['volume-profile', symbol, period],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/advanced-technicals/volume-profile/${symbol}?period=${period}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const { data: cvdData, isLoading: cvdLoading, refetch: cvdRefetch } = useQuery<CVDData>({
    queryKey: ['cvd', symbol, period],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/advanced-technicals/cvd/${symbol}?period=${period}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase())
    }
  }

  const handleRefresh = () => {
    vpRefetch()
    cvdRefetch()
  }

  // Prepare volume profile chart data
  const volumeChartData = vpData?.distribution?.price_levels?.map((price, idx) => ({
    price: price.toFixed(2),
    volume: vpData.distribution.volume_at_price[idx],
    isPOC: Math.abs(price - vpData.volume_profile.poc.price) < 0.5,
  })) || []

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <BarChart3 className="w-7 h-7 text-blue-500" />
            Volume Analysis
          </h1>
          <p className="text-slate-400 mt-1">
            Volume Profile, POC, VAH, VAL, CVD, and Order Flow Analysis
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
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
              />
              <Button onClick={handleSearch} className="bg-blue-600 hover:bg-blue-700">
                <Search className="w-4 h-4 mr-2" />
                Analyze
              </Button>
            </div>
            <div className="flex gap-2">
              {['1W', '1M', '3M', '6M', '1Y'].map((p) => (
                <Button
                  key={p}
                  variant={period === p ? 'default' : 'outline'}
                  onClick={() => setPeriod(p)}
                  className={period === p ? 'bg-blue-600' : 'border-slate-700'}
                >
                  {p}
                </Button>
              ))}
            </div>
            <Button variant="outline" onClick={handleRefresh} className="border-slate-700">
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {vpLoading || cvdLoading ? (
        <div className="text-center py-12 text-slate-400">Loading Volume analysis...</div>
      ) : vpData && cvdData ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Current Price</span>
                  <span className="text-2xl font-bold text-white">${(vpData.current_price ?? 0).toFixed(2)}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">POC (Point of Control)</span>
                  <span className="text-2xl font-bold text-amber-500">${(vpData.volume_profile?.poc?.price ?? 0).toFixed(2)}</span>
                  <span className="text-xs text-slate-500">{(vpData.volume_profile?.poc?.pct_of_total ?? 0).toFixed(1)}% of volume</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Value Area High</span>
                  <span className="text-2xl font-bold text-green-500">${(vpData.volume_profile?.vah ?? 0).toFixed(2)}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Value Area Low</span>
                  <span className="text-2xl font-bold text-red-500">${(vpData.volume_profile?.val ?? 0).toFixed(2)}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">CVD Trend</span>
                  <div className="flex items-center gap-2">
                    {cvdData.cvd.trend_5d === 'RISING' ? (
                      <TrendingUp className="w-5 h-5 text-green-500" />
                    ) : (
                      <TrendingDown className="w-5 h-5 text-red-500" />
                    )}
                    <span className={`text-lg font-bold ${cvdData.cvd.trend_5d === 'RISING' ? 'text-green-500' : 'text-red-500'}`}>
                      {cvdData.cvd.trend_5d}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Signals */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-500" />
                Volume Profile Signals
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Position</span>
                    <span className={`font-medium ${
                      vpData.signals.position === 'ABOVE_POC' ? 'text-green-500' :
                      vpData.signals.position === 'BELOW_POC' ? 'text-red-500' : 'text-amber-500'
                    }`}>{vpData.signals.position.replace(/_/g, ' ')}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Bias</span>
                    <span className={`font-medium ${
                      vpData.signals.bias === 'BULLISH' ? 'text-green-500' :
                      vpData.signals.bias === 'BEARISH' ? 'text-red-500' : 'text-slate-400'
                    }`}>{vpData.signals.bias}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Setup Quality</span>
                    <span className="text-white font-medium">{vpData.signals.setup_quality}</span>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Distance from POC</span>
                    <span className="text-white">{(vpData.signals?.distance_from_poc_pct ?? 0).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Near HVN</span>
                    <span className={vpData.signals.near_hvn ? 'text-green-500' : 'text-slate-500'}>
                      {vpData.signals.near_hvn ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                    <span className="text-slate-400">Near LVN</span>
                    <span className={vpData.signals.near_lvn ? 'text-amber-500' : 'text-slate-500'}>
                      {vpData.signals.near_lvn ? 'Yes (Fast Move Area)' : 'No'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <p className="text-blue-400">{vpData.signals.recommendation}</p>
              </div>
            </CardContent>
          </Card>

          {/* Volume Profile Chart */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white">Volume Profile Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={1}>
                  <BarChart data={volumeChartData} layout="vertical">
                    <XAxis type="number" stroke="#64748b" />
                    <YAxis dataKey="price" type="category" stroke="#64748b" width={60} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Bar
                      dataKey="volume"
                      fill="#3b82f6"
                      radius={[0, 4, 4, 0]}
                    />
                    <ReferenceLine y={(vpData.volume_profile?.poc?.price ?? 0).toFixed(2)} stroke="#f59e0b" strokeWidth={2} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* High/Low Volume Nodes */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <ArrowUpRight className="w-5 h-5 text-green-500" />
                  High Volume Nodes (HVN)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-400 text-sm mb-4">
                  Areas of high trading activity - act as support/resistance
                </p>
                <div className="space-y-2">
                  {vpData.high_volume_nodes.map((hvn, idx) => (
                    <div key={idx} className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                      <div>
                        <span className="text-white font-medium">${(hvn.price ?? 0).toFixed(2)}</span>
                        <span className={`ml-2 text-sm ${hvn.type === 'support' ? 'text-green-500' : 'text-red-500'}`}>
                          ({hvn.type})
                        </span>
                      </div>
                      <span className="text-slate-400">{(hvn.distance_pct ?? 0) > 0 ? '+' : ''}{(hvn.distance_pct ?? 0).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <ArrowDownRight className="w-5 h-5 text-amber-500" />
                  Low Volume Nodes (LVN)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-400 text-sm mb-4">
                  Areas of low liquidity - price moves quickly through these zones
                </p>
                <div className="space-y-2">
                  {vpData.low_volume_nodes.map((lvn, idx) => (
                    <div key={idx} className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                      <div>
                        <span className="text-white font-medium">${(lvn.price ?? 0).toFixed(2)}</span>
                      </div>
                      <span className="text-slate-400">{(lvn.distance_pct ?? 0) > 0 ? '+' : ''}{(lvn.distance_pct ?? 0).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* CVD Analysis */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-cyan-500" />
                Cumulative Volume Delta (CVD)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400 text-sm">Current CVD</span>
                  <p className="text-2xl font-bold text-white">{(cvdData.cvd?.current ?? 0).toLocaleString()}</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400 text-sm">5-Day Change</span>
                  <p className={`text-2xl font-bold ${(cvdData.cvd?.change_5d ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {(cvdData.cvd?.change_5d ?? 0) >= 0 ? '+' : ''}{(cvdData.cvd?.change_5d ?? 0).toLocaleString()}
                  </p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400 text-sm">Interpretation</span>
                  <p className={`text-lg font-bold ${
                    cvdData.interpretation.includes('BULLISH') ? 'text-green-500' :
                    cvdData.interpretation.includes('BEARISH') ? 'text-red-500' : 'text-slate-400'
                  }`}>{cvdData.interpretation.replace(/_/g, ' ')}</p>
                </div>
              </div>

              <div className={`p-4 rounded-lg border ${
                cvdData.interpretation.includes('BULLISH') ? 'bg-green-500/10 border-green-500/20' :
                cvdData.interpretation.includes('BEARISH') ? 'bg-red-500/10 border-red-500/20' :
                'bg-slate-800/50 border-slate-700'
              }`}>
                <p className="text-slate-300">{cvdData.description}</p>
              </div>

              {cvdData.divergences.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-white font-medium mb-3">CVD Divergences Detected</h3>
                  <div className="space-y-2">
                    {cvdData.divergences.map((div, idx) => (
                      <div key={idx} className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                        <div className="flex justify-between">
                          <span className="text-amber-500 font-medium">{div.type}</span>
                          <span className="text-white">${(div.price ?? 0).toFixed(2)}</span>
                        </div>
                        <p className="text-sm text-slate-400 mt-1">{div.signal} (Strength: {div.strength}%)</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  )
}
