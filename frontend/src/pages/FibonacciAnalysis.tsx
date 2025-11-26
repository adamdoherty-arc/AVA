import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  TrendingUp, TrendingDown, Search, RefreshCw, Target, ArrowRight
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface FibonacciData {
  symbol: string
  current_price: number
  period: string
  trend: string
  swing_high: number
  swing_low: number
  retracement_levels: Array<{
    level: string
    price: number
    distance_pct: number
    is_golden_zone: boolean
  }>
  extension_levels: Array<{
    level: string
    price: number
  }>
  golden_zone: {
    top: number
    bottom: number
    description: string
  }
  signal: {
    status: string
    at_level: {
      at_fib_level: boolean
      level?: string
      price?: number
      distance_pct?: number
    }
    recommendation: string
  }
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002'

export default function FibonacciAnalysis() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')
  const [period, setPeriod] = useState('3M')

  const { data, isLoading, error, refetch } = useQuery<FibonacciData>({
    queryKey: ['fibonacci', symbol, period],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/advanced-technicals/fibonacci/${symbol}?period=${period}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase())
    }
  }

  const getPricePosition = (currentPrice: number, levels: FibonacciData['retracement_levels']) => {
    for (let i = 0; i < levels.length - 1; i++) {
      if (currentPrice >= levels[i + 1].price && currentPrice <= levels[i].price) {
        return `Between ${levels[i + 1].level} and ${levels[i].level}`
      }
    }
    return 'Outside levels'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <TrendingUp className="w-7 h-7 text-amber-500" />
            Fibonacci Analysis
          </h1>
          <p className="text-slate-400 mt-1">
            Retracement Levels, Extensions, and Golden Zone Analysis
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
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-amber-500"
              />
              <Button onClick={handleSearch} className="bg-amber-600 hover:bg-amber-700">
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
                  className={period === p ? 'bg-amber-600' : 'border-slate-700'}
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
        <div className="text-center py-12 text-slate-400">Loading Fibonacci analysis...</div>
      ) : error ? (
        <div className="text-center py-12 text-red-400">Error loading data. Please try again.</div>
      ) : data ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Current Price</span>
                <p className="text-2xl font-bold text-white">${data.current_price.toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Trend</span>
                <div className="flex items-center gap-2 mt-1">
                  {data.trend === 'BULLISH' ? (
                    <TrendingUp className="w-5 h-5 text-green-500" />
                  ) : (
                    <TrendingDown className="w-5 h-5 text-red-500" />
                  )}
                  <span className={`text-xl font-bold ${data.trend === 'BULLISH' ? 'text-green-500' : 'text-red-500'}`}>
                    {data.trend}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Swing High</span>
                <p className="text-2xl font-bold text-green-500">${data.swing_high.toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Swing Low</span>
                <p className="text-2xl font-bold text-red-500">${data.swing_low.toFixed(2)}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <span className="text-slate-400 text-sm">Range</span>
                <p className="text-2xl font-bold text-white">${(data.swing_high - data.swing_low).toFixed(2)}</p>
                <span className="text-xs text-slate-500">
                  ({((data.swing_high - data.swing_low) / data.swing_low * 100).toFixed(1)}%)
                </span>
              </CardContent>
            </Card>
          </div>

          {/* Signal Alert */}
          <Card className={`border ${
            data.signal.status === 'GOLDEN_ZONE' ? 'bg-amber-500/10 border-amber-500/30' :
            data.signal.status === 'AT_FIB_LEVEL' ? 'bg-blue-500/10 border-blue-500/30' :
            'bg-slate-900/50 border-slate-800'
          }`}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Target className={`w-6 h-6 ${
                    data.signal.status === 'GOLDEN_ZONE' ? 'text-amber-500' :
                    data.signal.status === 'AT_FIB_LEVEL' ? 'text-blue-500' : 'text-slate-400'
                  }`} />
                  <div>
                    <span className={`font-bold ${
                      data.signal.status === 'GOLDEN_ZONE' ? 'text-amber-500' :
                      data.signal.status === 'AT_FIB_LEVEL' ? 'text-blue-500' : 'text-slate-400'
                    }`}>
                      {data.signal.status.replace(/_/g, ' ')}
                    </span>
                    {data.signal.at_level?.at_fib_level && (
                      <span className="text-slate-400 ml-2">
                        at {data.signal.at_level.level} (${data.signal.at_level.price?.toFixed(2)})
                      </span>
                    )}
                  </div>
                </div>
                <span className="text-slate-300">{data.signal.recommendation}</span>
              </div>
            </CardContent>
          </Card>

          {/* Golden Zone */}
          <Card className="bg-gradient-to-r from-amber-500/10 to-amber-600/5 border-amber-500/30">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-amber-500" />
                Golden Zone (50% - 61.8%)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 bg-slate-900/50 rounded-lg">
                  <span className="text-slate-400 text-sm">Zone Top (61.8%)</span>
                  <p className="text-2xl font-bold text-amber-500">${data.golden_zone.top.toFixed(2)}</p>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-lg">
                  <span className="text-slate-400 text-sm">Zone Bottom (50%)</span>
                  <p className="text-2xl font-bold text-amber-500">${data.golden_zone.bottom.toFixed(2)}</p>
                </div>
                <div className="p-4 bg-slate-900/50 rounded-lg flex items-center">
                  <p className="text-sm text-amber-400">{data.golden_zone.description}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Fibonacci Levels */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Retracement Levels */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Retracement Levels</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {data.retracement_levels.map((level, idx) => {
                    const isNearPrice = Math.abs(level.distance_pct) < 2
                    return (
                      <div
                        key={idx}
                        className={`p-3 rounded-lg border flex items-center justify-between ${
                          level.is_golden_zone
                            ? 'bg-amber-500/10 border-amber-500/30'
                            : isNearPrice
                            ? 'bg-blue-500/10 border-blue-500/30'
                            : 'bg-slate-800/50 border-slate-700'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            level.is_golden_zone ? 'bg-amber-500' :
                            level.price > data.current_price ? 'bg-green-500' : 'bg-red-500'
                          }`} />
                          <span className={`font-medium ${
                            level.is_golden_zone ? 'text-amber-500' : 'text-white'
                          }`}>
                            {level.level}
                          </span>
                          {level.is_golden_zone && (
                            <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded">
                              Golden
                            </span>
                          )}
                        </div>
                        <div className="text-right">
                          <span className="text-white font-bold">${level.price.toFixed(2)}</span>
                          <span className={`ml-2 text-sm ${
                            level.distance_pct > 0 ? 'text-green-500' : 'text-red-500'
                          }`}>
                            {level.distance_pct > 0 ? '+' : ''}{level.distance_pct.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Price Position Indicator */}
                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400 text-sm">Current Position: </span>
                  <span className="text-white">
                    {getPricePosition(data.current_price, data.retracement_levels)}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Extension Levels */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Extension Levels</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-400 text-sm mb-4">
                  Potential price targets if the move continues beyond 100%
                </p>
                <div className="space-y-2">
                  {data.extension_levels.map((level, idx) => {
                    const distancePct = ((level.price - data.current_price) / data.current_price) * 100
                    return (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-slate-800/50 border border-slate-700 flex items-center justify-between"
                      >
                        <div className="flex items-center gap-3">
                          <ArrowRight className="w-4 h-4 text-purple-500" />
                          <span className="text-purple-400 font-medium">{level.level}</span>
                        </div>
                        <div className="text-right">
                          <span className="text-white font-bold">${level.price.toFixed(2)}</span>
                          <span className={`ml-2 text-sm ${distancePct > 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {distancePct > 0 ? '+' : ''}{distancePct.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Trading Notes */}
                <div className="mt-6 space-y-3">
                  <h3 className="text-white font-medium">Trading Notes</h3>
                  <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                    <p className="text-sm text-green-400">
                      <strong>Support:</strong> Look for bounces at 50%, 61.8%, and 78.6% levels
                    </p>
                  </div>
                  <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <p className="text-sm text-red-400">
                      <strong>Resistance:</strong> Watch for rejection at 38.2% and 23.6% levels
                    </p>
                  </div>
                  <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                    <p className="text-sm text-amber-400">
                      <strong>Golden Zone:</strong> 50%-61.8% is the highest probability reversal area
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  )
}
