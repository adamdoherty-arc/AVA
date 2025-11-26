import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  Brain, TrendingUp, TrendingDown, AlertTriangle, Target,
  BarChart3, Layers, ArrowUpRight, ArrowDownRight, Search, RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface OrderBlock {
  type: string
  top: number
  bottom: number
  midpoint: number
  strength: number
  mitigated: boolean
  distance_pct: number
  zone: string
}

interface FairValueGap {
  type: string
  top: number
  bottom: number
  gap_pct: number
  filled: boolean
  fill_percentage: number
  distance_pct: number
}

interface LiquidityPool {
  type: string
  price: number
  touches: number
  strength: string
  swept: boolean
  distance_pct: number
}

interface SmartMoneyData {
  symbol: string
  current_price: number
  market_structure: {
    current_trend: string
    bias: string
    description: string
    recent_bos: Array<{ direction: string; price: number; type: string }>
    recent_choch: Array<{ direction: string; price: number; type: string }>
  }
  order_blocks: OrderBlock[]
  fair_value_gaps: FairValueGap[]
  liquidity_pools: LiquidityPool[]
  key_levels: {
    nearest_support: { price: number; type: string; strength: number } | null
    nearest_resistance: { price: number; type: string; strength: number } | null
  }
  signals: Array<{ type: string; indicator: string; price: number; strength: number; description: string }>
  summary: {
    total_order_blocks: number
    bullish_obs: number
    bearish_obs: number
    unfilled_fvgs: number
    liquidity_pools: number
  }
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002'

export default function SmartMoneyConcepts() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')
  const [timeframe, setTimeframe] = useState('1D')

  const { data, isLoading, error, refetch } = useQuery<SmartMoneyData>({
    queryKey: ['smart-money', symbol, timeframe],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/smart-money/${symbol}?timeframe=${timeframe}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase())
    }
  }

  const getTrendIcon = (trend: string) => {
    if (trend === 'BULLISH') return <TrendingUp className="w-5 h-5 text-green-500" />
    if (trend === 'BEARISH') return <TrendingDown className="w-5 h-5 text-red-500" />
    return <BarChart3 className="w-5 h-5 text-slate-400" />
  }

  const getTrendColor = (trend: string) => {
    if (trend === 'BULLISH') return 'text-green-500'
    if (trend === 'BEARISH') return 'text-red-500'
    return 'text-slate-400'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Brain className="w-7 h-7 text-purple-500" />
            Smart Money Concepts (ICT)
          </h1>
          <p className="text-slate-400 mt-1">
            Order Blocks, Fair Value Gaps, Market Structure, Liquidity Pools
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
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-purple-500"
              />
              <Button onClick={handleSearch} className="bg-purple-600 hover:bg-purple-700">
                <Search className="w-4 h-4 mr-2" />
                Analyze
              </Button>
            </div>
            <div className="flex gap-2">
              {['1H', '4H', '1D', '1W'].map((tf) => (
                <Button
                  key={tf}
                  variant={timeframe === tf ? 'default' : 'outline'}
                  onClick={() => setTimeframe(tf)}
                  className={timeframe === tf ? 'bg-purple-600' : 'border-slate-700'}
                >
                  {tf}
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
        <div className="text-center py-12 text-slate-400">Loading Smart Money analysis...</div>
      ) : error ? (
        <div className="text-center py-12 text-red-400">Error loading data. Please try again.</div>
      ) : data ? (
        <>
          {/* Market Structure Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Current Price</span>
                  <span className="text-2xl font-bold text-white">${data.current_price.toFixed(2)}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Market Structure</span>
                  <div className="flex items-center gap-2">
                    {getTrendIcon(data.market_structure.current_trend)}
                    <span className={`text-lg font-bold ${getTrendColor(data.market_structure.current_trend)}`}>
                      {data.market_structure.current_trend}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-slate-500 mt-2">{data.market_structure.description}</p>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Order Blocks</span>
                  <div className="text-right">
                    <span className="text-lg font-bold text-white">{data.summary.total_order_blocks}</span>
                    <div className="text-xs">
                      <span className="text-green-500">{data.summary.bullish_obs} Bull</span>
                      {' / '}
                      <span className="text-red-500">{data.summary.bearish_obs} Bear</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Unfilled FVGs</span>
                  <span className="text-lg font-bold text-amber-500">{data.summary.unfilled_fvgs}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Trading Signals */}
          {data.signals.length > 0 && (
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-amber-500" />
                  Active Signals
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {data.signals.map((signal, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border ${
                        signal.type === 'BUY'
                          ? 'bg-green-500/10 border-green-500/30'
                          : 'bg-red-500/10 border-red-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {signal.type === 'BUY' ? (
                            <ArrowUpRight className="w-5 h-5 text-green-500" />
                          ) : (
                            <ArrowDownRight className="w-5 h-5 text-red-500" />
                          )}
                          <div>
                            <span className={`font-bold ${signal.type === 'BUY' ? 'text-green-500' : 'text-red-500'}`}>
                              {signal.type}
                            </span>
                            <span className="text-slate-400 ml-2 text-sm">via {signal.indicator}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-white">${signal.price.toFixed(2)}</div>
                          <div className="text-xs text-slate-500">Strength: {signal.strength}%</div>
                        </div>
                      </div>
                      <p className="text-sm text-slate-400 mt-2">{signal.description}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Key Levels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-green-500" />
                  Nearest Support
                </CardTitle>
              </CardHeader>
              <CardContent>
                {data.key_levels.nearest_support ? (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Price</span>
                      <span className="text-green-500 font-bold">${data.key_levels.nearest_support.price.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Type</span>
                      <span className="text-white">{data.key_levels.nearest_support.type.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Strength</span>
                      <span className="text-white">{data.key_levels.nearest_support.strength}%</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-500">No support zone identified</p>
                )}
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-red-500" />
                  Nearest Resistance
                </CardTitle>
              </CardHeader>
              <CardContent>
                {data.key_levels.nearest_resistance ? (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Price</span>
                      <span className="text-red-500 font-bold">${data.key_levels.nearest_resistance.price.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Type</span>
                      <span className="text-white">{data.key_levels.nearest_resistance.type.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Strength</span>
                      <span className="text-white">{data.key_levels.nearest_resistance.strength}%</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-500">No resistance zone identified</p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Order Blocks */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-purple-500" />
                Order Blocks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3">
                {data.order_blocks.length > 0 ? (
                  data.order_blocks.slice(0, 8).map((ob, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border ${
                        ob.type === 'BULLISH_OB'
                          ? 'bg-green-500/5 border-green-500/20'
                          : 'bg-red-500/5 border-red-500/20'
                      } ${ob.mitigated ? 'opacity-50' : ''}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            ob.type === 'BULLISH_OB' ? 'bg-green-500' : 'bg-red-500'
                          }`} />
                          <div>
                            <span className={`font-medium ${
                              ob.type === 'BULLISH_OB' ? 'text-green-500' : 'text-red-500'
                            }`}>
                              {ob.type.replace(/_/g, ' ')}
                            </span>
                            {ob.mitigated && <span className="text-slate-500 ml-2 text-xs">(Mitigated)</span>}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-white">${ob.bottom.toFixed(2)} - ${ob.top.toFixed(2)}</div>
                          <div className="text-xs text-slate-500">
                            {ob.distance_pct > 0 ? '+' : ''}{ob.distance_pct.toFixed(1)}% from price
                          </div>
                        </div>
                      </div>
                      <div className="mt-2 flex justify-between text-sm">
                        <span className="text-slate-400">Strength: {ob.strength}%</span>
                        <span className="text-slate-400">Zone: {ob.zone}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500">No order blocks detected</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Fair Value Gaps */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-amber-500" />
                Fair Value Gaps (FVG)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3">
                {data.fair_value_gaps.length > 0 ? (
                  data.fair_value_gaps.slice(0, 8).map((fvg, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border ${
                        fvg.type === 'BULLISH_FVG'
                          ? 'bg-green-500/5 border-green-500/20'
                          : 'bg-red-500/5 border-red-500/20'
                      } ${fvg.filled ? 'opacity-50' : ''}`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            fvg.type === 'BULLISH_FVG' ? 'bg-green-500' : 'bg-red-500'
                          }`} />
                          <div>
                            <span className={`font-medium ${
                              fvg.type === 'BULLISH_FVG' ? 'text-green-500' : 'text-red-500'
                            }`}>
                              {fvg.type.replace(/_/g, ' ')}
                            </span>
                            {fvg.filled && <span className="text-slate-500 ml-2 text-xs">(Filled)</span>}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-white">${fvg.bottom.toFixed(2)} - ${fvg.top.toFixed(2)}</div>
                          <div className="text-xs text-slate-500">
                            Gap: {fvg.gap_pct.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      <div className="mt-2">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-slate-400">Fill Progress</span>
                          <span className="text-slate-400">{fvg.fill_percentage.toFixed(0)}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-700 rounded-full">
                          <div
                            className={`h-full rounded-full ${
                              fvg.type === 'BULLISH_FVG' ? 'bg-green-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${fvg.fill_percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500">No fair value gaps detected</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Liquidity Pools */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-500" />
                Liquidity Pools
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {data.liquidity_pools.length > 0 ? (
                  data.liquidity_pools.slice(0, 6).map((pool, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-lg border ${
                        pool.type === 'BUY_SIDE_LIQUIDITY'
                          ? 'bg-green-500/5 border-green-500/20'
                          : 'bg-red-500/5 border-red-500/20'
                      } ${pool.swept ? 'opacity-50' : ''}`}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`font-medium ${
                          pool.type === 'BUY_SIDE_LIQUIDITY' ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {pool.type.replace(/_/g, ' ')}
                        </span>
                        <span className="text-white font-bold">${pool.price.toFixed(2)}</span>
                      </div>
                      <div className="mt-2 flex justify-between text-sm">
                        <span className="text-slate-400">Touches: {pool.touches}</span>
                        <span className="text-slate-400">Strength: {pool.strength}</span>
                      </div>
                      {pool.swept && (
                        <div className="mt-2 text-xs text-slate-500">Already swept</div>
                      )}
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500 col-span-2">No liquidity pools identified</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Market Structure Events */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-500" />
                Market Structure Events
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-slate-400 text-sm font-medium mb-3">Break of Structure (BOS)</h3>
                  {data.market_structure.recent_bos.length > 0 ? (
                    <div className="space-y-2">
                      {data.market_structure.recent_bos.map((bos, idx) => (
                        <div key={idx} className="flex justify-between items-center p-2 bg-slate-800/50 rounded">
                          <span className={bos.direction === 'BULLISH' ? 'text-green-500' : 'text-red-500'}>
                            {bos.direction} BOS
                          </span>
                          <span className="text-white">${bos.price.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">No recent BOS</p>
                  )}
                </div>
                <div>
                  <h3 className="text-slate-400 text-sm font-medium mb-3">Change of Character (CHoCH)</h3>
                  {data.market_structure.recent_choch.length > 0 ? (
                    <div className="space-y-2">
                      {data.market_structure.recent_choch.map((choch, idx) => (
                        <div key={idx} className="flex justify-between items-center p-2 bg-slate-800/50 rounded">
                          <span className={choch.direction === 'BULLISH' ? 'text-green-500' : 'text-red-500'}>
                            {choch.direction} CHoCH
                          </span>
                          <span className="text-white">${choch.price.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500 text-sm">No recent CHoCH</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  )
}
