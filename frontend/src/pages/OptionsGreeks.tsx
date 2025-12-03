import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  Calculator, TrendingUp, TrendingDown, Search, RefreshCw, Target, Gauge,
  ArrowUpRight, ArrowDownRight, AlertCircle, Lightbulb, BarChart3
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts'
import { API_HOST } from '@/config/api'

interface IVRData {
  symbol: string
  current_price: number
  ivr: {
    value: number
    current_iv: number
    iv_min_52w: number
    iv_max_52w: number
    interpretation: string
    strategy: string
    recommendation: string
  }
  trading_guidance: {
    sell_premium: boolean
    buy_premium: boolean
    optimal_strategies: string[]
  }
}

interface ExpectedMoveData {
  symbol: string
  current_price: number
  iv: number
  expected_move: {
    dte: number
    '1_std_dev': { move: number; move_pct: number; upper_bound: number; lower_bound: number; probability: number }
    '2_std_dev': { move: number; move_pct: number; upper_bound: number; lower_bound: number; probability: number }
  }
  timeframes: {
    weekly: { move: number; move_pct: number; range: string }
    monthly: { move: number; move_pct: number; range: string }
  }
  trading_guidance: {
    strike_selection: {
      safe_put_strike: number
      safe_call_strike: number
      aggressive_put_strike: number
      aggressive_call_strike: number
    }
  }
}

interface PCRData {
  symbol: string
  current_price: number
  volume_pcr: {
    value: number
    put_volume: number
    call_volume: number
    sentiment: string
    interpretation: string
  }
  oi_pcr: {
    value: number
    put_oi: number
    call_oi: number
    sentiment: string
    interpretation: string
  }
  contrarian_view: string
  trading_guidance: {
    sentiment_score: number
    recommendation: string
  }
}

interface ComprehensiveData {
  symbol: string
  current_price: number
  iv_analysis: {
    current_iv: number
    ivr: number
    ivp: number
    interpretation: string
    strategy_bias: string
  }
  expected_move: {
    dte: number
    move_dollars: number
    move_pct: number
    upper: number
    lower: number
  }
  sentiment: {
    pcr: number
    sentiment: string
    interpretation: string
  }
  available_expirations: string[]
}

const API_BASE = API_HOST

export default function OptionsGreeks() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')

  const { data: comprehensiveData, isLoading, error, refetch } = useQuery<ComprehensiveData>({
    queryKey: ['options-comprehensive', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/options-indicators/comprehensive/${symbol}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const { data: emData } = useQuery<ExpectedMoveData>({
    queryKey: ['expected-move', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/options-indicators/expected-move/${symbol}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const { data: pcrData } = useQuery<PCRData>({
    queryKey: ['pcr', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/options-indicators/put-call-ratio/${symbol}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const handleSearch = () => {
    if (searchInput.trim()) {
      setSymbol(searchInput.trim().toUpperCase())
    }
  }

  const getIVRColor = (ivr: number) => {
    if (ivr >= 80) return 'text-red-500'
    if (ivr >= 50) return 'text-amber-500'
    if (ivr >= 20) return 'text-slate-400'
    return 'text-green-500'
  }

  const getIVRGradient = (ivr: number) => {
    if (ivr >= 80) return 'from-red-500/20 to-red-500/5'
    if (ivr >= 50) return 'from-amber-500/20 to-amber-500/5'
    if (ivr >= 20) return 'from-slate-500/20 to-slate-500/5'
    return 'from-green-500/20 to-green-500/5'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Calculator className="w-7 h-7 text-emerald-500" />
            Options Greeks Dashboard
          </h1>
          <p className="text-slate-400 mt-1">
            IVR, IVP, Expected Move, Put/Call Ratio, and Strategy Recommendations
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
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              />
              <Button onClick={handleSearch} className="bg-emerald-600 hover:bg-emerald-700">
                <Search className="w-4 h-4 mr-2" />
                Analyze
              </Button>
            </div>
            <Button variant="outline" onClick={() => refetch()} className="border-slate-700">
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        <div className="text-center py-12 text-slate-400">Loading Options analysis...</div>
      ) : error ? (
        <div className="text-center py-12 text-red-400">Error loading data. Please try again.</div>
      ) : comprehensiveData ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Current Price</span>
                  <span className="text-2xl font-bold text-white">${(comprehensiveData.current_price ?? 0).toFixed(2)}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Current IV</span>
                  <span className="text-2xl font-bold text-white">{(comprehensiveData.iv_analysis?.current_iv ?? 0).toFixed(1)}%</span>
                </div>
              </CardContent>
            </Card>

            <Card className={`bg-gradient-to-br ${getIVRGradient(comprehensiveData.iv_analysis?.ivr ?? 0)} border-slate-800`}>
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">IV Rank (IVR)</span>
                  <span className={`text-2xl font-bold ${getIVRColor(comprehensiveData.iv_analysis?.ivr ?? 0)}`}>
                    {(comprehensiveData.iv_analysis?.ivr ?? 0).toFixed(0)}%
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">IV Percentile (IVP)</span>
                  <span className="text-2xl font-bold text-white">{(comprehensiveData.iv_analysis?.ivp ?? 0).toFixed(0)}%</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-4">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-sm">Expected Move</span>
                  <span className="text-2xl font-bold text-amber-500">
                    {(comprehensiveData.expected_move?.move_pct ?? 0).toFixed(1)}%
                  </span>
                  <span className="text-xs text-slate-500">{comprehensiveData.expected_move?.dte ?? 0} DTE</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* IVR Gauge */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Gauge className="w-5 h-5 text-emerald-500" />
                Implied Volatility Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                {/* IVR Gauge Visualization */}
                <div className="space-y-4">
                  <div className="relative h-8 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`absolute left-0 top-0 h-full rounded-full transition-all duration-500 ${
                        (comprehensiveData.iv_analysis?.ivr ?? 0) >= 80 ? 'bg-red-500' :
                        (comprehensiveData.iv_analysis?.ivr ?? 0) >= 50 ? 'bg-amber-500' :
                        (comprehensiveData.iv_analysis?.ivr ?? 0) >= 20 ? 'bg-slate-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${comprehensiveData.iv_analysis?.ivr ?? 0}%` }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-white font-bold">{(comprehensiveData.iv_analysis?.ivr ?? 0).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>0% - Low IV</span>
                    <span>50% - Medium</span>
                    <span>100% - High IV</span>
                  </div>

                  <div className={`p-4 rounded-lg border ${
                    (comprehensiveData.iv_analysis?.ivr ?? 0) >= 50
                      ? 'bg-amber-500/10 border-amber-500/20'
                      : 'bg-green-500/10 border-green-500/20'
                  }`}>
                    <div className="flex items-start gap-3">
                      <AlertCircle className={`w-5 h-5 mt-0.5 ${
                        (comprehensiveData.iv_analysis?.ivr ?? 0) >= 50 ? 'text-amber-500' : 'text-green-500'
                      }`} />
                      <div>
                        <p className="font-medium text-white">{comprehensiveData.iv_analysis.interpretation}</p>
                        <p className="text-sm text-slate-400 mt-1">
                          Strategy Bias: <span className="text-white">{comprehensiveData.iv_analysis.strategy_bias}</span>
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Strategy Recommendations */}
                <div>
                  <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                    <Lightbulb className="w-4 h-4 text-amber-500" />
                    Recommended Strategies
                  </h3>
                  <div className="space-y-2">
                    {(comprehensiveData.iv_analysis?.ivr ?? 0) >= 50 ? (
                      <>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingDown className="w-4 h-4 text-green-500" />
                          <span className="text-slate-300">Sell premium - Iron Condors</span>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingDown className="w-4 h-4 text-green-500" />
                          <span className="text-slate-300">Credit Spreads</span>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingDown className="w-4 h-4 text-green-500" />
                          <span className="text-slate-300">Cash-Secured Puts</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-blue-500" />
                          <span className="text-slate-300">Buy premium - Long Straddles</span>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-blue-500" />
                          <span className="text-slate-300">Debit Spreads</span>
                        </div>
                        <div className="p-3 bg-slate-800/50 rounded-lg flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-blue-500" />
                          <span className="text-slate-300">Calendar Spreads</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Expected Move */}
          {emData && (
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-amber-500" />
                  Expected Move Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-slate-400 text-sm font-medium mb-4">1 Standard Deviation (68% Probability)</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Expected Move</span>
                        <span className="text-amber-500 font-bold">
                          ${(emData.expected_move?.['1_std_dev']?.move ?? 0).toFixed(2)} ({(emData.expected_move?.['1_std_dev']?.move_pct ?? 0).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="flex justify-between p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                        <span className="text-slate-400">Upper Bound</span>
                        <span className="text-green-500 font-bold">${(emData.expected_move?.['1_std_dev']?.upper_bound ?? 0).toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                        <span className="text-slate-400">Lower Bound</span>
                        <span className="text-red-500 font-bold">${(emData.expected_move?.['1_std_dev']?.lower_bound ?? 0).toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-slate-400 text-sm font-medium mb-4">2 Standard Deviations (95% Probability)</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Expected Move</span>
                        <span className="text-amber-500 font-bold">
                          ${(emData.expected_move?.['2_std_dev']?.move ?? 0).toFixed(2)} ({(emData.expected_move?.['2_std_dev']?.move_pct ?? 0).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="flex justify-between p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                        <span className="text-slate-400">Upper Bound</span>
                        <span className="text-green-500 font-bold">${(emData.expected_move?.['2_std_dev']?.upper_bound ?? 0).toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                        <span className="text-slate-400">Lower Bound</span>
                        <span className="text-red-500 font-bold">${(emData.expected_move?.['2_std_dev']?.lower_bound ?? 0).toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Strike Selection */}
                <div className="mt-6">
                  <h3 className="text-white font-medium mb-3">Recommended Strike Selection</h3>
                  <div className="grid md:grid-cols-4 gap-4">
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <span className="text-slate-400 text-sm">Safe Put Strike</span>
                      <p className="text-xl font-bold text-green-500">${(emData.trading_guidance?.strike_selection?.safe_put_strike ?? 0).toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <span className="text-slate-400 text-sm">Aggressive Put</span>
                      <p className="text-xl font-bold text-amber-500">${(emData.trading_guidance?.strike_selection?.aggressive_put_strike ?? 0).toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <span className="text-slate-400 text-sm">Safe Call Strike</span>
                      <p className="text-xl font-bold text-green-500">${(emData.trading_guidance?.strike_selection?.safe_call_strike ?? 0).toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <span className="text-slate-400 text-sm">Aggressive Call</span>
                      <p className="text-xl font-bold text-amber-500">${(emData.trading_guidance?.strike_selection?.aggressive_call_strike ?? 0).toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Put/Call Ratio */}
          {pcrData && (
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-cyan-500" />
                  Put/Call Ratio Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-slate-400 text-sm font-medium mb-4">Volume-Based PCR</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">PCR Value</span>
                        <span className={`font-bold ${
                          pcrData.volume_pcr.value > 1 ? 'text-red-500' :
                          pcrData.volume_pcr.value < 0.7 ? 'text-green-500' : 'text-slate-400'
                        }`}>{pcrData.volume_pcr.value?.toFixed(2) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Put Volume</span>
                        <span className="text-red-400">{(pcrData.volume_pcr?.put_volume ?? 0).toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Call Volume</span>
                        <span className="text-green-400">{(pcrData.volume_pcr?.call_volume ?? 0).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-slate-400 text-sm font-medium mb-4">Open Interest PCR</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">PCR Value</span>
                        <span className={`font-bold ${
                          pcrData.oi_pcr.value > 1 ? 'text-red-500' :
                          pcrData.oi_pcr.value < 0.7 ? 'text-green-500' : 'text-slate-400'
                        }`}>{pcrData.oi_pcr.value?.toFixed(2) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Put OI</span>
                        <span className="text-red-400">{(pcrData.oi_pcr?.put_oi ?? 0).toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between p-3 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Call OI</span>
                        <span className="text-green-400">{(pcrData.oi_pcr?.call_oi ?? 0).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sentiment Summary */}
                <div className="mt-6 grid md:grid-cols-2 gap-4">
                  <div className={`p-4 rounded-lg border ${
                    pcrData.volume_pcr.sentiment === 'BEARISH' ? 'bg-red-500/10 border-red-500/20' :
                    pcrData.volume_pcr.sentiment === 'BULLISH' ? 'bg-green-500/10 border-green-500/20' :
                    'bg-slate-800/50 border-slate-700'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      {pcrData.volume_pcr.sentiment === 'BEARISH' ? (
                        <ArrowDownRight className="w-5 h-5 text-red-500" />
                      ) : pcrData.volume_pcr.sentiment === 'BULLISH' ? (
                        <ArrowUpRight className="w-5 h-5 text-green-500" />
                      ) : null}
                      <span className="font-medium text-white">{pcrData.volume_pcr.sentiment} Sentiment</span>
                    </div>
                    <p className="text-sm text-slate-400">{pcrData.volume_pcr.interpretation}</p>
                  </div>

                  <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                    <h4 className="text-white font-medium mb-2">Contrarian View</h4>
                    <p className="text-sm text-slate-400">{pcrData.contrarian_view}</p>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                  <p className="text-emerald-400">{pcrData.trading_guidance.recommendation}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      ) : null}
    </div>
  )
}
