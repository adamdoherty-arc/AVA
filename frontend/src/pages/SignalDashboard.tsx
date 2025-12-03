import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import {
  Zap, TrendingUp, TrendingDown, Search, RefreshCw, Target, Activity,
  ArrowUpRight, ArrowDownRight, BarChart3, Brain, Layers, Calculator
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { API_HOST } from '@/config/api'

interface AllIndicatorsData {
  symbol: string
  current_price: number
  period: string
  bollinger_bands: { signal: string; position: string; volatility_state: string; recommendation: string }
  stochastic: { signal: string; zone: string; k: number; d: number; recommendation: string }
  obv: { signal: string; trend: string; recommendation: string }
  vwap: { signal: string; position: string; vwap: number; distance_pct: number; recommendation: string }
  mfi: { signal: string; zone: string; value: number; recommendation: string }
  adx: { signal: string; direction: string; trend_strength: string; adx_value: number; recommendation: string }
  cci: { signal: string; zone: string; cci: number; recommendation: string }
  summary: { bullish_signals: number; bearish_signals: number; neutral_signals: number }
}

interface SmartMoneyData {
  symbol: string
  current_price: number
  market_structure: { current_trend: string; bias: string }
  signals: Array<{ type: string; indicator: string; price: number; strength: number; description: string }>
  summary: { total_order_blocks: number; bullish_obs: number; bearish_obs: number; unfilled_fvgs: number }
}

interface IVRData {
  symbol: string
  ivr: { value: number; interpretation: string; strategy: string }
}

const API_BASE = API_HOST

export default function SignalDashboard() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('AAPL')

  const { data: indicatorsData, isLoading: indicatorsLoading, refetch: refetchIndicators } = useQuery<AllIndicatorsData>({
    queryKey: ['all-indicators', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/advanced-technicals/all-indicators/${symbol}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const { data: smcData, isLoading: smcLoading, refetch: refetchSmc } = useQuery<SmartMoneyData>({
    queryKey: ['smart-money-signals', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/smart-money/${symbol}`)
      return response.data
    },
    refetchInterval: 60000,
  })

  const { data: ivrData, refetch: refetchIvr } = useQuery<IVRData>({
    queryKey: ['ivr-signal', symbol],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/api/options-indicators/ivr/${symbol}`)
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
    refetchIndicators()
    refetchSmc()
    refetchIvr()
  }

  // Calculate composite score
  const calculateCompositeScore = () => {
    if (!indicatorsData || !smcData) return 50

    let score = 50 // Start neutral

    // Technical indicators weight: 40%
    const bullishSignals = indicatorsData.summary?.bullish_signals || 0
    const bearishSignals = indicatorsData.summary?.bearish_signals || 0
    const totalSignals = bullishSignals + bearishSignals
    if (totalSignals > 0) {
      const techScore = (bullishSignals / totalSignals) * 100
      score += (techScore - 50) * 0.4
    }

    // Smart Money weight: 40%
    if (smcData.market_structure?.current_trend === 'BULLISH') {
      score += 20
    } else if (smcData.market_structure?.current_trend === 'BEARISH') {
      score -= 20
    }

    // SMC signals
    const smcSignals = smcData.signals || []
    const buySignals = smcSignals.filter(s => s.type === 'BUY').length
    const sellSignals = smcSignals.filter(s => s.type === 'SELL').length
    score += (buySignals - sellSignals) * 5

    // IV weight: 20%
    if (ivrData?.ivr?.value) {
      // High IVR slightly bearish (mean reversion), low IVR slightly bullish
      if (ivrData.ivr.value > 70) {
        score -= 5
      } else if (ivrData.ivr.value < 30) {
        score += 5
      }
    }

    return Math.max(0, Math.min(100, Math.round(score)))
  }

  const compositeScore = calculateCompositeScore()

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-green-500'
    if (score >= 55) return 'text-emerald-400'
    if (score >= 45) return 'text-slate-400'
    if (score >= 30) return 'text-amber-500'
    return 'text-red-500'
  }

  const getScoreBias = (score: number) => {
    if (score >= 70) return 'STRONG BULLISH'
    if (score >= 55) return 'BULLISH'
    if (score >= 45) return 'NEUTRAL'
    if (score >= 30) return 'BEARISH'
    return 'STRONG BEARISH'
  }

  const getSignalIcon = (signal: string) => {
    if (signal.includes('BUY') || signal.includes('BULLISH')) {
      return <ArrowUpRight className="w-4 h-4 text-green-500" />
    }
    if (signal.includes('SELL') || signal.includes('BEARISH')) {
      return <ArrowDownRight className="w-4 h-4 text-red-500" />
    }
    return <Activity className="w-4 h-4 text-slate-400" />
  }

  const getSignalColor = (signal: string) => {
    if (signal.includes('BUY') || signal.includes('BULLISH') || signal.includes('STRONG_BUY')) {
      return 'text-green-500'
    }
    if (signal.includes('SELL') || signal.includes('BEARISH') || signal.includes('STRONG_SELL')) {
      return 'text-red-500'
    }
    return 'text-slate-400'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Zap className="w-7 h-7 text-yellow-500" />
            Signal Dashboard
          </h1>
          <p className="text-slate-400 mt-1">
            Aggregated signals from all technical indicators with composite scoring
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
                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-yellow-500"
              />
              <Button onClick={handleSearch} className="bg-yellow-600 hover:bg-yellow-700">
                <Search className="w-4 h-4 mr-2" />
                Analyze
              </Button>
            </div>
            <Button variant="outline" onClick={handleRefresh} className="border-slate-700">
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {indicatorsLoading || smcLoading ? (
        <div className="text-center py-12 text-slate-400">Loading signals...</div>
      ) : indicatorsData && smcData ? (
        <>
          {/* Composite Score */}
          <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
            <CardContent className="p-8">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-slate-400 text-lg">Composite Signal Score</h2>
                  <div className="flex items-center gap-4 mt-2">
                    <span className={`text-6xl font-bold ${getScoreColor(compositeScore)}`}>
                      {compositeScore}
                    </span>
                    <div>
                      <span className={`text-2xl font-bold ${getScoreColor(compositeScore)}`}>
                        {getScoreBias(compositeScore)}
                      </span>
                      <p className="text-slate-400">for {symbol}</p>
                    </div>
                  </div>
                </div>

                {/* Score Gauge */}
                <div className="w-48">
                  <div className="relative h-4 bg-gradient-to-r from-red-500 via-slate-500 to-green-500 rounded-full">
                    <div
                      className="absolute top-1/2 -translate-y-1/2 w-4 h-6 bg-white rounded shadow-lg"
                      style={{ left: `calc(${compositeScore}% - 8px)` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Bearish</span>
                    <span>Neutral</span>
                    <span>Bullish</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Signal Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-green-500/10 border-green-500/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-green-400">Bullish Signals</span>
                  <span className="text-3xl font-bold text-green-500">
                    {indicatorsData.summary?.bullish_signals || 0}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/50 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Neutral Signals</span>
                  <span className="text-3xl font-bold text-slate-400">
                    {indicatorsData.summary?.neutral_signals || 0}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-red-500/10 border-red-500/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-red-400">Bearish Signals</span>
                  <span className="text-3xl font-bold text-red-500">
                    {indicatorsData.summary?.bearish_signals || 0}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className={`border ${
              smcData.market_structure?.current_trend === 'BULLISH' ? 'bg-green-500/10 border-green-500/30' :
              smcData.market_structure?.current_trend === 'BEARISH' ? 'bg-red-500/10 border-red-500/30' :
              'bg-slate-800/50 border-slate-700'
            }`}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Market Structure</span>
                  <span className={`text-xl font-bold ${
                    smcData.market_structure?.current_trend === 'BULLISH' ? 'text-green-500' :
                    smcData.market_structure?.current_trend === 'BEARISH' ? 'text-red-500' : 'text-slate-400'
                  }`}>
                    {smcData.market_structure?.current_trend || 'N/A'}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* All Indicator Signals */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Standard Indicators */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  Technical Indicators
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {/* RSI / MFI */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.mfi?.signal || '')}
                      <span className="text-slate-300">MFI ({indicatorsData.mfi?.value?.toFixed(0)})</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.mfi?.signal || '')}>
                      {indicatorsData.mfi?.signal || 'N/A'}
                    </span>
                  </div>

                  {/* Stochastic */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.stochastic?.signal || '')}
                      <span className="text-slate-300">
                        Stochastic (K:{indicatorsData.stochastic?.k?.toFixed(0)}, D:{indicatorsData.stochastic?.d?.toFixed(0)})
                      </span>
                    </div>
                    <span className={getSignalColor(indicatorsData.stochastic?.signal || '')}>
                      {indicatorsData.stochastic?.signal || 'N/A'}
                    </span>
                  </div>

                  {/* Bollinger Bands */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.bollinger_bands?.signal || '')}
                      <span className="text-slate-300">Bollinger Bands</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.bollinger_bands?.signal || '')}>
                      {indicatorsData.bollinger_bands?.signal || 'N/A'}
                    </span>
                  </div>

                  {/* VWAP */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.vwap?.signal || '')}
                      <span className="text-slate-300">VWAP (${indicatorsData.vwap?.vwap?.toFixed(2)})</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.vwap?.signal || '')}>
                      {indicatorsData.vwap?.signal || 'N/A'}
                    </span>
                  </div>

                  {/* OBV */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.obv?.signal || '')}
                      <span className="text-slate-300">OBV ({indicatorsData.obv?.trend})</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.obv?.signal || '')}>
                      {indicatorsData.obv?.signal || 'N/A'}
                    </span>
                  </div>

                  {/* ADX */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.adx?.signal || '')}
                      <span className="text-slate-300">ADX ({indicatorsData.adx?.adx_value?.toFixed(0)})</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.adx?.signal || '')}>
                      {indicatorsData.adx?.signal || 'N/A'} - {indicatorsData.adx?.trend_strength}
                    </span>
                  </div>

                  {/* CCI */}
                  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      {getSignalIcon(indicatorsData.cci?.signal || '')}
                      <span className="text-slate-300">CCI ({indicatorsData.cci?.cci?.toFixed(0)})</span>
                    </div>
                    <span className={getSignalColor(indicatorsData.cci?.signal || '')}>
                      {indicatorsData.cci?.signal || 'N/A'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Smart Money & Options */}
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-500" />
                  Smart Money & Options
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Smart Money Summary */}
                  <div className="p-4 bg-slate-800/50 rounded-lg">
                    <h3 className="text-slate-400 text-sm mb-3">Smart Money Concepts</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="text-slate-500 text-xs">Bullish OBs</span>
                        <p className="text-lg font-bold text-green-500">{smcData.summary?.bullish_obs || 0}</p>
                      </div>
                      <div>
                        <span className="text-slate-500 text-xs">Bearish OBs</span>
                        <p className="text-lg font-bold text-red-500">{smcData.summary?.bearish_obs || 0}</p>
                      </div>
                      <div>
                        <span className="text-slate-500 text-xs">Unfilled FVGs</span>
                        <p className="text-lg font-bold text-amber-500">{smcData.summary?.unfilled_fvgs || 0}</p>
                      </div>
                      <div>
                        <span className="text-slate-500 text-xs">Total Zones</span>
                        <p className="text-lg font-bold text-white">{smcData.summary?.total_order_blocks || 0}</p>
                      </div>
                    </div>
                  </div>

                  {/* IVR */}
                  {ivrData && (
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <h3 className="text-slate-400 text-sm mb-3">Options - IV Rank</h3>
                      <div className="flex items-center justify-between">
                        <span className="text-2xl font-bold text-white">{ivrData.ivr?.value?.toFixed(0)}%</span>
                        <span className={`text-sm ${
                          ivrData.ivr?.value >= 50 ? 'text-amber-500' : 'text-green-500'
                        }`}>
                          {ivrData.ivr?.interpretation}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">{ivrData.ivr?.strategy}</p>
                    </div>
                  )}

                  {/* Active SMC Signals */}
                  {smcData.signals && smcData.signals.length > 0 && (
                    <div className="p-4 bg-slate-800/50 rounded-lg">
                      <h3 className="text-slate-400 text-sm mb-3">Active SMC Signals</h3>
                      <div className="space-y-2">
                        {smcData.signals.slice(0, 3).map((signal, idx) => (
                          <div key={idx} className={`p-2 rounded flex justify-between items-center ${
                            signal.type === 'BUY' ? 'bg-green-500/10' : 'bg-red-500/10'
                          }`}>
                            <div className="flex items-center gap-2">
                              {signal.type === 'BUY' ? (
                                <ArrowUpRight className="w-4 h-4 text-green-500" />
                              ) : (
                                <ArrowDownRight className="w-4 h-4 text-red-500" />
                              )}
                              <span className="text-sm text-slate-300">{signal.indicator}</span>
                            </div>
                            <span className="text-sm text-white">${(signal.price ?? 0).toFixed(2)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-yellow-500" />
                Trading Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <h3 className="text-slate-400 text-sm mb-2">Bollinger Bands</h3>
                  <p className="text-white text-sm">{indicatorsData.bollinger_bands?.recommendation}</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <h3 className="text-slate-400 text-sm mb-2">ADX / Trend</h3>
                  <p className="text-white text-sm">{indicatorsData.adx?.recommendation}</p>
                </div>
                <div className="p-4 bg-slate-800/50 rounded-lg">
                  <h3 className="text-slate-400 text-sm mb-2">VWAP</h3>
                  <p className="text-white text-sm">{indicatorsData.vwap?.recommendation}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  )
}
