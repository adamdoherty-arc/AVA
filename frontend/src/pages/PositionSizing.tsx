import { useState, useMemo } from 'react'
import { Calculator, DollarSign, Percent, AlertTriangle, TrendingUp, Shield, Target, PieChart } from 'lucide-react'
import clsx from 'clsx'

interface CalculationResult {
    positionSize: number
    contracts: number
    maxLoss: number
    maxLossPct: number
    riskRewardRatio: number
    kellySize: number
    safeSize: number
}

export default function PositionSizing() {
    const [accountSize, setAccountSize] = useState(100000)
    const [riskPerTrade, setRiskPerTrade] = useState(2)
    const [entryPrice, setEntryPrice] = useState(5.00)
    const [stopLoss, setStopLoss] = useState(2.50)
    const [targetPrice, setTargetPrice] = useState(10.00)
    const [winRate, setWinRate] = useState(55)
    const [contractMultiplier] = useState(100)

    const result = useMemo<CalculationResult>(() => {
        const riskAmount = accountSize * (riskPerTrade / 100)
        const riskPerContract = (entryPrice - stopLoss) * contractMultiplier
        const contracts = Math.floor(riskAmount / riskPerContract)
        const positionSize = contracts * entryPrice * contractMultiplier
        const maxLoss = contracts * riskPerContract
        const maxLossPct = (maxLoss / accountSize) * 100
        const potentialProfit = (targetPrice - entryPrice) * contractMultiplier * contracts
        const riskRewardRatio = potentialProfit / maxLoss

        // Kelly Criterion
        const winProb = winRate / 100
        const avgWin = targetPrice - entryPrice
        const avgLoss = entryPrice - stopLoss
        const kellyPct = ((winProb * avgWin) - ((1 - winProb) * avgLoss)) / avgWin
        const kellySize = Math.max(0, kellyPct * accountSize)
        const safeSize = kellySize * 0.5 // Half-Kelly

        return {
            positionSize,
            contracts: Math.max(0, contracts),
            maxLoss,
            maxLossPct,
            riskRewardRatio: riskRewardRatio || 0,
            kellySize,
            safeSize
        }
    }, [accountSize, riskPerTrade, entryPrice, stopLoss, targetPrice, winRate, contractMultiplier])

    return (
        <div className="space-y-6">
            {/* Header */}
            <header>
                <h1 className="page-title flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg">
                        <Calculator className="w-5 h-5 text-white" />
                    </div>
                    Position Sizing Calculator
                </h1>
                <p className="page-subtitle">Calculate optimal position sizes based on risk management rules</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Input Section */}
                <div className="space-y-4">
                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <DollarSign className="w-5 h-5 text-emerald-400" />
                            Account Settings
                        </h3>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Account Size</label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
                                    <input
                                        type="number"
                                        value={accountSize}
                                        onChange={e => setAccountSize(Number(e.target.value))}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-8 pr-3 py-2"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Risk Per Trade: {riskPerTrade}%</label>
                                <input
                                    type="range"
                                    min={0.5}
                                    max={10}
                                    step={0.5}
                                    value={riskPerTrade}
                                    onChange={e => setRiskPerTrade(Number(e.target.value))}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-slate-500 mt-1">
                                    <span>Conservative (0.5%)</span>
                                    <span>Aggressive (10%)</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Target className="w-5 h-5 text-blue-400" />
                            Trade Parameters
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Entry Price</label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
                                    <input
                                        type="number"
                                        step="0.01"
                                        value={entryPrice}
                                        onChange={e => setEntryPrice(Number(e.target.value))}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-8 pr-3 py-2"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Stop Loss</label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
                                    <input
                                        type="number"
                                        step="0.01"
                                        value={stopLoss}
                                        onChange={e => setStopLoss(Number(e.target.value))}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-8 pr-3 py-2"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Target Price</label>
                                <div className="relative">
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
                                    <input
                                        type="number"
                                        step="0.01"
                                        value={targetPrice}
                                        onChange={e => setTargetPrice(Number(e.target.value))}
                                        className="w-full bg-slate-700/50 border border-slate-600 rounded-lg pl-8 pr-3 py-2"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-2">Win Rate %</label>
                                <input
                                    type="number"
                                    value={winRate}
                                    onChange={e => setWinRate(Number(e.target.value))}
                                    className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Section */}
                <div className="space-y-4">
                    <div className="card p-6 bg-gradient-to-br from-primary/10 to-secondary/10 border-primary/30">
                        <h3 className="text-lg font-semibold mb-4">Recommended Position</h3>
                        <div className="text-center mb-6">
                            <p className="text-5xl font-bold text-primary">{result.contracts}</p>
                            <p className="text-slate-400">contracts</p>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <p className="text-slate-400">Position Value</p>
                                <p className="font-mono font-bold">${result.positionSize.toLocaleString()}</p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <p className="text-slate-400">Max Loss</p>
                                <p className="font-mono font-bold text-rose-400">${result.maxLoss.toLocaleString()}</p>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <Shield className="w-4 h-4" />
                                <span className="text-sm">Risk %</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                result.maxLossPct <= 2 ? "text-emerald-400" :
                                result.maxLossPct <= 5 ? "text-amber-400" : "text-rose-400"
                            )}>
                                {result.maxLossPct.toFixed(1)}%
                            </p>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <TrendingUp className="w-4 h-4" />
                                <span className="text-sm">Risk/Reward</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                result.riskRewardRatio >= 2 ? "text-emerald-400" :
                                result.riskRewardRatio >= 1 ? "text-amber-400" : "text-rose-400"
                            )}>
                                1:{result.riskRewardRatio.toFixed(1)}
                            </p>
                        </div>
                    </div>

                    <div className="card p-4">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <PieChart className="w-5 h-5 text-purple-400" />
                            Kelly Criterion
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <p className="text-slate-400 text-sm">Full Kelly</p>
                                <p className="font-mono font-bold text-amber-400">
                                    ${result.kellySize.toLocaleString()}
                                </p>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3">
                                <p className="text-slate-400 text-sm">Half Kelly (Safer)</p>
                                <p className="font-mono font-bold text-emerald-400">
                                    ${result.safeSize.toLocaleString()}
                                </p>
                            </div>
                        </div>
                        <p className="text-xs text-slate-500 mt-3">
                            Kelly Criterion calculates optimal bet size based on win rate and payoff ratio
                        </p>
                    </div>

                    {result.maxLossPct > 5 && (
                        <div className="card p-4 bg-rose-500/10 border-rose-500/30">
                            <div className="flex items-center gap-2 text-rose-400">
                                <AlertTriangle className="w-5 h-5" />
                                <span className="font-semibold">High Risk Warning</span>
                            </div>
                            <p className="text-sm text-slate-400 mt-2">
                                Position risk exceeds 5% of account. Consider reducing position size or widening stop loss.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
