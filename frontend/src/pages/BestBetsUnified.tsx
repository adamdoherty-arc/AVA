import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Trophy, RefreshCw, TrendingUp, Target, Percent, Clock,
    AlertCircle, Filter, DollarSign, Activity, ChevronDown,
    ChevronUp, Zap, Award, Star
} from 'lucide-react'
import clsx from 'clsx'

interface BetOpportunity {
    id: string
    sport: string
    event_name: string
    bet_type: string
    selection: string
    odds: number
    implied_probability: number
    model_probability: number
    ev_percentage: number
    confidence: number
    overall_score: number
    book: string
    updated_at: string
    game_time: string
    home_team: string
    away_team: string
}

interface BestBetsData {
    opportunities: BetOpportunity[]
    sport_summary: Record<string, number>
    avg_ev: number
    avg_confidence: number
}

const SPORTS = ['All', 'NFL', 'NBA', 'NCAA', 'NCAAB', 'MLB', 'NHL']

export default function BestBetsUnified() {
    const [selectedSports, setSelectedSports] = useState<string[]>(['All'])
    const [minEV, setMinEV] = useState(5)
    const [minConfidence, setMinConfidence] = useState(60)
    const [topN, setTopN] = useState(20)
    const [sortBy, setSortBy] = useState<'score' | 'ev' | 'confidence'>('score')
    const [showFilters, setShowFilters] = useState(false)

    const { data, isLoading, refetch, error } = useQuery<BestBetsData>({
        queryKey: ['best-bets', selectedSports, minEV, minConfidence, topN],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/best-bets/unified', {
                params: {
                    sports: selectedSports.includes('All') ? undefined : selectedSports.join(','),
                    min_ev: minEV / 100,
                    min_confidence: minConfidence,
                    top_n: topN
                }
            })
            return data
        },
        staleTime: 60000, // 1 minute
    })

    const toggleSport = (sport: string) => {
        if (sport === 'All') {
            setSelectedSports(['All'])
        } else {
            const newSports = selectedSports.includes(sport)
                ? selectedSports.filter(s => s !== sport)
                : [...selectedSports.filter(s => s !== 'All'), sport]
            setSelectedSports(newSports.length === 0 ? ['All'] : newSports)
        }
    }

    const opportunities = data?.opportunities || []
    const sortedOpportunities = [...opportunities].sort((a, b) => {
        if (sortBy === 'ev') return b.ev_percentage - a.ev_percentage
        if (sortBy === 'confidence') return b.confidence - a.confidence
        return b.overall_score - a.overall_score
    })

    const getSportIcon = (sport: string) => {
        const icons: Record<string, string> = {
            'NFL': '🏈',
            'NBA': '🏀',
            'NCAA': '🏈',
            'NCAAB': '🏀',
            'MLB': '⚾',
            'NHL': '🏒'
        }
        return icons[sport] || '🎯'
    }

    const getScoreColor = (score: number) => {
        if (score >= 80) return 'text-emerald-400'
        if (score >= 60) return 'text-amber-400'
        return 'text-red-400'
    }

    const formatOdds = (odds: number) => {
        if (odds >= 2) return `+${((odds - 1) * 100).toFixed(0)}`
        return `-${(100 / (odds - 1)).toFixed(0)}`
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg">
                            <Trophy className="w-5 h-5 text-white" />
                        </div>
                        Best Bets Unified
                    </h1>
                    <p className="page-subtitle">Top profitable opportunities across all sports</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={clsx("btn-icon", showFilters && "bg-primary/20 text-primary")}
                    >
                        <Filter className="w-5 h-5" />
                    </button>
                    <button
                        onClick={() => refetch()}
                        disabled={isLoading}
                        className="btn-icon"
                    >
                        <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                    </button>
                </div>
            </header>

            {/* Sport Summary */}
            {data?.sport_summary && (
                <div className="flex gap-2 flex-wrap">
                    {SPORTS.map(sport => {
                        const count = sport === 'All'
                            ? Object.values(data.sport_summary).reduce((a, b) => a + b, 0)
                            : data.sport_summary[sport] || 0
                        const isSelected = selectedSports.includes(sport)
                        return (
                            <button
                                key={sport}
                                onClick={() => toggleSport(sport)}
                                className={clsx(
                                    "px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                    isSelected
                                        ? "bg-primary text-white"
                                        : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50"
                                )}
                            >
                                <span>{getSportIcon(sport)}</span>
                                <span>{sport}</span>
                                <span className={clsx(
                                    "px-1.5 py-0.5 rounded text-xs",
                                    isSelected ? "bg-white/20" : "bg-slate-600/50"
                                )}>
                                    {count}
                                </span>
                            </button>
                        )
                    })}
                </div>
            )}

            {/* Filters Panel */}
            {showFilters && (
                <div className="card p-4">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Min Expected Value %</label>
                            <div className="flex items-center gap-2">
                                <input
                                    type="range"
                                    min={0}
                                    max={30}
                                    value={minEV}
                                    onChange={e => setMinEV(Number(e.target.value))}
                                    className="flex-1"
                                />
                                <span className="text-sm font-mono w-12 text-right">{minEV}%</span>
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Min Confidence</label>
                            <div className="flex items-center gap-2">
                                <input
                                    type="range"
                                    min={0}
                                    max={100}
                                    step={5}
                                    value={minConfidence}
                                    onChange={e => setMinConfidence(Number(e.target.value))}
                                    className="flex-1"
                                />
                                <span className="text-sm font-mono w-12 text-right">{minConfidence}</span>
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Show Top N</label>
                            <select
                                value={topN}
                                onChange={e => setTopN(Number(e.target.value))}
                                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                            >
                                {[10, 20, 30, 50].map(n => (
                                    <option key={n} value={n}>{n} bets</option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-slate-400 mb-2">Sort By</label>
                            <select
                                value={sortBy}
                                onChange={e => setSortBy(e.target.value as any)}
                                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                            >
                                <option value="score">Overall Score</option>
                                <option value="ev">Expected Value</option>
                                <option value="confidence">Confidence</option>
                            </select>
                        </div>
                    </div>
                </div>
            )}

            {/* Summary Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Target className="w-4 h-4" />
                        <span className="text-sm">Total Opportunities</span>
                    </div>
                    <p className="text-2xl font-bold">{opportunities.length}</p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-sm">Avg Expected Value</span>
                    </div>
                    <p className="text-2xl font-bold text-emerald-400">
                        {data?.avg_ev?.toFixed(1) || '0'}%
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-sm">Avg Confidence</span>
                    </div>
                    <p className="text-2xl font-bold text-amber-400">
                        {data?.avg_confidence?.toFixed(0) || '0'}
                    </p>
                </div>
                <div className="card p-4">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                        <Zap className="w-4 h-4" />
                        <span className="text-sm">High Value (80+)</span>
                    </div>
                    <p className="text-2xl font-bold text-primary">
                        {opportunities.filter(o => o.overall_score >= 80).length}
                    </p>
                </div>
            </div>

            {/* Best Bets List */}
            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Analyzing opportunities...</span>
                </div>
            ) : error ? (
                <div className="card p-8 flex items-center justify-center text-red-400">
                    <AlertCircle className="w-6 h-6 mr-2" />
                    Failed to load best bets
                </div>
            ) : sortedOpportunities.length === 0 ? (
                <div className="card p-8 text-center text-slate-400">
                    <Trophy className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No opportunities found matching your filters</p>
                    <p className="text-sm mt-1">Try lowering the minimum EV or confidence</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {sortedOpportunities.map((bet, index) => (
                        <BetCard
                            key={bet.id}
                            bet={bet}
                            rank={index + 1}
                            getSportIcon={getSportIcon}
                            getScoreColor={getScoreColor}
                            formatOdds={formatOdds}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}

function BetCard({
    bet,
    rank,
    getSportIcon,
    getScoreColor,
    formatOdds
}: {
    bet: BetOpportunity
    rank: number
    getSportIcon: (sport: string) => string
    getScoreColor: (score: number) => string
    formatOdds: (odds: number) => string
}) {
    const [expanded, setExpanded] = useState(false)

    return (
        <div className="card overflow-hidden">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full p-4 flex items-center gap-4 hover:bg-slate-800/50 transition-colors"
            >
                {/* Rank Badge */}
                <div className={clsx(
                    "w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold",
                    rank <= 3 ? "bg-amber-500/20 text-amber-400" : "bg-slate-700 text-slate-400"
                )}>
                    {rank}
                </div>

                {/* Sport Icon */}
                <div className="text-2xl">{getSportIcon(bet.sport)}</div>

                {/* Event Info */}
                <div className="flex-1 text-left">
                    <div className="flex items-center gap-2">
                        <span className="font-semibold">{bet.selection}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300">
                            {bet.bet_type}
                        </span>
                    </div>
                    <p className="text-sm text-slate-400">
                        {bet.away_team} @ {bet.home_team}
                    </p>
                </div>

                {/* Key Metrics */}
                <div className="flex items-center gap-6">
                    <div className="text-center">
                        <p className="text-xs text-slate-400">Score</p>
                        <p className={clsx("font-bold", getScoreColor(bet.overall_score ?? 0))}>
                            {(bet.overall_score ?? 0).toFixed(0)}
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-xs text-slate-400">EV</p>
                        <p className="font-bold text-emerald-400">+{(bet.ev_percentage ?? 0).toFixed(1)}%</p>
                    </div>
                    <div className="text-center">
                        <p className="text-xs text-slate-400">Odds</p>
                        <p className="font-mono font-bold">{formatOdds(bet.odds)}</p>
                    </div>
                    <ChevronDown className={clsx(
                        "w-5 h-5 text-slate-400 transition-transform",
                        expanded && "rotate-180"
                    )} />
                </div>
            </button>

            {expanded && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-700/50 grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Confidence</p>
                        <p className="font-medium">{(bet.confidence ?? 0).toFixed(0)}%</p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Implied Prob</p>
                        <p className="font-mono">{((bet.implied_probability ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Model Prob</p>
                        <p className="font-mono text-emerald-400">{((bet.model_probability ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Book</p>
                        <p className="font-medium">{bet.book}</p>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 mb-1">Game Time</p>
                        <p className="font-medium">{new Date(bet.game_time).toLocaleString()}</p>
                    </div>
                </div>
            )}
        </div>
    )
}
