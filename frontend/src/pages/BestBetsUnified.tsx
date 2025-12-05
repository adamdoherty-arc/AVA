import { useState, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Trophy, RefreshCw, TrendingUp, Target,
    AlertCircle, Filter, Activity, ChevronDown, Zap,
    Database, Check, Loader2, CloudDownload, ExternalLink
} from 'lucide-react'
import clsx from 'clsx'

interface SyncProgress {
    step: 'idle' | 'games' | 'odds' | 'complete' | 'error'
    message: string
    details?: {
        gamesSync?: { total: number; sports: Record<string, number> }
        oddsSync?: { updated: number; quota?: { remaining: number; used: number } }
    }
}

interface BetOpportunity {
    id: string
    sport: string
    event_name: string
    bet_type: string
    selection: string
    odds: number
    odds_american: number
    implied_probability: number
    model_probability: number
    ai_edge: number
    kelly_fraction: number
    ev_percentage: number
    confidence: number
    overall_score: number
    book: string
    reasoning?: string
    spread?: number
    over_under?: number
    odds_home?: number
    odds_away?: number
    updated_at: string
    game_time: string
    home_team: string
    away_team: string
    kalshi_url?: string
}

interface BestBetsData {
    opportunities: BetOpportunity[]
    sport_summary: Record<string, number>
    avg_ev: number
    avg_confidence: number
}

const SPORTS = ['All', 'NFL', 'NBA', 'NCAAF', 'NCAAB']

export default function BestBetsUnified() {
    const [selectedSports, setSelectedSports] = useState<string[]>(['All'])
    const [minEV, setMinEV] = useState(0)
    const [minConfidence, setMinConfidence] = useState(50)
    const [topN, setTopN] = useState(20)
    const [sortBy, setSortBy] = useState<'score' | 'ev' | 'confidence'>('score')
    const [showFilters, setShowFilters] = useState(false)
    const [syncProgress, setSyncProgress] = useState<SyncProgress>({ step: 'idle', message: '' })

    const { data, isLoading, refetch, error } = useQuery<BestBetsData>({
        queryKey: ['best-bets', selectedSports, minEV, minConfidence, topN],
        queryFn: async () => {
            // Use v2 endpoint which has proper async handling and AI predictions
            const { data } = await axiosInstance.get('/sports/v2/best-bets', {
                params: {
                    sports: selectedSports.includes('All') ? 'NFL,NBA,NCAAF,NCAAB' : selectedSports.join(','),
                    min_ev: minEV,
                    min_edge: 1.0,
                    min_confidence: minConfidence > 50 ? 'medium' : 'low',
                    limit: topN
                }
            })
            return data
        },
        staleTime: 60000,
    })

    // Full sync mutation - syncs games from ESPN then odds from The Odds API
    const syncMutation = useMutation({
        mutationFn: async () => {
            // Step 1: Sync games from ESPN for all sports
            setSyncProgress({ step: 'games', message: 'Syncing games from ESPN...' })
            const gamesResult = await axiosInstance.post('/sports/sync', null, {
                params: { sport: 'ALL' }
            })

            const gamesData = gamesResult.data
            setSyncProgress({
                step: 'games',
                message: `Synced ${gamesData.total_synced} games`,
                details: { gamesSync: { total: gamesData.total_synced, sports: gamesData.results } }
            })

            // Step 2: Sync real odds from The Odds API
            setSyncProgress({
                step: 'odds',
                message: 'Fetching live odds from The Odds API...',
                details: { gamesSync: { total: gamesData.total_synced, sports: gamesData.results } }
            })

            const oddsResult = await axiosInstance.post('/sports/sync-real-odds', null, {
                params: { sports: 'NFL,NBA,NCAAF,NCAAB' }
            })

            const oddsData = oddsResult.data

            return {
                games: gamesData,
                odds: oddsData
            }
        },
        onSuccess: (data) => {
            const totalGames = data.games.total_synced || 0
            const oddsUpdated = data.odds.totals?.updated || 0
            const quota = data.odds.quota

            setSyncProgress({
                step: 'complete',
                message: `Synced ${totalGames} games, updated ${oddsUpdated} odds`,
                details: {
                    gamesSync: { total: totalGames, sports: data.games.results || {} },
                    oddsSync: { updated: oddsUpdated, quota }
                }
            })

            // Refetch best bets with fresh data
            setTimeout(() => refetch(), 500)

            // Reset progress after 5 seconds
            setTimeout(() => setSyncProgress({ step: 'idle', message: '' }), 5000)
        },
        onError: (error: Error) => {
            setSyncProgress({
                step: 'error',
                message: `Sync failed: ${error.message}`
            })
            setTimeout(() => setSyncProgress({ step: 'idle', message: '' }), 5000)
        }
    })

    const handleSync = useCallback(() => {
        if (!syncMutation.isPending) {
            syncMutation.mutate()
        }
    }, [syncMutation])

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
            'NCAAF': '🏈',
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

    const formatAmericanOdds = (odds: number | undefined) => {
        if (odds === undefined || odds === null) return 'N/A'
        return odds > 0 ? `+${odds}` : `${odds}`
    }

    const getEdgeColor = (edge: number) => {
        if (edge >= 5) return 'text-emerald-400'
        if (edge >= 2) return 'text-amber-400'
        if (edge >= 0) return 'text-slate-400'
        return 'text-red-400'
    }

    const isSyncing = syncMutation.isPending

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

                    {/* Sync Button with Progress */}
                    <button
                        onClick={handleSync}
                        disabled={isSyncing}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-2 rounded-lg font-medium text-sm transition-all",
                            isSyncing
                                ? "bg-blue-600/20 text-blue-400 cursor-wait"
                                : syncProgress.step === 'complete'
                                ? "bg-emerald-600/20 text-emerald-400"
                                : syncProgress.step === 'error'
                                ? "bg-red-600/20 text-red-400"
                                : "bg-slate-700/50 text-slate-300 hover:bg-slate-600/50"
                        )}
                    >
                        {isSyncing ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span className="hidden sm:inline">
                                    {syncProgress.step === 'games' ? 'Syncing Games...' : 'Syncing Odds...'}
                                </span>
                            </>
                        ) : syncProgress.step === 'complete' ? (
                            <>
                                <Check className="w-4 h-4" />
                                <span className="hidden sm:inline">Synced!</span>
                            </>
                        ) : (
                            <>
                                <CloudDownload className="w-4 h-4" />
                                <span className="hidden sm:inline">Sync All</span>
                            </>
                        )}
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

            {/* Sync Progress Banner */}
            {syncProgress.step !== 'idle' && (
                <div className={clsx(
                    "rounded-lg p-4 flex items-center gap-3",
                    syncProgress.step === 'complete' ? "bg-emerald-900/30 border border-emerald-700/50" :
                    syncProgress.step === 'error' ? "bg-red-900/30 border border-red-700/50" :
                    "bg-blue-900/30 border border-blue-700/50"
                )}>
                    {isSyncing ? (
                        <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
                    ) : syncProgress.step === 'complete' ? (
                        <Check className="w-5 h-5 text-emerald-400" />
                    ) : syncProgress.step === 'error' ? (
                        <AlertCircle className="w-5 h-5 text-red-400" />
                    ) : null}

                    <div className="flex-1">
                        <p className={clsx(
                            "font-medium",
                            syncProgress.step === 'complete' ? "text-emerald-300" :
                            syncProgress.step === 'error' ? "text-red-300" :
                            "text-blue-300"
                        )}>
                            {syncProgress.message}
                        </p>

                        {syncProgress.details && syncProgress.step === 'complete' && (
                            <div className="mt-2 flex flex-wrap gap-4 text-sm">
                                {syncProgress.details.gamesSync && (
                                    <div className="flex items-center gap-2 text-slate-400">
                                        <Database className="w-4 h-4" />
                                        <span>
                                            {(Object.entries(syncProgress.details.gamesSync.sports || {}) as [string, { synced?: number }][])
                                                .filter(([, v]) => v && typeof v === 'object' && v.synced && v.synced > 0)
                                                .map(([sport, v]) => `${sport}: ${v?.synced || 0}`)
                                                .join(', ') || `${syncProgress.details.gamesSync.total} games`}
                                        </span>
                                    </div>
                                )}
                                {syncProgress.details.oddsSync?.quota && (
                                    <div className="flex items-center gap-2 text-slate-400">
                                        <Activity className="w-4 h-4" />
                                        <span>API Quota: {syncProgress.details.oddsSync.quota.remaining} remaining</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Progress Steps */}
                    {isSyncing && (
                        <div className="flex items-center gap-2">
                            <div className={clsx(
                                "w-2 h-2 rounded-full",
                                syncProgress.step === 'games' ? "bg-blue-400 animate-pulse" : "bg-emerald-400"
                            )} />
                            <span className="text-xs text-slate-400">Games</span>
                            <div className={clsx(
                                "w-2 h-2 rounded-full",
                                syncProgress.step === 'odds' ? "bg-blue-400 animate-pulse" :
                                syncProgress.step === 'games' ? "bg-slate-600" : "bg-emerald-400"
                            )} />
                            <span className="text-xs text-slate-400">Odds</span>
                        </div>
                    )}
                </div>
            )}

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
                                onChange={e => setSortBy(e.target.value as 'score' | 'ev' | 'confidence')}
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
                    <button
                        onClick={handleSync}
                        disabled={isSyncing}
                        className="mt-4 px-4 py-2 bg-primary/20 text-primary rounded-lg hover:bg-primary/30 transition-colors"
                    >
                        {isSyncing ? 'Syncing...' : 'Sync Games & Odds'}
                    </button>
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
                            formatAmericanOdds={formatAmericanOdds}
                            getEdgeColor={getEdgeColor}
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
    formatOdds,
    formatAmericanOdds,
    getEdgeColor
}: {
    bet: BetOpportunity
    rank: number
    getSportIcon: (sport: string) => string
    getScoreColor: (score: number) => string
    formatOdds: (odds: number) => string
    formatAmericanOdds: (odds: number | undefined) => string
    getEdgeColor: (edge: number) => string
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
                        <p className={clsx("font-bold", getScoreColor(bet.overall_score))}>
                            {bet.overall_score.toFixed(0)}
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-xs text-slate-400">EV</p>
                        <p className="font-bold text-emerald-400">+{bet.ev_percentage.toFixed(1)}%</p>
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
                <div className="px-4 pb-4 pt-2 border-t border-slate-700/50 space-y-3">
                    {/* Row 1: Core Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                        <div>
                            <p className="text-xs text-slate-400 mb-1">American Odds</p>
                            <p className="font-mono font-bold">{formatAmericanOdds(bet.odds_american)}</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">AI Edge</p>
                            <p className={clsx("font-mono font-bold", getEdgeColor(bet.ai_edge || 0))}>
                                {bet.ai_edge !== undefined ? `${bet.ai_edge > 0 ? '+' : ''}${bet.ai_edge.toFixed(1)}%` : 'N/A'}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Kelly Bet Size</p>
                            <p className="font-mono text-amber-400">
                                {bet.kelly_fraction !== undefined ? `${(bet.kelly_fraction * 100).toFixed(1)}%` : 'N/A'}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Implied Prob</p>
                            <p className="font-mono">{(bet.implied_probability * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Model Prob</p>
                            <p className="font-mono text-emerald-400">{(bet.model_probability * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Confidence</p>
                            <p className="font-medium">{bet.confidence.toFixed(0)}%</p>
                        </div>
                    </div>

                    {/* Row 2: Game Details */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {bet.spread !== undefined && bet.spread !== null && (
                            <div>
                                <p className="text-xs text-slate-400 mb-1">Spread</p>
                                <p className="font-mono">{bet.spread > 0 ? `+${bet.spread}` : bet.spread}</p>
                            </div>
                        )}
                        {bet.over_under !== undefined && bet.over_under !== null && (
                            <div>
                                <p className="text-xs text-slate-400 mb-1">O/U Total</p>
                                <p className="font-mono">{bet.over_under}</p>
                            </div>
                        )}
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Source</p>
                            <p className="font-medium text-xs">{bet.book}</p>
                        </div>
                        <div>
                            <p className="text-xs text-slate-400 mb-1">Game Time</p>
                            <p className="font-medium text-xs">{new Date(bet.game_time).toLocaleString()}</p>
                        </div>
                        {bet.kalshi_url && (
                            <div>
                                <p className="text-xs text-slate-400 mb-1">Trade on Kalshi</p>
                                <a
                                    href={bet.kalshi_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-purple-600/20 hover:bg-purple-600/40 text-purple-400 rounded-lg text-sm font-medium transition-colors"
                                    onClick={(e) => e.stopPropagation()}
                                >
                                    <ExternalLink className="w-3.5 h-3.5" />
                                    View on Kalshi
                                </a>
                            </div>
                        )}
                    </div>

                    {/* Row 3: AI Reasoning */}
                    {bet.reasoning && (
                        <div className="bg-slate-800/50 rounded-lg p-3">
                            <p className="text-xs text-slate-400 mb-1">AI Analysis</p>
                            <p className="text-sm text-slate-300">{bet.reasoning}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
