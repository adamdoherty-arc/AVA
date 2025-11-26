import { useState, useEffect } from 'react'
import { Calendar, Trophy, AlertCircle, Brain, Target, TrendingUp, RefreshCw, Zap } from 'lucide-react'
import clsx from 'clsx'
import { api } from '../services/api'

export function GameCards() {
    const [selectedLeague, setSelectedLeague] = useState<string>('All')
    const [games, setGames] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const fetchGames = async () => {
            setLoading(true)
            setError(null)
            try {
                const response = await api.getMarkets(selectedLeague === 'All' ? undefined : selectedLeague)

                const mappedGames = response.markets.map((market: any) => ({
                    id: market.id,
                    homeTeam: market.home_team || extractTeam(market.title, 'home'),
                    awayTeam: market.away_team || extractTeam(market.title, 'away'),
                    homeScore: 0,
                    awayScore: 0,
                    status: getStatus(market.game_date),
                    league: market.market_type?.toUpperCase() || 'SPORTS',
                    date: formatDate(market.game_date),
                    prediction: {
                        winner: market.predicted_outcome || 'TBD',
                        confidence: market.confidence_score ? market.confidence_score / 100 : 0,
                        ev: market.edge_percentage || 0,
                        explanation: market.reasoning || 'No analysis available.'
                    }
                }))
                setGames(mappedGames)
            } catch (err) {
                console.error("Failed to fetch games:", err)
                setError("Failed to load games. Please try again.")
            } finally {
                setLoading(false)
            }
        }

        fetchGames()
    }, [selectedLeague])

    const extractTeam = (title: string, side: 'home' | 'away') => {
        if (title.includes(' vs ')) {
            const parts = title.split(' vs ')
            return side === 'home' ? parts[0] : parts[1]
        }
        return side === 'home' ? 'Home' : 'Away'
    }

    const getStatus = (dateStr?: string) => {
        if (!dateStr) return 'Scheduled'
        const gameDate = new Date(dateStr)
        const now = new Date()
        if (now > gameDate) return 'Live'
        return 'Scheduled'
    }

    const formatDate = (dateStr?: string) => {
        if (!dateStr) return 'TBD'
        return new Date(dateStr).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        })
    }

    const leagues = ['All', 'NFL', 'NBA', 'NCAAF']

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-green-500/20">
                        <Trophy className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Live Games</h1>
                        <p className="text-sm text-slate-400">Real-time scores and AI predictions</p>
                    </div>
                </div>
                <button
                    onClick={() => window.location.reload()}
                    className="btn-primary flex items-center gap-2"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </header>

            {/* Stats Row */}
            <div className="grid grid-cols-4 gap-4">
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center">
                        <Trophy className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Total Games</p>
                        <p className="text-xl font-bold text-white">{games.length}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                        <Zap className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Live Now</p>
                        <p className="text-xl font-bold text-emerald-400">{games.filter(g => g.status === 'Live').length}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                        <Brain className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Predictions</p>
                        <p className="text-xl font-bold text-white">{games.filter(g => g.prediction.winner !== 'TBD').length}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                        <Target className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Avg Confidence</p>
                        <p className="text-xl font-bold text-white">
                            {games.length > 0
                                ? `${Math.round(games.reduce((acc, g) => acc + g.prediction.confidence, 0) / games.length * 100)}%`
                                : '-'}
                        </p>
                    </div>
                </div>
            </div>

            {/* League Filter */}
            <div className="glass-card p-4">
                <div className="flex items-center gap-4">
                    <span className="text-sm text-slate-400">Filter by league:</span>
                    <div className="flex gap-2">
                        {leagues.map(league => (
                            <button
                                key={league}
                                onClick={() => setSelectedLeague(league)}
                                className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                                    selectedLeague === league
                                        ? "bg-green-500 text-white shadow-lg shadow-green-500/20"
                                        : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white border border-slate-700/50"
                                )}
                            >
                                {league}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Error State */}
            {error && (
                <div className="glass-card p-5 border border-red-500/30 bg-red-500/5 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center flex-shrink-0">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                    </div>
                    <div>
                        <h4 className="font-medium text-red-400">Error Loading Games</h4>
                        <p className="text-sm text-slate-400">{error}</p>
                    </div>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div className="glass-card p-12 text-center">
                    <div className="relative w-16 h-16 mx-auto mb-6">
                        <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                        <div className="absolute inset-0 rounded-full border-4 border-t-green-500 border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">Loading Games</h3>
                    <p className="text-slate-400">Fetching latest scores and predictions...</p>
                </div>
            )}

            {/* Game Grid */}
            {!loading && !error && (
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
                    {games.length > 0 ? (
                        games.map(game => (
                            <GameCard key={game.id} game={game} />
                        ))
                    ) : (
                        <div className="col-span-full glass-card p-16 text-center">
                            <div className="w-20 h-20 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-6 border border-slate-700/50">
                                <Trophy className="w-10 h-10 text-slate-500" />
                            </div>
                            <h3 className="text-xl font-bold text-white mb-2">No Games Found</h3>
                            <p className="text-slate-400">Try selecting a different league or check back later.</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

function GameCard({ game }: { game: any }) {
    const getLeagueColor = (league: string) => {
        switch (league.toUpperCase()) {
            case 'NFL': return 'badge-danger'
            case 'NBA': return 'badge-warning'
            case 'NCAAF': return 'badge-info'
            default: return 'badge-neutral'
        }
    }

    const getStatusStyle = (status: string) => {
        switch (status) {
            case 'Live': return 'bg-emerald-500/20 text-emerald-400 animate-pulse'
            case 'Final': return 'bg-slate-700/50 text-slate-400'
            default: return 'bg-blue-500/20 text-blue-400'
        }
    }

    const getTeamInitials = (name: string) => {
        if (!name) return '?'
        return name.split(' ').map(w => w[0]).join('').slice(0, 3).toUpperCase()
    }

    return (
        <div className="glass-card overflow-hidden hover:border-green-500/30 transition-all duration-200 group">
            {/* Header */}
            <div className="bg-slate-900/50 px-5 py-3 flex justify-between items-center border-b border-slate-700/50">
                <div className="flex items-center gap-2">
                    <Trophy size={14} className="text-slate-400" />
                    <span className={getLeagueColor(game.league)}>
                        {game.league}
                    </span>
                </div>
                <span className={clsx(
                    "px-2.5 py-1 rounded-lg text-xs font-bold",
                    getStatusStyle(game.status)
                )}>
                    {game.status}
                </span>
            </div>

            {/* Teams & Scores */}
            <div className="p-5">
                <div className="flex justify-between items-center mb-5">
                    {/* Home Team */}
                    <div className="text-center flex-1">
                        <div className="w-14 h-14 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl mb-2 mx-auto flex items-center justify-center text-lg font-bold text-white border border-slate-600/50">
                            {getTeamInitials(game.homeTeam)}
                        </div>
                        <div className="font-bold text-white text-sm truncate max-w-[100px] mx-auto" title={game.homeTeam}>
                            {game.homeTeam}
                        </div>
                    </div>

                    {/* Score */}
                    <div className="px-4">
                        <div className="text-3xl font-bold text-slate-500">
                            {game.homeScore} <span className="text-slate-600">-</span> {game.awayScore}
                        </div>
                    </div>

                    {/* Away Team */}
                    <div className="text-center flex-1">
                        <div className="w-14 h-14 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl mb-2 mx-auto flex items-center justify-center text-lg font-bold text-white border border-slate-600/50">
                            {getTeamInitials(game.awayTeam)}
                        </div>
                        <div className="font-bold text-white text-sm truncate max-w-[100px] mx-auto" title={game.awayTeam}>
                            {game.awayTeam}
                        </div>
                    </div>
                </div>

                {/* AI Prediction */}
                <div className="bg-gradient-to-r from-primary/10 to-violet-500/10 rounded-xl p-4 border border-primary/20">
                    <div className="flex items-center justify-between mb-2">
                        <span className="flex items-center gap-2 text-sm font-medium text-primary">
                            <Brain size={14} />
                            AI Prediction
                        </span>
                        <span className="badge-info">
                            {(game.prediction.confidence * 100).toFixed(0)}% Conf
                        </span>
                    </div>
                    <p className="text-sm text-white mb-2">
                        Picking <span className="font-bold text-primary">{game.prediction.winner}</span>
                    </p>
                    <p className="text-xs text-slate-400 line-clamp-2">
                        {game.prediction.explanation}
                    </p>
                </div>
            </div>

            {/* Footer */}
            <div className="px-5 py-3 bg-slate-900/30 border-t border-slate-700/50 flex justify-between items-center">
                <span className="flex items-center gap-1.5 text-xs text-slate-400">
                    <Calendar size={12} />
                    {game.date}
                </span>
                <span className={clsx(
                    "text-xs font-mono font-bold",
                    game.prediction.ev > 0 ? "text-emerald-400" : game.prediction.ev < 0 ? "text-red-400" : "text-slate-400"
                )}>
                    EV: {game.prediction.ev > 0 ? '+' : ''}{game.prediction.ev.toFixed(1)}%
                </span>
            </div>
        </div>
    )
}

export default GameCards
