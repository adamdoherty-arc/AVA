import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Trophy, Clock, TrendingUp, Target, Zap, RefreshCw,
    ChevronRight, Star, AlertCircle, Activity, X, DollarSign, Flame, Medal, Brain
} from 'lucide-react'

export default function SportsBettingHub() {
    const [selectedLeague, setSelectedLeague] = useState('All')
    const [betSlip, setBetSlip] = useState<Bet[]>([])

    // Fetch live games
    const { data: liveGames, isLoading: liveLoading } = useQuery({
        queryKey: ['live-games'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/live')
            return data
        },
        refetchInterval: 30000
    })

    // Fetch upcoming games
    const { data: upcomingGames, isLoading: upcomingLoading } = useQuery({
        queryKey: ['upcoming-games'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/upcoming?limit=20')
            return data
        },
        refetchInterval: 60000
    })

    // Fetch best bets
    const { data: bestBets } = useQuery({
        queryKey: ['best-bets'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/best-bets')
            return data.bets || []  // Extract the bets array from response
        },
        staleTime: 300000
    })

    const isLoading = liveLoading || upcomingLoading

    const leagues = [
        { id: 'All', icon: Trophy, label: 'All Sports' },
        { id: 'NFL', icon: Trophy, label: 'NFL' },
        { id: 'NBA', icon: Trophy, label: 'NBA' },
        { id: 'NCAAF', icon: Medal, label: 'NCAAF' },
        { id: 'NCAAB', icon: Medal, label: 'NCAAB' },
        { id: 'MLB', icon: Trophy, label: 'MLB' },
        { id: 'NHL', icon: Trophy, label: 'NHL' },
    ]

    const allGames = [
        ...(liveGames || []).map((g: Game) => ({ ...g, isLive: true })),
        ...(upcomingGames || []).map((g: Game) => ({ ...g, isLive: false }))
    ]

    const filteredGames = selectedLeague === 'All'
        ? allGames
        : allGames.filter((g: Game) => g.league === selectedLeague)

    const addToBetSlip = (game: Game, betType: string, selection: string, odds: number) => {
        const newBet: Bet = {
            id: `${game.id}-${betType}-${Date.now()}`,
            gameId: game.id,
            teams: `${game.away_team} @ ${game.home_team}`,
            betType,
            selection,
            odds,
            stake: 0
        }
        setBetSlip([...betSlip, newBet])
    }

    const removeBet = (id: string) => {
        setBetSlip(betSlip.filter(b => b.id !== id))
    }

    const updateStake = (id: string, stake: number) => {
        setBetSlip(betSlip.map(b => b.id === id ? { ...b, stake } : b))
    }

    const totalStake = betSlip.reduce((sum, b) => sum + b.stake, 0)
    const potentialWin = betSlip.reduce((sum, b) => {
        const decimalOdds = b.odds > 0 ? (b.odds / 100) + 1 : (100 / Math.abs(b.odds)) + 1
        return sum + (b.stake * decimalOdds)
    }, 0)

    return (
        <div className="flex gap-6 h-[calc(100vh-8rem)]">
            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {/* Header */}
                <header className="flex items-center justify-between mb-6">
                    <div>
                        <h1 className="page-title flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
                                <Trophy className="w-5 h-5 text-white" />
                            </div>
                            Sports Betting Hub
                        </h1>
                        <p className="page-subtitle">AI-powered predictions and live odds</p>
                    </div>
                    <div className="flex items-center gap-3">
                        {liveGames?.length > 0 && (
                            <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 rounded-xl text-sm font-medium">
                                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                                {liveGames.length} Live Games
                            </div>
                        )}
                        <button className="btn-icon">
                            <RefreshCw className="w-5 h-5" />
                        </button>
                    </div>
                </header>

                {/* Stats Row */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="glass-card p-4 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                            <Activity className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">Active Games</p>
                            <p className="text-lg font-bold text-white">{allGames.length}</p>
                        </div>
                    </div>
                    <div className="glass-card p-4 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                            <Flame className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">Live Now</p>
                            <p className="text-lg font-bold text-white">{liveGames?.length || 0}</p>
                        </div>
                    </div>
                    <div className="glass-card p-4 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                            <Star className="w-5 h-5 text-amber-400" />
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">AI Picks</p>
                            <p className="text-lg font-bold text-white">{bestBets?.length || 0}</p>
                        </div>
                    </div>
                    <div className="glass-card p-4 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                            <Target className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-xs text-slate-400">Your Bets</p>
                            <p className="text-lg font-bold text-white">{betSlip.length}</p>
                        </div>
                    </div>
                </div>

                {/* League Filter */}
                <div className="flex gap-2 mb-6 overflow-x-auto pb-2 scrollbar-hide">
                    {leagues.map(league => (
                        <button
                            key={league.id}
                            onClick={() => setSelectedLeague(league.id)}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium text-sm transition-all whitespace-nowrap ${
                                selectedLeague === league.id
                                    ? 'bg-primary text-white shadow-lg shadow-primary/30'
                                    : 'bg-slate-800/60 text-slate-400 hover:bg-slate-700/60 hover:text-white border border-slate-700/50'
                            }`}
                        >
                            <league.icon className="w-4 h-4" />
                            {league.label}
                        </button>
                    ))}
                </div>

                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto space-y-6 pr-2">
                    {/* Best Bets Section */}
                    {bestBets?.length > 0 && (
                        <div className="glass-card p-5 bg-gradient-to-br from-amber-500/5 to-orange-500/5 border-amber-500/20">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                                    <Star className="w-4 h-4 text-amber-400" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">Top AI Picks</h3>
                                    <p className="text-xs text-slate-400">Highest confidence predictions today</p>
                                </div>
                            </div>
                            <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
                                {bestBets.slice(0, 5).map((bet: BestBet, idx: number) => (
                                    <div key={bet.id || idx} className="flex-shrink-0 glass-card p-4 min-w-[240px] group hover:border-amber-500/40 transition-all">
                                        <div className="flex items-center gap-2 mb-2">
                                            <span className="text-[10px] font-bold text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">{bet.sport}</span>
                                            <span className="text-[10px] text-slate-500">{bet.bet_type}</span>
                                        </div>
                                        <div className="text-sm font-medium text-white mb-1 group-hover:text-amber-400 transition-colors">
                                            {bet.matchup}
                                        </div>
                                        <div className="text-xs text-emerald-400 font-medium mb-3">{bet.pick} {bet.line}</div>
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-medium text-amber-400">{bet.confidence.toFixed(0)}%</span>
                                                <span className="text-[10px] text-slate-500">conf</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className={`text-xs font-mono font-medium px-2 py-0.5 rounded ${bet.ev >= 10 ? 'text-emerald-400 bg-emerald-500/15' : 'text-amber-400 bg-amber-500/15'}`}>
                                                    +{bet.ev.toFixed(1)}% EV
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Loading */}
                    {isLoading && (
                        <div className="glass-card p-12 text-center">
                            <div className="w-12 h-12 mx-auto mb-4 relative">
                                <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                                <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
                            </div>
                            <p className="text-slate-400">Loading games...</p>
                        </div>
                    )}

                    {/* Games Grid */}
                    {!isLoading && (
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                            {filteredGames.length > 0 ? (
                                filteredGames.map((game: Game) => (
                                    <GameCard
                                        key={game.id}
                                        game={game}
                                        onAddBet={addToBetSlip}
                                    />
                                ))
                            ) : (
                                <div className="col-span-2 glass-card p-12 text-center">
                                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/50 flex items-center justify-center">
                                        <Trophy className="w-8 h-8 text-slate-500" />
                                    </div>
                                    <h3 className="text-lg font-semibold text-white mb-2">No Games Found</h3>
                                    <p className="text-slate-400">No games available for {selectedLeague}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Bet Slip Sidebar */}
            <div className="w-80 flex-shrink-0">
                <div className="glass-card h-full flex flex-col overflow-hidden">
                    {/* Header */}
                    <div className="p-5 border-b border-slate-700/50 bg-slate-900/30">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg shadow-primary/20">
                                    <Target className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-semibold text-white">Bet Slip</h3>
                                    <p className="text-xs text-slate-400">{betSlip.length} selections</p>
                                </div>
                            </div>
                            {betSlip.length > 0 && (
                                <button
                                    onClick={() => setBetSlip([])}
                                    className="text-xs text-slate-400 hover:text-red-400 transition-colors"
                                >
                                    Clear All
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Bets */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        {betSlip.length === 0 ? (
                            <div className="text-center py-12">
                                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/50 flex items-center justify-center">
                                    <Target className="w-8 h-8 text-slate-500" />
                                </div>
                                <h4 className="text-sm font-medium text-white mb-1">No Selections</h4>
                                <p className="text-xs text-slate-400">Click on odds to add bets</p>
                            </div>
                        ) : (
                            betSlip.map(bet => (
                                <div key={bet.id} className="glass-card p-4 group">
                                    <div className="flex justify-between items-start mb-3">
                                        <div className="flex-1">
                                            <div className="text-sm font-medium text-white">{bet.selection}</div>
                                            <div className="text-xs text-slate-400">{bet.teams}</div>
                                            <div className="mt-1.5">
                                                <span className="text-xs badge-info">{bet.betType}</span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => removeBet(bet.id)}
                                            className="p-1 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded transition-all"
                                        >
                                            <X className="w-4 h-4" />
                                        </button>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="relative flex-1">
                                            <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                            <input
                                                type="number"
                                                value={bet.stake || ''}
                                                onChange={(e) => updateStake(bet.id, parseFloat(e.target.value) || 0)}
                                                placeholder="0.00"
                                                className="input-field pl-8 py-2 text-sm"
                                            />
                                        </div>
                                        <span className={`text-sm font-mono font-medium px-2.5 py-1.5 rounded-lg ${
                                            bet.odds > 0 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'
                                        }`}>
                                            {bet.odds > 0 ? '+' : ''}{bet.odds}
                                        </span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    {betSlip.length > 0 && (
                        <div className="p-5 border-t border-slate-700/50 bg-slate-900/30 space-y-4">
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-400">Total Stake</span>
                                    <span className="font-medium text-white">${totalStake.toFixed(2)}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-400">Potential Win</span>
                                    <span className="font-medium text-emerald-400">${potentialWin.toFixed(2)}</span>
                                </div>
                            </div>
                            <button
                                disabled={totalStake === 0}
                                className="btn-primary w-full py-3 flex items-center justify-center gap-2 text-base disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Zap className="w-5 h-5" />
                                Place Bet
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

// Types
interface Game {
    id: string
    home_team: string
    away_team: string
    home_score?: number
    away_score?: number
    league: string
    game_time: string
    status: string
    isLive?: boolean
    odds?: {
        spread_home?: number
        spread_home_odds?: number
        total?: number
        moneyline_home?: number
        moneyline_away?: number
    }
    ai_prediction?: {
        pick: string
        confidence: number
        probability: number
        spread: number
        ev?: number
        reasoning: string
    }
}

interface Bet {
    id: string
    gameId: string
    teams: string
    betType: string
    selection: string
    odds: number
    stake: number
}

interface BestBet {
    id: string
    sport: string
    matchup: string
    pick: string
    confidence: number
    ev: number
    odds: number
    line: string
    bet_type: string
    reasoning: string
}

// Game Card Component
function GameCard({ game, onAddBet }: { game: Game; onAddBet: (game: Game, betType: string, selection: string, odds: number) => void }) {
    return (
        <div className={`glass-card overflow-hidden group transition-all hover:border-primary/40 ${
            game.isLive ? 'border-emerald-500/40' : ''
        }`}>
            {/* Header */}
            <div className="px-4 py-3 flex justify-between items-center border-b border-slate-700/50 bg-slate-900/30">
                <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-lg bg-slate-800/80 flex items-center justify-center">
                        <Trophy className="w-3 h-3 text-slate-400" />
                    </div>
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                        {game.league}
                    </span>
                </div>
                {game.isLive ? (
                    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/15 rounded-lg text-xs font-medium text-emerald-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        LIVE
                    </div>
                ) : (
                    <div className="flex items-center gap-1.5 text-xs text-slate-400">
                        <Clock className="w-3 h-3" />
                        {game.game_time}
                    </div>
                )}
            </div>

            {/* Teams */}
            <div className="p-4">
                <div className="flex justify-between items-center mb-4">
                    <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-slate-800/80 flex items-center justify-center text-xs font-bold text-slate-400">
                                {game.away_team.substring(0, 2)}
                            </div>
                            <span className={`font-medium ${
                                game.ai_prediction?.pick === game.away_team ? 'text-primary' : 'text-white'
                            }`}>{game.away_team}</span>
                            {game.ai_prediction?.pick === game.away_team && (
                                <span className="text-xs px-2 py-0.5 bg-primary/20 text-primary rounded">AI Pick</span>
                            )}
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-slate-800/80 flex items-center justify-center text-xs font-bold text-slate-400">
                                {game.home_team.substring(0, 2)}
                            </div>
                            <span className={`font-medium ${
                                game.ai_prediction?.pick === game.home_team ? 'text-primary' : 'text-white'
                            }`}>{game.home_team}</span>
                            {game.ai_prediction?.pick === game.home_team && (
                                <span className="text-xs px-2 py-0.5 bg-primary/20 text-primary rounded">AI Pick</span>
                            )}
                        </div>
                    </div>
                    {game.isLive && (
                        <div className="text-right space-y-2">
                            <div className="text-2xl font-bold text-white">{game.away_score || 0}</div>
                            <div className="text-2xl font-bold text-white">{game.home_score || 0}</div>
                        </div>
                    )}
                </div>

                {/* AI Prediction Section */}
                {game.ai_prediction && (
                    <div className="mb-4 bg-gradient-to-r from-primary/10 to-violet-500/10 rounded-xl p-3 border border-primary/20">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2 text-xs font-medium text-primary">
                                <Brain className="w-3.5 h-3.5" />
                                AI Prediction
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs px-2 py-0.5 bg-slate-800/80 rounded text-slate-300">
                                    {game.ai_prediction.confidence}% conf
                                </span>
                                {game.ai_prediction.ev && game.ai_prediction.ev > 0 && (
                                    <span className="text-xs px-2 py-0.5 bg-emerald-500/15 rounded text-emerald-400">
                                        +{game.ai_prediction.ev.toFixed(1)}% EV
                                    </span>
                                )}
                            </div>
                        </div>
                        <p className="text-xs text-slate-400 line-clamp-2">
                            {game.ai_prediction.reasoning}
                        </p>
                    </div>
                )}

                {/* Odds - Always show, handle null values */}
                <div className="grid grid-cols-3 gap-2">
                    {/* Spread */}
                    <button
                        onClick={() => game.odds?.spread_home != null && onAddBet(game, 'Spread', `${game.home_team} ${game.odds?.spread_home}`, game.odds?.spread_home_odds || -110)}
                        disabled={game.odds?.spread_home == null}
                        className="bg-slate-800/60 hover:bg-primary/20 border border-slate-700/50 hover:border-primary/50 rounded-xl p-3 text-center transition-all group/btn disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-slate-800/60"
                    >
                        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Spread</div>
                        <div className="font-mono text-sm font-medium text-white group-hover/btn:text-primary">
                            {game.odds?.spread_home != null
                                ? `${game.odds.spread_home > 0 ? '+' : ''}${game.odds.spread_home}`
                                : '-'}
                        </div>
                    </button>

                    {/* Total */}
                    <button
                        onClick={() => game.odds?.total != null && onAddBet(game, 'Total', `O ${game.odds?.total}`, -110)}
                        disabled={game.odds?.total == null}
                        className="bg-slate-800/60 hover:bg-primary/20 border border-slate-700/50 hover:border-primary/50 rounded-xl p-3 text-center transition-all group/btn disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-slate-800/60"
                    >
                        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Total</div>
                        <div className="font-mono text-sm font-medium text-white group-hover/btn:text-primary">
                            {game.odds?.total != null ? `O/U ${game.odds.total}` : '-'}
                        </div>
                    </button>

                    {/* Moneyline */}
                    <button
                        onClick={() => game.odds?.moneyline_home != null && onAddBet(game, 'ML', game.home_team, game.odds?.moneyline_home || -110)}
                        disabled={game.odds?.moneyline_home == null}
                        className="bg-slate-800/60 hover:bg-primary/20 border border-slate-700/50 hover:border-primary/50 rounded-xl p-3 text-center transition-all group/btn disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-slate-800/60"
                    >
                        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">ML</div>
                        <div className={`font-mono text-sm font-medium group-hover/btn:text-primary ${
                            (game.odds?.moneyline_home || 0) > 0 ? 'text-emerald-400' : 'text-white'
                        }`}>
                            {game.odds?.moneyline_home != null
                                ? `${game.odds.moneyline_home > 0 ? '+' : ''}${game.odds.moneyline_home}`
                                : '-'}
                        </div>
                    </button>
                </div>
            </div>
        </div>
    )
}
