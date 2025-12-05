import { useState, useEffect, useMemo, useCallback, useRef, memo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Trophy, Clock, TrendingUp, Target, Zap, RefreshCw,
    ChevronRight, Star, AlertCircle, Activity, X, DollarSign, Flame, Medal, Brain,
    ChevronLeft, ChevronDown, ChevronUp, Send, Bell, Settings, Radio, Eye,
    ArrowUpRight, ArrowDownRight, AlertTriangle, Sparkles, Timer, LineChart,
    TrendingDown, Filter, SortAsc, SortDesc, Wifi, WifiOff, Volume2, LayoutGrid,
    Grid2X2, Grid3X3, Rows, Percent, Scale, Calculator, Award, Crown, Shield,
    Layers, Cpu, BarChart3, CircleDot, Gauge, Play, Pause, Power, Signal,
    GitMerge, Network, Search
} from 'lucide-react'
import { useBetSlipStore } from '@/store/betSlipStore'
import { toast } from 'sonner'
import {
    useStreamingPrediction,
    useLiveGameStream,
    useOddsMovementStream
} from '@/hooks/useStreamingPrediction'

// Constants
const LIVE_GAME_REFRESH_MS = 5000
const UPCOMING_REFRESH_MS = 30000
const HIGH_EV_THRESHOLD = 20
const ELITE_CONFIDENCE = 80
const STRONG_CONFIDENCE = 65

// Kelly Criterion calculation
function calculateKellyBet(probability: number, odds: number, bankroll: number = 1000): { fraction: number; suggestedBet: number; tier: string } {
    const decimalOdds = odds > 0 ? (odds / 100) + 1 : (100 / Math.abs(odds)) + 1
    const b = decimalOdds - 1
    const p = probability
    const q = 1 - p
    const kelly = (b * p - q) / b
    const fraction = Math.max(0, Math.min(0.25, kelly * 0.25)) // Quarter Kelly, capped at 25%
    const suggestedBet = Math.round(bankroll * fraction)

    let tier = 'pass'
    if (fraction >= 0.05) tier = 'strong'
    if (fraction >= 0.10) tier = 'elite'
    if (fraction < 0.02) tier = 'pass'

    return { fraction, suggestedBet, tier }
}

// Confidence tier helper
function getConfidenceTier(confidence: number): { tier: string; color: string; bg: string; icon: typeof Crown } {
    if (confidence >= ELITE_CONFIDENCE) return { tier: 'Elite', color: 'text-amber-400', bg: 'bg-amber-500/20', icon: Crown }
    if (confidence >= STRONG_CONFIDENCE) return { tier: 'Strong', color: 'text-emerald-400', bg: 'bg-emerald-500/20', icon: Shield }
    return { tier: 'Moderate', color: 'text-blue-400', bg: 'bg-blue-500/20', icon: Target }
}

// Edge calculation
function calculateEdge(modelProb: number, impliedProb: number): number {
    return ((modelProb - impliedProb) / impliedProb) * 100
}

export default function SportsBettingHub() {
    const [selectedLeague, setSelectedLeague] = useState('All')
    const [betSlipOpen, setBetSlipOpen] = useState(false)
    const [showTelegramSetup, setShowTelegramSetup] = useState(false)
    const [telegramChatId, setTelegramChatId] = useState(() =>
        localStorage.getItem('telegram_chat_id') || ''
    )
    const [autoAlerts, setAutoAlerts] = useState(() =>
        localStorage.getItem('auto_alerts') === 'true'
    )
    const [selectedGameForStream, setSelectedGameForStream] = useState<string | null>(null)
    const [showAIPanel, setShowAIPanel] = useState(false)
    const [showAlertsPanel, setShowAlertsPanel] = useState(false)
    const [sortBy, setSortBy] = useState<'time' | 'ev' | 'confidence' | 'kelly'>('ev')
    const [sortDesc, setSortDesc] = useState(true)
    const [alertHistory, setAlertHistory] = useState<Alert[]>([])
    const [tilesPerRow, setTilesPerRow] = useState<1 | 2 | 3 | 4>(() => {
        const saved = localStorage.getItem('sports_tiles_per_row')
        return (saved ? parseInt(saved) : 2) as 1 | 2 | 3 | 4
    })
    const [bankroll, setBankroll] = useState(() => {
        const saved = localStorage.getItem('sports_bankroll')
        return saved ? parseFloat(saved) : 1000
    })
    const [autoSync, setAutoSync] = useState(() => localStorage.getItem('sports_auto_sync') !== 'false')
    const [smartFilter, setSmartFilter] = useState<'all' | 'live' | 'high_ev' | 'elite' | 'strong_kelly'>('all')
    const [teamSearchQuery, setTeamSearchQuery] = useState('')
    const [showTeamDropdown, setShowTeamDropdown] = useState(false)
    const [showOddsMovement, setShowOddsMovement] = useState(false)
    const [oddsAlerts, setOddsAlerts] = useState<OddsAlert[]>([])
    const lastSyncRef = useRef<Date>(new Date())

    const queryClient = useQueryClient()
    const { legs, addLeg, removeLeg, stakeAmount, setStakeAmount, calculatePotentialPayout, clearSlip } = useBetSlipStore()

    // SSE Streaming
    const streamingPrediction = useStreamingPrediction({
        onPrediction: (pred) => console.log('Prediction received:', pred),
        onRecommendation: (rec) => {
            if (rec.action === 'STRONG BET' && autoAlerts && telegramChatId) {
                sendTelegramMutation.mutate(
                    `🎯 STRONG BET Alert!\n${rec.side}\nEdge: +${rec.edge.toFixed(1)}%\nSuggested: ${rec.suggestedBetSize}`
                )
            }
        },
        onError: (err) => toast.error(`AI Error: ${err}`)
    })

    // Fetch data using v2 endpoints which have proper async handling
    const { data: allSportsData, isLoading: dataLoading, refetch: refetchAll, dataUpdatedAt } = useQuery({
        queryKey: ['sports-all-data'],
        queryFn: async () => {
            lastSyncRef.current = new Date()
            // Fetch from multiple v2 endpoints in parallel for reliability
            const [liveRes, upcomingRes, betsRes] = await Promise.all([
                axiosInstance.get('/sports/v2/games/live?sports=NFL,NBA,NCAAF,NCAAB').catch(() => ({ data: [] })),
                axiosInstance.get('/sports/v2/games/upcoming?sports=NFL,NBA,NCAAF,NCAAB&limit=50').catch(() => ({ data: [] })),
                axiosInstance.get('/sports/v2/best-bets?min_edge=1.0&limit=20').catch(() => ({ data: { opportunities: [] } }))
            ])
            // Normalize field names from API (snake_case) to frontend (camelCase)
            const normalizeGame = (g: any) => ({
                ...g,
                isLive: g.is_live || false,
                gameTime: g.game_time,
                homeTeam: g.home_team,
                awayTeam: g.away_team,
                homeScore: g.home_score || 0,
                awayScore: g.away_score || 0,
                gameStatus: g.game_status,
                timeRemaining: g.time_remaining,
            })
            return {
                live_games: (liveRes.data || []).map(normalizeGame),
                upcoming_games: (upcomingRes.data || []).map(normalizeGame),
                best_bets: betsRes.data.opportunities || [],
                summary: {
                    total_live: liveRes.data?.length || 0,
                    total_upcoming: upcomingRes.data?.length || 0,
                    total_best_bets: betsRes.data.opportunities?.length || 0
                }
            }
        },
        refetchInterval: autoSync ? LIVE_GAME_REFRESH_MS : false,
        staleTime: 3000
    })

    // Extract data from combined response
    const liveGames = allSportsData?.live_games || []
    const upcomingGames = allSportsData?.upcoming_games || []
    const bestBetsData = { opportunities: allSportsData?.best_bets || [] }
    const dataSummary = allSportsData?.summary || {}

    // Backwards-compatible loading states
    const liveLoading = dataLoading
    const upcomingLoading = dataLoading

    // Unified refetch function
    const refetchLive = refetchAll
    const refetchUpcoming = refetchAll
    const refetchBets = refetchAll

    // Connection status based on polling state (WebSocket endpoint not available)
    const wsConnected = !liveLoading && !upcomingLoading
    const wsReconnecting = liveLoading || upcomingLoading

    const bestBets = bestBetsData?.opportunities || []

    // Create a lookup map from best-bets to enrich game data
    const bestBetsMap = useMemo(() => {
        const map = new Map<string, BestBet>()
        bestBets.forEach((bet: BestBet) => {
            // Extract game ID from bet ID (e.g., "NFL_401772892" -> "401772892")
            const gameId = bet.id.split('_')[1]
            if (gameId) map.set(gameId, bet)
        })
        return map
    }, [bestBets])

    // Sync odds mutation
    const syncOddsMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/sports/sync-odds')
            return data
        },
        onSuccess: (data) => {
            toast.success(`Synced ${data.synced || 0} odds from Kalshi`)
            refetchLive()
            refetchUpcoming()
            refetchBets()
        },
        onError: () => toast.error('Failed to sync odds')
    })

    // Telegram
    const sendTelegramMutation = useMutation({
        mutationFn: async (message: string) => {
            const { data } = await axiosInstance.post('/notifications/telegram', {
                chat_id: telegramChatId,
                message
            })
            return data
        },
        onSuccess: () => toast.success('Alert sent to Telegram'),
        onError: () => toast.error('Failed to send Telegram alert')
    })

    const addAlert = useCallback((alert: Omit<Alert, 'timestamp' | 'id'>) => {
        const newAlert: Alert = { ...alert, id: Date.now().toString(), timestamp: new Date() }
        setAlertHistory(prev => [newAlert, ...prev.slice(0, 49)])
        if (autoAlerts && telegramChatId) {
            sendTelegramMutation.mutate(newAlert.message)
        }
    }, [autoAlerts, telegramChatId, sendTelegramMutation])

    const isLoading = liveLoading || upcomingLoading

    const leagues = [
        { id: 'All', icon: Trophy, label: 'All' },
        { id: 'NFL', icon: Trophy, label: 'NFL' },
        { id: 'NBA', icon: Trophy, label: 'NBA' },
        { id: 'NCAAF', icon: Medal, label: 'NCAAF' },
        { id: 'NCAAB', icon: Medal, label: 'NCAAB' },
    ]

    // Enrich games with best-bets data and sort
    const allGames = useMemo(() => {
        const games = [
            ...(liveGames || []).map((g: Game) => ({ ...g, isLive: true })),
            ...(upcomingGames || []).map((g: Game) => ({ ...g, isLive: false }))
        ]

        // Enrich with best-bets data
        const enrichedGames = games.map((game: Game) => {
            const bestBet = bestBetsMap.get(game.id)
            if (bestBet) {
                return {
                    ...game,
                    enrichedData: {
                        ev_percentage: bestBet.ev_percentage,
                        model_probability: bestBet.model_probability,
                        implied_probability: bestBet.implied_probability,
                        overall_score: bestBet.overall_score,
                        bet_type: bestBet.bet_type,
                        selection: bestBet.selection
                    }
                }
            }
            return game
        })

        // Sort
        return enrichedGames.sort((a, b) => {
            const multiplier = sortDesc ? -1 : 1
            switch (sortBy) {
                case 'ev':
                    const evA = a.enrichedData?.ev_percentage || a.ai_prediction?.ev || 0
                    const evB = b.enrichedData?.ev_percentage || b.ai_prediction?.ev || 0
                    return (evB - evA) * multiplier
                case 'confidence':
                    const confA = a.ai_prediction?.confidence || 0
                    const confB = b.ai_prediction?.confidence || 0
                    return (confB - confA) * multiplier
                case 'kelly':
                    // Normalize probability: handle both 0-1 and 0-100 formats
                    const rawProbA = a.ai_prediction?.probability ?? 50
                    const rawProbB = b.ai_prediction?.probability ?? 50
                    const probA = a.enrichedData?.model_probability || (rawProbA > 1 ? rawProbA / 100 : rawProbA)
                    const probB = b.enrichedData?.model_probability || (rawProbB > 1 ? rawProbB / 100 : rawProbB)
                    const kellyA = calculateKellyBet(probA, -110, bankroll).fraction
                    const kellyB = calculateKellyBet(probB, -110, bankroll).fraction
                    return (kellyB - kellyA) * multiplier
                case 'time':
                default:
                    if (a.isLive && !b.isLive) return -1
                    if (!a.isLive && b.isLive) return 1
                    return 0
            }
        })
    }, [liveGames, upcomingGames, bestBetsMap, sortBy, sortDesc, bankroll])

    const filteredGames = useMemo(() => {
        let games = selectedLeague === 'All'
            ? allGames
            : allGames.filter((g: Game) => g.league === selectedLeague)

        // Apply smart filter
        if (smartFilter !== 'all') {
            games = games.filter((g: Game) => {
                const ev = g.enrichedData?.ev_percentage || g.ai_prediction?.ev || 0
                const confidence = g.ai_prediction?.confidence || 0
                // Normalize probability: handle both 0-1 and 0-100 formats
                const rawFilterProb = g.ai_prediction?.probability ?? 50
                const modelProb = g.enrichedData?.model_probability || (rawFilterProb > 1 ? rawFilterProb / 100 : rawFilterProb)
                const kelly = calculateKellyBet(modelProb, -110, bankroll)

                switch (smartFilter) {
                    case 'live':
                        return g.isLive === true
                    case 'high_ev':
                        return ev >= HIGH_EV_THRESHOLD
                    case 'elite':
                        return confidence >= ELITE_CONFIDENCE
                    case 'strong_kelly':
                        return kelly.tier === 'elite' || kelly.tier === 'strong'
                    default:
                        return true
                }
            })
        }

        // Apply team search filter
        if (teamSearchQuery.trim()) {
            const query = teamSearchQuery.toLowerCase().trim()
            games = games.filter((g: Game) => {
                const homeTeam = (g.homeTeam || g.home_team || '').toLowerCase()
                const awayTeam = (g.awayTeam || g.away_team || '').toLowerCase()
                return homeTeam.includes(query) || awayTeam.includes(query)
            })
        }

        // Always sort live games to the top
        games = games.sort((a: Game, b: Game) => {
            if (a.isLive && !b.isLive) return -1
            if (!a.isLive && b.isLive) return 1
            return 0
        })

        return games
    }, [allGames, selectedLeague, smartFilter, bankroll, teamSearchQuery])

    // Stats
    const stats = useMemo(() => {
        const highEVCount = allGames.filter(g => {
            const ev = g.enrichedData?.ev_percentage || g.ai_prediction?.ev || 0
            return ev >= HIGH_EV_THRESHOLD
        }).length
        const eliteCount = allGames.filter(g => (g.ai_prediction?.confidence || 0) >= ELITE_CONFIDENCE).length
        const liveCount = allGames.filter(g => g.isLive).length
        const avgEV = allGames.length > 0
            ? allGames.reduce((sum, g) => sum + (g.enrichedData?.ev_percentage || g.ai_prediction?.ev || 0), 0) / allGames.length
            : 0
        return { highEVCount, eliteCount, liveCount, total: allGames.length, avgEV }
    }, [allGames])

    // Unique teams for autosearch dropdown
    const uniqueTeams = useMemo(() => {
        const teamSet = new Set<string>()
        allGames.forEach((g: Game) => {
            const home = g.homeTeam || g.home_team || ''
            const away = g.awayTeam || g.away_team || ''
            if (home) teamSet.add(home)
            if (away) teamSet.add(away)
        })
        return Array.from(teamSet).sort()
    }, [allGames])

    // Filtered teams for dropdown
    const filteredTeams = useMemo(() => {
        if (!teamSearchQuery.trim()) return uniqueTeams.slice(0, 10)
        const query = teamSearchQuery.toLowerCase().trim()
        return uniqueTeams.filter(team =>
            team.toLowerCase().includes(query)
        ).slice(0, 10)
    }, [uniqueTeams, teamSearchQuery])

    // Memoized callbacks for GameCard performance optimization
    const handleAddToBetSlip = useCallback((game: Game, betType: string, selection: string, odds: number) => {
        addLeg({
            gameId: game.id,
            sport: game.league,
            homeTeam: game.home_team,
            awayTeam: game.away_team,
            betType: betType.toLowerCase().includes('ml') ? 'moneyline' : betType.toLowerCase().includes('spread') ? 'spread' : 'total_over',
            selection: selection.toLowerCase().includes(game.home_team.toLowerCase()) ? 'home' : 'away',
            line: game.odds?.spread_home,
            odds: odds,
            gameTime: new Date(game.game_time),
            aiProbability: game.ai_prediction?.probability,
            aiEdge: (game.enrichedData?.ev_percentage || game.ai_prediction?.ev || 0) / 100,
            aiConfidence: (game.ai_prediction?.confidence || 0) > STRONG_CONFIDENCE ? 'high' : 'medium'
        })
        setBetSlipOpen(true)
        toast.success(`Added ${selection} to bet slip`)
    }, [addLeg])

    const handleRefresh = useCallback(() => {
        syncOddsMutation.mutate()
        refetchAll()
    }, [syncOddsMutation, refetchAll])

    const startAIAnalysis = useCallback((gameId: string, sport: string) => {
        setSelectedGameForStream(gameId)
        setShowAIPanel(true)
        streamingPrediction.startPrediction(gameId, sport, true)
    }, [streamingPrediction])

    // Memoized telegram sender for GameCard
    const handleSendTelegram = useCallback((msg: string) => {
        if (telegramChatId) sendTelegramMutation.mutate(msg)
    }, [telegramChatId, sendTelegramMutation])

    const potentialWin = calculatePotentialPayout()
    const timeSinceUpdate = useMemo(() => {
        const diff = Date.now() - (dataUpdatedAt || Date.now())
        return Math.floor(diff / 1000)
    }, [dataUpdatedAt])

    return (
        <div className="flex h-[calc(100vh-6rem)]">
            <div className={`flex-1 flex flex-col overflow-hidden transition-all duration-300 ${
                betSlipOpen ? 'mr-80' : showAIPanel ? 'mr-96' : 'mr-0'
            }`}>
                {/* Header */}
                <header className="flex items-center justify-between mb-4 px-1">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 via-teal-500 to-cyan-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                            <Trophy className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-lg font-bold text-white flex items-center gap-2">
                                Sports Hub
                                <span className="flex items-center gap-1 text-xs font-normal text-emerald-400">
                                    <Radio className="w-3 h-3 animate-pulse" />
                                    AI Powered
                                </span>
                                {/* Connection Status */}
                                <motion.span
                                    className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full ${
                                        wsConnected
                                            ? 'bg-emerald-500/20 text-emerald-400'
                                            : wsReconnecting
                                            ? 'bg-amber-500/20 text-amber-400'
                                            : 'bg-red-500/20 text-red-400'
                                    }`}
                                    animate={{ opacity: wsReconnecting ? [1, 0.5, 1] : 1 }}
                                    transition={{ repeat: wsReconnecting ? Infinity : 0, duration: 1 }}
                                >
                                    {wsConnected ? <Wifi className="w-2.5 h-2.5" /> : wsReconnecting ? <Signal className="w-2.5 h-2.5" /> : <WifiOff className="w-2.5 h-2.5" />}
                                    {wsConnected ? 'Live' : wsReconnecting ? 'Connecting...' : 'Offline'}
                                </motion.span>
                            </h1>
                            <p className="text-xs text-slate-400 flex items-center gap-2">
                                Kelly Criterion • EV Analysis • Ensemble AI
                                <span className="text-emerald-400/70">• {timeSinceUpdate}s ago</span>
                                {oddsAlerts.length > 0 && (
                                    <span className="flex items-center gap-1 text-amber-400 bg-amber-500/15 px-1.5 py-0.5 rounded">
                                        <BarChart3 className="w-2.5 h-2.5" />
                                        {oddsAlerts.length} moves
                                    </span>
                                )}
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Live Stats */}
                        <motion.div
                            className="flex items-center gap-3 px-4 py-2 bg-slate-800/70 rounded-xl border border-slate-700/50 text-xs backdrop-blur-sm"
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="flex items-center gap-1.5" title="Total Games">
                                <Activity className="w-3.5 h-3.5 text-blue-400" />
                                <span className="text-white font-bold">{stats.total}</span>
                            </div>
                            <span className="w-px h-5 bg-slate-600"></span>
                            <div className="flex items-center gap-1.5" title="Live Games">
                                <span className="relative">
                                    <span className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
                                    <span className="relative w-2 h-2 rounded-full bg-emerald-500 block"></span>
                                </span>
                                <span className="text-emerald-400 font-bold">{stats.liveCount}</span>
                            </div>
                            <span className="w-px h-5 bg-slate-600"></span>
                            <div className="flex items-center gap-1.5" title="High EV Opportunities">
                                <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
                                <span className="text-amber-400 font-bold">{stats.highEVCount}</span>
                                <span className="text-slate-500">+EV</span>
                            </div>
                            <span className="w-px h-5 bg-slate-600"></span>
                            <div className="flex items-center gap-1.5" title="Elite Confidence">
                                <Crown className="w-3.5 h-3.5 text-purple-400" />
                                <span className="text-purple-400 font-bold">{stats.eliteCount}</span>
                            </div>
                            <span className="w-px h-5 bg-slate-600"></span>
                            <div className="flex items-center gap-1.5" title="Average EV">
                                <Percent className="w-3.5 h-3.5 text-cyan-400" />
                                <span className="text-cyan-400 font-bold">+{stats.avgEV.toFixed(0)}%</span>
                            </div>
                        </motion.div>

                        {/* Bankroll Setting */}
                        <div className="relative group">
                            <button className="p-2 bg-slate-800/60 rounded-lg border border-slate-700/50 text-slate-400 hover:text-white transition-colors">
                                <Calculator className="w-4 h-4" />
                            </button>
                            <div className="absolute right-0 top-full mt-2 w-48 p-3 bg-slate-900 rounded-lg border border-slate-700 shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                                <label className="text-xs text-slate-400 mb-1 block">Bankroll</label>
                                <div className="flex items-center gap-2">
                                    <span className="text-slate-500">$</span>
                                    <input
                                        type="number"
                                        value={bankroll}
                                        onChange={(e) => {
                                            const val = parseFloat(e.target.value) || 1000
                                            setBankroll(val)
                                            localStorage.setItem('sports_bankroll', val.toString())
                                        }}
                                        className="input-field py-1 text-sm flex-1"
                                    />
                                </div>
                                <p className="text-[10px] text-slate-500 mt-1">For Kelly bet sizing</p>
                            </div>
                        </div>

                        <motion.button
                            onClick={() => setShowAlertsPanel(!showAlertsPanel)}
                            className={`relative p-2 rounded-lg border transition-all ${
                                showAlertsPanel
                                    ? 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                                    : 'bg-slate-800/60 border-slate-700/50 text-slate-400 hover:text-white'
                            }`}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Bell className="w-4 h-4" />
                            {alertHistory.length > 0 && (
                                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-amber-500 text-[10px] font-bold text-black flex items-center justify-center">
                                    {alertHistory.length}
                                </span>
                            )}
                        </motion.button>

                        <motion.button
                            onClick={() => setShowAIPanel(!showAIPanel)}
                            className={`p-2 rounded-lg border transition-all ${
                                showAIPanel
                                    ? 'bg-primary/20 border-primary/50 text-primary'
                                    : 'bg-slate-800/60 border-slate-700/50 text-slate-400 hover:text-white'
                            }`}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Brain className="w-4 h-4" />
                        </motion.button>

                        <button
                            onClick={() => setShowTelegramSetup(!showTelegramSetup)}
                            className={`p-2 rounded-lg border transition-colors ${
                                telegramChatId
                                    ? 'bg-blue-500/20 border-blue-500/50 text-blue-400'
                                    : 'bg-slate-800/60 border-slate-700/50 text-slate-400 hover:text-white'
                            }`}
                        >
                            <Send className="w-4 h-4" />
                        </button>

                        {/* Auto-Sync Toggle */}
                        <motion.button
                            onClick={() => {
                                const newVal = !autoSync
                                setAutoSync(newVal)
                                localStorage.setItem('sports_auto_sync', newVal.toString())
                                toast.success(newVal ? 'Auto-sync enabled' : 'Auto-sync disabled')
                            }}
                            className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                                autoSync
                                    ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400'
                                    : 'bg-slate-800/60 border-slate-700/50 text-slate-400'
                            }`}
                            whileTap={{ scale: 0.95 }}
                        >
                            {autoSync ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
                            Auto
                        </motion.button>

                        {/* Odds Movement Panel Toggle */}
                        <motion.button
                            onClick={() => setShowOddsMovement(!showOddsMovement)}
                            className={`relative p-2 rounded-lg border transition-all ${
                                showOddsMovement
                                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                                    : 'bg-slate-800/60 border-slate-700/50 text-slate-400 hover:text-white'
                            }`}
                            whileTap={{ scale: 0.95 }}
                        >
                            <BarChart3 className="w-4 h-4" />
                            {oddsAlerts.filter(a => a.type === 'significant').length > 0 && (
                                <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-cyan-500 animate-pulse" />
                            )}
                        </motion.button>

                        <motion.button
                            onClick={handleRefresh}
                            disabled={syncOddsMutation.isPending}
                            className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all ${
                                syncOddsMutation.isPending
                                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                                    : 'bg-slate-800/60 hover:bg-slate-700/60 border-slate-700/50 text-slate-400 hover:text-white'
                            }`}
                            whileTap={{ scale: 0.95 }}
                        >
                            <RefreshCw className={`w-4 h-4 ${syncOddsMutation.isPending ? 'animate-spin' : ''}`} />
                            {syncOddsMutation.isPending && (
                                <span className="text-xs font-medium">Syncing...</span>
                            )}
                        </motion.button>

                        <motion.button
                            onClick={() => setBetSlipOpen(!betSlipOpen)}
                            className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium text-sm transition-all ${
                                betSlipOpen
                                    ? 'bg-gradient-to-r from-primary to-purple-600 text-white shadow-lg shadow-primary/30'
                                    : 'bg-slate-800/60 text-slate-400 hover:text-white border border-slate-700/50'
                            }`}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Target className="w-4 h-4" />
                            Slip
                            {legs.length > 0 && (
                                <motion.span
                                    className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center text-xs"
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    key={legs.length}
                                >
                                    {legs.length}
                                </motion.span>
                            )}
                        </motion.button>
                    </div>
                </header>

                {/* Sync Progress Indicator */}
                <AnimatePresence>
                    {syncOddsMutation.isPending && (
                        <motion.div
                            className="mb-4 glass-card overflow-hidden"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                        >
                            <div className="p-3 flex items-center gap-3">
                                <div className="relative">
                                    <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center">
                                        <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin" />
                                    </div>
                                    <div className="absolute inset-0 rounded-full border-2 border-cyan-400/30 animate-ping" />
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-sm font-medium text-white">Syncing odds from Kalshi...</span>
                                        <span className="text-xs text-cyan-400">Live Data</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                                            initial={{ width: '0%' }}
                                            animate={{ width: ['0%', '30%', '60%', '90%', '100%'] }}
                                            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Telegram Setup */}
                <AnimatePresence>
                    {showTelegramSetup && (
                        <motion.div
                            className="mb-4 glass-card p-4 bg-blue-500/5 border-blue-500/20"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                    <Send className="w-4 h-4 text-blue-400" />
                                    <span className="font-medium text-white text-sm">Telegram Alerts</span>
                                </div>
                                <button onClick={() => setShowTelegramSetup(false)} className="text-slate-400 hover:text-white">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={telegramChatId}
                                    onChange={(e) => setTelegramChatId(e.target.value)}
                                    placeholder="Enter Telegram Chat ID"
                                    className="input-field flex-1 py-2 text-sm"
                                />
                                <button
                                    onClick={() => {
                                        localStorage.setItem('telegram_chat_id', telegramChatId)
                                        toast.success('Telegram Chat ID saved')
                                    }}
                                    className="btn-primary px-4 py-2 text-sm"
                                >
                                    Save
                                </button>
                            </div>
                            <label className="flex items-center gap-2 cursor-pointer mt-3">
                                <input
                                    type="checkbox"
                                    checked={autoAlerts}
                                    onChange={(e) => {
                                        setAutoAlerts(e.target.checked)
                                        localStorage.setItem('auto_alerts', e.target.checked.toString())
                                    }}
                                    className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary focus:ring-primary"
                                />
                                <span className="text-xs text-slate-300">Auto-send high EV alerts (+{HIGH_EV_THRESHOLD}% EV)</span>
                            </label>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Alerts Panel */}
                <AnimatePresence>
                    {showAlertsPanel && (
                        <motion.div
                            className="mb-4 glass-card p-4 bg-amber-500/5 border-amber-500/20 max-h-48 overflow-y-auto"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <span className="font-medium text-white text-sm">Alert History ({alertHistory.length})</span>
                                <button onClick={() => setAlertHistory([])} className="text-xs text-slate-400 hover:text-red-400">Clear</button>
                            </div>
                            {alertHistory.length === 0 ? (
                                <p className="text-xs text-slate-500 text-center py-4">No alerts yet</p>
                            ) : (
                                <div className="space-y-2">
                                    {alertHistory.slice(0, 10).map(alert => (
                                        <div key={alert.id} className="flex items-center gap-2 p-2 bg-slate-800/50 rounded-lg text-xs">
                                            <TrendingUp className="w-3 h-3 text-emerald-400" />
                                            <span className="flex-1 text-slate-300">{alert.message}</span>
                                            <span className="text-slate-500">{alert.timestamp.toLocaleTimeString()}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Odds Movement Panel */}
                <AnimatePresence>
                    {showOddsMovement && (
                        <motion.div
                            className="mb-4 glass-card p-4 bg-cyan-500/5 border-cyan-500/20 max-h-56 overflow-y-auto"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                    <BarChart3 className="w-4 h-4 text-cyan-400" />
                                    <span className="font-medium text-white text-sm">Odds Movement Tracker</span>
                                    <span className="text-[10px] text-cyan-400 bg-cyan-500/20 px-1.5 py-0.5 rounded">
                                        {oddsAlerts.length} moves
                                    </span>
                                </div>
                                <button onClick={() => setOddsAlerts([])} className="text-xs text-slate-400 hover:text-red-400">Clear</button>
                            </div>
                            {oddsAlerts.length === 0 ? (
                                <div className="text-center py-6">
                                    <BarChart3 className="w-8 h-8 mx-auto mb-2 text-slate-600" />
                                    <p className="text-xs text-slate-500">No odds movements detected</p>
                                    <p className="text-[10px] text-slate-600 mt-1">Sharp moves will appear here in real-time</p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {oddsAlerts.slice(0, 15).map(alert => (
                                        <motion.div
                                            key={alert.id}
                                            className={`flex items-center gap-2 p-2 rounded-lg text-xs ${
                                                alert.type === 'significant'
                                                    ? 'bg-cyan-500/10 border border-cyan-500/30'
                                                    : 'bg-slate-800/50'
                                            }`}
                                            initial={{ x: -10, opacity: 0 }}
                                            animate={{ x: 0, opacity: 1 }}
                                        >
                                            {alert.direction === 'up' ? (
                                                <ArrowUpRight className={`w-3.5 h-3.5 ${alert.type === 'significant' ? 'text-cyan-400' : 'text-emerald-400'}`} />
                                            ) : (
                                                <ArrowDownRight className={`w-3.5 h-3.5 ${alert.type === 'significant' ? 'text-cyan-400' : 'text-red-400'}`} />
                                            )}
                                            <span className="flex-1 text-slate-300">{alert.message}</span>
                                            {alert.magnitude > 0 && (
                                                <span className={`font-mono font-medium ${
                                                    alert.magnitude > 5 ? 'text-cyan-400' : 'text-slate-400'
                                                }`}>
                                                    {alert.direction === 'up' ? '+' : '-'}{alert.magnitude.toFixed(1)}%
                                                </span>
                                            )}
                                            <span className="text-slate-500 text-[10px]">{alert.timestamp.toLocaleTimeString()}</span>
                                        </motion.div>
                                    ))}
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Filter Bar */}
                <div className="flex items-center justify-between gap-4 mb-4">
                    <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-hide">
                        {leagues.map(league => (
                            <motion.button
                                key={league.id}
                                onClick={() => setSelectedLeague(league.id)}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium text-xs transition-all whitespace-nowrap ${
                                    selectedLeague === league.id
                                        ? 'bg-gradient-to-r from-primary to-purple-600 text-white shadow-md shadow-primary/20'
                                        : 'bg-slate-800/60 text-slate-400 hover:bg-slate-700/60 hover:text-white border border-slate-700/50'
                                }`}
                                whileTap={{ scale: 0.95 }}
                            >
                                <league.icon className="w-3 h-3" />
                                {league.label}
                            </motion.button>
                        ))}
                    </div>

                    {/* Team Search with Autosearch */}
                    <div className="relative">
                        <div className="flex items-center gap-2 bg-slate-800/60 rounded-lg px-2.5 py-1.5 border border-slate-700/50 focus-within:border-primary/50 transition-colors">
                            <Search className="w-3.5 h-3.5 text-slate-400" />
                            <input
                                type="text"
                                placeholder="Search teams..."
                                value={teamSearchQuery}
                                onChange={(e) => {
                                    setTeamSearchQuery(e.target.value)
                                    setShowTeamDropdown(true)
                                }}
                                onFocus={() => setShowTeamDropdown(true)}
                                onBlur={() => setTimeout(() => setShowTeamDropdown(false), 200)}
                                className="bg-transparent text-xs text-white placeholder-slate-500 w-28 focus:outline-none"
                            />
                            {teamSearchQuery && (
                                <button
                                    onClick={() => {
                                        setTeamSearchQuery('')
                                        setShowTeamDropdown(false)
                                    }}
                                    className="text-slate-400 hover:text-white"
                                >
                                    <X className="w-3 h-3" />
                                </button>
                            )}
                        </div>

                        {/* Dropdown */}
                        <AnimatePresence>
                            {showTeamDropdown && filteredTeams.length > 0 && (
                                <motion.div
                                    initial={{ opacity: 0, y: -5 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -5 }}
                                    className="absolute top-full left-0 mt-1 w-48 bg-slate-800 rounded-lg border border-slate-700/50 shadow-xl z-50 max-h-60 overflow-y-auto"
                                >
                                    {filteredTeams.map((team) => (
                                        <button
                                            key={team}
                                            onClick={() => {
                                                setTeamSearchQuery(team)
                                                setShowTeamDropdown(false)
                                            }}
                                            className="w-full px-3 py-2 text-left text-xs text-slate-300 hover:bg-slate-700/50 hover:text-white transition-colors first:rounded-t-lg last:rounded-b-lg"
                                        >
                                            {team}
                                        </button>
                                    ))}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* Smart Filter */}
                        <div className="flex items-center gap-1 bg-slate-800/60 rounded-lg p-0.5 border border-slate-700/50">
                            <button
                                onClick={() => setSmartFilter('all')}
                                className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                                    smartFilter === 'all' ? 'bg-primary/30 text-primary' : 'text-slate-500 hover:text-white'
                                }`}
                            >
                                All
                            </button>
                            <button
                                onClick={() => setSmartFilter('live')}
                                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                                    smartFilter === 'live' ? 'bg-red-500/30 text-red-400' : 'text-slate-500 hover:text-white'
                                }`}
                            >
                                <Radio className="w-2.5 h-2.5 animate-pulse" />
                                Live
                            </button>
                            <button
                                onClick={() => setSmartFilter('high_ev')}
                                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                                    smartFilter === 'high_ev' ? 'bg-amber-500/30 text-amber-400' : 'text-slate-500 hover:text-white'
                                }`}
                            >
                                <Flame className="w-2.5 h-2.5" />
                                +EV
                            </button>
                            <button
                                onClick={() => setSmartFilter('elite')}
                                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                                    smartFilter === 'elite' ? 'bg-purple-500/30 text-purple-400' : 'text-slate-500 hover:text-white'
                                }`}
                            >
                                <Crown className="w-2.5 h-2.5" />
                                Elite
                            </button>
                            <button
                                onClick={() => setSmartFilter('strong_kelly')}
                                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors ${
                                    smartFilter === 'strong_kelly' ? 'bg-emerald-500/30 text-emerald-400' : 'text-slate-500 hover:text-white'
                                }`}
                            >
                                <Calculator className="w-2.5 h-2.5" />
                                Kelly
                            </button>
                        </div>

                        <div className="w-px h-5 bg-slate-700"></div>

                        <div className="flex items-center gap-1 bg-slate-800/60 rounded-lg p-0.5 border border-slate-700/50">
                            {[1, 2, 3, 4].map((num) => (
                                <button
                                    key={num}
                                    onClick={() => {
                                        setTilesPerRow(num as 1 | 2 | 3 | 4)
                                        localStorage.setItem('sports_tiles_per_row', num.toString())
                                    }}
                                    className={`p-1.5 rounded transition-colors ${
                                        tilesPerRow === num ? 'bg-primary/30 text-primary' : 'text-slate-500 hover:text-white'
                                    }`}
                                >
                                    {num === 1 ? <Rows className="w-3 h-3" /> :
                                     num === 2 ? <Grid2X2 className="w-3 h-3" /> :
                                     num === 3 ? <Grid3X3 className="w-3 h-3" /> :
                                     <LayoutGrid className="w-3 h-3" />}
                                </button>
                            ))}
                        </div>

                        <span className="text-xs text-slate-500">Sort:</span>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                            className="text-xs bg-slate-800/60 border border-slate-700/50 rounded-lg px-2 py-1 text-slate-300"
                        >
                            <option value="ev">Expected Value</option>
                            <option value="confidence">AI Confidence</option>
                            <option value="kelly">Kelly Bet Size</option>
                            <option value="time">Game Time</option>
                        </select>
                        <button
                            onClick={() => setSortDesc(!sortDesc)}
                            className="p-1 bg-slate-800/60 rounded border border-slate-700/50 text-slate-400 hover:text-white"
                        >
                            {sortDesc ? <SortDesc className="w-3 h-3" /> : <SortAsc className="w-3 h-3" />}
                        </button>
                    </div>
                </div>

                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                    {/* Top Picks Carousel */}
                    {bestBets.length > 0 && (
                        <motion.div
                            className="glass-card p-4 bg-gradient-to-br from-amber-500/5 via-orange-500/5 to-red-500/5 border-amber-500/20"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="flex items-center gap-2 mb-3">
                                <div className="p-1.5 bg-amber-500/20 rounded-lg">
                                    <Crown className="w-4 h-4 text-amber-400" />
                                </div>
                                <span className="font-semibold text-white text-sm">AI Top Picks</span>
                                <span className="text-xs text-slate-400">• Best +EV Opportunities</span>
                                <span className="ml-auto text-[10px] text-emerald-400 bg-emerald-500/15 px-2 py-0.5 rounded flex items-center gap-1">
                                    <Percent className="w-2.5 h-2.5" />
                                    Avg +{dataSummary?.avg_ev?.toFixed(0) || 0}% EV
                                </span>
                            </div>
                            <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
                                {bestBets.slice(0, 8).map((bet: BestBet, idx: number) => {
                                    const kelly = calculateKellyBet(bet.model_probability, -110, bankroll)
                                    const confidenceTier = getConfidenceTier(bet.confidence)
                                    const TierIcon = confidenceTier.icon

                                    return (
                                        <motion.div
                                            key={bet.id || idx}
                                            onClick={() => {
                                                addLeg({
                                                    gameId: bet.id,
                                                    sport: bet.sport,
                                                    homeTeam: bet.home_team || '',
                                                    awayTeam: bet.away_team || '',
                                                    betType: 'spread',
                                                    selection: 'home',
                                                    odds: -110,
                                                    gameTime: new Date(bet.game_time || Date.now()),
                                                    aiProbability: bet.model_probability,
                                                    aiEdge: bet.ev_percentage / 100,
                                                    aiConfidence: bet.confidence > STRONG_CONFIDENCE ? 'high' : 'medium',
                                                })
                                                setBetSlipOpen(true)
                                                toast.success('Added AI pick to slip')
                                            }}
                                            className={`flex-shrink-0 glass-card p-3 min-w-[240px] cursor-pointer transition-all hover:scale-[1.02] ${
                                                kelly.tier === 'elite'
                                                    ? 'border-amber-500/40 bg-amber-500/5'
                                                    : kelly.tier === 'strong'
                                                    ? 'border-emerald-500/40 bg-emerald-500/5'
                                                    : 'border-slate-700/50'
                                            }`}
                                            whileTap={{ scale: 0.98 }}
                                        >
                                            {/* Header */}
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-[10px] font-bold text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                                                    {bet.sport}
                                                </span>
                                                <div className={`flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded ${confidenceTier.bg} ${confidenceTier.color}`}>
                                                    <TierIcon className="w-2.5 h-2.5" />
                                                    {confidenceTier.tier}
                                                </div>
                                            </div>

                                            {/* Matchup */}
                                            <div className="text-xs text-slate-400 mb-1 truncate">{bet.event_name}</div>
                                            <div className="text-sm font-semibold text-primary mb-2">{bet.selection}</div>

                                            {/* Stats Grid */}
                                            <div className="grid grid-cols-3 gap-2 mb-2">
                                                <div className="text-center p-1.5 bg-slate-800/50 rounded">
                                                    <div className="text-[10px] text-slate-500">EV</div>
                                                    <div className="text-xs font-bold text-emerald-400">+{bet.ev_percentage?.toFixed(0)}%</div>
                                                </div>
                                                <div className="text-center p-1.5 bg-slate-800/50 rounded">
                                                    <div className="text-[10px] text-slate-500">Model</div>
                                                    <div className="text-xs font-bold text-blue-400">{(bet.model_probability * 100).toFixed(0)}%</div>
                                                </div>
                                                <div className="text-center p-1.5 bg-slate-800/50 rounded">
                                                    <div className="text-[10px] text-slate-500">Kelly</div>
                                                    <div className={`text-xs font-bold ${
                                                        kelly.tier === 'elite' ? 'text-amber-400' :
                                                        kelly.tier === 'strong' ? 'text-emerald-400' : 'text-slate-400'
                                                    }`}>
                                                        ${kelly.suggestedBet}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Probability Bar */}
                                            <div className="relative h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                <div
                                                    className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                                                    style={{ width: `${bet.model_probability * 100}%` }}
                                                />
                                                <div
                                                    className="absolute top-0 h-full w-0.5 bg-red-500"
                                                    style={{ left: `${bet.implied_probability * 100}%` }}
                                                    title={`Market: ${(bet.implied_probability * 100).toFixed(0)}%`}
                                                />
                                            </div>
                                            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                                                <span>Model: {(bet.model_probability * 100).toFixed(0)}%</span>
                                                <span>Market: {(bet.implied_probability * 100).toFixed(0)}%</span>
                                            </div>
                                        </motion.div>
                                    )
                                })}
                            </div>
                        </motion.div>
                    )}

                    {/* Skeleton Loading */}
                    {isLoading && (
                        <div className={`grid gap-3 ${
                            tilesPerRow === 1 ? 'grid-cols-1' :
                            tilesPerRow === 2 ? 'grid-cols-1 md:grid-cols-2' :
                            tilesPerRow === 3 ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' :
                            'grid-cols-1 md:grid-cols-2 xl:grid-cols-4'
                        }`}>
                            {Array.from({ length: 6 }).map((_, i) => (
                                <div key={i} className="glass-card overflow-hidden animate-pulse">
                                    {/* Header skeleton */}
                                    <div className="px-3 py-2 flex justify-between items-center border-b border-slate-700/50 bg-slate-900/30">
                                        <div className="flex items-center gap-2">
                                            <div className="w-10 h-4 bg-slate-700 rounded" />
                                            <div className="w-8 h-4 bg-slate-700 rounded" />
                                        </div>
                                        <div className="w-16 h-4 bg-slate-700 rounded" />
                                    </div>
                                    {/* Body skeleton */}
                                    <div className="p-3">
                                        <div className="flex justify-between items-center mb-4">
                                            <div className="space-y-2 flex-1">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-7 h-7 bg-slate-700 rounded-lg" />
                                                    <div className="w-32 h-4 bg-slate-700 rounded" />
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-7 h-7 bg-slate-700 rounded-lg" />
                                                    <div className="w-28 h-4 bg-slate-700 rounded" />
                                                </div>
                                            </div>
                                        </div>
                                        {/* Odds skeleton */}
                                        <div className="grid grid-cols-3 gap-2 mb-3">
                                            <div className="h-12 bg-slate-800/50 rounded-lg" />
                                            <div className="h-12 bg-slate-800/50 rounded-lg" />
                                            <div className="h-12 bg-slate-800/50 rounded-lg" />
                                        </div>
                                        {/* AI prediction skeleton */}
                                        <div className="h-24 bg-gradient-to-r from-slate-800/30 to-slate-800/50 rounded-lg" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Games Grid */}
                    {!isLoading && (
                        <div className={`grid gap-3 ${
                            tilesPerRow === 1 ? 'grid-cols-1' :
                            tilesPerRow === 2 ? 'grid-cols-1 md:grid-cols-2' :
                            tilesPerRow === 3 ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' :
                            'grid-cols-1 md:grid-cols-2 xl:grid-cols-4'
                        }`}>
                            <AnimatePresence mode="popLayout">
                                {filteredGames.length > 0 ? (
                                    filteredGames.map((game: Game, index: number) => (
                                        <motion.div
                                            key={game.id}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, scale: 0.95 }}
                                            transition={{ delay: index * 0.03 }}
                                        >
                                            <GameCard
                                                game={game}
                                                bankroll={bankroll}
                                                onAddBet={handleAddToBetSlip}
                                                onSendTelegram={handleSendTelegram}
                                                onStartAIAnalysis={() => startAIAnalysis(game.id, game.league)}
                                            />
                                        </motion.div>
                                    ))
                                ) : (
                                    <div className="col-span-full glass-card p-8 text-center">
                                        <Trophy className="w-8 h-8 mx-auto mb-3 text-slate-500" />
                                        <h3 className="text-sm font-semibold text-white mb-1">No Games Found</h3>
                                        <p className="text-xs text-slate-400">No games for {selectedLeague}</p>
                                    </div>
                                )}
                            </AnimatePresence>
                        </div>
                    )}
                </div>
            </div>

            {/* AI Panel */}
            <AnimatePresence>
                {showAIPanel && !betSlipOpen && (
                    <motion.div
                        className="fixed right-0 top-0 h-full w-96 z-40"
                        initial={{ x: '100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '100%' }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                    >
                        <AIStreamingPanel
                            streamingState={streamingPrediction}
                            selectedGameId={selectedGameForStream}
                            onClose={() => setShowAIPanel(false)}
                        />
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Bet Slip */}
            <AnimatePresence>
                {betSlipOpen && (
                    <motion.div
                        className="fixed right-0 top-0 h-full w-80 z-40"
                        initial={{ x: '100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '100%' }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                    >
                        <BetSlipPanel
                            legs={legs}
                            stakeAmount={stakeAmount}
                            potentialWin={potentialWin}
                            onRemoveLeg={removeLeg}
                            onSetStake={setStakeAmount}
                            onClear={clearSlip}
                            onClose={() => setBetSlipOpen(false)}
                        />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

// Types
interface Alert {
    id: string
    type: 'high_ev' | 'sharp_move' | 'odds_change'
    key: string
    message: string
    timestamp: Date
    bet?: BestBet
}

interface OddsAlert {
    id: string
    gameId: string
    type: 'stable' | 'moderate' | 'significant'
    direction: 'up' | 'down'
    magnitude: number
    message: string
    timestamp: Date
}

interface Game {
    id: string
    home_team: string
    away_team: string
    homeTeam?: string  // Alternative camelCase format
    awayTeam?: string  // Alternative camelCase format
    home_score?: number
    away_score?: number
    league: string
    game_time: string
    status: string
    isLive?: boolean
    home_rank?: number
    away_rank?: number
    // Live game state
    quarter?: number
    period?: number
    time_remaining?: string
    possession?: string
    down_distance?: string
    yard_line?: number
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
        model_contributions?: {
            name: string
            weight: number
            prediction: string
        }[]
    }
    enrichedData?: {
        ev_percentage: number
        model_probability: number
        implied_probability: number
        overall_score: number
        bet_type: string
        selection: string
        ensemble_agreement?: number
    }
}

interface BestBet {
    id: string
    sport: string
    event_name: string
    selection: string
    ev_percentage: number
    model_probability: number
    implied_probability: number
    confidence: number
    overall_score: number
    bet_type: string
    game_time: string
    home_team?: string
    away_team?: string
}

// Helper functions
function formatGameTime(gameTime: string): string {
    const date = new Date(gameTime)
    if (!isNaN(date.getTime()) && gameTime.includes('-')) {
        return date.toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        })
    }
    return gameTime
}

function probabilityToMoneyline(probability: number): number {
    if (probability >= 0.5) {
        return Math.round(-(probability / (1 - probability)) * 100)
    }
    return Math.round(((1 - probability) / probability) * 100)
}

// Game Card Component - Memoized for performance optimization
const GameCard = memo(function GameCard({
    game,
    bankroll,
    onAddBet,
    onSendTelegram,
    onStartAIAnalysis
}: {
    game: Game
    bankroll: number
    onAddBet: (game: Game, betType: string, selection: string, odds: number) => void
    onSendTelegram?: (message: string) => void
    onStartAIAnalysis?: () => void
}) {
    const [expanded, setExpanded] = useState(false)

    const ev = game.enrichedData?.ev_percentage || game.ai_prediction?.ev || 0
    // Normalize probability: handle both 0-1 and 0-100 formats
    const rawProb = game.ai_prediction?.probability ?? 50
    const normalizedAIProb = rawProb > 1 ? rawProb / 100 : rawProb
    const modelProb = game.enrichedData?.model_probability || normalizedAIProb
    const confidence = game.ai_prediction?.confidence || 0

    const hasHighEV = ev >= HIGH_EV_THRESHOLD
    const confidenceTier = getConfidenceTier(confidence)
    const TierIcon = confidenceTier.icon
    const kelly = calculateKellyBet(modelProb, -110, bankroll)

    const aiSpread = game.ai_prediction?.spread
    const aiMoneyline = modelProb ? probabilityToMoneyline(modelProb) : null
    const displaySpread = game.odds?.spread_home != null ? parseFloat(String(game.odds.spread_home)) : aiSpread
    const displayMoneyline = game.odds?.moneyline_home != null ? Number(game.odds.moneyline_home) : aiMoneyline
    const displayTotal = game.odds?.over_under != null ? parseFloat(String(game.odds.over_under)) : undefined
    const isAISpread = game.odds?.spread_home == null && aiSpread != null
    const isAIMoneyline = game.odds?.moneyline_home == null && aiMoneyline != null

    return (
        <motion.div
            className={`glass-card overflow-hidden transition-all ${
                game.isLive
                    ? 'border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                    : hasHighEV
                    ? 'border-amber-500/30 hover:border-amber-500/50'
                    : kelly.tier === 'strong' || kelly.tier === 'elite'
                    ? 'border-emerald-500/30 hover:border-emerald-500/50'
                    : 'hover:border-primary/40'
            }`}
            whileHover={{ scale: 1.005 }}
            layout
        >
            {/* Header */}
            <div className="px-3 py-2 flex justify-between items-center border-b border-slate-700/50 bg-slate-900/30">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-slate-400 uppercase">{game.league}</span>
                    {/* Ensemble Model Badge */}
                    {game.ai_prediction && (
                        <span className="flex items-center gap-0.5 text-[10px] font-medium text-cyan-400 bg-cyan-500/15 px-1.5 py-0.5 rounded" title="AI Ensemble Model">
                            <GitMerge className="w-2.5 h-2.5" />
                            {game.enrichedData?.ensemble_agreement ? `${(game.enrichedData.ensemble_agreement * 100).toFixed(0)}%` : 'AI'}
                        </span>
                    )}
                    {hasHighEV && (
                        <span className="flex items-center gap-0.5 text-[10px] font-bold text-amber-400 bg-amber-500/20 px-1.5 py-0.5 rounded">
                            <Flame className="w-2.5 h-2.5" />
                            +{ev.toFixed(0)}%
                        </span>
                    )}
                    {confidence >= STRONG_CONFIDENCE && (
                        <span className={`flex items-center gap-0.5 text-[10px] font-bold px-1.5 py-0.5 rounded ${confidenceTier.bg} ${confidenceTier.color}`}>
                            <TierIcon className="w-2.5 h-2.5" />
                            {confidenceTier.tier}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    {game.isLive ? (
                        <div className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/15 rounded text-[10px] font-medium text-emerald-400">
                            <span className="relative">
                                <span className="absolute inset-0 w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping"></span>
                                <span className="relative w-1.5 h-1.5 rounded-full bg-emerald-500 block"></span>
                            </span>
                            LIVE
                        </div>
                    ) : (
                        <span className="text-[10px] text-slate-400 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatGameTime(game.game_time)}
                        </span>
                    )}
                    <div className="flex items-center gap-1">
                        {onStartAIAnalysis && (
                            <motion.button
                                onClick={(e) => { e.stopPropagation(); onStartAIAnalysis() }}
                                className="p-1 text-primary/70 hover:text-primary hover:bg-primary/10 rounded transition-colors"
                                whileTap={{ scale: 0.9 }}
                            >
                                <Brain className="w-3.5 h-3.5" />
                            </motion.button>
                        )}
                        {game.ai_prediction && onSendTelegram && (
                            <motion.button
                                onClick={(e) => {
                                    e.stopPropagation()
                                    const emoji = game.league === 'NFL' || game.league === 'NCAAF' ? '🏈' : '🏀'
                                    onSendTelegram(`${emoji} AI Pick!\n${game.away_team} @ ${game.home_team}\nPick: ${game.ai_prediction?.pick}\nEV: +${ev.toFixed(1)}%\nKelly: $${kelly.suggestedBet}`)
                                }}
                                className="p-1 text-blue-400/70 hover:text-blue-400 hover:bg-blue-500/10 rounded transition-colors"
                                whileTap={{ scale: 0.9 }}
                            >
                                <Send className="w-3 h-3" />
                            </motion.button>
                        )}
                    </div>
                </div>
            </div>

            {/* Teams & Game State */}
            <div className="p-3 cursor-pointer" onClick={() => setExpanded(!expanded)}>
                {/* Live Game Banner with Period/Time */}
                {game.isLive && (
                    <div className="mb-3 flex items-center justify-between px-3 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
                            <span className="text-sm font-bold text-emerald-400">
                                {game.quarter ? `Q${game.quarter}` : game.period ? `P${game.period}` : 'LIVE'}
                            </span>
                            {game.time_remaining && (
                                <span className="text-xs text-emerald-300 font-mono">{game.time_remaining}</span>
                            )}
                        </div>
                        {game.possession && (
                            <div className="flex items-center gap-1 text-xs text-slate-300">
                                <CircleDot className="w-3 h-3 text-amber-400" />
                                <span className="font-medium">{game.possession}</span>
                            </div>
                        )}
                        {game.down_distance && (
                            <span className="text-[10px] text-slate-400 bg-slate-800/50 px-2 py-0.5 rounded">
                                {game.down_distance}
                            </span>
                        )}
                    </div>
                )}

                <div className="flex justify-between items-center mb-3">
                    <div className="flex-1 space-y-1.5">
                        {/* Away Team Row */}
                        <div className="flex items-center gap-2">
                            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-[10px] font-bold text-slate-400">
                                {game.away_team.substring(0, 2).toUpperCase()}
                            </div>
                            <span className={`text-sm font-medium ${
                                game.ai_prediction?.pick === game.away_team ? 'text-primary' : 'text-white'
                            }`}>{game.away_team}</span>
                            {game.ai_prediction?.pick === game.away_team && (
                                <span className="text-[10px] px-1.5 py-0.5 bg-primary/20 text-primary rounded font-medium">AI</span>
                            )}
                            {game.away_rank && (
                                <span className="text-[10px] px-1 py-0.5 bg-amber-500/20 text-amber-400 rounded font-bold">
                                    #{game.away_rank}
                                </span>
                            )}
                        </div>
                        {/* Home Team Row */}
                        <div className="flex items-center gap-2">
                            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-[10px] font-bold text-slate-400">
                                {game.home_team.substring(0, 2).toUpperCase()}
                            </div>
                            <span className={`text-sm font-medium ${
                                game.ai_prediction?.pick === game.home_team ? 'text-primary' : 'text-white'
                            }`}>{game.home_team}</span>
                            {game.ai_prediction?.pick === game.home_team && (
                                <span className="text-[10px] px-1.5 py-0.5 bg-primary/20 text-primary rounded font-medium">AI</span>
                            )}
                            {game.home_rank && (
                                <span className="text-[10px] px-1 py-0.5 bg-amber-500/20 text-amber-400 rounded font-bold">
                                    #{game.home_rank}
                                </span>
                            )}
                        </div>
                    </div>
                    {/* Score Display for Live Games */}
                    {game.isLive && (
                        <div className="text-right space-y-1.5 ml-2">
                            <div className={`text-2xl font-bold tabular-nums ${
                                (game.away_score || 0) > (game.home_score || 0) ? 'text-emerald-400' : 'text-white'
                            }`}>{game.away_score || 0}</div>
                            <div className={`text-2xl font-bold tabular-nums ${
                                (game.home_score || 0) > (game.away_score || 0) ? 'text-emerald-400' : 'text-white'
                            }`}>{game.home_score || 0}</div>
                        </div>
                    )}
                </div>

                {/* Odds Grid - Always visible for scheduled games */}
                {!game.isLive && (displaySpread !== undefined || displayMoneyline !== undefined || displayTotal !== undefined) && (
                    <div className="grid grid-cols-3 gap-2 mb-3 p-2 rounded-lg bg-slate-800/40 border border-slate-700/30">
                        {/* Spread */}
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 uppercase mb-0.5">Spread</div>
                            <div className={`text-xs font-bold ${isAISpread ? 'text-cyan-400' : 'text-white'}`}>
                                {displaySpread != null ? (displaySpread > 0 ? `+${displaySpread}` : displaySpread) : '—'}
                                {isAISpread && <span className="text-[8px] ml-0.5 text-cyan-400/60">AI</span>}
                            </div>
                        </div>
                        {/* Moneyline */}
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 uppercase mb-0.5">ML</div>
                            <div className={`text-xs font-bold ${isAIMoneyline ? 'text-cyan-400' : 'text-white'}`}>
                                {displayMoneyline != null ? (displayMoneyline > 0 ? `+${displayMoneyline}` : displayMoneyline) : '—'}
                                {isAIMoneyline && <span className="text-[8px] ml-0.5 text-cyan-400/60">AI</span>}
                            </div>
                        </div>
                        {/* Total */}
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 uppercase mb-0.5">Total</div>
                            <div className="text-xs font-bold text-white">
                                {displayTotal != null ? `O/U ${displayTotal}` : '—'}
                            </div>
                        </div>
                    </div>
                )}

                {/* Live Game Odds - Compact row */}
                {game.isLive && (displaySpread !== undefined || displayMoneyline !== undefined || displayTotal !== undefined) && (
                    <div className="flex justify-between items-center text-[10px] text-slate-400 mb-2 px-1">
                        {displaySpread != null && (
                            <span className="flex items-center gap-1">
                                <span className="text-slate-500">SPD:</span>
                                <span className={isAISpread ? 'text-cyan-400' : 'text-slate-300'}>
                                    {displaySpread > 0 ? `+${displaySpread}` : displaySpread}
                                </span>
                            </span>
                        )}
                        {displayMoneyline != null && (
                            <span className="flex items-center gap-1">
                                <span className="text-slate-500">ML:</span>
                                <span className={isAIMoneyline ? 'text-cyan-400' : 'text-slate-300'}>
                                    {displayMoneyline > 0 ? `+${displayMoneyline}` : displayMoneyline}
                                </span>
                            </span>
                        )}
                        {displayTotal != null && (
                            <span className="flex items-center gap-1">
                                <span className="text-slate-500">O/U:</span>
                                <span className="text-slate-300">{displayTotal}</span>
                            </span>
                        )}
                    </div>
                )}

                {/* AI Prediction with Kelly & Probability Gauge */}
                {game.ai_prediction && (
                    <div className={`mb-3 rounded-lg p-2.5 border ${
                        hasHighEV ? 'bg-gradient-to-r from-amber-500/10 to-orange-500/10 border-amber-500/30' :
                        kelly.tier === 'elite' ? 'bg-gradient-to-r from-amber-500/10 to-yellow-500/10 border-amber-500/30' :
                        kelly.tier === 'strong' ? 'bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border-emerald-500/30' :
                        'bg-gradient-to-r from-primary/10 to-violet-500/10 border-primary/20'
                    }`}>
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <Brain className="w-3.5 h-3.5 text-primary" />
                                <span className="text-xs font-semibold text-primary">{game.ai_prediction.pick}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="flex items-center gap-1">
                                    <div className="w-10 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                        <motion.div
                                            className={`h-full rounded-full ${
                                                confidence >= ELITE_CONFIDENCE ? 'bg-gradient-to-r from-amber-500 to-amber-400' :
                                                confidence >= STRONG_CONFIDENCE ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
                                                'bg-gradient-to-r from-primary to-purple-400'
                                            }`}
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.min(100, confidence)}%` }}
                                        />
                                    </div>
                                    <span className="text-[10px] text-slate-300 font-medium">{confidence}%</span>
                                </div>
                                {ev > 0 && (
                                    <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold bg-emerald-500/20 text-emerald-400">
                                        +{ev.toFixed(0)}%
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Probability Gauge Ring */}
                        <div className="flex items-center gap-3 mb-2">
                            <div className="relative w-14 h-14 flex-shrink-0">
                                {/* Background circle */}
                                <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                                    <circle
                                        cx="18" cy="18" r="15.5"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="3"
                                        className="text-slate-700"
                                    />
                                    {/* Model probability arc */}
                                    <motion.circle
                                        cx="18" cy="18" r="15.5"
                                        fill="none"
                                        stroke="url(#probGradient)"
                                        strokeWidth="3"
                                        strokeLinecap="round"
                                        strokeDasharray={`${modelProb * 97.4} 97.4`}
                                        initial={{ strokeDasharray: "0 97.4" }}
                                        animate={{ strokeDasharray: `${modelProb * 97.4} 97.4` }}
                                        transition={{ duration: 0.8, ease: "easeOut" }}
                                    />
                                    {/* Market probability marker */}
                                    {game.enrichedData?.implied_probability && (
                                        <circle
                                            cx={18 + 15.5 * Math.cos((game.enrichedData.implied_probability - 0.25) * 2 * Math.PI)}
                                            cy={18 + 15.5 * Math.sin((game.enrichedData.implied_probability - 0.25) * 2 * Math.PI)}
                                            r="2"
                                            fill="#ef4444"
                                            className="drop-shadow-sm"
                                        />
                                    )}
                                    <defs>
                                        <linearGradient id="probGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stopColor="#10b981" />
                                            <stop offset="100%" stopColor="#34d399" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                                {/* Center text */}
                                <div className="absolute inset-0 flex flex-col items-center justify-center">
                                    <span className="text-sm font-bold text-white">{(modelProb * 100).toFixed(0)}%</span>
                                    <span className="text-[8px] text-slate-500">Win</span>
                                </div>
                            </div>

                            {/* Model vs Market Comparison Bar */}
                            <div className="flex-1 space-y-1">
                                <div className="flex items-center justify-between text-[9px]">
                                    <span className="text-emerald-400 font-medium">Model: {(modelProb * 100).toFixed(0)}%</span>
                                    <span className="text-slate-500">Market: {((game.enrichedData?.implied_probability || 0.5238) * 100).toFixed(0)}%</span>
                                </div>
                                <div className="relative h-2 bg-slate-700/50 rounded-full overflow-hidden">
                                    {/* Model probability bar */}
                                    <motion.div
                                        className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${modelProb * 100}%` }}
                                        transition={{ duration: 0.6 }}
                                    />
                                    {/* Market probability marker line */}
                                    <div
                                        className="absolute top-0 h-full w-0.5 bg-red-500 shadow-[0_0_4px_#ef4444]"
                                        style={{ left: `${(game.enrichedData?.implied_probability || 0.5238) * 100}%` }}
                                    />
                                </div>
                                <div className="flex items-center justify-between text-[9px]">
                                    <span className={`font-bold ${
                                        (modelProb - (game.enrichedData?.implied_probability || 0.5238)) > 0.05 ? 'text-emerald-400' :
                                        (modelProb - (game.enrichedData?.implied_probability || 0.5238)) > 0 ? 'text-blue-400' : 'text-red-400'
                                    }`}>
                                        Edge: {modelProb > (game.enrichedData?.implied_probability || 0.5238) ? '+' : ''}{((modelProb - (game.enrichedData?.implied_probability || 0.5238)) * 100).toFixed(1)}%
                                    </span>
                                    <span className={`font-medium ${
                                        kelly.tier === 'elite' ? 'text-amber-400' :
                                        kelly.tier === 'strong' ? 'text-emerald-400' : 'text-slate-400'
                                    }`}>
                                        Kelly: ${kelly.suggestedBet}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Stats Row */}
                        <div className="grid grid-cols-4 gap-2 text-[10px]">
                            <div className="text-center p-1 bg-slate-800/40 rounded">
                                <div className="text-slate-500">Conf</div>
                                <div className={`font-bold ${confidenceTier.color}`}>{confidence}%</div>
                            </div>
                            <div className="text-center p-1 bg-slate-800/40 rounded">
                                <div className="text-slate-500">Spread</div>
                                <div className="font-bold text-white">
                                    {game.ai_prediction.spread > 0 ? '+' : ''}{game.ai_prediction.spread.toFixed(1)}
                                </div>
                            </div>
                            <div className="text-center p-1 bg-slate-800/40 rounded">
                                <div className="text-slate-500">EV</div>
                                <div className={`font-bold ${ev > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                                    {ev > 0 ? '+' : ''}{ev.toFixed(0)}%
                                </div>
                            </div>
                            <div className="text-center p-1 bg-slate-800/40 rounded">
                                <div className="text-slate-500">Score</div>
                                <div className="font-bold text-purple-400">
                                    {(game.enrichedData?.overall_score || kelly.fraction * 100).toFixed(0)}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Quick Add AI Pick Button - Show for strong recommendations */}
                {game.ai_prediction && (kelly.tier === 'elite' || kelly.tier === 'strong' || hasHighEV) && (
                    <motion.button
                        onClick={(e) => {
                            e.stopPropagation()
                            onAddBet(
                                game,
                                'AI Pick',
                                game.ai_prediction!.pick,
                                displayMoneyline || -110
                            )
                        }}
                        className={`w-full mb-2 py-2 px-3 rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all ${
                            kelly.tier === 'elite'
                                ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/40 text-amber-400 hover:from-amber-500/30 hover:to-orange-500/30'
                                : kelly.tier === 'strong'
                                ? 'bg-gradient-to-r from-emerald-500/20 to-teal-500/20 border border-emerald-500/40 text-emerald-400 hover:from-emerald-500/30 hover:to-teal-500/30'
                                : 'bg-gradient-to-r from-primary/20 to-purple-500/20 border border-primary/40 text-primary hover:from-primary/30 hover:to-purple-500/30'
                        }`}
                        whileTap={{ scale: 0.98 }}
                        whileHover={{ scale: 1.01 }}
                    >
                        <Zap className="w-3.5 h-3.5" />
                        <span>Add AI Pick: {game.ai_prediction.pick}</span>
                        <span className="ml-auto text-[10px] opacity-80">
                            ${kelly.suggestedBet} Kelly
                        </span>
                    </motion.button>
                )}

                {/* Odds Buttons */}
                <div className="grid grid-cols-3 gap-1.5">
                    <motion.button
                        onClick={(e) => {
                            e.stopPropagation()
                            if (displaySpread != null) {
                                onAddBet(game, 'Spread', `${game.home_team} ${displaySpread}`, game.odds?.spread_home_odds || -110)
                            }
                        }}
                        disabled={displaySpread == null}
                        className={`bg-slate-800/60 hover:bg-primary/20 border rounded-lg p-2 text-center transition-all disabled:opacity-40 ${
                            isAISpread ? 'border-primary/30' : 'border-slate-700/50 hover:border-primary/50'
                        }`}
                        whileTap={{ scale: 0.95 }}
                    >
                        <div className="text-[9px] uppercase text-slate-500 mb-0.5 flex items-center justify-center gap-0.5">
                            Spread {isAISpread && <Brain className="w-2 h-2 text-primary" />}
                        </div>
                        <div className={`font-mono text-xs font-medium ${isAISpread ? 'text-primary' : 'text-white'}`}>
                            {displaySpread != null ? `${displaySpread > 0 ? '+' : ''}${displaySpread.toFixed(1)}` : '—'}
                        </div>
                    </motion.button>
                    <motion.button
                        onClick={(e) => {
                            e.stopPropagation()
                            if (displayTotal != null) onAddBet(game, 'Total', `O ${displayTotal}`, -110)
                        }}
                        disabled={displayTotal == null}
                        className="bg-slate-800/60 hover:bg-primary/20 border border-slate-700/50 hover:border-primary/50 rounded-lg p-2 text-center transition-all disabled:opacity-40"
                        whileTap={{ scale: 0.95 }}
                    >
                        <div className="text-[9px] uppercase text-slate-500 mb-0.5">Total</div>
                        <div className="font-mono text-xs font-medium text-white">
                            {displayTotal != null ? `O/U ${displayTotal}` : '—'}
                        </div>
                    </motion.button>
                    <motion.button
                        onClick={(e) => {
                            e.stopPropagation()
                            if (displayMoneyline != null) onAddBet(game, 'ML', game.home_team, displayMoneyline)
                        }}
                        disabled={displayMoneyline == null}
                        className={`bg-slate-800/60 hover:bg-primary/20 border rounded-lg p-2 text-center transition-all disabled:opacity-40 ${
                            isAIMoneyline ? 'border-primary/30' : 'border-slate-700/50 hover:border-primary/50'
                        }`}
                        whileTap={{ scale: 0.95 }}
                    >
                        <div className="text-[9px] uppercase text-slate-500 mb-0.5 flex items-center justify-center gap-0.5">
                            ML {isAIMoneyline && <Brain className="w-2 h-2 text-primary" />}
                        </div>
                        <div className={`font-mono text-xs font-medium ${
                            isAIMoneyline ? 'text-primary' : (displayMoneyline || 0) > 0 ? 'text-emerald-400' : 'text-white'
                        }`}>
                            {displayMoneyline != null ? `${displayMoneyline > 0 ? '+' : ''}${displayMoneyline}` : '—'}
                        </div>
                    </motion.button>
                </div>

                <div className="flex items-center justify-center mt-2 text-[10px] text-slate-500">
                    <motion.div animate={{ rotate: expanded ? 180 : 0 }}>
                        <ChevronDown className="w-3 h-3" />
                    </motion.div>
                </div>
            </div>

            {/* Expanded */}
            <AnimatePresence>
                {expanded && game.ai_prediction?.reasoning && (
                    <motion.div
                        className="px-3 pb-3"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                    >
                        <div className="pt-2 border-t border-slate-700/50">
                            <div className="text-[10px] text-slate-500 uppercase mb-1">AI Reasoning</div>
                            <p className="text-xs text-slate-400 leading-relaxed">{game.ai_prediction.reasoning}</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    )
})

// AI Streaming Panel
function AIStreamingPanel({
    streamingState,
    selectedGameId,
    onClose
}: {
    streamingState: ReturnType<typeof useStreamingPrediction>
    selectedGameId: string | null
    onClose: () => void
}) {
    const { isLoading, currentStep, progress, prediction, factors, reasoning, recommendation, error, isStreamingReasoning } = streamingState

    return (
        <div className="h-full glass-card rounded-none border-l border-slate-700/50 flex flex-col overflow-hidden bg-slate-900/95 backdrop-blur-xl">
            <div className="p-4 border-b border-slate-700/50 bg-gradient-to-r from-primary/10 to-purple-500/10">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center">
                            <Brain className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white text-sm">AI Analysis</h3>
                            <p className="text-[10px] text-slate-400">Real-time streaming</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-1 text-slate-400 hover:text-white">
                        <X className="w-4 h-4" />
                    </button>
                </div>
                {isLoading && (
                    <div className="mt-3">
                        <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                            <span>{currentStep?.replace('_', ' ')}</span>
                            <span>{progress}%</span>
                        </div>
                        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-gradient-to-r from-primary to-purple-500 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {!selectedGameId && !isLoading && (
                    <div className="text-center py-8">
                        <Brain className="w-12 h-12 mx-auto mb-3 text-slate-600" />
                        <h4 className="text-sm font-medium text-white mb-1">Select a Game</h4>
                        <p className="text-xs text-slate-400">Click the brain icon to start</p>
                    </div>
                )}

                {error && (
                    <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                        <p className="text-xs text-red-400">{error}</p>
                    </div>
                )}

                {prediction && (
                    <div className="glass-card p-3">
                        <div className="text-[10px] text-slate-500 uppercase mb-2">Prediction</div>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="text-sm text-white">{prediction.awayTeam}</span>
                                <span className="text-xs font-mono text-slate-300">{(prediction.awayWinProbability * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-sm text-white">{prediction.homeTeam}</span>
                                <span className="text-xs font-mono text-slate-300">{(prediction.homeWinProbability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                )}

                {factors.length > 0 && (
                    <div className="glass-card p-3">
                        <div className="text-[10px] text-slate-500 uppercase mb-2">Key Factors</div>
                        <div className="space-y-2">
                            {factors.map((factor, idx) => (
                                <div key={idx} className="flex justify-between text-xs">
                                    <span className="text-slate-300">{factor.factor}</span>
                                    <span className={factor.impact.startsWith('+') ? 'text-emerald-400' : factor.impact.startsWith('-') ? 'text-red-400' : 'text-slate-400'}>
                                        {factor.impact}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {(reasoning || isStreamingReasoning) && (
                    <div className="glass-card p-3">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] text-slate-500 uppercase">AI Reasoning</span>
                            {isStreamingReasoning && <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>}
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                            {reasoning}{isStreamingReasoning && <span className="animate-pulse">▋</span>}
                        </p>
                    </div>
                )}

                {recommendation && (
                    <div className={`glass-card p-3 ${
                        recommendation.action === 'STRONG BET' ? 'border-emerald-500/40 bg-emerald-500/5' :
                        recommendation.action === 'LEAN' ? 'border-amber-500/40 bg-amber-500/5' : ''
                    }`}>
                        <div className="text-[10px] text-slate-500 uppercase mb-2">Recommendation</div>
                        <div className="flex items-center gap-2 mb-2">
                            {recommendation.action === 'STRONG BET' ? <Flame className="w-4 h-4 text-emerald-400" /> :
                             recommendation.action === 'LEAN' ? <TrendingUp className="w-4 h-4 text-amber-400" /> :
                             <AlertCircle className="w-4 h-4 text-slate-400" />}
                            <span className={`text-sm font-bold ${
                                recommendation.action === 'STRONG BET' ? 'text-emerald-400' :
                                recommendation.action === 'LEAN' ? 'text-amber-400' : 'text-slate-400'
                            }`}>{recommendation.action}</span>
                        </div>
                        <div className="space-y-1 text-xs">
                            <div className="flex justify-between"><span className="text-slate-400">Side</span><span className="text-white">{recommendation.side}</span></div>
                            <div className="flex justify-between"><span className="text-slate-400">Edge</span><span className="text-emerald-400">+{recommendation.edge}%</span></div>
                            <div className="flex justify-between"><span className="text-slate-400">Bet Size</span><span className="text-white">{recommendation.suggestedBetSize}</span></div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

// Bet Slip Panel
function BetSlipPanel({
    legs,
    stakeAmount,
    potentialWin,
    onRemoveLeg,
    onSetStake,
    onClear,
    onClose
}: {
    legs: any[]
    stakeAmount: number
    potentialWin: number
    onRemoveLeg: (id: string) => void
    onSetStake: (amount: number) => void
    onClear: () => void
    onClose: () => void
}) {
    return (
        <div className="h-full glass-card rounded-none border-l border-slate-700/50 flex flex-col overflow-hidden bg-slate-900/95 backdrop-blur-xl">
            <div className="p-4 border-b border-slate-700/50 bg-slate-900/50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center">
                            <Target className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-white text-sm">Bet Slip</h3>
                            <p className="text-[10px] text-slate-400">{legs.length} selections</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {legs.length > 0 && <button onClick={onClear} className="text-[10px] text-slate-400 hover:text-red-400">Clear</button>}
                        <button onClick={onClose} className="p-1 text-slate-400 hover:text-white"><ChevronRight className="w-4 h-4" /></button>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {legs.length === 0 ? (
                    <div className="text-center py-8">
                        <Target className="w-10 h-10 mx-auto mb-3 text-slate-600" />
                        <h4 className="text-xs font-medium text-white mb-0.5">No Selections</h4>
                        <p className="text-[10px] text-slate-400">Click odds to add bets</p>
                    </div>
                ) : (
                    legs.map(leg => (
                        <motion.div key={leg.id} className="glass-card p-3" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                            <div className="flex justify-between items-start mb-2">
                                <div className="flex-1">
                                    <div className="text-xs font-medium text-white">
                                        {leg.selection === 'home' ? leg.homeTeam : leg.awayTeam}
                                    </div>
                                    <div className="text-[10px] text-slate-400">{leg.awayTeam} @ {leg.homeTeam}</div>
                                    {leg.aiEdge && leg.aiEdge > 0.02 && (
                                        <span className="text-[10px] text-emerald-400 flex items-center gap-0.5">
                                            <TrendingUp className="w-2.5 h-2.5" />+{(leg.aiEdge * 100).toFixed(1)}% edge
                                        </span>
                                    )}
                                </div>
                                <button onClick={() => onRemoveLeg(leg.id)} className="p-1 text-slate-500 hover:text-red-400">
                                    <X className="w-3 h-3" />
                                </button>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-slate-400">{leg.betType}</span>
                                <span className={`text-xs font-mono font-medium px-2 py-0.5 rounded ${
                                    leg.odds > 0 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-slate-700 text-white'
                                }`}>{leg.odds > 0 ? '+' : ''}{leg.odds}</span>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>

            {legs.length > 0 && (
                <div className="p-4 border-t border-slate-700/50 bg-slate-900/50 space-y-3">
                    <div className="flex items-center gap-2">
                        <div className="relative flex-1">
                            <DollarSign className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-500" />
                            <input
                                type="number"
                                value={stakeAmount || ''}
                                onChange={(e) => onSetStake(parseFloat(e.target.value) || 0)}
                                placeholder="Stake"
                                className="input-field pl-7 py-2 text-sm w-full"
                            />
                        </div>
                        <div className="flex gap-1">
                            {[10, 25, 50, 100].map(amt => (
                                <button
                                    key={amt}
                                    onClick={() => onSetStake(amt)}
                                    className="px-2 py-1.5 text-[10px] bg-slate-800 hover:bg-slate-700 text-white rounded"
                                >
                                    ${amt}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-slate-400">Stake</span>
                            <span className="text-white font-medium">${stakeAmount.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-400">To Win</span>
                            <span className="text-emerald-400 font-semibold">${potentialWin.toFixed(2)}</span>
                        </div>
                    </div>
                    <motion.button
                        disabled={stakeAmount === 0}
                        className="btn-primary w-full py-2.5 flex items-center justify-center gap-2 text-sm disabled:opacity-50"
                        whileTap={{ scale: 0.98 }}
                    >
                        <Zap className="w-4 h-4" />
                        Place Bet
                    </motion.button>
                </div>
            )}
        </div>
    )
}
