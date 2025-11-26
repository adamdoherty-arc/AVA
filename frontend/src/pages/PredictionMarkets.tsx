import { useState, useEffect } from 'react'
import { TrendingUp, Clock, DollarSign, AlertCircle, Sparkles, Filter, RefreshCw, BarChart2 } from 'lucide-react'
import { api } from '../services/api'
import clsx from 'clsx'

export function PredictionMarkets() {
    const [markets, setMarkets] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedSector, setSelectedSector] = useState<string>('All')

    useEffect(() => {
        const fetchMarkets = async () => {
            setLoading(true)
            setError(null)
            try {
                const response = await api.getPredictionMarkets(selectedSector === 'All' ? undefined : selectedSector)

                const mappedMarkets = response.markets.map((market: any) => ({
                    id: market.id,
                    title: market.title,
                    volume: formatVolume(market.volume || 0),
                    yesPrice: market.yes_price || 0,
                    noPrice: market.no_price || 0,
                    closeDate: formatDate(market.close_time),
                    category: market.market_type || 'Unknown'
                }))
                setMarkets(mappedMarkets)
            } catch (err) {
                console.error("Failed to fetch markets:", err)
                setError("Failed to load markets. Please try again.")
            } finally {
                setLoading(false)
            }
        }

        fetchMarkets()
    }, [selectedSector])

    const formatVolume = (vol: number) => {
        if (vol >= 1000000) return `$${(vol / 1000000).toFixed(1)}M`
        if (vol >= 1000) return `$${(vol / 1000).toFixed(1)}K`
        return `$${vol.toFixed(0)}`
    }

    const formatDate = (dateStr?: string) => {
        if (!dateStr) return 'TBD'
        return new Date(dateStr).toLocaleDateString()
    }

    const sectors = ['All', 'Politics', 'Economics', 'Crypto', 'Tech']

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
                        <Sparkles className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Prediction Markets</h1>
                        <p className="text-sm text-slate-400">Trade on future events with Kalshi & PolyMarket</p>
                    </div>
                </div>
                <button
                    onClick={() => window.location.reload()}
                    className="btn-secondary flex items-center gap-2"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </header>

            {/* Stats Row */}
            <div className="grid grid-cols-4 gap-4">
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
                        <BarChart2 className="w-5 h-5 text-amber-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Total Markets</p>
                        <p className="text-xl font-bold text-white">{markets.length}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                        <DollarSign className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Top Volume</p>
                        <p className="text-xl font-bold text-white">{markets[0]?.volume || '-'}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                        <Filter className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Category</p>
                        <p className="text-xl font-bold text-white">{selectedSector}</p>
                    </div>
                </div>
                <div className="glass-card p-4 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                        <Clock className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wide">Active</p>
                        <p className="text-xl font-bold text-white">{markets.filter(m => m.closeDate !== 'TBD').length}</p>
                    </div>
                </div>
            </div>

            {/* Sector Filter */}
            <div className="glass-card p-4">
                <div className="flex items-center gap-4">
                    <span className="text-sm text-slate-400">Filter by sector:</span>
                    <div className="flex gap-2">
                        {sectors.map(sector => (
                            <button
                                key={sector}
                                onClick={() => setSelectedSector(sector)}
                                className={clsx(
                                    "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                                    selectedSector === sector
                                        ? "bg-amber-500 text-white shadow-lg shadow-amber-500/20"
                                        : "bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white border border-slate-700/50"
                                )}
                            >
                                {sector}
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
                        <h4 className="font-medium text-red-400">Error Loading Markets</h4>
                        <p className="text-sm text-slate-400">{error}</p>
                    </div>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div className="glass-card p-12 text-center">
                    <div className="relative w-16 h-16 mx-auto mb-6">
                        <div className="absolute inset-0 rounded-full border-4 border-slate-700"></div>
                        <div className="absolute inset-0 rounded-full border-4 border-t-amber-500 border-r-transparent border-b-transparent border-l-transparent animate-spin"></div>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">Loading Markets</h3>
                    <p className="text-slate-400">Fetching latest prediction markets...</p>
                </div>
            )}

            {/* Market Grid */}
            {!loading && !error && (
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
                    {markets.length > 0 ? (
                        markets.map(market => (
                            <MarketCard key={market.id} market={market} />
                        ))
                    ) : (
                        <div className="col-span-full glass-card p-16 text-center">
                            <div className="w-20 h-20 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-6 border border-slate-700/50">
                                <Sparkles className="w-10 h-10 text-slate-500" />
                            </div>
                            <h3 className="text-xl font-bold text-white mb-2">No Markets Found</h3>
                            <p className="text-slate-400">Try selecting a different category or check back later.</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

function MarketCard({ market }: { market: any }) {
    const getCategoryColor = (cat: string) => {
        switch (cat.toLowerCase()) {
            case 'politics': return 'badge-danger'
            case 'economics': return 'badge-warning'
            case 'crypto': return 'badge-purple'
            case 'tech': return 'badge-info'
            default: return 'badge-neutral'
        }
    }

    return (
        <div className="glass-card p-5 hover:border-amber-500/30 transition-all duration-200 group">
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
                <span className={getCategoryColor(market.category)}>
                    {market.category}
                </span>
                <span className="flex items-center gap-1.5 text-xs text-slate-400">
                    <Clock size={12} />
                    {market.closeDate}
                </span>
            </div>

            {/* Title */}
            <h3 className="text-lg font-bold text-white mb-5 line-clamp-2 min-h-[3.5rem]" title={market.title}>
                {market.title}
            </h3>

            {/* Yes/No Buttons */}
            <div className="grid grid-cols-2 gap-3 mb-5">
                <button className="flex flex-col items-center justify-center p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 hover:border-emerald-500/40 transition-all duration-200 group/yes">
                    <span className="text-xs font-bold text-emerald-400 mb-1 uppercase tracking-wider">Yes</span>
                    <span className="text-2xl font-bold text-emerald-400 group-hover/yes:scale-110 transition-transform">
                        {Math.round(market.yesPrice * 100)}¢
                    </span>
                </button>
                <button className="flex flex-col items-center justify-center p-4 rounded-xl bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 hover:border-red-500/40 transition-all duration-200 group/no">
                    <span className="text-xs font-bold text-red-400 mb-1 uppercase tracking-wider">No</span>
                    <span className="text-2xl font-bold text-red-400 group-hover/no:scale-110 transition-transform">
                        {Math.round(market.noPrice * 100)}¢
                    </span>
                </button>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
                <span className="flex items-center gap-1.5 text-sm text-slate-400">
                    <TrendingUp size={14} className="text-amber-400" />
                    Vol: <span className="text-white font-medium">{market.volume}</span>
                </span>
                <button className="flex items-center gap-1.5 text-sm text-primary hover:text-primary/80 transition-colors font-medium">
                    <DollarSign size={14} />
                    Trade
                </button>
            </div>
        </div>
    )
}

export default PredictionMarkets
