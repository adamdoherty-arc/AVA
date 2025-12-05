/**
 * Stocks Tile Hub Page
 * Interactive tile grid with AI scores, watchlist management, and streaming analysis
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Activity,
  Filter,
  RefreshCw,
  Plus,
  Search,
  LayoutGrid,
  Star,
  X,
  ChevronDown,
  Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';

import { AnimatedStockCard } from '@/components/stocks/AnimatedStockCard';
import type { StockTileData } from '@/components/stocks/AnimatedStockCard';
import { useStockWatchlistStore, DEFAULT_WATCHLISTS } from '@/store/stockWatchlistStore';
import { useStockStreamingAnalysis } from '@/hooks/useStockStreamingAnalysis';
import { BACKEND_URL } from '@/config/api';

// Filter types
type FilterType = 'all' | 'bullish' | 'bearish' | 'high_iv' | 'breakout' | 'favorites';
type SortType = 'ai_score' | 'change' | 'iv' | 'name';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

const StatsCard: React.FC<StatsCardProps> = ({ title, value, icon, trend }) => (
  <Card className="bg-card/50">
    <CardContent className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
        <div
          className={cn(
            'p-2 rounded-lg',
            trend === 'up' && 'bg-green-500/10 text-green-500',
            trend === 'down' && 'bg-red-500/10 text-red-500',
            trend === 'neutral' && 'bg-muted text-muted-foreground'
          )}
        >
          {icon}
        </div>
      </div>
    </CardContent>
  </Card>
);

// Skeleton Card Component for loading states
const SkeletonStockCard: React.FC = () => (
  <Card className="overflow-hidden animate-pulse">
    <CardContent className="p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="h-6 w-16 bg-muted rounded" />
        <div className="h-5 w-20 bg-muted rounded" />
      </div>
      <div className="h-4 w-32 bg-muted rounded mb-2" />
      <div className="flex items-center justify-between mb-3">
        <div className="h-8 w-24 bg-muted rounded" />
        <div className="h-6 w-16 bg-muted rounded" />
      </div>
      <div className="h-2 w-full bg-muted rounded mb-3" />
      <div className="grid grid-cols-2 gap-2">
        <div className="h-4 bg-muted rounded" />
        <div className="h-4 bg-muted rounded" />
        <div className="h-4 bg-muted rounded" />
        <div className="h-4 bg-muted rounded" />
      </div>
    </CardContent>
  </Card>
);

// Analysis Panel Component
const AnalysisPanel: React.FC<{
  stock: StockTileData | null;
  isOpen: boolean;
  onClose: () => void;
}> = ({ stock, isOpen, onClose }) => {
  const analysis = useStockStreamingAnalysis();

  useEffect(() => {
    if (stock && isOpen) {
      analysis.startAnalysis(stock.symbol);
    }
    return () => analysis.reset();
  }, [stock?.symbol, isOpen]);

  if (!stock) return null;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            AI Analysis: {stock.symbol}
          </SheetTitle>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* Price Summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Price Data</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end justify-between">
                <div>
                  <div className="text-3xl font-bold">
                    ${analysis.priceData?.current_price.toFixed(2) || stock.current_price.toFixed(2)}
                  </div>
                  <div
                    className={cn(
                      'text-sm',
                      stock.daily_change_pct >= 0 ? 'text-green-500' : 'text-red-500'
                    )}
                  >
                    {stock.daily_change_pct >= 0 ? '+' : ''}
                    {stock.daily_change_pct.toFixed(2)}%
                  </div>
                </div>
                <div className="text-right text-sm text-muted-foreground">
                  <div>IV: {analysis.priceData?.iv_estimate.toFixed(1) || stock.iv_estimate.toFixed(1)}%</div>
                  <div>Vol Regime: {stock.vol_regime}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Score */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="w-4 h-4" />
                AI Score
                {analysis.isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-4xl font-bold">
                    {analysis.aiScore?.ai_score.toFixed(0) || stock.ai_score.toFixed(0)}
                  </span>
                  <Badge
                    variant="outline"
                    className={cn(
                      'text-lg px-3 py-1',
                      stock.ai_score >= 65 && 'bg-green-500/20 text-green-500 border-green-500/30',
                      stock.ai_score >= 50 && stock.ai_score < 65 && 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30',
                      stock.ai_score < 50 && 'bg-red-500/20 text-red-500 border-red-500/30'
                    )}
                  >
                    {analysis.aiScore?.recommendation || stock.recommendation}
                  </Badge>
                </div>

                {/* Score Components */}
                {(analysis.aiScore?.components || stock.score_components) && (
                  <div className="space-y-2">
                    {Object.entries(analysis.aiScore?.components || stock.score_components || {}).map(
                      ([name, data]) => (
                        <div key={name} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="capitalize">{name.replace('_', ' ')}</span>
                            <span>{data.raw_score.toFixed(0)}</span>
                          </div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-primary rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${data.raw_score}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Technical Signals */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Technical Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-muted-foreground">Trend</div>
                  <div className="flex items-center gap-1">
                    {stock.trend === 'bullish' ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : stock.trend === 'bearish' ? (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    ) : (
                      <Activity className="w-4 h-4 text-yellow-500" />
                    )}
                    <span className="font-medium capitalize">{stock.trend}</span>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">RSI (14)</div>
                  <div className="font-medium">{stock.rsi_14.toFixed(1)}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Support</div>
                  <div className="font-medium text-green-500">${stock.support_price.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Resistance</div>
                  <div className="font-medium text-red-500">${stock.resistance_price.toFixed(2)}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Prediction */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Price Prediction</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-muted-foreground">1 Day</div>
                  <div
                    className={cn(
                      'font-medium',
                      stock.predicted_change_1d >= 0 ? 'text-green-500' : 'text-red-500'
                    )}
                  >
                    {stock.predicted_change_1d >= 0 ? '+' : ''}
                    {stock.predicted_change_1d.toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">5 Day</div>
                  <div
                    className={cn(
                      'font-medium',
                      stock.predicted_change_5d >= 0 ? 'text-green-500' : 'text-red-500'
                    )}
                  >
                    {stock.predicted_change_5d >= 0 ? '+' : ''}
                    {stock.predicted_change_5d.toFixed(1)}%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Reasoning */}
          {analysis.reasoning && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">AI Reasoning</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{analysis.reasoning}</p>
              </CardContent>
            </Card>
          )}

          {/* LLM Deep Analysis (when available) */}
          {analysis.llmAnalysis && (
            <Card className="border-primary/30 bg-primary/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Brain className="w-4 h-4 text-primary" />
                  AI Deep Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm">{analysis.llmAnalysis}</p>
              </CardContent>
            </Card>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
};

// Main Component
export const StocksTileHub: React.FC = () => {
  const [stocks, setStocks] = useState<StockTileData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterType>('all');
  const [sort, setSort] = useState<SortType>('ai_score');
  const [search, setSearch] = useState('');
  const [gridCols, setGridCols] = useState(3);
  const [selectedStock, setSelectedStock] = useState<StockTileData | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [showNewWatchlist, setShowNewWatchlist] = useState(false);

  const {
    activeWatchlist,
    setActiveWatchlist,
    customWatchlists,
    favorites,
    toggleFavorite,
    createWatchlist,
  } = useStockWatchlistStore();

  const allWatchlists = { ...DEFAULT_WATCHLISTS, ...customWatchlists };

  // Fetch data
  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Uses centralized config from @/config/api
      const apiUrl = BACKEND_URL;
      const response = await fetch(
        `${apiUrl}/api/stocks/tiles/all-data?watchlist=${encodeURIComponent(activeWatchlist)}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch stocks');
      }

      const data = await response.json();
      setStocks(data.stocks || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load stocks');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [activeWatchlist]);

  // Filter and sort stocks
  const filteredStocks = useMemo(() => {
    let result = [...stocks];

    // Apply search
    if (search) {
      const searchLower = search.toLowerCase();
      result = result.filter(
        (s) =>
          s.symbol.toLowerCase().includes(searchLower) ||
          s.company_name.toLowerCase().includes(searchLower)
      );
    }

    // Apply filter
    switch (filter) {
      case 'bullish':
        result = result.filter((s) => s.trend === 'bullish' || s.ai_score >= 65);
        break;
      case 'bearish':
        result = result.filter((s) => s.trend === 'bearish' || s.ai_score < 35);
        break;
      case 'high_iv':
        result = result.filter((s) => s.iv_estimate > 40);
        break;
      case 'breakout':
        result = result.filter((s) => Math.abs(s.predicted_change_5d) > 3);
        break;
      case 'favorites':
        result = result.filter((s) => favorites.includes(s.symbol));
        break;
    }

    // Apply sort
    switch (sort) {
      case 'ai_score':
        result.sort((a, b) => b.ai_score - a.ai_score);
        break;
      case 'change':
        result.sort((a, b) => b.daily_change_pct - a.daily_change_pct);
        break;
      case 'iv':
        result.sort((a, b) => b.iv_estimate - a.iv_estimate);
        break;
      case 'name':
        result.sort((a, b) => a.symbol.localeCompare(b.symbol));
        break;
    }

    return result;
  }, [stocks, filter, sort, search, favorites]);

  // Stats
  const stats = useMemo(() => {
    if (stocks.length === 0) {
      return { avgScore: 0, bullish: 0, bearish: 0, neutral: 0 };
    }
    return {
      avgScore: stocks.reduce((sum, s) => sum + s.ai_score, 0) / stocks.length,
      bullish: stocks.filter((s) => s.trend === 'bullish').length,
      bearish: stocks.filter((s) => s.trend === 'bearish').length,
      neutral: stocks.filter((s) => s.trend === 'neutral').length,
    };
  }, [stocks]);

  // Handle stock click
  const handleStockClick = (stock: StockTileData) => {
    setSelectedStock(stock);
    setShowAnalysis(true);
  };

  // Create new watchlist
  const handleCreateWatchlist = () => {
    if (newWatchlistName.trim()) {
      createWatchlist(newWatchlistName.trim(), []);
      setNewWatchlistName('');
      setShowNewWatchlist(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-primary" />
            Stocks Hub
          </h1>
          <p className="text-muted-foreground">
            AI-powered stock analysis and scoring
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
            <RefreshCw className={cn('w-4 h-4 mr-1', loading && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatsCard
          title="Avg AI Score"
          value={stats.avgScore.toFixed(0)}
          icon={<Brain className="w-5 h-5" />}
          trend={stats.avgScore >= 60 ? 'up' : stats.avgScore < 40 ? 'down' : 'neutral'}
        />
        <StatsCard
          title="Bullish"
          value={stats.bullish}
          icon={<TrendingUp className="w-5 h-5" />}
          trend="up"
        />
        <StatsCard
          title="Bearish"
          value={stats.bearish}
          icon={<TrendingDown className="w-5 h-5" />}
          trend="down"
        />
        <StatsCard
          title="Total Stocks"
          value={stocks.length}
          icon={<Activity className="w-5 h-5" />}
          trend="neutral"
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Watchlist Selector */}
        <Select value={activeWatchlist} onValueChange={setActiveWatchlist}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select watchlist" />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(allWatchlists).map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* New Watchlist Button */}
        <Dialog open={showNewWatchlist} onOpenChange={setShowNewWatchlist}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Plus className="w-4 h-4 mr-1" />
              New Watchlist
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Watchlist</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 mt-4">
              <Input
                placeholder="Watchlist name"
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateWatchlist()}
              />
              <Button onClick={handleCreateWatchlist} className="w-full">
                Create Watchlist
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search stocks..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Filter */}
        <Select value={filter} onValueChange={(v) => setFilter(v as FilterType)}>
          <SelectTrigger className="w-[150px]">
            <Filter className="w-4 h-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Stocks</SelectItem>
            <SelectItem value="bullish">Bullish</SelectItem>
            <SelectItem value="bearish">Bearish</SelectItem>
            <SelectItem value="high_iv">High IV</SelectItem>
            <SelectItem value="breakout">Breakout</SelectItem>
            <SelectItem value="favorites">Favorites</SelectItem>
          </SelectContent>
        </Select>

        {/* Sort */}
        <Select value={sort} onValueChange={(v) => setSort(v as SortType)}>
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="Sort by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ai_score">AI Score</SelectItem>
            <SelectItem value="change">% Change</SelectItem>
            <SelectItem value="iv">IV Rank</SelectItem>
            <SelectItem value="name">Name</SelectItem>
          </SelectContent>
        </Select>

        {/* Grid Size */}
        <div className="flex items-center gap-1 border rounded-md p-1">
          {[2, 3, 4].map((cols) => (
            <Button
              key={cols}
              variant={gridCols === cols ? 'default' : 'ghost'}
              size="sm"
              className="px-2"
              onClick={() => setGridCols(cols)}
            >
              <LayoutGrid className="w-4 h-4" />
            </Button>
          ))}
        </div>
      </div>

      {/* Error State */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="p-4 flex items-center gap-2 text-destructive">
            <X className="w-5 h-5" />
            {error}
          </CardContent>
        </Card>
      )}

      {/* Loading State - Skeleton Grid */}
      {loading && (
        <div
          className={cn(
            'grid gap-4',
            gridCols === 2 && 'grid-cols-1 md:grid-cols-2',
            gridCols === 3 && 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
            gridCols === 4 && 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
          )}
        >
          {Array.from({ length: 9 }).map((_, i) => (
            <SkeletonStockCard key={i} />
          ))}
        </div>
      )}

      {/* Stock Grid */}
      {!loading && !error && (
        <motion.div
          className={cn(
            'grid gap-4',
            gridCols === 2 && 'grid-cols-1 md:grid-cols-2',
            gridCols === 3 && 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
            gridCols === 4 && 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
          )}
        >
          <AnimatePresence>
            {filteredStocks.map((stock) => (
              <div key={stock.symbol} className="relative">
                <AnimatedStockCard
                  stock={stock}
                  onClick={handleStockClick}
                  showAIScore={true}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-2 right-2 z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleFavorite(stock.symbol);
                  }}
                >
                  <Star
                    className={cn(
                      'w-4 h-4',
                      favorites.includes(stock.symbol)
                        ? 'fill-yellow-500 text-yellow-500'
                        : 'text-muted-foreground'
                    )}
                  />
                </Button>
              </div>
            ))}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Empty State */}
      {!loading && !error && filteredStocks.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          No stocks found matching your criteria
        </div>
      )}

      {/* Analysis Panel */}
      <AnalysisPanel
        stock={selectedStock}
        isOpen={showAnalysis}
        onClose={() => setShowAnalysis(false)}
      />
    </div>
  );
};

export default StocksTileHub;
