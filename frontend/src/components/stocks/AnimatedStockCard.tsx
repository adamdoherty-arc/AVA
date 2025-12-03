/**
 * Animated Stock Card Component
 * Interactive tile with AI score, price animations, and technical signals
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Brain,
  ChevronRight,
  BarChart3,
  Zap,
} from 'lucide-react';

export interface StockTileData {
  symbol: string;
  company_name: string;
  sector: string;
  current_price: number;
  daily_change_pct: number;
  ai_score: number;
  recommendation: string;
  confidence: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  trend_strength: number;
  rsi_14: number;
  iv_estimate: number;
  vol_regime: string;
  predicted_change_1d: number;
  predicted_change_5d: number;
  support_price: number;
  resistance_price: number;
  market_cap?: number;
  score_components?: Record<string, { raw_score: number; weight: number }>;
}

interface AnimatedStockCardProps {
  stock: StockTileData;
  onClick?: (stock: StockTileData) => void;
  showAIScore?: boolean;
  compact?: boolean;
  className?: string;
}

// Animated price display with color flash
const AnimatedPrice: React.FC<{ price: number; change: number }> = ({
  price,
  change,
}) => {
  const controls = useAnimation();
  const prevPriceRef = useRef(price);

  useEffect(() => {
    if (prevPriceRef.current !== price) {
      const isUp = price > prevPriceRef.current;
      controls.start({
        scale: [1, 1.1, 1],
        color: [
          'inherit',
          isUp ? '#22c55e' : '#ef4444',
          'inherit',
        ],
        transition: { duration: 0.4 },
      });
      prevPriceRef.current = price;
    }
  }, [price, controls]);

  return (
    <motion.div animate={controls} className="text-right">
      <div className="text-2xl font-bold tabular-nums">${price.toFixed(2)}</div>
      <div
        className={cn(
          'text-sm font-medium',
          change >= 0 ? 'text-green-500' : 'text-red-500'
        )}
      >
        {change >= 0 ? '+' : ''}
        {change.toFixed(2)}%
      </div>
    </motion.div>
  );
};

// Market status indicator
const MarketStatusIndicator: React.FC<{ trend: string }> = ({ trend }) => {
  const color =
    trend === 'bullish'
      ? 'bg-green-500'
      : trend === 'bearish'
      ? 'bg-red-500'
      : 'bg-yellow-500';

  return (
    <motion.div
      className={cn('w-2 h-2 rounded-full', color)}
      animate={{
        scale: [1, 1.2, 1],
        opacity: [1, 0.7, 1],
      }}
      transition={{
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  );
};

// AI Score Meter with gradient
const AIScoreMeter: React.FC<{
  score: number;
  recommendation: string;
  confidence: number;
}> = ({ score, recommendation, confidence }) => {
  // Color based on score
  const getColor = () => {
    if (score >= 80) return 'from-emerald-500 to-green-400';
    if (score >= 65) return 'from-green-500 to-lime-400';
    if (score >= 50) return 'from-yellow-500 to-amber-400';
    if (score >= 35) return 'from-orange-500 to-amber-400';
    return 'from-red-500 to-rose-400';
  };

  const getBadgeColor = () => {
    if (score >= 80) return 'bg-emerald-500/20 text-emerald-500 border-emerald-500/30';
    if (score >= 65) return 'bg-green-500/20 text-green-500 border-green-500/30';
    if (score >= 50) return 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30';
    if (score >= 35) return 'bg-orange-500/20 text-orange-500 border-orange-500/30';
    return 'bg-red-500/20 text-red-500 border-red-500/30';
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1 text-muted-foreground">
          <Brain className="w-3 h-3" />
          AI Score
        </span>
        <span className="font-bold text-lg">{score.toFixed(0)}</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <motion.div
          className={cn('h-full rounded-full bg-gradient-to-r', getColor())}
          initial={{ width: 0 }}
          animate={{ width: `${score}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>
      <div className="flex items-center justify-between">
        <Badge variant="outline" className={cn('text-xs', getBadgeColor())}>
          {recommendation.replace('_', ' ')}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {(confidence * 100).toFixed(0)}% conf
        </span>
      </div>
    </div>
  );
};

// Technical Signal Badges
const TechnicalSignals: React.FC<{
  rsi: number;
  trend: string;
  trendStrength: number;
  volRegime: string;
}> = ({ rsi, trend, trendStrength, volRegime }) => {
  const getRsiBadge = () => {
    if (rsi > 70) return { text: 'Overbought', color: 'text-red-400 bg-red-500/10' };
    if (rsi < 30) return { text: 'Oversold', color: 'text-green-400 bg-green-500/10' };
    return { text: `RSI ${rsi.toFixed(0)}`, color: 'text-muted-foreground bg-muted' };
  };

  const getTrendIcon = () => {
    if (trend === 'bullish') return <TrendingUp className="w-3 h-3 text-green-500" />;
    if (trend === 'bearish') return <TrendingDown className="w-3 h-3 text-red-500" />;
    return <Minus className="w-3 h-3 text-yellow-500" />;
  };

  const rsiBadge = getRsiBadge();

  return (
    <div className="flex flex-wrap gap-1">
      <Badge variant="outline" className={cn('text-xs', rsiBadge.color)}>
        {rsiBadge.text}
      </Badge>
      <Badge variant="outline" className="text-xs flex items-center gap-1">
        {getTrendIcon()}
        {(trendStrength * 100).toFixed(0)}%
      </Badge>
      {volRegime !== 'normal' && (
        <Badge variant="outline" className="text-xs text-purple-400 bg-purple-500/10">
          <Activity className="w-3 h-3 mr-1" />
          {volRegime}
        </Badge>
      )}
    </div>
  );
};

// Prediction Display
const PredictionDisplay: React.FC<{
  change1d: number;
  change5d: number;
}> = ({ change1d, change5d }) => {
  if (Math.abs(change5d) < 0.5) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -5 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-1 text-xs"
    >
      <Zap className="w-3 h-3 text-yellow-500" />
      <span className="text-muted-foreground">5d:</span>
      <span
        className={cn(
          'font-medium',
          change5d > 0 ? 'text-green-500' : 'text-red-500'
        )}
      >
        {change5d > 0 ? '+' : ''}
        {change5d.toFixed(1)}%
      </span>
    </motion.div>
  );
};

// Main Component
export const AnimatedStockCard: React.FC<AnimatedStockCardProps> = ({
  stock,
  onClick,
  showAIScore = true,
  compact = false,
  className,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [prevPrice, setPrevPrice] = useState(stock.current_price);
  const [priceFlash, setPriceFlash] = useState<'up' | 'down' | null>(null);

  // Detect price changes
  useEffect(() => {
    if (stock.current_price !== prevPrice) {
      setPriceFlash(stock.current_price > prevPrice ? 'up' : 'down');
      setTimeout(() => setPriceFlash(null), 1000);
      setPrevPrice(stock.current_price);
    }
  }, [stock.current_price, prevPrice]);

  // Score-based ring color
  const getScoreRing = () => {
    if (stock.ai_score >= 70) return 'ring-1 ring-green-500/30';
    if (stock.ai_score <= 30) return 'ring-1 ring-red-500/30';
    return '';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      className={className}
    >
      <Card
        className={cn(
          'relative overflow-hidden cursor-pointer transition-shadow',
          'hover:shadow-lg hover:shadow-primary/10',
          getScoreRing(),
          priceFlash === 'up' && 'ring-2 ring-green-500',
          priceFlash === 'down' && 'ring-2 ring-red-500'
        )}
        onClick={() => onClick?.(stock)}
      >
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MarketStatusIndicator trend={stock.trend} />
              <div>
                <div className="font-bold text-lg">{stock.symbol}</div>
                <div className="text-xs text-muted-foreground truncate max-w-[120px]">
                  {stock.company_name}
                </div>
              </div>
            </div>
            <Badge variant="outline" className="text-xs">
              {stock.sector}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Price Display */}
          <div className="flex items-end justify-between">
            <div className="text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <BarChart3 className="w-3 h-3" />
                IV: {stock.iv_estimate.toFixed(1)}%
              </div>
            </div>
            <AnimatedPrice
              price={stock.current_price}
              change={stock.daily_change_pct}
            />
          </div>

          {/* AI Score */}
          {showAIScore && !compact && (
            <AIScoreMeter
              score={stock.ai_score}
              recommendation={stock.recommendation}
              confidence={stock.confidence}
            />
          )}

          {/* Technical Signals */}
          {!compact && (
            <TechnicalSignals
              rsi={stock.rsi_14}
              trend={stock.trend}
              trendStrength={stock.trend_strength}
              volRegime={stock.vol_regime}
            />
          )}

          {/* Prediction */}
          {!compact && (
            <PredictionDisplay
              change1d={stock.predicted_change_1d}
              change5d={stock.predicted_change_5d}
            />
          )}

          {/* Support/Resistance (on hover) */}
          <AnimatePresence>
            {isHovered && !compact && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="text-xs text-muted-foreground border-t pt-2"
              >
                <div className="flex justify-between">
                  <span>Support: ${stock.support_price.toFixed(2)}</span>
                  <span>Resistance: ${stock.resistance_price.toFixed(2)}</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Expand indicator on hover */}
          <AnimatePresence>
            {isHovered && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="flex items-center justify-center text-xs text-muted-foreground"
              >
                <span>View Analysis</span>
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default AnimatedStockCard;
