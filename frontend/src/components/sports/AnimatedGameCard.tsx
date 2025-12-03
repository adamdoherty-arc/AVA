/**
 * Animated Game Card Component
 * Modern, real-time updating game card with animations
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  Zap,
  Clock,
  Radio,
  ChevronRight,
  Brain,
} from 'lucide-react';

export interface GameData {
  game_id: string;
  sport: string;
  home_team: string;
  away_team: string;
  home_abbr: string;
  away_abbr: string;
  home_score: number;
  away_score: number;
  home_logo?: string;
  away_logo?: string;
  quarter?: number;
  period?: number;
  clock?: string;
  time_remaining?: string;
  status: string;
  status_detail?: string;
  is_live: boolean;
  is_completed?: boolean;
  possession?: string;
  is_red_zone?: boolean;
  down_distance?: string;
  venue?: string;
  tv?: string;
  // AI Prediction data
  ai_home_prob?: number;
  ai_confidence?: string;
  ai_edge?: number;
  ai_recommendation?: string;
  // Odds data
  home_odds?: number;
  away_odds?: number;
  spread?: number;
  total?: number;
  // Movement
  odds_movement?: {
    home_change: number;
    away_change: number;
  };
}

interface AnimatedGameCardProps {
  game: GameData;
  onClick?: (game: GameData) => void;
  showAIPrediction?: boolean;
  compact?: boolean;
  className?: string;
}

// Animated score display with flip effect
const AnimatedScore: React.FC<{ score: number; isScoring?: boolean }> = ({
  score,
  isScoring,
}) => {
  const controls = useAnimation();
  const prevScoreRef = useRef(score);

  useEffect(() => {
    if (prevScoreRef.current !== score) {
      controls.start({
        scale: [1, 1.3, 1],
        color: ['inherit', '#22c55e', 'inherit'],
        transition: { duration: 0.5 },
      });
      prevScoreRef.current = score;
    }
  }, [score, controls]);

  return (
    <motion.span
      animate={controls}
      className={cn(
        'text-3xl font-bold tabular-nums',
        isScoring && 'text-green-500'
      )}
    >
      {score}
    </motion.span>
  );
};

// Live indicator with pulse animation
const LiveIndicator: React.FC<{ isLive: boolean }> = ({ isLive }) => {
  if (!isLive) return null;

  return (
    <div className="flex items-center gap-1.5">
      <motion.div
        className="w-2 h-2 rounded-full bg-red-500"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [1, 0.7, 1],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      <span className="text-xs font-semibold text-red-500 uppercase">Live</span>
    </div>
  );
};

// AI Confidence meter with gradient
const AIConfidenceMeter: React.FC<{
  probability: number;
  confidence: string;
  edge?: number;
}> = ({ probability, confidence, edge }) => {
  const probPercent = probability * 100;

  // Color based on edge/confidence
  const getColor = () => {
    if (edge && edge > 0.05) return 'from-green-500 to-emerald-400';
    if (edge && edge > 0.02) return 'from-yellow-500 to-amber-400';
    return 'from-blue-500 to-cyan-400';
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1 text-muted-foreground">
          <Brain className="w-3 h-3" />
          AI Confidence
        </span>
        <span className="font-semibold">{probPercent.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <motion.div
          className={cn('h-full rounded-full bg-gradient-to-r', getColor())}
          initial={{ width: 0 }}
          animate={{ width: `${probPercent}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      {edge !== undefined && edge > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-1"
        >
          <Zap className="w-3 h-3 text-yellow-500" />
          <span className="text-xs text-yellow-500 font-medium">
            +{(edge * 100).toFixed(1)}% Edge
          </span>
        </motion.div>
      )}
    </div>
  );
};

// Odds with movement indicator
const OddsDisplay: React.FC<{
  homeOdds?: number;
  awayOdds?: number;
  movement?: { home_change: number; away_change: number };
}> = ({ homeOdds, awayOdds, movement }) => {
  const formatOdds = (odds?: number) => {
    if (odds === undefined) return '-';
    return odds > 0 ? `+${odds}` : odds.toString();
  };

  const getMovementIcon = (change?: number) => {
    if (!change) return null;
    if (change > 0.02) return <TrendingUp className="w-3 h-3 text-green-500" />;
    if (change < -0.02) return <TrendingDown className="w-3 h-3 text-red-500" />;
    return <Minus className="w-3 h-3 text-muted-foreground" />;
  };

  return (
    <div className="grid grid-cols-2 gap-2 text-xs">
      <div className="flex items-center justify-center gap-1 bg-muted/50 rounded px-2 py-1">
        <span className="font-mono">{formatOdds(homeOdds)}</span>
        {movement && getMovementIcon(movement.home_change)}
      </div>
      <div className="flex items-center justify-center gap-1 bg-muted/50 rounded px-2 py-1">
        <span className="font-mono">{formatOdds(awayOdds)}</span>
        {movement && getMovementIcon(movement.away_change)}
      </div>
    </div>
  );
};

// Red zone indicator with urgency animation
const RedZoneIndicator: React.FC = () => (
  <motion.div
    className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-orange-500 to-red-500"
    animate={{
      backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
    }}
    transition={{
      duration: 2,
      repeat: Infinity,
      ease: 'linear',
    }}
    style={{ backgroundSize: '200% 100%' }}
  />
);

// Main component
export const AnimatedGameCard: React.FC<AnimatedGameCardProps> = ({
  game,
  onClick,
  showAIPrediction = true,
  compact = false,
  className,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [prevScores, setPrevScores] = useState({
    home: game.home_score,
    away: game.away_score,
  });
  const [isScoring, setIsScoring] = useState<'home' | 'away' | null>(null);

  // Detect score changes
  useEffect(() => {
    if (game.home_score !== prevScores.home) {
      setIsScoring('home');
      setTimeout(() => setIsScoring(null), 2000);
    }
    if (game.away_score !== prevScores.away) {
      setIsScoring('away');
      setTimeout(() => setIsScoring(null), 2000);
    }
    setPrevScores({ home: game.home_score, away: game.away_score });
  }, [game.home_score, game.away_score, prevScores]);

  // Format game time/status
  const gameStatus = useMemo(() => {
    if (game.is_completed) return 'Final';
    if (game.is_live) {
      const period = game.quarter || game.period || 1;
      const periodLabel = game.sport === 'NBA' || game.sport === 'NCAAB' ? 'Q' : 'Q';
      return `${periodLabel}${period} ${game.clock || game.time_remaining || ''}`;
    }
    return game.status_detail || game.status || 'Scheduled';
  }, [game]);

  // Determine possession indicator
  const possessionIndicator = game.possession === 'home' ? '◀' : game.possession === 'away' ? '▶' : '';

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
          game.is_live && 'ring-1 ring-red-500/30',
          isScoring && 'ring-2 ring-green-500'
        )}
        onClick={() => onClick?.(game)}
      >
        {/* Red zone indicator */}
        {game.is_red_zone && <RedZoneIndicator />}

        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <Badge variant="outline" className="text-xs">
              {game.sport}
            </Badge>
            <div className="flex items-center gap-2">
              <LiveIndicator isLive={game.is_live} />
              {game.tv && (
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <Radio className="w-3 h-3" />
                  {game.tv}
                </span>
              )}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Teams and Scores */}
          <div className="space-y-3">
            {/* Away Team */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {game.away_logo && (
                  <img
                    src={game.away_logo}
                    alt={game.away_team}
                    className="w-8 h-8 object-contain"
                  />
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{game.away_team}</span>
                    {game.possession === 'away' && (
                      <motion.span
                        animate={{ opacity: [1, 0.5, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="text-yellow-500"
                      >
                        ●
                      </motion.span>
                    )}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {game.away_abbr}
                  </span>
                </div>
              </div>
              <AnimatedScore
                score={game.away_score}
                isScoring={isScoring === 'away'}
              />
            </div>

            {/* Home Team */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {game.home_logo && (
                  <img
                    src={game.home_logo}
                    alt={game.home_team}
                    className="w-8 h-8 object-contain"
                  />
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{game.home_team}</span>
                    {game.possession === 'home' && (
                      <motion.span
                        animate={{ opacity: [1, 0.5, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="text-yellow-500"
                      >
                        ●
                      </motion.span>
                    )}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {game.home_abbr}
                  </span>
                </div>
              </div>
              <AnimatedScore
                score={game.home_score}
                isScoring={isScoring === 'home'}
              />
            </div>
          </div>

          {/* Game Status */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span>{gameStatus}</span>
            </div>
            {game.down_distance && (
              <span className="text-xs text-muted-foreground">
                {game.down_distance}
              </span>
            )}
          </div>

          {/* Odds Display */}
          {(game.home_odds || game.away_odds) && !compact && (
            <OddsDisplay
              homeOdds={game.home_odds}
              awayOdds={game.away_odds}
              movement={game.odds_movement}
            />
          )}

          {/* AI Prediction */}
          {showAIPrediction && game.ai_home_prob !== undefined && !compact && (
            <AIConfidenceMeter
              probability={game.ai_home_prob}
              confidence={game.ai_confidence || 'medium'}
              edge={game.ai_edge}
            />
          )}

          {/* AI Recommendation Badge */}
          {game.ai_recommendation && (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
              >
                <Badge
                  variant={
                    game.ai_recommendation.toLowerCase().includes('bet')
                      ? 'default'
                      : 'secondary'
                  }
                  className="w-full justify-center gap-1"
                >
                  <Brain className="w-3 h-3" />
                  {game.ai_recommendation}
                </Badge>
              </motion.div>
            </AnimatePresence>
          )}

          {/* Expand indicator on hover */}
          <AnimatePresence>
            {isHovered && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="flex items-center justify-center text-xs text-muted-foreground"
              >
                <span>View Details</span>
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default AnimatedGameCard;
