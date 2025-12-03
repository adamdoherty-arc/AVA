/**
 * AI Prediction Panel Component
 * Real-time streaming AI analysis display with modern animations
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Brain,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  Sparkles,
  Target,
  BarChart3,
  X,
} from 'lucide-react';
import { useStreamingPrediction } from '@/hooks/useStreamingPrediction';
import { useBetSlipStore } from '@/store/betSlipStore';

interface AIPredictionPanelProps {
  gameId: string;
  sport: string;
  homeTeam: string;
  awayTeam: string;
  className?: string;
  onClose?: () => void;
}

// Step indicator with animation
const StepIndicator: React.FC<{
  label: string;
  isActive: boolean;
  isComplete: boolean;
}> = ({ label, isActive, isComplete }) => (
  <motion.div
    className={cn(
      'flex items-center gap-2 text-xs px-2 py-1 rounded',
      isActive && 'bg-blue-500/10 text-blue-500',
      isComplete && 'bg-green-500/10 text-green-500',
      !isActive && !isComplete && 'text-muted-foreground'
    )}
    animate={isActive ? { scale: [1, 1.02, 1] } : {}}
    transition={{ duration: 0.5, repeat: isActive ? Infinity : 0 }}
  >
    {isComplete ? (
      <CheckCircle className="w-3 h-3" />
    ) : isActive ? (
      <Loader2 className="w-3 h-3 animate-spin" />
    ) : (
      <div className="w-3 h-3 rounded-full border" />
    )}
    <span>{label}</span>
  </motion.div>
);

// Animated typing effect for reasoning
const TypewriterText: React.FC<{
  text: string;
  isStreaming: boolean;
}> = ({ text, isStreaming }) => (
  <div className="text-sm text-muted-foreground whitespace-pre-wrap">
    {text}
    {isStreaming && (
      <motion.span
        animate={{ opacity: [1, 0, 1] }}
        transition={{ duration: 0.8, repeat: Infinity }}
        className="inline-block w-2 h-4 bg-primary ml-0.5"
      />
    )}
  </div>
);

// Factor chip with impact indicator
const FactorChip: React.FC<{
  factor: string;
  impact: string;
  description: string;
}> = ({ factor, impact, description }) => {
  const isPositive = impact.startsWith('+');
  const isNegative = impact.startsWith('-');

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={cn(
        'flex items-center justify-between p-2 rounded-lg',
        'bg-muted/50 hover:bg-muted transition-colors'
      )}
    >
      <div className="flex items-center gap-2">
        {isPositive && <TrendingUp className="w-4 h-4 text-green-500" />}
        {isNegative && <TrendingDown className="w-4 h-4 text-red-500" />}
        {!isPositive && !isNegative && <BarChart3 className="w-4 h-4 text-muted-foreground" />}
        <div>
          <p className="text-sm font-medium">{factor}</p>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </div>
      <Badge
        variant="outline"
        className={cn(
          'font-mono',
          isPositive && 'text-green-500 border-green-500/30',
          isNegative && 'text-red-500 border-red-500/30'
        )}
      >
        {impact}
      </Badge>
    </motion.div>
  );
};

// Probability gauge
const ProbabilityGauge: React.FC<{
  homeTeam: string;
  awayTeam: string;
  homeProb: number;
}> = ({ homeTeam, awayTeam, homeProb }) => {
  const awayProb = 1 - homeProb;
  const homePct = homeProb * 100;
  const awayPct = awayProb * 100;

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="font-medium">{awayTeam}</span>
        <span className="font-medium">{homeTeam}</span>
      </div>
      <div className="relative h-8 rounded-lg overflow-hidden bg-muted flex">
        <motion.div
          className="h-full bg-gradient-to-r from-blue-600 to-blue-500 flex items-center justify-end pr-2"
          initial={{ width: 0 }}
          animate={{ width: `${awayPct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <span className="text-xs font-bold text-white">
            {awayPct.toFixed(1)}%
          </span>
        </motion.div>
        <motion.div
          className="h-full bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-start pl-2"
          initial={{ width: 0 }}
          animate={{ width: `${homePct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <span className="text-xs font-bold text-white">
            {homePct.toFixed(1)}%
          </span>
        </motion.div>
      </div>
    </div>
  );
};

// Recommendation card
const RecommendationCard: React.FC<{
  recommendation: any;
}> = ({ recommendation }) => {
  if (!recommendation) return null;

  const getActionColor = () => {
    switch (recommendation.action) {
      case 'STRONG BET':
        return 'bg-green-500/10 border-green-500/30 text-green-500';
      case 'LEAN':
        return 'bg-yellow-500/10 border-yellow-500/30 text-yellow-500';
      default:
        return 'bg-red-500/10 border-red-500/30 text-red-500';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn('p-4 rounded-lg border', getActionColor())}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5" />
          <span className="font-bold text-lg">{recommendation.action}</span>
        </div>
        <Badge variant="outline" className="font-mono">
          {recommendation.edge > 0 ? '+' : ''}{recommendation.edge}% Edge
        </Badge>
      </div>
      <p className="text-sm mb-2">
        Take <span className="font-bold">{recommendation.side}</span>
      </p>
      <div className="flex items-center justify-between text-xs opacity-80">
        <span>Suggested: {recommendation.suggestedBetSize}</span>
        <span>Kelly: {(recommendation.kellyFraction * 100).toFixed(1)}%</span>
      </div>
    </motion.div>
  );
};

export const AIPredictionPanel: React.FC<AIPredictionPanelProps> = ({
  gameId,
  sport,
  homeTeam,
  awayTeam,
  className,
  onClose,
}) => {
  const {
    isLoading,
    isModelLoading,
    isDataFetching,
    isStreamingReasoning,
    currentStep,
    progress,
    prediction,
    factors,
    reasoning,
    recommendation,
    error,
    startPrediction,
    cancelPrediction,
    reset,
  } = useStreamingPrediction();

  const addLeg = useBetSlipStore((state) => state.addLeg);

  // Auto-start prediction on mount
  React.useEffect(() => {
    startPrediction(gameId, sport, true);
    return () => cancelPrediction();
  }, [gameId, sport]);

  const handleAddToBetSlip = () => {
    if (!prediction || !recommendation) return;

    addLeg({
      gameId,
      sport,
      homeTeam: prediction.homeTeam,
      awayTeam: prediction.awayTeam,
      betType: 'moneyline',
      selection: recommendation.side === prediction.homeTeam ? 'home' : 'away',
      odds: -110, // Default odds, would be fetched from actual odds
      gameTime: new Date(),
      aiProbability: prediction.homeWinProbability,
      aiEdge: recommendation.edge / 100,
      aiConfidence: prediction.confidence,
      aiReasoning: reasoning,
    });
  };

  return (
    <Card className={cn('overflow-hidden', className)}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Brain className="h-5 w-5 text-purple-500" />
            AI Analysis
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => {
                reset();
                startPrediction(gameId, sport, true);
              }}
              disabled={isLoading}
            >
              <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
            </Button>
            {onClose && (
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <Progress value={progress} className="h-1 mt-2" />
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Analysis Steps */}
        <div className="flex flex-wrap gap-2">
          <StepIndicator
            label="Models"
            isActive={isModelLoading}
            isComplete={!!currentStep && ['data_fetching', 'prediction', 'factors', 'reasoning_token', 'reasoning_complete', 'recommendation', 'complete'].includes(currentStep)}
          />
          <StepIndicator
            label="Data"
            isActive={isDataFetching}
            isComplete={!!currentStep && ['prediction', 'factors', 'reasoning_token', 'reasoning_complete', 'recommendation', 'complete'].includes(currentStep)}
          />
          <StepIndicator
            label="Predict"
            isActive={currentStep === 'prediction'}
            isComplete={!!currentStep && ['factors', 'reasoning_token', 'reasoning_complete', 'recommendation', 'complete'].includes(currentStep)}
          />
          <StepIndicator
            label="Analyze"
            isActive={isStreamingReasoning}
            isComplete={!!currentStep && ['recommendation', 'complete'].includes(currentStep)}
          />
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-2 text-red-500 bg-red-500/10 rounded-lg p-3">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* Prediction Result */}
        <AnimatePresence mode="wait">
          {prediction && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Probability Gauge */}
              <ProbabilityGauge
                homeTeam={prediction.homeTeam}
                awayTeam={prediction.awayTeam}
                homeProb={prediction.homeWinProbability}
              />

              {/* Confidence Badge */}
              <div className="flex items-center justify-center gap-2">
                <Target className="w-4 h-4 text-purple-500" />
                <span className="text-sm">
                  <span className="font-medium capitalize">{prediction.confidence}</span>
                  {' '}Confidence
                </span>
                <Badge variant="outline" className="text-xs">
                  {prediction.modelVersion}
                </Badge>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Factors */}
        <AnimatePresence>
          {factors.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-2"
            >
              <h4 className="text-sm font-medium flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Key Factors
              </h4>
              <div className="space-y-1">
                {factors.map((factor, idx) => (
                  <FactorChip
                    key={idx}
                    factor={factor.factor}
                    impact={factor.impact}
                    description={factor.description}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Streaming Reasoning */}
        {(reasoning || isStreamingReasoning) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-2"
          >
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-500" />
              AI Reasoning
              {isStreamingReasoning && (
                <Badge variant="secondary" className="text-xs">
                  Streaming...
                </Badge>
              )}
            </h4>
            <ScrollArea className="h-32 rounded-lg border p-3 bg-muted/30">
              <TypewriterText text={reasoning} isStreaming={isStreamingReasoning} />
            </ScrollArea>
          </motion.div>
        )}

        {/* Recommendation */}
        {recommendation && (
          <RecommendationCard recommendation={recommendation} />
        )}

        {/* Add to Bet Slip Button */}
        {recommendation && recommendation.action !== 'PASS' && (
          <Button
            className="w-full"
            onClick={handleAddToBetSlip}
          >
            <Zap className="w-4 h-4 mr-2" />
            Add to Bet Slip
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

export default AIPredictionPanel;
