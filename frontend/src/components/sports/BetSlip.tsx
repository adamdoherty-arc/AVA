/**
 * BetSlip Component
 * Modern, animated bet slip with AI analysis integration
 */

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  X,
  Trash2,
  ChevronDown,
  ChevronUp,
  Brain,
  Zap,
  AlertTriangle,
  TrendingUp,
  DollarSign,
  Info,
  Sparkles,
  Loader2,
  Check,
  RefreshCw,
  History,
} from 'lucide-react';
import { useBetSlipStore, selectTotalOdds, selectHasValueBet } from '@/store/betSlipStore';

// Format American odds for display
const formatOdds = (odds: number): string => {
  return odds > 0 ? `+${odds}` : odds.toString();
};

// Format currency
const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(amount);
};

// Individual Bet Leg Component
const BetLegCard: React.FC<{
  leg: any;
  onRemove: () => void;
  onAnalyze?: () => void;
}> = ({ leg, onRemove, onAnalyze }) => {
  const hasEdge = leg.aiEdge && leg.aiEdge > 0.02;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="relative"
    >
      <Card className={cn('relative', hasEdge && 'ring-1 ring-green-500/50')}>
        <CardContent className="p-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">
                  {leg.sport}
                </Badge>
                {hasEdge && (
                  <Badge className="bg-green-500/10 text-green-500 text-xs">
                    <Zap className="w-3 h-3 mr-1" />
                    +{(leg.aiEdge * 100).toFixed(1)}% Edge
                  </Badge>
                )}
              </div>

              <div className="mt-1">
                <p className="text-sm font-medium">
                  {leg.awayTeam} @ {leg.homeTeam}
                </p>
                <p className="text-xs text-muted-foreground">
                  {leg.betType === 'moneyline' && `${leg.selection === 'home' ? leg.homeTeam : leg.awayTeam} ML`}
                  {leg.betType === 'spread' && `${leg.selection === 'home' ? leg.homeTeam : leg.awayTeam} ${leg.line > 0 ? '+' : ''}${leg.line}`}
                  {leg.betType === 'total_over' && `Over ${leg.line}`}
                  {leg.betType === 'total_under' && `Under ${leg.line}`}
                </p>
              </div>

              <div className="flex items-center gap-2 mt-2">
                <span className="text-lg font-bold font-mono">
                  {formatOdds(leg.odds)}
                </span>
                <span className="text-xs text-muted-foreground">
                  ({(leg.impliedProbability * 100).toFixed(1)}%)
                </span>
              </div>

              {leg.aiProbability && (
                <div className="flex items-center gap-1 mt-1">
                  <Brain className="w-3 h-3 text-purple-500" />
                  <span className="text-xs text-purple-500">
                    AI: {(leg.aiProbability * 100).toFixed(1)}% ({leg.aiConfidence})
                  </span>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={onRemove}
              >
                <X className="h-4 w-4" />
              </Button>
              {onAnalyze && !leg.aiProbability && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={onAnalyze}
                >
                  <Brain className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// Parlay Analysis Display
const ParlayAnalysisDisplay: React.FC<{
  analysis: any;
  isLoading: boolean;
}> = ({ analysis, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Analyzing parlay...</span>
      </div>
    );
  }

  if (!analysis) return null;

  const getRecommendationColor = () => {
    switch (analysis.recommendation) {
      case 'BET':
        return 'bg-green-500/10 text-green-500';
      case 'CONSIDER':
        return 'bg-yellow-500/10 text-yellow-500';
      default:
        return 'bg-red-500/10 text-red-500';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-3"
    >
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">AI Analysis</span>
        <Badge className={cn('text-xs', getRecommendationColor())}>
          <Sparkles className="w-3 h-3 mr-1" />
          {analysis.recommendation}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-muted/50 rounded p-2">
          <span className="text-muted-foreground">True Prob</span>
          <p className="font-mono font-bold">
            {(analysis.adjustedProbability * 100).toFixed(2)}%
          </p>
        </div>
        <div className="bg-muted/50 rounded p-2">
          <span className="text-muted-foreground">Expected Value</span>
          <p className={cn('font-mono font-bold', analysis.expectedValue > 0 ? 'text-green-500' : 'text-red-500')}>
            {analysis.expectedValue > 0 ? '+' : ''}{(analysis.expectedValue * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Kelly Suggests:</span>
        <span className="font-medium">{analysis.suggestedBetSize}</span>
      </div>

      {analysis.correlationWarnings && analysis.correlationWarnings.length > 0 && (
        <div className="flex items-start gap-2 text-xs text-yellow-500 bg-yellow-500/10 rounded p-2">
          <AlertTriangle className="w-3 h-3 mt-0.5" />
          <span>{analysis.correlationWarnings[0]}</span>
        </div>
      )}
    </motion.div>
  );
};

// Streaming Reasoning Display
const StreamingReasoning: React.FC<{
  reasoning: string;
  isStreaming: boolean;
}> = ({ reasoning, isStreaming }) => {
  if (!reasoning && !isStreaming) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="mt-3"
    >
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-4 h-4 text-purple-500" />
        <span className="text-sm font-medium">AI Reasoning</span>
        {isStreaming && <Loader2 className="w-3 h-3 animate-spin" />}
      </div>
      <ScrollArea className="h-24 rounded border p-2">
        <p className="text-xs text-muted-foreground whitespace-pre-wrap">
          {reasoning}
          {isStreaming && <span className="animate-pulse">|</span>}
        </p>
      </ScrollArea>
    </motion.div>
  );
};

// Main BetSlip Component
export const BetSlip: React.FC = () => {
  const {
    legs,
    stakeAmount,
    betMode,
    parlayAnalysis,
    isAnalyzing,
    streamingReasoning,
    isStreamingReasoning,
    isOpen,
    isMinimized,
    bankroll,
    riskLevel,
    recentBets,
    // Actions
    removeLeg,
    clearSlip,
    setStakeAmount,
    setBetMode,
    setRiskLevel,
    toggleSlip,
    minimizeSlip,
    expandSlip,
    analyzeParlay,
    placeBet,
    calculatePotentialPayout,
    calculateKellyBet,
  } = useBetSlipStore();

  const totalOdds = useBetSlipStore(selectTotalOdds);
  const hasValueBet = useBetSlipStore(selectHasValueBet);
  const potentialPayout = calculatePotentialPayout();
  const kellyBet = calculateKellyBet();

  const [activeTab, setActiveTab] = useState<'slip' | 'history'>('slip');
  const [isPlacing, setIsPlacing] = useState(false);

  // Handle bet placement
  const handlePlaceBet = async () => {
    setIsPlacing(true);
    const result = await placeBet();
    setIsPlacing(false);

    if (result.success) {
      // Show success notification (would integrate with toast system)
      console.log('Bet placed:', result.betId);
    }
  };

  // Don't render if empty and closed
  if (legs.length === 0 && !isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 300, opacity: 0 }}
          className="fixed right-4 bottom-4 w-80 z-50"
        >
          <Card className="shadow-2xl border-2">
            {/* Header */}
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <DollarSign className="h-5 w-5" />
                  Bet Slip
                  {legs.length > 0 && (
                    <Badge variant="secondary" className="ml-1">
                      {legs.length}
                    </Badge>
                  )}
                </CardTitle>
                <div className="flex items-center gap-1">
                  {legs.length > 0 && (
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={clearSlip}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={isMinimized ? expandSlip : minimizeSlip}
                  >
                    {isMinimized ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </Button>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={toggleSlip}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            <AnimatePresence mode="wait">
              {!isMinimized && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: 'auto' }}
                  exit={{ height: 0 }}
                  className="overflow-hidden"
                >
                  <CardContent className="pb-4">
                    <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'slip' | 'history')}>
                      <TabsList className="grid w-full grid-cols-2 mb-3">
                        <TabsTrigger value="slip" className="text-xs">
                          Bet Slip
                        </TabsTrigger>
                        <TabsTrigger value="history" className="text-xs">
                          <History className="w-3 h-3 mr-1" />
                          History
                        </TabsTrigger>
                      </TabsList>

                      <TabsContent value="slip" className="space-y-3">
                        {/* Mode Toggle */}
                        <div className="flex items-center gap-2">
                          <Button
                            variant={betMode === 'parlay' ? 'default' : 'outline'}
                            size="sm"
                            className="flex-1 text-xs"
                            onClick={() => setBetMode('parlay')}
                          >
                            Parlay
                          </Button>
                          <Button
                            variant={betMode === 'singles' ? 'default' : 'outline'}
                            size="sm"
                            className="flex-1 text-xs"
                            onClick={() => setBetMode('singles')}
                          >
                            Singles
                          </Button>
                        </div>

                        {/* Bet Legs */}
                        <ScrollArea className="max-h-48">
                          <div className="space-y-2">
                            <AnimatePresence>
                              {legs.map((leg) => (
                                <BetLegCard
                                  key={leg.id}
                                  leg={leg}
                                  onRemove={() => removeLeg(leg.id)}
                                />
                              ))}
                            </AnimatePresence>
                          </div>
                        </ScrollArea>

                        {legs.length === 0 ? (
                          <div className="text-center py-6 text-muted-foreground">
                            <DollarSign className="h-8 w-8 mx-auto mb-2 opacity-50" />
                            <p className="text-sm">Add selections to build your bet</p>
                          </div>
                        ) : (
                          <>
                            {/* Parlay Analysis */}
                            {betMode === 'parlay' && legs.length >= 2 && (
                              <>
                                <Separator />
                                <div className="flex items-center justify-between">
                                  <div>
                                    <p className="text-sm font-medium">
                                      {legs.length}-Leg Parlay
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                      Odds: {formatOdds(Math.round((totalOdds - 1) * 100))}
                                    </p>
                                  </div>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={analyzeParlay}
                                    disabled={isAnalyzing}
                                  >
                                    {isAnalyzing ? (
                                      <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                      <>
                                        <Brain className="h-4 w-4 mr-1" />
                                        Analyze
                                      </>
                                    )}
                                  </Button>
                                </div>

                                <ParlayAnalysisDisplay
                                  analysis={parlayAnalysis}
                                  isLoading={isAnalyzing}
                                />
                              </>
                            )}

                            {/* Streaming Reasoning */}
                            <StreamingReasoning
                              reasoning={streamingReasoning}
                              isStreaming={isStreamingReasoning}
                            />

                            <Separator />

                            {/* Stake Input */}
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-sm">Stake</span>
                                {kellyBet > 0 && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-6 text-xs text-purple-500"
                                    onClick={() => setStakeAmount(Math.round(kellyBet * 100) / 100)}
                                  >
                                    <Sparkles className="h-3 w-3 mr-1" />
                                    Use Kelly: {formatCurrency(kellyBet)}
                                  </Button>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                <Input
                                  type="number"
                                  value={stakeAmount}
                                  onChange={(e) => setStakeAmount(parseFloat(e.target.value) || 0)}
                                  className="text-right font-mono"
                                  min={0}
                                  step={5}
                                />
                              </div>
                              <div className="flex gap-1">
                                {[10, 25, 50, 100].map((amount) => (
                                  <Button
                                    key={amount}
                                    variant="outline"
                                    size="sm"
                                    className="flex-1 text-xs"
                                    onClick={() => setStakeAmount(amount)}
                                  >
                                    ${amount}
                                  </Button>
                                ))}
                              </div>
                            </div>

                            {/* Potential Payout */}
                            <div className="bg-muted/50 rounded-lg p-3">
                              <div className="flex items-center justify-between">
                                <span className="text-sm text-muted-foreground">To Win</span>
                                <span className="text-lg font-bold text-green-500">
                                  {formatCurrency(potentialPayout)}
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-xs text-muted-foreground">
                                <span>Total Payout</span>
                                <span>{formatCurrency(potentialPayout + stakeAmount)}</span>
                              </div>
                            </div>

                            {/* Value Bet Indicator */}
                            {hasValueBet && (
                              <div className="flex items-center gap-2 text-xs text-green-500 bg-green-500/10 rounded-lg p-2">
                                <TrendingUp className="h-4 w-4" />
                                <span>AI detected value opportunity in your slip!</span>
                              </div>
                            )}

                            {/* Place Bet Button */}
                            <Button
                              className="w-full"
                              size="lg"
                              onClick={handlePlaceBet}
                              disabled={isPlacing || stakeAmount <= 0}
                            >
                              {isPlacing ? (
                                <>
                                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                  Placing...
                                </>
                              ) : (
                                <>
                                  Place Bet - {formatCurrency(stakeAmount)}
                                </>
                              )}
                            </Button>
                          </>
                        )}
                      </TabsContent>

                      <TabsContent value="history">
                        <ScrollArea className="h-64">
                          <div className="space-y-2">
                            {recentBets.length === 0 ? (
                              <div className="text-center py-8 text-muted-foreground">
                                <History className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                <p className="text-sm">No betting history</p>
                              </div>
                            ) : (
                              recentBets.slice(0, 10).map((bet) => (
                                <Card key={bet.id} className="p-2">
                                  <div className="flex items-center justify-between">
                                    <div>
                                      <p className="text-xs font-medium">
                                        {bet.legs.length}-leg {bet.legs.length > 1 ? 'parlay' : 'single'}
                                      </p>
                                      <p className="text-xs text-muted-foreground">
                                        {new Date(bet.placedAt).toLocaleDateString()}
                                      </p>
                                    </div>
                                    <div className="text-right">
                                      <p className="text-sm font-mono">
                                        {formatCurrency(bet.stake)}
                                      </p>
                                      <Badge
                                        variant={
                                          bet.status === 'won' ? 'default' :
                                          bet.status === 'lost' ? 'destructive' : 'secondary'
                                        }
                                        className="text-xs"
                                      >
                                        {bet.status}
                                      </Badge>
                                    </div>
                                  </div>
                                </Card>
                              ))
                            )}
                          </div>
                        </ScrollArea>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Floating Action Button to open bet slip when closed
export const BetSlipFAB: React.FC = () => {
  const { legs, isOpen, toggleSlip } = useBetSlipStore();

  if (isOpen || legs.length === 0) return null;

  return (
    <motion.button
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      exit={{ scale: 0 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={cn(
        'fixed right-4 bottom-4 z-50',
        'w-14 h-14 rounded-full',
        'bg-primary text-primary-foreground',
        'shadow-lg flex items-center justify-center',
        'hover:shadow-xl transition-shadow'
      )}
      onClick={toggleSlip}
    >
      <div className="relative">
        <DollarSign className="h-6 w-6" />
        <Badge
          className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center text-xs"
        >
          {legs.length}
        </Badge>
      </div>
    </motion.button>
  );
};

export default BetSlip;
