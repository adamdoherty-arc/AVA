/**
 * AIPickCard Component - World-Class Edition (Tailwind CSS)
 *
 * Modern, animated CSP recommendation card with:
 * - Framer Motion animations (stagger, spring, gestures)
 * - Ensemble AI voting indicators (DeepSeek + Qwen consensus)
 * - Monte Carlo confidence intervals visualization
 * - Greeks probability display
 * - Liquidity score badges
 * - Expandable detailed analysis
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronDown, ChevronUp, TrendingUp, CheckCircle,
  AlertTriangle, XCircle, Info, Gauge, Brain, BarChart3,
  ThumbsUp, ThumbsDown, Vote, Zap, Activity
} from 'lucide-react';

// Types for enhanced AI Pick data
export interface ConfidenceInterval {
  lower_bound: number;
  median: number;
  upper_bound: number;
  std_dev: number;
  range_width?: number;
}

export interface EnsembleVote {
  deepseek_vote: 'strong_buy' | 'buy' | 'hold' | 'avoid';
  qwen_vote: 'strong_buy' | 'buy' | 'hold' | 'avoid';
  consensus: 'strong_buy' | 'buy' | 'hold' | 'avoid';
  agreement_score: number;
}

export interface GreeksSnapshot {
  delta?: number;
  gamma?: number;
  theta?: number;
  vega?: number;
  iv?: number;
  probability_of_profit?: number;
}

export interface AIPick {
  symbol: string;
  strike: number;
  expiration: string;
  dte: number;
  premium: number;
  premium_pct: number;
  monthly_return: number;
  annual_return: number;
  mcdm_score: number;
  ai_score: number;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
  reasoning: string;
  key_factors: string[];
  concerns: string[];

  // Enhanced data
  greeks?: GreeksSnapshot;
  stock_price?: number;
  bid?: number;
  ask?: number;
  volume?: number;
  open_interest?: number;
  premium_scenarios?: ConfidenceInterval;
  ensemble_vote?: EnsembleVote;
  liquidity_score?: 'excellent' | 'good' | 'fair' | 'poor' | 'unknown';
  bid_ask_spread_pct?: number;

  // Legacy fields (for backwards compatibility)
  delta?: number;
  iv?: number;
  theta?: number;
}

interface AIPickCardProps {
  pick: AIPick;
  rank: number;
  index?: number;
}

// Animation variants - using proper Framer Motion types
const cardVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.4,
      ease: 'easeOut' as const,
    },
  },
  hover: {
    y: -4,
    transition: { duration: 0.2 },
  },
  tap: { scale: 0.98 },
};

const expandVariants = {
  hidden: { opacity: 0, height: 0 },
  visible: {
    opacity: 1,
    height: 'auto',
    transition: {
      duration: 0.3,
      ease: 'easeOut' as const,
    },
  },
};

// Helper functions
const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'low':
      return { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' };
    case 'medium':
      return { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' };
    case 'high':
      return { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' };
    default:
      return { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' };
  }
};

const getRiskIcon = (risk: string) => {
  switch (risk) {
    case 'low':
      return <CheckCircle className="w-3.5 h-3.5" />;
    case 'medium':
      return <AlertTriangle className="w-3.5 h-3.5" />;
    case 'high':
      return <XCircle className="w-3.5 h-3.5" />;
    default:
      return <Info className="w-3.5 h-3.5" />;
  }
};

const getScoreColor = (score: number) => {
  if (score >= 80) return 'text-green-400';
  if (score >= 65) return 'text-lime-400';
  if (score >= 50) return 'text-amber-400';
  return 'text-red-400';
};

const getScoreBgColor = (score: number) => {
  if (score >= 80) return 'from-green-500/20 to-green-500/5';
  if (score >= 65) return 'from-lime-500/20 to-lime-500/5';
  if (score >= 50) return 'from-amber-500/20 to-amber-500/5';
  return 'from-red-500/20 to-red-500/5';
};

const getVoteColor = (vote: string) => {
  switch (vote) {
    case 'strong_buy':
      return { bg: 'bg-emerald-500/20', text: 'text-emerald-400' };
    case 'buy':
      return { bg: 'bg-green-500/20', text: 'text-green-400' };
    case 'hold':
      return { bg: 'bg-amber-500/20', text: 'text-amber-400' };
    case 'avoid':
      return { bg: 'bg-red-500/20', text: 'text-red-400' };
    default:
      return { bg: 'bg-slate-500/20', text: 'text-slate-400' };
  }
};

const getVoteLabel = (vote: string) => {
  switch (vote) {
    case 'strong_buy':
      return 'Strong Buy';
    case 'buy':
      return 'Buy';
    case 'hold':
      return 'Hold';
    case 'avoid':
      return 'Avoid';
    default:
      return vote;
  }
};

const getLiquidityColor = (liquidity: string) => {
  switch (liquidity) {
    case 'excellent':
      return { bg: 'bg-emerald-500/20', text: 'text-emerald-400' };
    case 'good':
      return { bg: 'bg-green-500/20', text: 'text-green-400' };
    case 'fair':
      return { bg: 'bg-amber-500/20', text: 'text-amber-400' };
    case 'poor':
      return { bg: 'bg-red-500/20', text: 'text-red-400' };
    default:
      return { bg: 'bg-slate-500/20', text: 'text-slate-400' };
  }
};

export const AIPickCard: React.FC<AIPickCardProps> = ({ pick, rank, index = 0 }) => {
  const [expanded, setExpanded] = useState(false);

  // Get delta from greeks or legacy field
  const delta = pick.greeks?.delta ?? pick.delta;
  const iv = pick.greeks?.iv ?? pick.iv;
  const theta = pick.greeks?.theta ?? pick.theta;
  const probabilityOfProfit = pick.greeks?.probability_of_profit;
  const riskColors = getRiskColor(pick.risk_level);

  return (
    <motion.div
      custom={index}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover="hover"
      whileTap="tap"
      layout
      className="relative"
    >
      <div className={`
        relative overflow-hidden rounded-xl
        bg-gradient-to-br from-slate-800/95 to-slate-900/95
        backdrop-blur-sm border border-slate-700/50
        hover:border-slate-600/50 transition-all duration-300
        hover:shadow-xl hover:shadow-purple-500/10
      `}>
        {/* Top Accent Bar */}
        <div className={`h-1 bg-gradient-to-r ${getScoreBgColor(pick.ai_score)}`} />

        <div className="p-4">
          {/* Header Row */}
          <div className="flex justify-between items-start mb-3">
            <div className="flex items-center gap-3">
              {/* Rank Badge */}
              <motion.div
                animate={rank <= 3 ? { scale: [1, 1.05, 1] } : {}}
                transition={rank <= 3 ? { repeat: Infinity, duration: 2 } : {}}
                className={`
                  w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm
                  ${rank <= 3
                    ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-slate-900 shadow-lg shadow-amber-500/30'
                    : 'bg-slate-700/50 text-slate-400'
                  }
                `}
              >
                {rank}
              </motion.div>

              {/* Symbol & Details */}
              <div>
                <h4 className="text-lg font-bold text-white tracking-wide">
                  {pick.symbol}
                </h4>
                <p className="text-xs text-slate-400">
                  ${pick.strike} | {pick.expiration} | {pick.dte} DTE
                </p>
              </div>
            </div>

            {/* AI Score */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="text-right"
            >
              <div className={`text-2xl font-bold ${getScoreColor(pick.ai_score)}`}>
                {pick.ai_score}
              </div>
              <div className="text-xs text-slate-500">AI Score</div>
            </motion.div>
          </div>

          {/* Metrics Row */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {/* Premium % */}
            <span className="px-2 py-1 rounded-md bg-blue-500/20 text-blue-400 text-xs font-semibold">
              {pick.premium_pct.toFixed(2)}%
            </span>

            {/* Annual Return */}
            <span className="px-2 py-1 rounded-md bg-green-500/15 text-green-400 text-xs font-semibold flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              {pick.annual_return.toFixed(1)}% APY
            </span>

            {/* Risk Level */}
            <span className={`px-2 py-1 rounded-md ${riskColors.bg} ${riskColors.text} text-xs font-semibold flex items-center gap-1`}>
              {getRiskIcon(pick.risk_level)}
              {pick.risk_level.toUpperCase()}
            </span>

            {/* Liquidity Score */}
            {pick.liquidity_score && pick.liquidity_score !== 'unknown' && (
              <span className={`px-2 py-1 rounded-md ${getLiquidityColor(pick.liquidity_score).bg} ${getLiquidityColor(pick.liquidity_score).text} text-xs font-semibold flex items-center gap-1`}>
                <Gauge className="w-3 h-3" />
                {pick.liquidity_score.toUpperCase()}
              </span>
            )}

            {/* Probability of Profit */}
            {probabilityOfProfit && (
              <span className="px-2 py-1 rounded-md bg-purple-500/20 text-purple-400 text-xs font-semibold flex items-center gap-1">
                <Activity className="w-3 h-3" />
                {probabilityOfProfit.toFixed(0)}% POP
              </span>
            )}
          </div>

          {/* Ensemble Voting Section */}
          {pick.ensemble_vote && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-3 p-2.5 bg-slate-800/60 rounded-lg border border-slate-700/50"
            >
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  Ensemble AI Consensus
                </span>
                <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400">
                  {pick.ensemble_vote.agreement_score.toFixed(0)}% Agreement
                </span>
              </div>

              <div className="flex items-center gap-3">
                {/* DeepSeek Vote */}
                <div className="flex items-center gap-1.5">
                  <BarChart3 className="w-3.5 h-3.5 text-blue-400" />
                  <span className={`text-xs font-medium ${getVoteColor(pick.ensemble_vote.deepseek_vote).text}`}>
                    {getVoteLabel(pick.ensemble_vote.deepseek_vote)}
                  </span>
                </div>

                {/* Qwen Vote */}
                <div className="flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-pink-400" />
                  <span className={`text-xs font-medium ${getVoteColor(pick.ensemble_vote.qwen_vote).text}`}>
                    {getVoteLabel(pick.ensemble_vote.qwen_vote)}
                  </span>
                </div>

                {/* Consensus */}
                <div className={`ml-auto px-2 py-1 rounded-md ${getVoteColor(pick.ensemble_vote.consensus).bg} flex items-center gap-1.5`}>
                  <Vote className={`w-3.5 h-3.5 ${getVoteColor(pick.ensemble_vote.consensus).text}`} />
                  <span className={`text-xs font-bold ${getVoteColor(pick.ensemble_vote.consensus).text}`}>
                    {getVoteLabel(pick.ensemble_vote.consensus)}
                  </span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Monte Carlo Confidence Interval */}
          {pick.premium_scenarios && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="mb-3 p-2.5 bg-slate-800/60 rounded-lg border border-slate-700/50"
            >
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
                Premium Scenarios (Monte Carlo)
              </div>
              <div className="flex justify-between text-center">
                <div>
                  <div className="text-xs text-red-400">Worst (5%)</div>
                  <div className="text-sm font-semibold text-slate-200">
                    ${pick.premium_scenarios.lower_bound.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-green-400">Expected</div>
                  <div className="text-sm font-bold text-green-400">
                    ${pick.premium_scenarios.median.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-blue-400">Best (95%)</div>
                  <div className="text-sm font-semibold text-slate-200">
                    ${pick.premium_scenarios.upper_bound.toFixed(2)}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Confidence Bar */}
          <div className="mb-3">
            <div className="flex justify-between mb-1">
              <span className="text-xs text-slate-400">Confidence</span>
              <span className="text-xs text-slate-400">{pick.confidence}%</span>
            </div>
            <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pick.confidence}%` }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className={`h-full rounded-full bg-gradient-to-r ${getScoreBgColor(pick.confidence)}`}
              />
            </div>
          </div>

          {/* Key Factors Preview */}
          {pick.key_factors.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {pick.key_factors.slice(0, 3).map((factor, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="px-2 py-0.5 rounded border border-green-500/30 text-green-400 text-[10px]"
                >
                  {factor}
                </motion.span>
              ))}
            </div>
          )}

          {/* Expand Button */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-center pt-2 mt-1 border-t border-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors"
          >
            {expanded ? (
              <ChevronUp className="w-5 h-5" />
            ) : (
              <ChevronDown className="w-5 h-5" />
            )}
          </button>

          {/* Expanded Details */}
          <AnimatePresence>
            {expanded && (
              <motion.div
                variants={expandVariants}
                initial="hidden"
                animate="visible"
                exit="hidden"
                className="pt-3"
              >
                {/* Reasoning */}
                <div className="mb-3">
                  <h5 className="text-xs font-semibold text-blue-400 mb-1">AI Reasoning</h5>
                  <p className="text-sm text-slate-300 p-2.5 bg-slate-800/50 rounded-lg border-l-2 border-blue-500">
                    {pick.reasoning}
                  </p>
                </div>

                {/* Greeks & Details Grid */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  {delta !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">Delta</div>
                      <div className="text-sm font-semibold text-slate-200">{delta.toFixed(3)}</div>
                    </div>
                  )}
                  {iv !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">IV</div>
                      <div className="text-sm font-semibold text-slate-200">{(iv * 100).toFixed(1)}%</div>
                    </div>
                  )}
                  {pick.stock_price !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">Stock Price</div>
                      <div className="text-sm font-semibold text-slate-200">${pick.stock_price.toFixed(2)}</div>
                    </div>
                  )}
                  {pick.bid !== undefined && pick.ask !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">Bid/Ask</div>
                      <div className="text-sm font-semibold text-slate-200">
                        ${pick.bid.toFixed(2)} / ${pick.ask.toFixed(2)}
                      </div>
                    </div>
                  )}
                  {pick.volume !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">Volume</div>
                      <div className="text-sm font-semibold text-slate-200">{pick.volume.toLocaleString()}</div>
                    </div>
                  )}
                  {pick.open_interest !== undefined && (
                    <div className="text-center">
                      <div className="text-xs text-slate-500">Open Interest</div>
                      <div className="text-sm font-semibold text-slate-200">{pick.open_interest.toLocaleString()}</div>
                    </div>
                  )}
                </div>

                {/* Concerns */}
                {pick.concerns.length > 0 && (
                  <div>
                    <h5 className="text-xs font-semibold text-amber-400 mb-1">Potential Concerns</h5>
                    <div className="space-y-1">
                      {pick.concerns.map((concern, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1 }}
                          className="flex items-center gap-1.5 text-xs text-amber-300"
                        >
                          <AlertTriangle className="w-3 h-3" />
                          {concern}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* MCDM Score */}
                <div className="mt-3 pt-2 border-t border-slate-700/50">
                  <span className="text-xs text-slate-500">
                    MCDM Score: {pick.mcdm_score}/100
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

export default AIPickCard;
