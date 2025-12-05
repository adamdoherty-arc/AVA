/**
 * FeatureImportance Component - XAI Visualization
 *
 * Displays SHAP-like feature importance for AI recommendations.
 * Shows why a particular CSP was recommended with:
 * - Horizontal bar chart of feature importance
 * - Color-coded contribution direction
 * - Expandable explanations
 * - Animated transitions
 *
 * Created: 2025-12-04
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useExplainPick } from '../hooks/useAIPicks';

interface FeatureImportance {
  feature: string;
  importance: number;
  contribution: 'positive' | 'negative' | 'neutral';
  explanation: string;
}

interface FeatureImportanceProps {
  symbol: string;
  onClose?: () => void;
}

const getContributionColor = (contribution: string): string => {
  switch (contribution) {
    case 'positive':
      return '#4ade80';
    case 'negative':
      return '#f87171';
    case 'neutral':
      return '#fbbf24';
    default:
      return '#94a3b8';
  }
};

const getContributionIcon = (contribution: string): string => {
  switch (contribution) {
    case 'positive':
      return '+';
    case 'negative':
      return '-';
    case 'neutral':
      return '~';
    default:
      return '?';
  }
};

const formatFeatureName = (feature: string): string => {
  return feature
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export const FeatureImportanceChart: React.FC<FeatureImportanceProps> = ({
  symbol,
  onClose,
}) => {
  const [expandedFeature, setExpandedFeature] = useState<string | null>(null);
  const { data, isLoading, error } = useExplainPick(symbol);

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-slate-800/90 backdrop-blur-sm rounded-xl p-6 border border-slate-700"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-slate-300">Analyzing {symbol} features...</span>
        </div>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-slate-700 rounded w-24 mb-2" />
              <div className="h-6 bg-slate-700/50 rounded" style={{ width: `${80 - i * 15}%` }} />
            </div>
          ))}
        </div>
      </motion.div>
    );
  }

  if (error || !data) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-800/90 backdrop-blur-sm rounded-xl p-6 border border-red-500/30"
      >
        <div className="text-red-400 flex items-center gap-2">
          <span>Failed to load feature analysis for {symbol}</span>
        </div>
      </motion.div>
    );
  }

  const features = data.features || [];
  const maxImportance = Math.max(...features.map(f => f.importance), 0.01);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.98 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="bg-gradient-to-br from-slate-800/95 to-slate-900/95 backdrop-blur-sm rounded-xl border border-slate-700 overflow-hidden"
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">
              AI Explainability: {symbol}
            </h3>
            <p className="text-sm text-slate-400">
              Feature importance analysis (XAI)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Score badge */}
          <div className="text-right">
            <div className="text-2xl font-bold text-white">{data.ai_score}</div>
            <div className="text-xs text-slate-400">AI Score</div>
          </div>

          {onClose && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Feature bars */}
      <div className="p-6 space-y-4">
        {features.map((feature, index) => (
          <motion.div
            key={feature.feature}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="group"
          >
            <div
              className="cursor-pointer"
              onClick={() => setExpandedFeature(
                expandedFeature === feature.feature ? null : feature.feature
              )}
            >
              {/* Label row */}
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span
                    className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                    style={{
                      backgroundColor: `${getContributionColor(feature.contribution)}20`,
                      color: getContributionColor(feature.contribution),
                    }}
                  >
                    {getContributionIcon(feature.contribution)}
                  </span>
                  <span className="text-sm font-medium text-slate-300">
                    {formatFeatureName(feature.feature)}
                  </span>
                </div>
                <span
                  className="text-sm font-semibold"
                  style={{ color: getContributionColor(feature.contribution) }}
                >
                  {(feature.importance * 100).toFixed(0)}%
                </span>
              </div>

              {/* Bar */}
              <div className="h-7 bg-slate-700/50 rounded-lg overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(feature.importance / maxImportance) * 100}%` }}
                  transition={{ duration: 0.5, delay: index * 0.1, ease: 'easeOut' }}
                  className="h-full rounded-lg flex items-center px-2"
                  style={{
                    backgroundColor: `${getContributionColor(feature.contribution)}40`,
                    borderLeft: `3px solid ${getContributionColor(feature.contribution)}`,
                  }}
                >
                  <span className="text-xs text-slate-200 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                    {feature.explanation.slice(0, 50)}...
                  </span>
                </motion.div>
              </div>
            </div>

            {/* Expanded explanation */}
            <AnimatePresence>
              {expandedFeature === feature.feature && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="mt-2 ml-7 p-3 bg-slate-800/50 rounded-lg border-l-2"
                  style={{ borderColor: getContributionColor(feature.contribution) }}
                >
                  <p className="text-sm text-slate-300">{feature.explanation}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Summary footer */}
      {data.summary && (
        <div className="px-6 py-4 bg-slate-800/50 border-t border-slate-700/50">
          <div className="flex items-start gap-3">
            <div className="p-1.5 bg-blue-500/20 rounded mt-0.5">
              <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <div className="text-xs font-medium text-slate-400 mb-1">SUMMARY</div>
              <p className="text-sm text-slate-300">{data.summary}</p>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="px-6 py-3 bg-slate-900/50 flex items-center justify-center gap-6 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-green-400/30 border border-green-400" />
          <span className="text-slate-400">Positive Factor</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-yellow-400/30 border border-yellow-400" />
          <span className="text-slate-400">Neutral Factor</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-400/30 border border-red-400" />
          <span className="text-slate-400">Negative Factor</span>
        </div>
      </div>
    </motion.div>
  );
};

// Modal wrapper
export const FeatureImportanceModal: React.FC<{
  symbol: string | null;
  isOpen: boolean;
  onClose: () => void;
}> = ({ symbol, isOpen, onClose }) => {
  if (!isOpen || !symbol) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.95, y: 20 }}
          className="w-full max-w-lg"
          onClick={e => e.stopPropagation()}
        >
          <FeatureImportanceChart symbol={symbol} onClose={onClose} />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default FeatureImportanceChart;
