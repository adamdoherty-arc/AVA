/**
 * useAIPicks - Modern React Hooks for AI CSP Recommendations
 *
 * Features:
 * - React Query with automatic refetching
 * - WebSocket real-time updates
 * - Optimistic updates for mutations
 * - SSE streaming support
 * - Feature importance/XAI
 * - Performance tracking
 * - Sentiment analysis
 *
 * Created: 2025-12-04
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useEffect, useCallback, useRef } from 'react';
import { axiosInstance as axios } from '../lib/axios';
import { WS_URL, BACKEND_URL } from '@/config/api';

// ============ Types ============

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
  greeks?: GreeksSnapshot;
  stock_price?: number;
  bid?: number;
  ask?: number;
  volume?: number;
  open_interest?: number;
  premium_scenarios?: ConfidenceInterval;
  ensemble_vote?: EnsembleVote;
  liquidity_score?: 'excellent' | 'good' | 'fair' | 'poor' | 'unknown';
}

export interface ConfidenceInterval {
  lower_bound: number;
  median: number;
  upper_bound: number;
  std_dev: number;
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

export interface AIPicksResponse {
  picks: AIPick[];
  market_context: string;
  market_regime?: string;
  csp_environment_score?: number;
  generated_at: string;
  model: string;
  total_scanned: number;
  processing_time_ms: number;
  from_cache?: boolean;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  contribution: 'positive' | 'negative' | 'neutral';
  explanation: string;
}

export interface PerformanceStats {
  total_recommendations: number;
  evaluated: number;
  wins: number;
  losses: number;
  win_rate: number;
  assignment_rate: number;
  average_ai_score: number;
  average_confidence: number;
  score_accuracy: number;
}

export interface SentimentData {
  symbol: string;
  score: number;
  signal: 'bullish' | 'bearish' | 'neutral';
  sources: Array<{
    source: string;
    score: number;
    mentions?: number;
  }>;
}

// ============ Fetch Functions ============

const fetchAIPicks = async (params: {
  minDte?: number;
  maxDte?: number;
  minPremiumPct?: number;
  forceRefresh?: boolean;
}): Promise<AIPicksResponse> => {
  const { data } = await axios.get('/scanner/ai-picks', {
    params: {
      min_dte: params.minDte ?? 7,
      max_dte: params.maxDte ?? 45,
      min_premium_pct: params.minPremiumPct ?? 0.5,
      force_refresh: params.forceRefresh ?? false,
    },
  });
  return data;
};

const fetchExplainPick = async (symbol: string): Promise<{
  symbol: string;
  features: FeatureImportance[];
  summary: string;
  ai_score: number;
  confidence: number;
}> => {
  const { data } = await axios.get(`/scanner/ai-picks/explain/${symbol}`);
  return data;
};

const fetchPerformanceStats = async (days: number = 30): Promise<PerformanceStats> => {
  const { data } = await axios.get('/scanner/ai-picks/performance', {
    params: { days },
  });
  return data;
};

const fetchSentiment = async (symbol: string): Promise<SentimentData> => {
  const { data } = await axios.get(`/scanner/ai-picks/sentiment/${symbol}`);
  return data;
};

const refreshPicks = async (params: {
  minDte?: number;
  maxDte?: number;
  minPremiumPct?: number;
}): Promise<AIPicksResponse> => {
  const { data } = await axios.post('/scanner/ai-picks/refresh', null, {
    params: {
      min_dte: params.minDte ?? 7,
      max_dte: params.maxDte ?? 45,
      min_premium_pct: params.minPremiumPct ?? 0.5,
    },
  });
  return data;
};

const recordOutcome = async (params: {
  symbol: string;
  expiration: string;
  wasProfitable: boolean;
  actualPremium?: number;
  wasAssigned?: boolean;
  profitLoss?: number;
}): Promise<{ status: string; message: string }> => {
  const { data } = await axios.post('/scanner/ai-picks/outcome', null, {
    params: {
      symbol: params.symbol,
      expiration: params.expiration,
      was_profitable: params.wasProfitable,
      actual_premium: params.actualPremium,
      was_assigned: params.wasAssigned ?? false,
      profit_loss: params.profitLoss,
    },
  });
  return data;
};

// ============ Main Hook ============

export function useAIPicks(params: {
  minDte?: number;
  maxDte?: number;
  minPremiumPct?: number;
  enabled?: boolean;
  refetchInterval?: number;
} = {}) {
  const queryClient = useQueryClient();

  // Main query for AI picks
  const picksQuery = useQuery({
    queryKey: ['ai-picks', params.minDte, params.maxDte, params.minPremiumPct],
    queryFn: () => fetchAIPicks(params),
    enabled: params.enabled !== false,
    refetchInterval: params.refetchInterval ?? 60000, // 1 minute default
    staleTime: 30000, // 30 seconds
  });

  // Refresh mutation with optimistic updates
  const refreshMutation = useMutation({
    mutationFn: refreshPicks,
    onMutate: async () => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['ai-picks'] });

      // Snapshot previous value
      const previousPicks = queryClient.getQueryData(['ai-picks', params.minDte, params.maxDte, params.minPremiumPct]);

      return { previousPicks };
    },
    onError: (err, variables, context) => {
      // Rollback on error
      if (context?.previousPicks) {
        queryClient.setQueryData(
          ['ai-picks', params.minDte, params.maxDte, params.minPremiumPct],
          context.previousPicks
        );
      }
    },
    onSettled: () => {
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: ['ai-picks'] });
    },
  });

  return {
    picks: picksQuery.data?.picks ?? [],
    marketContext: picksQuery.data?.market_context,
    marketRegime: picksQuery.data?.market_regime,
    cspEnvironmentScore: picksQuery.data?.csp_environment_score,
    generatedAt: picksQuery.data?.generated_at,
    totalScanned: picksQuery.data?.total_scanned,
    processingTime: picksQuery.data?.processing_time_ms,
    fromCache: picksQuery.data?.from_cache,

    isLoading: picksQuery.isLoading,
    isError: picksQuery.isError,
    error: picksQuery.error,
    isFetching: picksQuery.isFetching,
    isRefreshing: refreshMutation.isPending,

    refresh: () => refreshMutation.mutate(params),
    refetch: picksQuery.refetch,
  };
}

// ============ XAI Hook ============

export function useExplainPick(symbol: string | null) {
  return useQuery({
    queryKey: ['ai-pick-explain', symbol],
    queryFn: () => fetchExplainPick(symbol!),
    enabled: !!symbol,
    staleTime: 120000, // 2 minutes
  });
}

// ============ Performance Hook ============

export function useAIPerformance(days: number = 30) {
  return useQuery({
    queryKey: ['ai-performance', days],
    queryFn: () => fetchPerformanceStats(days),
    staleTime: 300000, // 5 minutes
    refetchInterval: 300000,
  });
}

// ============ Sentiment Hook ============

export function useSentiment(symbol: string | null) {
  return useQuery({
    queryKey: ['sentiment', symbol],
    queryFn: () => fetchSentiment(symbol!),
    enabled: !!symbol,
    staleTime: 60000, // 1 minute
  });
}

// ============ Outcome Recording Hook ============

export function useRecordOutcome() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: recordOutcome,
    onSuccess: () => {
      // Invalidate performance stats after recording
      queryClient.invalidateQueries({ queryKey: ['ai-performance'] });
    },
  });
}

// ============ WebSocket Hook ============

export function useAIPicksWebSocket(
  onUpdate: (picks: AIPick[]) => void,
  onProgress?: (step: string, message: string) => void
) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const connect = useCallback(() => {
    // Use centralized WS_URL config
    const wsUrl = `${WS_URL}/ws/ai-picks`;

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setConnectionError(null);
        console.log('[WebSocket] Connected to AI picks stream');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.event === 'picks_updated') {
            onUpdate(data.picks);
          } else if (data.event === 'progress' && onProgress) {
            onProgress(data.step, data.message);
          }
        } catch (e) {
          console.error('[WebSocket] Parse error:', e);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setConnectionError('WebSocket connection error');
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        console.log('[WebSocket] Disconnected');

        // Auto-reconnect after 5 seconds
        setTimeout(() => {
          if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
            connect();
          }
        }, 5000);
      };
    } catch (e) {
      setConnectionError('Failed to create WebSocket connection');
    }
  }, [onUpdate, onProgress]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    isConnected,
    connectionError,
    reconnect: connect,
    disconnect,
  };
}

// ============ SSE Streaming Hook ============

export function useAIPicksStream(onPick: (pick: AIPick, rank: number) => void) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const startStream = useCallback((params?: {
    minDte?: number;
    maxDte?: number;
    minPremiumPct?: number;
  }) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Use BACKEND_URL from centralized config
    const url = new URL('/api/scanner/ai-picks/stream', BACKEND_URL);
    if (params?.minDte) url.searchParams.set('min_dte', params.minDte.toString());
    if (params?.maxDte) url.searchParams.set('max_dte', params.maxDte.toString());
    if (params?.minPremiumPct) url.searchParams.set('min_premium_pct', params.minPremiumPct.toString());

    eventSourceRef.current = new EventSource(url.toString());
    setIsStreaming(true);

    eventSourceRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.event) {
          case 'start':
            setProgress('Starting AI analysis...');
            break;
          case 'progress':
            setProgress(data.message || `${data.step}...`);
            break;
          case 'pick':
            onPick(data.pick, data.rank);
            break;
          case 'complete':
            setProgress(null);
            setIsStreaming(false);
            eventSourceRef.current?.close();
            break;
          case 'error':
            console.error('[SSE] Error:', data.error);
            setIsStreaming(false);
            break;
        }
      } catch (e) {
        console.error('[SSE] Parse error:', e);
      }
    };

    eventSourceRef.current.onerror = () => {
      setIsStreaming(false);
      eventSourceRef.current?.close();
    };
  }, [onPick]);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
    setProgress(null);
  }, []);

  useEffect(() => {
    return () => stopStream();
  }, [stopStream]);

  return {
    isStreaming,
    progress,
    startStream,
    stopStream,
  };
}

// ============ Features Status Hook ============

export function useAIFeatures() {
  return useQuery({
    queryKey: ['ai-features'],
    queryFn: async () => {
      const { data } = await axios.get('/scanner/ai-picks/features');
      return data;
    },
    staleTime: 300000, // 5 minutes
  });
}

// ============ Circuit Breaker Hook ============

export function useCircuitBreaker() {
  return useQuery({
    queryKey: ['circuit-breaker'],
    queryFn: async () => {
      const { data } = await axios.get('/scanner/ai-picks/circuit-breaker');
      return data;
    },
    refetchInterval: 30000, // 30 seconds
  });
}

export default useAIPicks;
