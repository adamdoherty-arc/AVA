import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useRef, useState, useCallback, useEffect } from 'react';
import { axiosInstance } from '@/lib/axios';
import { WS_API_URL } from '@/config/api';

// ============================================
// SHARED QUERY OPTIONS - Resilient defaults
// ============================================

/**
 * Default query options for all hooks to prevent cascading failures.
 * These can be overridden per-hook as needed.
 */
const DEFAULT_QUERY_OPTIONS = {
    retry: 1,                      // Only retry once on failure
    retryDelay: 5000,             // Wait 5 seconds before retry
    refetchOnWindowFocus: false,   // Don't refetch on window focus
    networkMode: 'online' as const, // Only fetch when online
};

/** Polling query options for endpoints that refresh periodically */
const POLLING_QUERY_OPTIONS = {
    ...DEFAULT_QUERY_OPTIONS,
    staleTime: 30000,             // Data fresh for 30 seconds
    refetchInterval: 30000,       // Poll every 30 seconds
};

/** AI/Expensive query options for computationally expensive endpoints */
const AI_QUERY_OPTIONS = {
    ...DEFAULT_QUERY_OPTIONS,
    staleTime: 300000,            // Data fresh for 5 minutes
    retry: 0,                     // Don't retry expensive operations
};

// ============================================
// PORTFOLIO HOOKS
// ============================================

/**
 * Get positions from database cache - INSTANT loading, never blocks.
 *
 * The backend maintains a cache that syncs with Robinhood every 30 minutes.
 * The UI always reads from cache for instant loading.
 *
 * For real-time live data, use useLivePositions() instead.
 */
export const usePositions = () => {
    return useQuery({
        queryKey: ['positions'],
        queryFn: async () => {
            // Uses cached endpoint - reads from database, never blocks on Robinhood API
            const { data } = await axiosInstance.get('/portfolio/positions/cached');
            return data;
        },
        staleTime: 60000, // Data fresh for 1 minute (cache is updated every 30 min)
        retry: 1, // Only retry once - cache should be reliable
        retryDelay: 1000, // Quick retry
        refetchOnWindowFocus: false,
    });
};

/**
 * Get live positions directly from Robinhood API.
 * Use this only when you need real-time data and can wait for the API call.
 * For most UI cases, use usePositions() which reads from cache.
 */
export const useLivePositions = () => {
    return useQuery({
        queryKey: ['live-positions'],
        queryFn: async () => {
            // Direct Robinhood API call - can be slow (30-60s)
            const { data } = await axiosInstance.get('/portfolio/positions');
            return data;
        },
        staleTime: 30000, // Data considered fresh for 30 seconds
        retry: 2, // Retry twice on failure (Robinhood can be flaky)
        retryDelay: 3000, // Wait 3 seconds before retry
        refetchOnWindowFocus: false,
        enabled: false, // Manual trigger only - expensive call
    });
};

/**
 * Get the sync service status (last sync time, next sync, etc.)
 */
export const useSyncStatus = () => {
    return useQuery({
        queryKey: ['sync-status'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/sync/status');
            return data;
        },
        staleTime: 10000, // Fresh for 10 seconds
        refetchInterval: 30000, // Update every 30 seconds
        refetchOnWindowFocus: true,
    });
};

/**
 * Trigger a manual sync of positions from Robinhood to database.
 * Use after making trades to immediately update the cache.
 */
export const useTriggerSync = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/portfolio/sync/trigger');
            return data;
        },
        onSuccess: () => {
            // Invalidate positions cache to pick up new data
            queryClient.invalidateQueries({ queryKey: ['positions'] });
            queryClient.invalidateQueries({ queryKey: ['sync-status'] });
            queryClient.invalidateQueries({ queryKey: ['enriched-positions'] });
        },
    });
};

export const usePortfolioSummary = () => {
    return useQuery({
        queryKey: ['portfolio-summary'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/summary');
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
    });
};

export const useSyncPortfolio = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/portfolio/sync');
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['positions'] });
            queryClient.invalidateQueries({ queryKey: ['portfolio-summary'] });
            queryClient.invalidateQueries({ queryKey: ['enriched-positions'] });
            queryClient.invalidateQueries({ queryKey: ['recommendations'] });
        },
    });
};

export const useEnrichedPositions = () => {
    return useQuery({
        queryKey: ['enriched-positions'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/positions/enriched');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

export const usePositionRecommendations = () => {
    return useQuery({
        queryKey: ['recommendations'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/recommendations');
            return data;
        },
        ...AI_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes - AI analysis is expensive
    });
};

export const useSymbolMetadata = (symbol: string | null) => {
    return useQuery({
        queryKey: ['metadata', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/metadata/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 3600000, // 1 hour - metadata doesn't change often
    });
};

export const useSymbolRecommendation = (symbol: string | null) => {
    return useQuery({
        queryKey: ['recommendation', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/recommendations/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...AI_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
    });
};

export const useDeepAnalysis = (symbol: string | null, model: string = 'auto') => {
    return useQuery({
        queryKey: ['deep-analysis', symbol, model],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/deep-analysis/${symbol}?model=${model}`);
            return data;
        },
        enabled: false, // Manual trigger only - expensive LLM call
        ...AI_QUERY_OPTIONS,
    });
};

export const usePortfolioAnalysis = (model: string = 'auto') => {
    return useQuery({
        queryKey: ['portfolio-analysis', model],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/portfolio-analysis?model=${model}`);
            return data;
        },
        enabled: false, // Manual trigger only - expensive LLM call
        ...AI_QUERY_OPTIONS,
    });
};

export const useRunDeepAnalysis = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async ({ symbol, model = 'auto' }: { symbol: string; model?: string }) => {
            const { data } = await axiosInstance.get(`/portfolio/deep-analysis/${symbol}?model=${model}`);
            return data;
        },
        onSuccess: (data, variables) => {
            queryClient.setQueryData(['deep-analysis', variables.symbol, variables.model || 'auto'], data);
        },
    });
};

export const useRunPortfolioAnalysis = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (model: string = 'auto') => {
            const { data } = await axiosInstance.get(`/portfolio/portfolio-analysis?model=${model}`);
            return data;
        },
        onSuccess: (data, model) => {
            queryClient.setQueryData(['portfolio-analysis', model], data);
        },
    });
};

// ============================================
// DASHBOARD HOOKS
// ============================================

export const useDashboardSummary = () => {
    return useQuery({
        queryKey: ['dashboard-summary'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/dashboard/summary');
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
    });
};

export const usePerformanceHistory = (period: string = '1M') => {
    return useQuery({
        queryKey: ['performance-history', period],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/dashboard/performance?period=${period}`);
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 300000, // 5 minutes
    });
};

// ============================================
// RESEARCH HOOKS
// ============================================

export const useResearch = (symbol: string | null) => {
    return useQuery({
        queryKey: ['research', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/research/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 300000, // 5 minutes
    });
};

export const useAnalyzeSymbol = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (symbol: string) => {
            const { data } = await axiosInstance.post('/research/analyze', { symbol });
            return data;
        },
        onSuccess: (data, symbol) => {
            queryClient.setQueryData(['research', symbol], data);
        },
    });
};

export const useRefreshResearch = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (symbol: string) => {
            const { data } = await axiosInstance.get(`/research/${symbol}/refresh`);
            return data;
        },
        onSuccess: (data, symbol) => {
            queryClient.setQueryData(['research', symbol], data);
        },
    });
};

// ============================================
// SPORTS BETTING HOOKS
// ============================================

export const useLiveGames = () => {
    return useQuery({
        queryKey: ['live-games'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/live');
            return data;
        },
        staleTime: 30000, // Data considered fresh for 30 seconds
        refetchInterval: 30000, // Poll every 30 seconds for live games
        retry: 1, // Only retry once on failure
        retryDelay: 5000, // Wait 5 seconds before retry
        refetchOnWindowFocus: false, // Don't refetch on window focus to reduce requests
    });
};

export const useUpcomingGames = (limit: number = 20) => {
    return useQuery({
        queryKey: ['upcoming-games', limit],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/sports/upcoming?limit=${limit}`);
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

export const useSportsMarkets = (marketType?: string, limit: number = 50) => {
    return useQuery({
        queryKey: ['sports-markets', marketType, limit],
        queryFn: async () => {
            const params = new URLSearchParams();
            if (marketType && marketType !== 'All') params.append('market_type', marketType.toLowerCase());
            params.append('limit', limit.toString());
            const { data } = await axiosInstance.get(`/sports/markets?${params.toString()}`);
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

export const useBestBets = () => {
    return useQuery({
        queryKey: ['best-bets'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/best-bets');
            return data;
        },
        ...AI_QUERY_OPTIONS,
    });
};

export const usePredictGame = () => {
    return useMutation({
        mutationFn: async (params: { sport: string; home_team: string; away_team: string }) => {
            const { data } = await axiosInstance.post('/sports/predict', params);
            return data;
        },
    });
};

// ============================================
// OPTIONS/SCANNER HOOKS
// ============================================

export const useScanPremiums = () => {
    return useMutation({
        mutationFn: async (params: {
            symbols?: string[];
            max_price?: number;
            min_premium_pct?: number;
            dte?: number;
        }) => {
            const { data } = await axiosInstance.post('/scanner/scan', params);
            return data;
        },
    });
};

export const useOptionsChain = (symbol: string | null) => {
    return useQuery({
        queryKey: ['options-chain', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/options/${symbol}/chain`);
            return data;
        },
        enabled: !!symbol,
    });
};

// ============================================
// STRATEGY HOOKS
// ============================================

export const useStrategyAnalysis = (watchlist: string, strategies?: string[]) => {
    return useQuery({
        queryKey: ['strategy', watchlist, strategies],
        queryFn: async () => {
            const params = new URLSearchParams();
            params.append('watchlist', watchlist);
            if (strategies) {
                strategies.forEach(s => params.append('strategies', s));
            }
            const { data } = await axiosInstance.get(`/strategy/analyze?${params.toString()}`);
            return data;
        },
        enabled: false, // Don't run automatically, wait for user action
    });
};

// ============================================
// AGENT HOOKS
// ============================================

export const useAgents = () => {
    return useQuery({
        queryKey: ['agents'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/agents');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 300000, // 5 minutes - agent list doesn't change often
    });
};

export const useInvokeAgent = () => {
    return useMutation({
        mutationFn: async (params: { agentName: string; query: string; context?: Record<string, unknown> }) => {
            const { data } = await axiosInstance.post(`/agents/${params.agentName}/invoke`, {
                query: params.query,
                context: params.context,
            });
            return data;
        },
    });
};

// ============================================
// CHAT HOOKS
// ============================================

export const useChat = () => {
    return useMutation({
        mutationFn: async (params: { message: string; history?: Array<{ role: string; content: string }> }) => {
            const { data } = await axiosInstance.post('/chat/', {
                message: params.message,
                history: params.history || [],
            });
            return data;
        },
    });
};

export const useConversations = () => {
    return useQuery({
        queryKey: ['conversations'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/conversations');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

// ============================================
// WATCHLIST HOOKS
// ============================================

export const useWatchlists = () => {
    return useQuery({
        queryKey: ['watchlists'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/watchlists');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
    });
};

// ============================================
// PREDICTION MARKETS HOOKS
// ============================================

export const usePredictionMarkets = (sector?: string, limit: number = 50) => {
    return useQuery({
        queryKey: ['prediction-markets', sector, limit],
        queryFn: async () => {
            const params = new URLSearchParams();
            if (sector && sector !== 'All') params.append('sector', sector);
            params.append('limit', limit.toString());
            const { data } = await axiosInstance.get(`/predictions/markets?${params.toString()}`);
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

// ============================================
// HEALTH CHECK HOOKS
// ============================================

export const useHealthCheck = () => {
    return useQuery({
        queryKey: ['health'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/health');
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
        refetchInterval: 60000, // Override to 60 seconds for health checks
    });
};

// ============================================
// ADVANCED ANALYTICS HOOKS
// ============================================

export const useAdvancedRiskMetrics = () => {
    return useQuery({
        queryKey: ['advanced-risk-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/analytics/risk');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
        refetchInterval: 120000, // Refresh every 2 minutes
    });
};

export const useProbabilityMetrics = () => {
    return useQuery({
        queryKey: ['probability-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/analytics/probability');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

export const useMultiAgentConsensus = (symbol: string | null) => {
    return useQuery({
        queryKey: ['multi-agent-consensus', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/analytics/consensus/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...AI_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes - AI analysis
    });
};

export const usePositionAlerts = () => {
    return useQuery({
        queryKey: ['position-alerts'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/analytics/alerts');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 60 seconds
        refetchInterval: 60000, // Check every minute
    });
};

export const useAnalyticsDashboard = () => {
    return useQuery({
        queryKey: ['analytics-dashboard'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/analytics/dashboard');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
        refetchInterval: 120000, // Refresh every 2 minutes
    });
};

// Streaming hook for real-time insights
export const useStreamingInsights = (enabled: boolean = false) => {
    return useQuery({
        queryKey: ['streaming-insights'],
        queryFn: async () => {
            // This is a placeholder - actual SSE implementation would use EventSource
            const { data } = await axiosInstance.get('/portfolio/analytics/dashboard');
            return data;
        },
        enabled,
        ...POLLING_QUERY_OPTIONS,
    });
};

// ============================================
// PORTFOLIO V2 HOOKS - Modern Infrastructure
// ============================================

export const usePositionsV2 = (forceRefresh: boolean = false) => {
    return useQuery({
        queryKey: ['positions-v2', forceRefresh],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/v2/positions?force_refresh=${forceRefresh}`);
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
    });
};

export const useEnrichedPositionsV2 = () => {
    return useQuery({
        queryKey: ['enriched-positions-v2'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/positions/enriched');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 300000, // 5 minutes
    });
};

export const useRefreshPositionsV2 = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/portfolio/v2/positions/refresh');
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['positions-v2'] });
            queryClient.invalidateQueries({ queryKey: ['enriched-positions-v2'] });
        },
    });
};

// Advanced VaR Analysis
export const useVaRAnalysis = (method: 'parametric' | 'monte_carlo' | 'both' = 'both', simulations: number = 10000) => {
    return useQuery({
        queryKey: ['var-analysis', method, simulations],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/v2/risk/var?method=${method}&simulations=${simulations}`);
            return data;
        },
        enabled: false, // Manual trigger
        ...AI_QUERY_OPTIONS,
    });
};

export const useRunVaRAnalysis = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async ({ method = 'both', simulations = 10000 }: { method?: string; simulations?: number }) => {
            const { data } = await axiosInstance.get(`/portfolio/v2/risk/var?method=${method}&simulations=${simulations}`);
            return data;
        },
        onSuccess: (data, variables) => {
            queryClient.setQueryData(['var-analysis', variables.method || 'both', variables.simulations || 10000], data);
        },
    });
};

// Stress Testing
export const useStressTests = () => {
    return useQuery({
        queryKey: ['stress-tests'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/risk/stress-test');
            return data;
        },
        enabled: false, // Manual trigger
        ...AI_QUERY_OPTIONS,
    });
};

export const useRunStressTests = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/risk/stress-test');
            return data;
        },
        onSuccess: (data) => {
            queryClient.setQueryData(['stress-tests'], data);
        },
    });
};

// P/L Projection (What-If)
export const usePnLProjection = (underlyingMove: number = 0, ivChange: number = 0, days: number = 1) => {
    return useQuery({
        queryKey: ['pnl-projection', underlyingMove, ivChange, days],
        queryFn: async () => {
            const { data } = await axiosInstance.get(
                `/portfolio/v2/risk/pnl-projection?underlying_move=${underlyingMove}&iv_change=${ivChange}&days=${days}`
            );
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 60000, // 1 minute
    });
};

// Max Loss Analysis
export const useMaxLoss = () => {
    return useQuery({
        queryKey: ['max-loss'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/risk/max-loss');
            return data;
        },
        ...DEFAULT_QUERY_OPTIONS,
        staleTime: 300000, // 5 minutes
    });
};

// Background Task Management
export const useStartPortfolioAnalysis = () => {
    return useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/portfolio/v2/tasks/analyze-portfolio');
            return data;
        },
    });
};

export const useTaskStatus = (taskId: string | null) => {
    return useQuery({
        queryKey: ['task-status', taskId],
        queryFn: async () => {
            if (!taskId) return null;
            const { data } = await axiosInstance.get(`/portfolio/v2/tasks/${taskId}`);
            return data;
        },
        enabled: !!taskId,
        refetchInterval: (query) => {
            // Poll until complete
            const data = query.state.data;
            if (data?.status === 'completed' || data?.status === 'failed') {
                return false;
            }
            return 2000; // Poll every 2 seconds
        },
    });
};

// Health & Monitoring
export const usePortfolioV2Health = () => {
    return useQuery({
        queryKey: ['portfolio-v2-health'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/health');
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
        refetchInterval: 60000, // Check health every 60 seconds
    });
};

export const usePortfolioV2Metrics = () => {
    return useQuery({
        queryKey: ['portfolio-v2-metrics'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/metrics');
            return data;
        },
        ...POLLING_QUERY_OPTIONS,
    });
};

// Cache Management
export const useInvalidateCache = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (pattern: string = '*') => {
            const { data } = await axiosInstance.post(`/portfolio/v2/cache/invalidate?pattern=${pattern}`);
            return data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['positions-v2'] });
            queryClient.invalidateQueries({ queryKey: ['enriched-positions-v2'] });
        },
    });
};

// WebSocket Hook for Real-Time Positions
export const usePositionsWebSocket = (userId?: string, autoConnect: boolean = false) => {
    const queryClient = useQueryClient();
    const wsRef = useRef<WebSocket | null>(null);
    const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    const connect = useCallback(() => {
        // Prevent duplicate connections
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        // Uses centralized config from @/config/api
        const wsUrl = `${WS_API_URL}/portfolio/v2/ws/positions${userId ? `?user_id=${userId}` : ''}`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            console.debug('[WS] Positions WebSocket connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'positions_update' || data.type === 'initial_positions') {
                    queryClient.setQueryData(['positions-v2', false], data.data);
                }
            } catch (e) {
                console.error('[WS] Failed to parse message:', e);
            }
        };

        ws.onerror = (error) => {
            console.error('[WS] WebSocket error:', error);
        };

        ws.onclose = () => {
            setIsConnected(false);
            // Clear heartbeat on any close (normal or abnormal) to prevent memory leak
            if (heartbeatRef.current) {
                clearInterval(heartbeatRef.current);
                heartbeatRef.current = null;
            }
            console.debug('[WS] Positions WebSocket disconnected');
        };

        // Send heartbeat every 25 seconds
        heartbeatRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);
    }, [userId, queryClient]);

    const disconnect = useCallback(() => {
        if (heartbeatRef.current) {
            clearInterval(heartbeatRef.current);
            heartbeatRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    const refresh = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send('refresh');
        }
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        if (autoConnect) {
            connect();
        }
        return () => {
            disconnect();
        };
    }, [autoConnect, connect, disconnect]);

    return {
        connect,
        disconnect,
        refresh,
        isConnected,
    };
};

// ============================================
// AI-POWERED ANALYSIS HOOKS
// ============================================

// Anomaly Detection
export const useAnomalyDetection = () => {
    return useQuery({
        queryKey: ['ai-anomalies'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/ai/anomalies');
            return data;
        },
        ...AI_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
    });
};

// Portfolio Risk Score (0-100)
export const useRiskScore = () => {
    return useQuery({
        queryKey: ['ai-risk-score'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/v2/ai/risk-score');
            return data;
        },
        ...AI_QUERY_OPTIONS,
        staleTime: 120000, // 2 minutes
    });
};

// AI Trade Recommendations
export const useAIRecommendations = (riskTolerance: 'conservative' | 'moderate' | 'aggressive' = 'moderate') => {
    return useQuery({
        queryKey: ['ai-recommendations', riskTolerance],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/v2/ai/recommendations?risk_tolerance=${riskTolerance}`);
            return data;
        },
        ...AI_QUERY_OPTIONS,
    });
};

// AI Price Prediction for Symbol
export const usePricePrediction = (symbol: string | null) => {
    return useQuery({
        queryKey: ['ai-prediction', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/v2/ai/predictions/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...AI_QUERY_OPTIONS,
    });
};

// Trend Signal for Symbol
export const useTrendSignal = (symbol: string | null) => {
    return useQuery({
        queryKey: ['ai-trend', symbol],
        queryFn: async () => {
            if (!symbol) return null;
            const { data } = await axiosInstance.get(`/portfolio/v2/ai/trend/${symbol}`);
            return data;
        },
        enabled: !!symbol,
        ...AI_QUERY_OPTIONS,
    });
};

// Comprehensive AI Analysis (combines all metrics)
export const useComprehensiveAnalysis = (riskTolerance: 'conservative' | 'moderate' | 'aggressive' = 'moderate') => {
    return useQuery({
        queryKey: ['ai-comprehensive', riskTolerance],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/portfolio/v2/ai/comprehensive?risk_tolerance=${riskTolerance}`);
            return data;
        },
        enabled: false, // Manual trigger recommended
        ...AI_QUERY_OPTIONS,
    });
};

export const useRunComprehensiveAnalysis = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (riskTolerance: 'conservative' | 'moderate' | 'aggressive' = 'moderate') => {
            const { data } = await axiosInstance.get(`/portfolio/v2/ai/comprehensive?risk_tolerance=${riskTolerance}`);
            return data;
        },
        onSuccess: (data, riskTolerance) => {
            queryClient.setQueryData(['ai-comprehensive', riskTolerance || 'moderate'], data);
        },
    });
};
