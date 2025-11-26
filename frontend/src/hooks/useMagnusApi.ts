import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { axiosInstance } from '@/lib/axios';

// ============================================
// PORTFOLIO HOOKS
// ============================================

export const usePositions = () => {
    return useQuery({
        queryKey: ['positions'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/positions');
            return data;
        },
        refetchInterval: 30000, // Poll every 30 seconds
    });
};

export const usePortfolioSummary = () => {
    return useQuery({
        queryKey: ['portfolio-summary'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/portfolio/summary');
            return data;
        },
        staleTime: 30000,
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
        staleTime: 30000,
    });
};

export const usePerformanceHistory = (period: string = '1M') => {
    return useQuery({
        queryKey: ['performance-history', period],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/dashboard/performance?period=${period}`);
            return data;
        },
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
        refetchInterval: 30000, // Poll every 30 seconds for live games
    });
};

export const useUpcomingGames = (limit: number = 20) => {
    return useQuery({
        queryKey: ['upcoming-games', limit],
        queryFn: async () => {
            const { data } = await axiosInstance.get(`/sports/upcoming?limit=${limit}`);
            return data;
        },
        staleTime: 60000,
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
    });
};

export const useBestBets = () => {
    return useQuery({
        queryKey: ['best-bets'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/sports/best-bets');
            return data;
        },
        staleTime: 300000,
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
        staleTime: 300000,
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
        refetchInterval: 60000,
    });
};
