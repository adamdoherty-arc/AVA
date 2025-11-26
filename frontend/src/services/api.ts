const API_BASE_URL = 'http://localhost:8002/api';

// Types
export interface Market {
    id: number;
    ticker: string;
    title: string;
    market_type: string;
    home_team: string;
    away_team: string;
    game_date: string;
    yes_price: number;
    no_price: number;
    volume: number;
    close_time: string;
    predicted_outcome: string;
    confidence_score: number;
    edge_percentage: number;
    overall_rank: number;
    recommended_action: string;
    recommended_stake_pct: number;
    reasoning: string;
}

export interface MarketResponse {
    markets: Market[];
    count: number;
}

export interface PortfolioSummary {
    total_value: number;
    buying_power: number;
    day_change: number;
    day_change_pct: number;
    allocations: {
        stocks: number;
        options: number;
        cash: number;
    };
    positions_count: number;
    last_updated: string;
}

export interface Position {
    symbol: string;
    quantity: number;
    avg_buy_price: number;
    current_price: number;
    cost_basis: number;
    current_value: number;
    pl: number;
    pl_pct: number;
    type: 'stock' | 'option';
}

export interface OptionPosition {
    symbol: string;
    strategy: string;
    type: string;
    option_type: string;
    strike: number;
    expiration: string;
    quantity: number;
    avg_price: number;
    current_price: number;
    total_premium: number;
    current_value: number;
    pl: number;
    pl_pct: number;
}

export interface PositionsResponse {
    summary: {
        total_equity: number;
        buying_power: number;
        total_positions: number;
    };
    stocks: Position[];
    options: OptionPosition[];
}

export interface ScannerResult {
    symbol: string;
    stock_price: number;
    strike: number;
    expiration: string;
    dte: number;
    premium: number;
    premium_pct: number;
    monthly_return: number;
    annual_return: number;
    iv: number;
    volume: number;
    open_interest: number;
}

export interface Game {
    id: string;
    league: string;
    home_team: string;
    away_team: string;
    home_score?: number;
    away_score?: number;
    status: string;
    is_live: boolean;
    game_time: string;
    odds: {
        spread_home: number;
        spread_home_odds: number;
        total: number;
        moneyline_home: number;
    };
}

export interface ResearchReport {
    symbol: string;
    fundamental: any;
    technical: any;
    sentiment: any;
    options: any;
    synthesis: string;
}

export const api = {
    getMarkets: async (marketType?: string, limit: number = 50): Promise<MarketResponse> => {
        const params = new URLSearchParams();
        if (marketType && marketType !== 'All') params.append('market_type', marketType.toLowerCase());
        params.append('limit', limit.toString());

        const response = await fetch(`${API_BASE_URL}/sports/markets?${params.toString()}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    getLiveGames: async (): Promise<any[]> => {
        const response = await fetch(`${API_BASE_URL}/sports/live`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    getUpcomingGames: async (limit: number = 10): Promise<any[]> => {
        const response = await fetch(`${API_BASE_URL}/sports/upcoming?limit=${limit}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    getPredictionMarkets: async (sector?: string, limit: number = 50): Promise<MarketResponse> => {
        const params = new URLSearchParams();
        if (sector && sector !== 'All') params.append('sector', sector);
        params.append('limit', limit.toString());

        const response = await fetch(`${API_BASE_URL}/predictions/markets?${params.toString()}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    getDashboardSummary: async (): Promise<any> => {
        const response = await fetch(`${API_BASE_URL}/dashboard/summary`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    chat: async (message: string, history: any[] = [], model: string = 'auto'): Promise<any> => {
        const response = await fetch(`${API_BASE_URL}/chat/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message, history, model }),
        });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    },

    getChatModels: async (): Promise<any> => {
        const response = await fetch(`${API_BASE_URL}/chat/models`);
        if (!response.ok) {
            throw new Error('Failed to fetch chat models');
        }
        return response.json();
    },

    // Portfolio endpoints
    getPositions: async (): Promise<PositionsResponse> => {
        const response = await fetch(`${API_BASE_URL}/portfolio/positions`);
        if (!response.ok) {
            throw new Error('Failed to fetch positions');
        }
        return response.json();
    },

    getPortfolioSummary: async (): Promise<PortfolioSummary> => {
        const response = await fetch(`${API_BASE_URL}/portfolio/summary`);
        if (!response.ok) {
            throw new Error('Failed to fetch portfolio summary');
        }
        return response.json();
    },

    syncPortfolio: async (): Promise<{ success: boolean; message: string }> => {
        const response = await fetch(`${API_BASE_URL}/portfolio/sync`, {
            method: 'POST',
        });
        if (!response.ok) {
            throw new Error('Failed to sync portfolio');
        }
        return response.json();
    },

    // Research endpoints
    analyzeSymbol: async (symbol: string): Promise<ResearchReport> => {
        const response = await fetch(`${API_BASE_URL}/research/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol }),
        });
        if (!response.ok) {
            throw new Error('Failed to analyze symbol');
        }
        return response.json();
    },

    // Options/Scanner endpoints
    scanPremiums: async (params: {
        symbols?: string[];
        max_price?: number;
        min_premium_pct?: number;
        dte?: number;
    }): Promise<ScannerResult[]> => {
        const response = await fetch(`${API_BASE_URL}/scanner/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!response.ok) {
            throw new Error('Failed to scan premiums');
        }
        return response.json();
    },

    getOptionsChain: async (symbol: string): Promise<any> => {
        const response = await fetch(`${API_BASE_URL}/options/${symbol}/chain`);
        if (!response.ok) {
            throw new Error('Failed to fetch options chain');
        }
        return response.json();
    },

    // Agents endpoints
    getAgents: async (): Promise<any[]> => {
        const response = await fetch(`${API_BASE_URL}/agents`);
        if (!response.ok) {
            throw new Error('Failed to fetch agents');
        }
        return response.json();
    },

    invokeAgent: async (agentName: string, query: string, context?: any): Promise<any> => {
        const response = await fetch(`${API_BASE_URL}/agents/${agentName}/invoke`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, context }),
        });
        if (!response.ok) {
            throw new Error('Failed to invoke agent');
        }
        return response.json();
    },

    // Watchlist endpoints
    getWatchlists: async (): Promise<any[]> => {
        const response = await fetch(`${API_BASE_URL}/watchlists`);
        if (!response.ok) {
            throw new Error('Failed to fetch watchlists');
        }
        return response.json();
    },

    // Health check
    healthCheck: async (): Promise<{ status: string; service: string }> => {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('Service unavailable');
        }
        return response.json();
    }
};
