/**
 * Stock Watchlist Store
 * Zustand store for managing stock watchlists with localStorage persistence
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface WatchedStock {
  symbol: string;
  addedAt: string;
  notes?: string;
}

export interface StockWatchlist {
  name: string;
  symbols: string[];
  isCustom: boolean;
  createdAt: string;
}

interface StockWatchlistState {
  // Current active watchlist
  activeWatchlist: string;

  // Custom watchlists (persisted)
  customWatchlists: Record<string, StockWatchlist>;

  // Recent searches
  recentSymbols: string[];

  // Favorited stocks
  favorites: string[];

  // Actions
  setActiveWatchlist: (name: string) => void;
  createWatchlist: (name: string, symbols: string[]) => void;
  deleteWatchlist: (name: string) => void;
  addToWatchlist: (watchlistName: string, symbol: string) => void;
  removeFromWatchlist: (watchlistName: string, symbol: string) => void;
  addRecentSymbol: (symbol: string) => void;
  toggleFavorite: (symbol: string) => void;
  isFavorite: (symbol: string) => boolean;
  getWatchlistSymbols: (name: string) => string[];
}

// Default watchlists (synced with backend - expanded diversity)
export const DEFAULT_WATCHLISTS: Record<string, StockWatchlist> = {
  'Tech Leaders': {
    name: 'Tech Leaders',
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'CRM'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Options Favorites': {
    name: 'Options Favorites',
    symbols: ['SPY', 'QQQ', 'IWM', 'AAPL', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'META'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'AI & Semiconductor': {
    name: 'AI & Semiconductor',
    symbols: ['NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'ASML', 'ARM', 'MU', 'MRVL'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'High IV Plays': {
    name: 'High IV Plays',
    symbols: ['GME', 'AMC', 'MARA', 'COIN', 'RIVN', 'PLTR', 'SOFI', 'NIO', 'RIOT'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Dividend Aristocrats': {
    name: 'Dividend Aristocrats',
    symbols: ['JNJ', 'PG', 'KO', 'PEP', 'ABT', 'ABBV', 'VZ', 'XOM', 'CVX'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Financials': {
    name: 'Financials',
    symbols: ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'BLK', 'SCHW'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Healthcare': {
    name: 'Healthcare',
    symbols: ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'ETF Universe': {
    name: 'ETF Universe',
    symbols: ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLF', 'XLE', 'XLK', 'GLD'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Consumer': {
    name: 'Consumer',
    symbols: ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'COST', 'WMT'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
  'Energy': {
    name: 'Energy',
    symbols: ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY'],
    isCustom: false,
    createdAt: new Date().toISOString(),
  },
};

export const useStockWatchlistStore = create<StockWatchlistState>()(
  persist(
    (set, get) => ({
      activeWatchlist: 'Tech Leaders',
      customWatchlists: {},
      recentSymbols: [],
      favorites: [],

      setActiveWatchlist: (name) => {
        set({ activeWatchlist: name });
      },

      createWatchlist: (name, symbols) => {
        set((state) => ({
          customWatchlists: {
            ...state.customWatchlists,
            [name]: {
              name,
              symbols,
              isCustom: true,
              createdAt: new Date().toISOString(),
            },
          },
          activeWatchlist: name,
        }));
      },

      deleteWatchlist: (name) => {
        // Can't delete default watchlists
        if (DEFAULT_WATCHLISTS[name]) return;

        set((state) => {
          const { [name]: deleted, ...rest } = state.customWatchlists;
          return {
            customWatchlists: rest,
            activeWatchlist:
              state.activeWatchlist === name
                ? 'Tech Leaders'
                : state.activeWatchlist,
          };
        });
      },

      addToWatchlist: (watchlistName, symbol) => {
        const state = get();
        const upperSymbol = symbol.toUpperCase();

        // If it's a custom watchlist
        if (state.customWatchlists[watchlistName]) {
          const watchlist = state.customWatchlists[watchlistName];
          if (!watchlist.symbols.includes(upperSymbol)) {
            set({
              customWatchlists: {
                ...state.customWatchlists,
                [watchlistName]: {
                  ...watchlist,
                  symbols: [...watchlist.symbols, upperSymbol],
                },
              },
            });
          }
        }
      },

      removeFromWatchlist: (watchlistName, symbol) => {
        const state = get();
        const upperSymbol = symbol.toUpperCase();

        // If it's a custom watchlist
        if (state.customWatchlists[watchlistName]) {
          const watchlist = state.customWatchlists[watchlistName];
          set({
            customWatchlists: {
              ...state.customWatchlists,
              [watchlistName]: {
                ...watchlist,
                symbols: watchlist.symbols.filter((s) => s !== upperSymbol),
              },
            },
          });
        }
      },

      addRecentSymbol: (symbol) => {
        const upperSymbol = symbol.toUpperCase();
        set((state) => {
          const filtered = state.recentSymbols.filter((s) => s !== upperSymbol);
          return {
            recentSymbols: [upperSymbol, ...filtered].slice(0, 10), // Keep last 10
          };
        });
      },

      toggleFavorite: (symbol) => {
        const upperSymbol = symbol.toUpperCase();
        set((state) => {
          if (state.favorites.includes(upperSymbol)) {
            return {
              favorites: state.favorites.filter((s) => s !== upperSymbol),
            };
          } else {
            return {
              favorites: [...state.favorites, upperSymbol],
            };
          }
        });
      },

      isFavorite: (symbol) => {
        return get().favorites.includes(symbol.toUpperCase());
      },

      getWatchlistSymbols: (name) => {
        const state = get();
        // Check custom watchlists first
        if (state.customWatchlists[name]) {
          return state.customWatchlists[name].symbols;
        }
        // Then check defaults
        if (DEFAULT_WATCHLISTS[name]) {
          return DEFAULT_WATCHLISTS[name].symbols;
        }
        return [];
      },
    }),
    {
      name: 'stock-watchlist-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        customWatchlists: state.customWatchlists,
        recentSymbols: state.recentSymbols,
        favorites: state.favorites,
        activeWatchlist: state.activeWatchlist,
      }),
    }
  )
);

// Helper to get all watchlists (default + custom)
export const getAllWatchlists = (): Record<string, StockWatchlist> => {
  const customWatchlists = useStockWatchlistStore.getState().customWatchlists;
  return { ...DEFAULT_WATCHLISTS, ...customWatchlists };
};

// Helper to get watchlist names
export const getWatchlistNames = (): string[] => {
  const customWatchlists = useStockWatchlistStore.getState().customWatchlists;
  return [...Object.keys(DEFAULT_WATCHLISTS), ...Object.keys(customWatchlists)];
};
