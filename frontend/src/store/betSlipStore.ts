/**
 * Bet Slip Store - Zustand State Management
 * Persistent bet slip with AI-powered analysis and parlay building
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Types
export type BetType = 'moneyline' | 'spread' | 'total_over' | 'total_under' | 'prop';
export type BetStatus = 'pending' | 'placed' | 'won' | 'lost' | 'pushed' | 'cancelled';

export interface BetLeg {
  id: string;
  gameId: string;
  sport: string;
  homeTeam: string;
  awayTeam: string;
  betType: BetType;
  selection: string; // e.g., "home", "away", "over", "under"
  line?: number; // spread or total line
  odds: number; // American odds
  decimalOdds: number;
  impliedProbability: number;
  // AI Analysis
  aiProbability?: number;
  aiEdge?: number;
  aiConfidence?: string;
  aiReasoning?: string;
  // Metadata
  addedAt: Date;
  gameTime: Date;
}

export interface ParlayAnalysis {
  combinedProbability: number;
  correlationFactor: number;
  adjustedProbability: number;
  expectedValue: number;
  parlayOdds: number;
  recommendation: 'BET' | 'CONSIDER' | 'PASS';
  correlationWarnings: string[];
  suggestedBetSize: string;
  kellyFraction: number;
}

export interface BetSlipState {
  // Bet Slip Data
  legs: BetLeg[];
  stakeAmount: number;
  betMode: 'singles' | 'parlay';

  // AI Analysis
  parlayAnalysis: ParlayAnalysis | null;
  isAnalyzing: boolean;
  analysisError: string | null;

  // Streaming State
  streamingReasoning: string;
  isStreamingReasoning: boolean;

  // UI State
  isOpen: boolean;
  isMinimized: boolean;

  // Bankroll Management
  bankroll: number;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';

  // History
  recentBets: Array<{
    id: string;
    legs: BetLeg[];
    stake: number;
    potentialPayout: number;
    status: BetStatus;
    placedAt: Date;
    settledAt?: Date;
    result?: 'win' | 'loss' | 'push';
    profit?: number;
  }>;
}

export interface BetSlipActions {
  // Leg Management
  addLeg: (leg: Omit<BetLeg, 'id' | 'addedAt' | 'decimalOdds' | 'impliedProbability'>) => void;
  removeLeg: (legId: string) => void;
  updateLegOdds: (legId: string, newOdds: number) => void;
  clearSlip: () => void;

  // Bet Configuration
  setStakeAmount: (amount: number) => void;
  setBetMode: (mode: 'singles' | 'parlay') => void;
  setRiskLevel: (level: 'conservative' | 'moderate' | 'aggressive') => void;
  setBankroll: (amount: number) => void;

  // UI Actions
  toggleSlip: () => void;
  minimizeSlip: () => void;
  expandSlip: () => void;

  // AI Analysis
  analyzeParlay: () => Promise<void>;
  streamAnalysis: (gameId: string) => Promise<void>;
  updateStreamingReasoning: (text: string) => void;

  // Bet Placement
  placeBet: () => Promise<{ success: boolean; betId?: string; error?: string }>;

  // History
  addToHistory: (bet: BetSlipState['recentBets'][0]) => void;

  // Calculations
  calculatePotentialPayout: () => number;
  calculateKellyBet: () => number;
}

// Helper functions
const americanToDecimal = (odds: number): number => {
  if (odds > 0) return (odds / 100) + 1;
  return (100 / Math.abs(odds)) + 1;
};

const decimalToImplied = (decimal: number): number => {
  return 1 / decimal;
};

const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// Safely convert gameTime to ISO string (handles invalid dates)
const safeToISOString = (dateValue: Date | string | undefined | null): string | undefined => {
  if (!dateValue) return undefined;

  try {
    if (dateValue instanceof Date) {
      return !isNaN(dateValue.getTime()) ? dateValue.toISOString() : undefined;
    }
    if (typeof dateValue === 'string') {
      const parsed = new Date(dateValue);
      return !isNaN(parsed.getTime()) ? parsed.toISOString() : dateValue;
    }
    return undefined;
  } catch {
    return undefined;
  }
};

// Telegram notification helper (fire-and-forget)
const sendTelegramNotification = async (leg: BetLeg, stakeAmount: number) => {
  try {
    const payload = {
      legs: [{
        game_id: leg.gameId,
        sport: leg.sport,
        home_team: leg.homeTeam,
        away_team: leg.awayTeam,
        bet_type: leg.betType,
        selection: leg.selection,
        odds: leg.odds,
        line: leg.line,
        game_time: safeToISOString(leg.gameTime),
        ai_probability: leg.aiProbability,
        ai_edge: leg.aiEdge,
        ai_confidence: leg.aiConfidence,
        ai_reasoning: leg.aiReasoning,
        stake: stakeAmount,
        potential_payout: stakeAmount * (leg.decimalOdds - 1),
      }],
      mode: 'singles',
      stake: stakeAmount,
    };

    await fetch('/api/sports/bet-slip/notify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    console.warn('Telegram notification failed (non-critical):', error);
  }
};

// Store
export const useBetSlipStore = create<BetSlipState & BetSlipActions>()(
  persist(
    immer((set, get) => ({
      // Initial State
      legs: [],
      stakeAmount: 10,
      betMode: 'parlay',
      parlayAnalysis: null,
      isAnalyzing: false,
      analysisError: null,
      streamingReasoning: '',
      isStreamingReasoning: false,
      isOpen: false,
      isMinimized: false,
      bankroll: 1000,
      riskLevel: 'moderate',
      recentBets: [],

      // Leg Management
      addLeg: (legData) => {
        const decimalOdds = americanToDecimal(legData.odds);
        const impliedProbability = decimalToImplied(decimalOdds);

        const newLeg: BetLeg = {
          ...legData,
          id: generateId(),
          addedAt: new Date(),
          decimalOdds,
          impliedProbability,
        };

        const state = get();
        // Check for duplicate (same game + bet type)
        const exists = state.legs.some(
          (leg) => leg.gameId === newLeg.gameId && leg.betType === newLeg.betType
        );

        if (!exists) {
          set((draft) => {
            draft.legs.push(newLeg);
            draft.isOpen = true;
            draft.parlayAnalysis = null; // Reset analysis when legs change
          });

          // Send Telegram notification (fire-and-forget)
          sendTelegramNotification(newLeg, state.stakeAmount);
        }
      },

      removeLeg: (legId) => {
        set((state) => {
          state.legs = state.legs.filter((leg) => leg.id !== legId);
          state.parlayAnalysis = null;
          if (state.legs.length === 0) {
            state.isOpen = false;
          }
        });
      },

      updateLegOdds: (legId, newOdds) => {
        set((state) => {
          const leg = state.legs.find((l) => l.id === legId);
          if (leg) {
            leg.odds = newOdds;
            leg.decimalOdds = americanToDecimal(newOdds);
            leg.impliedProbability = decimalToImplied(leg.decimalOdds);
            state.parlayAnalysis = null;
          }
        });
      },

      clearSlip: () => {
        set((state) => {
          state.legs = [];
          state.parlayAnalysis = null;
          state.streamingReasoning = '';
          state.isOpen = false;
        });
      },

      // Bet Configuration
      setStakeAmount: (amount) => set({ stakeAmount: amount }),
      setBetMode: (mode) => set({ betMode: mode }),
      setRiskLevel: (level) => set({ riskLevel: level }),
      setBankroll: (amount) => set({ bankroll: amount }),

      // UI Actions
      toggleSlip: () => set((state) => ({ isOpen: !state.isOpen })),
      minimizeSlip: () => set({ isMinimized: true }),
      expandSlip: () => set({ isMinimized: false }),

      // AI Analysis
      analyzeParlay: async () => {
        const state = get();
        if (state.legs.length < 2) {
          set({ analysisError: 'Need at least 2 legs for parlay analysis' });
          return;
        }

        set({ isAnalyzing: true, analysisError: null });

        try {
          const gameIds = state.legs.map((leg) => leg.gameId).join(',');
          const response = await fetch(
            `/api/sports/stream/parlay-builder?game_ids=${gameIds}&sport=${state.legs[0].sport}`
          );

          if (!response.ok) throw new Error('Failed to analyze parlay');

          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (!reader) throw new Error('No response body');

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n').filter((line) => line.startsWith('data:'));

            for (const line of lines) {
              try {
                const data = JSON.parse(line.slice(5));

                if (data.type === 'parlay_result') {
                  set((draft) => {
                    draft.parlayAnalysis = {
                      combinedProbability: data.combined_probability,
                      correlationFactor: 0.95, // From response
                      adjustedProbability: data.combined_probability,
                      expectedValue: data.expected_value,
                      parlayOdds: data.parlay_odds_american,
                      recommendation: data.recommendation,
                      correlationWarnings: [],
                      suggestedBetSize: `${(data.kelly_fraction * 100).toFixed(1)}% of bankroll`,
                      kellyFraction: data.kelly_fraction,
                    };
                  });
                }
              } catch {
                // Continue on parse errors
              }
            }
          }
        } catch (error) {
          set({ analysisError: (error as Error).message });
        } finally {
          set({ isAnalyzing: false });
        }
      },

      streamAnalysis: async (gameId) => {
        set({ isStreamingReasoning: true, streamingReasoning: '' });

        try {
          const response = await fetch(
            `/api/sports/stream/predict/${gameId}?include_reasoning=true`
          );

          if (!response.ok) throw new Error('Failed to stream analysis');

          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (!reader) throw new Error('No response body');

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n').filter((line) => line.startsWith('data:'));

            for (const line of lines) {
              try {
                const data = JSON.parse(line.slice(5));

                if (data.type === 'reasoning_token') {
                  set((draft) => {
                    draft.streamingReasoning = data.accumulated;
                  });
                } else if (data.type === 'prediction') {
                  // Update AI probability for the leg
                  set((draft) => {
                    const leg = draft.legs.find((l) => l.gameId === gameId);
                    if (leg) {
                      leg.aiProbability = data.home_win_probability;
                      leg.aiConfidence = data.confidence;
                    }
                  });
                } else if (data.type === 'recommendation') {
                  set((draft) => {
                    const leg = draft.legs.find((l) => l.gameId === gameId);
                    if (leg) {
                      leg.aiEdge = data.edge / 100;
                    }
                  });
                }
              } catch {
                // Continue on parse errors
              }
            }
          }
        } catch (error) {
          console.error('Stream analysis error:', error);
        } finally {
          set({ isStreamingReasoning: false });
        }
      },

      updateStreamingReasoning: (text) => {
        set({ streamingReasoning: text });
      },

      // Bet Placement
      placeBet: async () => {
        const state = get();

        if (state.legs.length === 0) {
          return { success: false, error: 'No legs in bet slip' };
        }

        try {
          // In production, this would call the actual betting API
          const betId = generateId();
          const potentialPayout = state.calculatePotentialPayout();

          const bet = {
            id: betId,
            legs: [...state.legs],
            stake: state.stakeAmount,
            potentialPayout,
            status: 'pending' as BetStatus,
            placedAt: new Date(),
          };

          set((draft) => {
            draft.recentBets.unshift(bet);
            draft.recentBets = draft.recentBets.slice(0, 50); // Keep last 50
            draft.legs = [];
            draft.parlayAnalysis = null;
            draft.isOpen = false;
          });

          return { success: true, betId };
        } catch (error) {
          return { success: false, error: (error as Error).message };
        }
      },

      // History
      addToHistory: (bet) => {
        set((state) => {
          state.recentBets.unshift(bet);
          state.recentBets = state.recentBets.slice(0, 50);
        });
      },

      // Calculations
      calculatePotentialPayout: () => {
        const state = get();

        if (state.betMode === 'singles') {
          // Sum of individual bet payouts
          return state.legs.reduce((total, leg) => {
            return total + (state.stakeAmount * (leg.decimalOdds - 1));
          }, 0);
        }

        // Parlay payout
        const multiplier = state.legs.reduce((mult, leg) => mult * leg.decimalOdds, 1);
        return state.stakeAmount * (multiplier - 1);
      },

      calculateKellyBet: () => {
        const state = get();

        if (!state.parlayAnalysis || state.parlayAnalysis.kellyFraction <= 0) {
          return 0;
        }

        // Apply fractional Kelly based on risk level
        const kellyMultipliers = {
          conservative: 0.15,
          moderate: 0.25,
          aggressive: 0.40,
        };

        const fractionalKelly = state.parlayAnalysis.kellyFraction * kellyMultipliers[state.riskLevel];
        return Math.min(state.bankroll * fractionalKelly, state.bankroll * 0.1); // Cap at 10% max
      },
    })),
    {
      name: 'bet-slip-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        legs: state.legs,
        stakeAmount: state.stakeAmount,
        betMode: state.betMode,
        bankroll: state.bankroll,
        riskLevel: state.riskLevel,
        recentBets: state.recentBets,
      }),
    }
  )
);

// Selectors
export const selectTotalOdds = (state: BetSlipState) => {
  if (state.legs.length === 0) return 1;
  return state.legs.reduce((mult, leg) => mult * leg.decimalOdds, 1);
};

export const selectLegCount = (state: BetSlipState) => state.legs.length;

export const selectHasValueBet = (state: BetSlipState) => {
  return state.legs.some((leg) => leg.aiEdge && leg.aiEdge > 0.02);
};

export const selectRecentWinRate = (state: BetSlipState) => {
  const settled = state.recentBets.filter((bet) => bet.result);
  if (settled.length === 0) return 0;
  const wins = settled.filter((bet) => bet.result === 'win').length;
  return wins / settled.length;
};

export default useBetSlipStore;
