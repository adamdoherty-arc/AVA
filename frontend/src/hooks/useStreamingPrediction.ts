/**
 * Streaming Prediction Hook
 * Real-time SSE streaming for AI prediction analysis with token-by-token reasoning
 */

import { useState, useCallback, useRef } from 'react';

export type StreamEventType =
  | 'start'
  | 'model_loading'
  | 'data_fetching'
  | 'prediction'
  | 'factors'
  | 'reasoning_token'
  | 'reasoning_complete'
  | 'recommendation'
  | 'complete'
  | 'error';

export interface PredictionResult {
  homeTeam: string;
  awayTeam: string;
  homeWinProbability: number;
  awayWinProbability: number;
  confidence: string;
  modelVersion: string;
}

export interface Factor {
  factor: string;
  impact: string;
  description: string;
}

export interface Recommendation {
  action: 'STRONG BET' | 'LEAN' | 'PASS';
  side: string;
  edge: number;
  suggestedBetSize: string;
  kellyFraction: number;
}

export interface StreamingState {
  // Loading states
  isLoading: boolean;
  isModelLoading: boolean;
  isDataFetching: boolean;
  isStreamingReasoning: boolean;

  // Progress
  currentStep: StreamEventType | null;
  progress: number;

  // Data
  prediction: PredictionResult | null;
  factors: Factor[];
  reasoning: string;
  recommendation: Recommendation | null;

  // Error
  error: string | null;
}

export interface UseStreamingPredictionOptions {
  onPrediction?: (prediction: PredictionResult) => void;
  onFactors?: (factors: Factor[]) => void;
  onReasoningToken?: (token: string, accumulated: string) => void;
  onRecommendation?: (recommendation: Recommendation) => void;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function useStreamingPrediction(options: UseStreamingPredictionOptions = {}) {
  const [state, setState] = useState<StreamingState>({
    isLoading: false,
    isModelLoading: false,
    isDataFetching: false,
    isStreamingReasoning: false,
    currentStep: null,
    progress: 0,
    prediction: null,
    factors: [],
    reasoning: '',
    recommendation: null,
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const startPrediction = useCallback(
    async (gameId: string, sport: string = 'NFL', includeReasoning: boolean = true) => {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();

      // Reset state
      setState({
        isLoading: true,
        isModelLoading: false,
        isDataFetching: false,
        isStreamingReasoning: false,
        currentStep: 'start',
        progress: 0,
        prediction: null,
        factors: [],
        reasoning: '',
        recommendation: null,
        error: null,
      });

      try {
        const response = await fetch(
          `/api/sports/stream/predict/${gameId}?sport=${sport}&include_reasoning=${includeReasoning}`,
          { signal: abortControllerRef.current.signal }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('No response body');
        }

        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE events
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data:')) {
              try {
                const data = JSON.parse(line.slice(5).trim());
                processEvent(data);
              } catch (e) {
                // Skip malformed JSON
                console.warn('Failed to parse SSE event:', e);
              }
            }
          }
        }
      } catch (error) {
        if ((error as Error).name === 'AbortError') {
          // Request was cancelled, don't treat as error
          return;
        }

        const errorMessage = (error as Error).message;
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorMessage,
        }));
        options.onError?.(errorMessage);
      }
    },
    [options]
  );

  const processEvent = useCallback(
    (data: any) => {
      const eventType = data.type as StreamEventType;

      switch (eventType) {
        case 'start':
          setState((prev) => ({
            ...prev,
            currentStep: 'start',
            progress: 5,
          }));
          break;

        case 'model_loading':
          setState((prev) => ({
            ...prev,
            isModelLoading: true,
            currentStep: 'model_loading',
            progress: 15,
          }));
          break;

        case 'data_fetching':
          setState((prev) => ({
            ...prev,
            isModelLoading: false,
            isDataFetching: true,
            currentStep: 'data_fetching',
            progress: 30,
          }));
          break;

        case 'prediction':
          const prediction: PredictionResult = {
            homeTeam: data.home_team,
            awayTeam: data.away_team,
            homeWinProbability: data.home_win_probability,
            awayWinProbability: data.away_win_probability,
            confidence: data.confidence,
            modelVersion: data.model_version,
          };

          setState((prev) => ({
            ...prev,
            isDataFetching: false,
            currentStep: 'prediction',
            progress: 50,
            prediction,
          }));

          options.onPrediction?.(prediction);
          break;

        case 'factors':
          const factors: Factor[] = data.factors.map((f: any) => ({
            factor: f.factor,
            impact: f.impact,
            description: f.description,
          }));

          setState((prev) => ({
            ...prev,
            currentStep: 'factors',
            progress: 60,
            factors,
          }));

          options.onFactors?.(factors);
          break;

        case 'reasoning_token':
          setState((prev) => ({
            ...prev,
            isStreamingReasoning: true,
            currentStep: 'reasoning_token',
            progress: 60 + Math.min(25, prev.reasoning.length / 10),
            reasoning: data.accumulated,
          }));

          options.onReasoningToken?.(data.token, data.accumulated);
          break;

        case 'reasoning_complete':
          setState((prev) => ({
            ...prev,
            isStreamingReasoning: false,
            currentStep: 'reasoning_complete',
            progress: 85,
            reasoning: data.full_reasoning,
          }));
          break;

        case 'recommendation':
          const recommendation: Recommendation = {
            action: data.action,
            side: data.side,
            edge: data.edge,
            suggestedBetSize: data.suggested_bet_size,
            kellyFraction: data.kelly_fraction,
          };

          setState((prev) => ({
            ...prev,
            currentStep: 'recommendation',
            progress: 95,
            recommendation,
          }));

          options.onRecommendation?.(recommendation);
          break;

        case 'complete':
          setState((prev) => ({
            ...prev,
            isLoading: false,
            currentStep: 'complete',
            progress: 100,
          }));

          options.onComplete?.();
          break;

        case 'error':
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: data.message,
          }));

          options.onError?.(data.message);
          break;
      }
    },
    [options]
  );

  const cancelPrediction = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setState((prev) => ({
      ...prev,
      isLoading: false,
      isStreamingReasoning: false,
    }));
  }, []);

  const reset = useCallback(() => {
    cancelPrediction();
    setState({
      isLoading: false,
      isModelLoading: false,
      isDataFetching: false,
      isStreamingReasoning: false,
      currentStep: null,
      progress: 0,
      prediction: null,
      factors: [],
      reasoning: '',
      recommendation: null,
      error: null,
    });
  }, [cancelPrediction]);

  return {
    ...state,
    startPrediction,
    cancelPrediction,
    reset,
  };
}

/**
 * Hook for streaming live game updates
 */
export function useLiveGameStream(gameId: string, sport: string = 'NFL') {
  const [liveData, setLiveData] = useState<{
    homeScore: number;
    awayScore: number;
    period: number;
    timeRemaining: string;
    pregameProb: number;
    liveProb: number;
    probabilityChange: number;
    momentum: string;
    confidence: number;
    isConnected: boolean;
  } | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const connect = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`/api/sports/stream/live/${gameId}?sport=${sport}`, {
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      setIsConnected(true);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n').filter((line) => line.startsWith('data:'));

        for (const line of lines) {
          try {
            const data = JSON.parse(line.slice(5).trim());

            if (data.type === 'live_update') {
              setLiveData({
                homeScore: data.home_score,
                awayScore: data.away_score,
                period: data.period,
                timeRemaining: data.time_remaining,
                pregameProb: data.pregame_prob,
                liveProb: data.live_prob,
                probabilityChange: data.probability_change,
                momentum: data.momentum,
                confidence: data.confidence,
                isConnected: true,
              });
            }
          } catch {
            // Continue on parse errors
          }
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('Live stream error:', error);
      }
    } finally {
      setIsConnected(false);
    }
  }, [gameId, sport]);

  const disconnect = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsConnected(false);
  }, []);

  return {
    liveData,
    isConnected,
    connect,
    disconnect,
  };
}

/**
 * Hook for streaming odds movement alerts
 */
export function useOddsMovementStream(gameId: string, sport: string = 'NFL') {
  const [oddsData, setOddsData] = useState<{
    homeOdds: number;
    awayOdds: number;
    homeImpliedProb: number;
    movementType: 'stable' | 'moderate' | 'significant';
    alert?: {
      type: string;
      direction: string;
      magnitude: number;
      message: string;
    };
  } | null>(null);

  const [alerts, setAlerts] = useState<
    Array<{
      type: string;
      direction: string;
      magnitude: number;
      message: string;
      timestamp: Date;
    }>
  >([]);

  const abortControllerRef = useRef<AbortController | null>(null);

  const startMonitoring = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`/api/sports/stream/odds-movement/${gameId}?sport=${sport}`, {
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n').filter((line) => line.startsWith('data:'));

        for (const line of lines) {
          try {
            const data = JSON.parse(line.slice(5).trim());

            if (data.type === 'odds_update') {
              setOddsData({
                homeOdds: data.home_odds,
                awayOdds: data.away_odds,
                homeImpliedProb: data.home_implied_prob,
                movementType: data.movement_type,
                alert: data.alert,
              });

              if (data.alert) {
                setAlerts((prev) => [
                  { ...data.alert, timestamp: new Date() },
                  ...prev.slice(0, 19), // Keep last 20 alerts
                ]);
              }
            }
          } catch {
            // Continue on parse errors
          }
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('Odds stream error:', error);
      }
    }
  }, [gameId, sport]);

  const stopMonitoring = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  return {
    oddsData,
    alerts,
    startMonitoring,
    stopMonitoring,
  };
}

export default useStreamingPrediction;
