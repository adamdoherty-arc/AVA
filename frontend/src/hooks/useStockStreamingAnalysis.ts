/**
 * Stock Streaming Analysis Hook
 * SSE connection for real-time AI analysis
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { BACKEND_URL } from '@/config/api';

export interface AnalysisState {
  status: 'idle' | 'connecting' | 'analyzing' | 'complete' | 'error';
  symbol: string | null;

  // Progressive data
  priceData: {
    current_price: number;
    iv_estimate: number;
  } | null;

  technicals: {
    trend: string;
    strength: number;
    indicators: Record<string, number>;
  } | null;

  aiScore: {
    ai_score: number;
    recommendation: string;
    confidence: number;
    components: Record<string, { raw_score: number; weight: number }>;
  } | null;

  reasoning: string | null;
  llmAnalysis: string | null;  // Rich LLM-powered analysis
  error: string | null;
}

const initialState: AnalysisState = {
  status: 'idle',
  symbol: null,
  priceData: null,
  technicals: null,
  aiScore: null,
  reasoning: null,
  llmAnalysis: null,
  error: null,
};

export const useStockStreamingAnalysis = () => {
  const [state, setState] = useState<AnalysisState>(initialState);
  const eventSourceRef = useRef<EventSource | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // Start analysis for a symbol
  const startAnalysis = useCallback(
    (symbol: string) => {
      // Cleanup any existing connection
      cleanup();

      // Reset state
      setState({
        ...initialState,
        status: 'connecting',
        symbol: symbol.toUpperCase(),
      });

      // Create new EventSource - uses centralized config
      const apiUrl = BACKEND_URL;
      const url = `${apiUrl}/api/stocks/tiles/stream/analyze/${symbol.toUpperCase()}`;

      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      // Handle events
      eventSource.addEventListener('start', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          status: 'analyzing',
          symbol: data.symbol,
        }));
      });

      eventSource.addEventListener('price_data', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          priceData: {
            current_price: data.current_price,
            iv_estimate: data.iv_estimate,
          },
        }));
      });

      eventSource.addEventListener('technicals', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          technicals: {
            trend: data.trend,
            strength: data.strength,
            indicators: data.indicators,
          },
        }));
      });

      eventSource.addEventListener('ai_score', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          aiScore: {
            ai_score: data.ai_score,
            recommendation: data.recommendation,
            confidence: data.confidence,
            components: data.components,
          },
        }));
      });

      eventSource.addEventListener('reasoning', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          reasoning: data.text,
        }));
      });

      // LLM-powered rich analysis (optional, arrives after reasoning)
      eventSource.addEventListener('llm_analysis', (e) => {
        const data = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          llmAnalysis: data.text,
        }));
      });

      eventSource.addEventListener('complete', () => {
        setState((prev) => ({
          ...prev,
          status: 'complete',
        }));
        cleanup();
      });

      eventSource.addEventListener('error', (e) => {
        try {
          const data = JSON.parse((e as MessageEvent).data);
          setState((prev) => ({
            ...prev,
            status: 'error',
            error: data.error || 'Analysis failed',
          }));
        } catch {
          setState((prev) => ({
            ...prev,
            status: 'error',
            error: 'Connection error',
          }));
        }
        cleanup();
      });

      // Handle connection errors
      eventSource.onerror = () => {
        if (eventSource.readyState === EventSource.CLOSED) {
          // Only set error if we weren't expecting close
          setState((prev) => {
            if (prev.status !== 'complete') {
              return {
                ...prev,
                status: 'error',
                error: 'Connection lost',
              };
            }
            return prev;
          });
        }
        cleanup();
      };
    },
    [cleanup]
  );

  // Stop analysis
  const stopAnalysis = useCallback(() => {
    cleanup();
    setState((prev) => ({
      ...prev,
      status: 'idle',
    }));
  }, [cleanup]);

  // Reset to initial state
  const reset = useCallback(() => {
    cleanup();
    setState(initialState);
  }, [cleanup]);

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  return {
    ...state,
    startAnalysis,
    stopAnalysis,
    reset,
    isLoading: state.status === 'connecting' || state.status === 'analyzing',
    isComplete: state.status === 'complete',
    hasError: state.status === 'error',
  };
};

export default useStockStreamingAnalysis;
