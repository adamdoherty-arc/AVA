/**
 * Sports WebSocket Hook
 * Real-time updates for live games, odds, and AI predictions
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { WS_URL } from '@/config/api';

// WebSocket message types
export type SportsUpdateType =
  | 'update'
  | 'score_change'
  | 'odds_move'
  | 'prediction'
  | 'alert'
  | 'initial';

export type SportsBroadcastChannel =
  | 'live_games'
  | 'odds_updates'
  | 'predictions'
  | 'alerts'
  | 'game_detail';

export interface SportsUpdate {
  channel: SportsBroadcastChannel;
  sport: string;
  data: any;
  timestamp: string;
  update_type: SportsUpdateType;
  game_id?: string;
}

export interface WebSocketMessage {
  type: string;
  channel?: string;
  sport?: string;
  game_id?: string;
  data?: any;
  message?: string;
  timestamp?: string;
}

export interface ConnectionState {
  isConnected: boolean;
  isReconnecting: boolean;
  reconnectAttempts: number;
  lastMessageTime: Date | null;
}

interface UseSportsWebSocketOptions {
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Reconnect interval in ms (default: 3000) */
  reconnectInterval?: number;
  /** Maximum reconnect attempts (default: 10) */
  maxReconnectAttempts?: number;
  /** Channels to subscribe to on connect */
  channels?: SportsBroadcastChannel[];
  /** Sports to filter updates by */
  sports?: string[];
  /** Specific game IDs to watch */
  gameIds?: string[];
  /** Callback for score changes */
  onScoreChange?: (update: SportsUpdate) => void;
  /** Callback for odds movements */
  onOddsMove?: (update: SportsUpdate) => void;
  /** Callback for AI prediction updates */
  onPrediction?: (update: SportsUpdate) => void;
  /** Callback for alerts */
  onAlert?: (update: SportsUpdate) => void;
}

export function useSportsWebSocket(options: UseSportsWebSocketOptions = {}) {
  const {
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    channels = ['live_games'],
    sports = [],
    gameIds = [],
    onScoreChange,
    onOddsMove,
    onPrediction,
    onAlert,
  } = options;

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Use ref for reconnect attempts to avoid stale closure in connect callback
  const reconnectAttemptsRef = useRef<number>(0);

  // State
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    isConnected: false,
    isReconnecting: false,
    reconnectAttempts: 0,
    lastMessageTime: null,
  });

  const [liveGames, setLiveGames] = useState<Map<string, any>>(new Map());
  const [latestUpdates, setLatestUpdates] = useState<SportsUpdate[]>([]);

  // Get WebSocket URL from centralized config
  const getWebSocketUrl = useCallback(() => {
    // Uses WS_URL from centralized config (port 8002)
    return `${WS_URL}/ws/sports`;
  }, []);

  // Generate unique client ID
  const getClientId = useCallback(() => {
    let clientId = sessionStorage.getItem('ws_client_id');
    if (!clientId) {
      clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('ws_client_id', clientId);
    }
    return clientId;
  }, []);

  // Send message to WebSocket
  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Subscribe to a channel
  const subscribe = useCallback((channel: SportsBroadcastChannel) => {
    sendMessage({ action: 'subscribe', channel });
  }, [sendMessage]);

  // Subscribe to a sport
  const subscribeSport = useCallback((sport: string) => {
    sendMessage({ action: 'subscribe_sport', sport });
  }, [sendMessage]);

  // Subscribe to a specific game
  const subscribeGame = useCallback((gameId: string) => {
    sendMessage({ action: 'subscribe_game', game_id: gameId });
  }, [sendMessage]);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      setConnectionState(prev => ({
        ...prev,
        lastMessageTime: new Date(),
      }));

      // Handle different message types
      switch (message.type) {
        case 'connected':
          console.log('Connected to sports WebSocket:', message.message);
          break;

        case 'subscribed':
        case 'subscribed_sport':
        case 'subscribed_game':
          console.log('Subscription confirmed:', message);
          break;

        case 'pong':
          // Heartbeat response
          break;

        default:
          // Handle data updates
          if (message.channel && message.data) {
            const update: SportsUpdate = {
              channel: message.channel as SportsBroadcastChannel,
              sport: message.sport || 'UNKNOWN',
              data: message.data,
              timestamp: message.timestamp || new Date().toISOString(),
              update_type: (message as any).update_type || 'update',
              game_id: (message as any).game_id,
            };

            // Update live games state
            if (update.game_id) {
              setLiveGames(prev => {
                const newMap = new Map(prev);
                newMap.set(update.game_id!, {
                  ...newMap.get(update.game_id!),
                  ...update.data,
                  lastUpdate: update.timestamp,
                });
                return newMap;
              });
            }

            // Add to latest updates (keep last 50)
            setLatestUpdates(prev => [update, ...prev].slice(0, 50));

            // Call specific callbacks based on update type
            switch (update.update_type) {
              case 'score_change':
                onScoreChange?.(update);
                break;
              case 'odds_move':
                onOddsMove?.(update);
                break;
              case 'prediction':
                onPrediction?.(update);
                break;
              case 'alert':
                onAlert?.(update);
                break;
            }
          }
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [onScoreChange, onOddsMove, onPrediction, onAlert]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const clientId = getClientId();
    const url = `${getWebSocketUrl()}?client_id=${clientId}`;

    try {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('Sports WebSocket connected');
        reconnectAttemptsRef.current = 0;
        setConnectionState({
          isConnected: true,
          isReconnecting: false,
          reconnectAttempts: 0,
          lastMessageTime: new Date(),
        });

        // Subscribe to channels
        channels.forEach(channel => subscribe(channel));

        // Subscribe to sports
        sports.forEach(sport => subscribeSport(sport));

        // Subscribe to specific games
        gameIds.forEach(gameId => subscribeGame(gameId));

        // Start heartbeat
        pingIntervalRef.current = setInterval(() => {
          sendMessage({ action: 'ping' });
        }, 30000);
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onclose = (event) => {
        console.log('Sports WebSocket closed:', event.code, event.reason);
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
        }));

        // Clear heartbeat
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Auto-reconnect using ref to avoid stale closure
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          setConnectionState(prev => ({
            ...prev,
            isReconnecting: true,
            reconnectAttempts: reconnectAttemptsRef.current,
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('Sports WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
    }
  }, [
    getWebSocketUrl,
    getClientId,
    channels,
    sports,
    gameIds,
    subscribe,
    subscribeSport,
    subscribeGame,
    sendMessage,
    handleMessage,
    autoReconnect,
    maxReconnectAttempts,
    reconnectInterval,
    // Removed connectionState.reconnectAttempts - using reconnectAttemptsRef to avoid stale closure
  ]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    reconnectAttemptsRef.current = 0;
    setConnectionState({
      isConnected: false,
      isReconnecting: false,
      reconnectAttempts: 0,
      lastMessageTime: null,
    });
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Get game by ID
  const getGame = useCallback((gameId: string) => {
    return liveGames.get(gameId);
  }, [liveGames]);

  // Get all games as array
  const getAllGames = useCallback(() => {
    return Array.from(liveGames.values());
  }, [liveGames]);

  return {
    // Connection state
    ...connectionState,

    // Data
    liveGames: getAllGames(),
    latestUpdates,

    // Methods
    connect,
    disconnect,
    subscribe,
    subscribeSport,
    subscribeGame,
    getGame,
    sendMessage,
  };
}

/**
 * Hook for subscribing to a specific game
 */
export function useGameUpdates(gameId: string) {
  const [gameData, setGameData] = useState<any>(null);
  const [scoreHistory, setScoreHistory] = useState<Array<{ time: Date; home: number; away: number }>>([]);

  const { isConnected, subscribeGame, getGame } = useSportsWebSocket({
    channels: ['live_games', 'predictions'],
    gameIds: [gameId],
    onScoreChange: (update) => {
      if (update.game_id === gameId) {
        setGameData((prev: any) => ({ ...prev, ...update.data }));
        setScoreHistory(prev => [
          ...prev,
          {
            time: new Date(),
            home: update.data.home_score,
            away: update.data.away_score,
          },
        ].slice(-100)); // Keep last 100 score updates
      }
    },
  });

  useEffect(() => {
    if (isConnected) {
      subscribeGame(gameId);
    }
  }, [isConnected, gameId, subscribeGame]);

  useEffect(() => {
    const game = getGame(gameId);
    if (game) {
      setGameData(game);
    }
  }, [gameId, getGame]);

  return {
    isConnected,
    gameData,
    scoreHistory,
  };
}

export default useSportsWebSocket;
