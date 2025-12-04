/**
 * Sync Store - Zustand State Management for Portfolio Synchronization
 *
 * Features:
 * - Persistent sync history and timestamps
 * - Multi-stage sync progress tracking
 * - AI-powered sync recommendations
 * - Connection health monitoring
 * - Sync statistics and analytics
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Types
export type SyncStage =
    | 'idle'
    | 'connecting'
    | 'authenticating'
    | 'fetching_stocks'
    | 'fetching_options'
    | 'processing_greeks'
    | 'calculating_analytics'
    | 'completing'
    | 'success'
    | 'error';

export type ConnectionHealth = 'healthy' | 'degraded' | 'offline' | 'unknown';

export interface SyncResult {
    id: string;
    timestamp: Date;
    duration_ms: number;
    stocks_count: number;
    options_count: number;
    total_value: number;
    changes: SyncChange[];
    success: boolean;
    error?: string;
}

export interface SyncChange {
    type: 'added' | 'removed' | 'updated';
    symbol: string;
    field?: string;
    old_value?: string | number;
    new_value?: string | number;
}

export interface SyncRecommendation {
    should_sync: boolean;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    reason: string;
    details: string[];
    next_recommended_sync: Date;
}

export interface SyncState {
    // Current sync status
    stage: SyncStage;
    progress: number; // 0-100
    currentAction: string;

    // Timestamps
    lastSyncAt: Date | null;
    lastSuccessfulSyncAt: Date | null;
    nextScheduledSync: Date | null;

    // History
    syncHistory: SyncResult[];
    maxHistorySize: number;

    // Health
    connectionHealth: ConnectionHealth;
    lastHealthCheck: Date | null;
    consecutiveFailures: number;

    // AI Recommendations
    recommendation: SyncRecommendation | null;

    // Statistics
    totalSyncs: number;
    successfulSyncs: number;
    averageSyncDuration: number;

    // Auto-sync settings
    autoSyncEnabled: boolean;
    autoSyncIntervalMinutes: number;

    // Actions
    startSync: () => void;
    updateStage: (stage: SyncStage, progress: number, action?: string) => void;
    completeSync: (result: Omit<SyncResult, 'id' | 'timestamp'>) => void;
    failSync: (error: string) => void;
    updateHealth: (health: ConnectionHealth) => void;
    updateRecommendation: (recommendation: SyncRecommendation) => void;
    setAutoSync: (enabled: boolean, intervalMinutes?: number) => void;
    clearHistory: () => void;
    reset: () => void;
}

// Stage progress mapping
export const SYNC_STAGES: Record<SyncStage, { progress: number; label: string; icon: string }> = {
    idle: { progress: 0, label: 'Ready to sync', icon: 'circle' },
    connecting: { progress: 10, label: 'Connecting to Robinhood...', icon: 'wifi' },
    authenticating: { progress: 20, label: 'Authenticating...', icon: 'shield' },
    fetching_stocks: { progress: 35, label: 'Fetching stock positions...', icon: 'trending-up' },
    fetching_options: { progress: 55, label: 'Fetching option positions...', icon: 'target' },
    processing_greeks: { progress: 70, label: 'Calculating Greeks...', icon: 'calculator' },
    calculating_analytics: { progress: 85, label: 'Running AI analytics...', icon: 'brain' },
    completing: { progress: 95, label: 'Finalizing...', icon: 'check' },
    success: { progress: 100, label: 'Sync complete!', icon: 'check-circle' },
    error: { progress: 0, label: 'Sync failed', icon: 'alert-circle' },
};

// Helper to generate unique ID
const generateId = () => `sync-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// Helper to calculate AI recommendation
const calculateRecommendation = (state: Partial<SyncState>): SyncRecommendation => {
    const now = new Date();
    const lastSync = state.lastSuccessfulSyncAt ? new Date(state.lastSuccessfulSyncAt) : null;
    const minutesSinceSync = lastSync
        ? Math.floor((now.getTime() - lastSync.getTime()) / (1000 * 60))
        : Infinity;

    const details: string[] = [];
    let urgency: SyncRecommendation['urgency'] = 'low';
    let shouldSync = false;
    let reason = 'Portfolio is up to date';

    // Check time since last sync
    if (minutesSinceSync === Infinity) {
        shouldSync = true;
        urgency = 'critical';
        reason = 'No sync history found';
        details.push('Initial sync required to load your portfolio');
    } else if (minutesSinceSync > 120) { // 2 hours
        shouldSync = true;
        urgency = 'high';
        reason = `Last sync was ${Math.floor(minutesSinceSync / 60)} hours ago`;
        details.push('Market conditions may have changed significantly');
        details.push('Option Greeks need recalculation');
    } else if (minutesSinceSync > 30) {
        shouldSync = true;
        urgency = 'medium';
        reason = `Last sync was ${minutesSinceSync} minutes ago`;
        details.push('Consider syncing for latest prices');
    } else if (minutesSinceSync > 5) {
        urgency = 'low';
        reason = 'Recently synced';
        details.push(`Synced ${minutesSinceSync} minutes ago`);
    }

    // Check for consecutive failures
    if ((state.consecutiveFailures || 0) >= 3) {
        shouldSync = true;
        urgency = 'critical';
        reason = 'Multiple sync failures detected';
        details.push('Check your Robinhood connection');
        details.push('Verify your credentials are valid');
    }

    // Check market hours (rough estimate - 9:30 AM to 4 PM ET)
    const hour = now.getHours();
    const isMarketHours = hour >= 9 && hour < 16;
    if (isMarketHours && minutesSinceSync > 15) {
        if (urgency === 'low') urgency = 'medium';
        details.push('Market is currently open - prices are moving');
    }

    // Calculate next recommended sync
    const nextSync = new Date(now.getTime() + (shouldSync ? 0 : 30 * 60 * 1000));

    return {
        should_sync: shouldSync,
        urgency,
        reason,
        details,
        next_recommended_sync: nextSync,
    };
};

export const useSyncStore = create<SyncState>()(
    persist(
        immer((set, get) => ({
            // Initial state
            stage: 'idle',
            progress: 0,
            currentAction: 'Ready to sync',

            lastSyncAt: null,
            lastSuccessfulSyncAt: null,
            nextScheduledSync: null,

            syncHistory: [],
            maxHistorySize: 50,

            connectionHealth: 'unknown',
            lastHealthCheck: null,
            consecutiveFailures: 0,

            recommendation: null,

            totalSyncs: 0,
            successfulSyncs: 0,
            averageSyncDuration: 0,

            autoSyncEnabled: false,
            autoSyncIntervalMinutes: 30,

            // Actions
            startSync: () => set((state) => {
                state.stage = 'connecting';
                state.progress = 10;
                state.currentAction = 'Connecting to Robinhood...';
                state.lastSyncAt = new Date();
            }),

            updateStage: (stage, progress, action) => set((state) => {
                state.stage = stage;
                state.progress = progress;
                state.currentAction = action || SYNC_STAGES[stage].label;
            }),

            completeSync: (result) => set((state) => {
                const syncResult: SyncResult = {
                    ...result,
                    id: generateId(),
                    timestamp: new Date(),
                };

                state.stage = 'success';
                state.progress = 100;
                state.currentAction = 'Sync complete!';
                state.lastSuccessfulSyncAt = new Date();
                state.consecutiveFailures = 0;
                state.connectionHealth = 'healthy';
                state.lastHealthCheck = new Date();

                // Update history
                state.syncHistory.unshift(syncResult);
                if (state.syncHistory.length > state.maxHistorySize) {
                    state.syncHistory = state.syncHistory.slice(0, state.maxHistorySize);
                }

                // Update statistics
                state.totalSyncs += 1;
                state.successfulSyncs += 1;
                state.averageSyncDuration =
                    (state.averageSyncDuration * (state.successfulSyncs - 1) + result.duration_ms) /
                    state.successfulSyncs;

                // Update recommendation
                state.recommendation = calculateRecommendation(state);

                // Reset to idle after 3 seconds
                setTimeout(() => {
                    set((s) => {
                        s.stage = 'idle';
                        s.progress = 0;
                        s.currentAction = 'Ready to sync';
                    });
                }, 3000);
            }),

            failSync: (error) => set((state) => {
                state.stage = 'error';
                state.progress = 0;
                state.currentAction = error;
                state.consecutiveFailures += 1;
                state.totalSyncs += 1;

                if (state.consecutiveFailures >= 3) {
                    state.connectionHealth = 'offline';
                } else if (state.consecutiveFailures >= 1) {
                    state.connectionHealth = 'degraded';
                }

                state.recommendation = calculateRecommendation(state);
            }),

            updateHealth: (health) => set((state) => {
                state.connectionHealth = health;
                state.lastHealthCheck = new Date();
            }),

            updateRecommendation: (recommendation) => set((state) => {
                state.recommendation = recommendation;
            }),

            setAutoSync: (enabled, intervalMinutes) => set((state) => {
                state.autoSyncEnabled = enabled;
                if (intervalMinutes !== undefined) {
                    state.autoSyncIntervalMinutes = intervalMinutes;
                }
                if (enabled) {
                    state.nextScheduledSync = new Date(
                        Date.now() + (intervalMinutes || state.autoSyncIntervalMinutes) * 60 * 1000
                    );
                } else {
                    state.nextScheduledSync = null;
                }
            }),

            clearHistory: () => set((state) => {
                state.syncHistory = [];
            }),

            reset: () => set((state) => {
                state.stage = 'idle';
                state.progress = 0;
                state.currentAction = 'Ready to sync';
                state.lastSyncAt = null;
                state.lastSuccessfulSyncAt = null;
                state.syncHistory = [];
                state.consecutiveFailures = 0;
                state.connectionHealth = 'unknown';
                state.recommendation = null;
                state.totalSyncs = 0;
                state.successfulSyncs = 0;
                state.averageSyncDuration = 0;
            }),
        })),
        {
            name: 'magnus-sync-store',
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({
                lastSyncAt: state.lastSyncAt,
                lastSuccessfulSyncAt: state.lastSuccessfulSyncAt,
                syncHistory: state.syncHistory.slice(0, 10), // Only persist last 10
                totalSyncs: state.totalSyncs,
                successfulSyncs: state.successfulSyncs,
                averageSyncDuration: state.averageSyncDuration,
                autoSyncEnabled: state.autoSyncEnabled,
                autoSyncIntervalMinutes: state.autoSyncIntervalMinutes,
            }),
        }
    )
);

// Selector hooks for common use cases
export const useSyncStatus = () => useSyncStore((state) => ({
    stage: state.stage,
    progress: state.progress,
    currentAction: state.currentAction,
    issyncing: !['idle', 'success', 'error'].includes(state.stage),
}));

export const useSyncHealth = () => useSyncStore((state) => ({
    connectionHealth: state.connectionHealth,
    lastHealthCheck: state.lastHealthCheck,
    consecutiveFailures: state.consecutiveFailures,
    recommendation: state.recommendation,
}));

export const useSyncStats = () => useSyncStore((state) => ({
    totalSyncs: state.totalSyncs,
    successfulSyncs: state.successfulSyncs,
    successRate: state.totalSyncs > 0
        ? Math.round((state.successfulSyncs / state.totalSyncs) * 100)
        : 100,
    averageSyncDuration: state.averageSyncDuration,
    lastSuccessfulSyncAt: state.lastSuccessfulSyncAt,
}));
