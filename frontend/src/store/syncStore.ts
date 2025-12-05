/**
 * Sync Store - Zustand State Management for Portfolio Synchronization
 * Simplified version without immer for better compatibility
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { useShallow } from 'zustand/react/shallow';

// Types - exported for use in components
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
    timestamp: string;
    duration_ms: number;
    stocks_count: number;
    options_count: number;
    success: boolean;
    error?: string;
}

export interface SyncRecommendation {
    should_sync: boolean;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    reason: string;
    details: string[];
}

// Stage configuration
export const SYNC_STAGES: Record<SyncStage, { progress: number; label: string }> = {
    idle: { progress: 0, label: 'Ready to sync' },
    connecting: { progress: 10, label: 'Connecting to Robinhood...' },
    authenticating: { progress: 20, label: 'Authenticating...' },
    fetching_stocks: { progress: 35, label: 'Fetching stock positions...' },
    fetching_options: { progress: 55, label: 'Fetching option positions...' },
    processing_greeks: { progress: 70, label: 'Calculating Greeks...' },
    calculating_analytics: { progress: 85, label: 'Running AI analytics...' },
    completing: { progress: 95, label: 'Finalizing...' },
    success: { progress: 100, label: 'Sync complete!' },
    error: { progress: 0, label: 'Sync failed' },
};

interface SyncState {
    // Current sync status
    stage: SyncStage;
    progress: number;
    currentAction: string;

    // Timestamps
    lastSuccessfulSyncAt: string | null;

    // History
    syncHistory: SyncResult[];

    // Health
    connectionHealth: ConnectionHealth;
    consecutiveFailures: number;

    // AI Recommendations
    recommendation: SyncRecommendation | null;

    // Statistics
    totalSyncs: number;
    successfulSyncs: number;
    averageSyncDuration: number;

    // Actions
    startSync: () => void;
    updateStage: (stage: SyncStage, progress: number, action?: string) => void;
    completeSync: (result: Omit<SyncResult, 'id' | 'timestamp'>) => void;
    failSync: (error: string) => void;
    updateHealth: (health: ConnectionHealth) => void;
    reset: () => void;
}

// Helper to generate unique ID
const generateId = () => `sync-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// Helper to calculate AI recommendation
const calculateRecommendation = (lastSync: string | null, failures: number): SyncRecommendation => {
    const now = new Date();
    const lastSyncDate = lastSync ? new Date(lastSync) : null;
    const minutesSinceSync = lastSyncDate
        ? Math.floor((now.getTime() - lastSyncDate.getTime()) / (1000 * 60))
        : Infinity;

    const details: string[] = [];
    let urgency: SyncRecommendation['urgency'] = 'low';
    let shouldSync = false;
    let reason = 'Portfolio is up to date';

    if (minutesSinceSync === Infinity) {
        shouldSync = true;
        urgency = 'critical';
        reason = 'No sync history found';
        details.push('Initial sync required to load your portfolio');
    } else if (minutesSinceSync > 120) {
        shouldSync = true;
        urgency = 'high';
        reason = `Last sync was ${Math.floor(minutesSinceSync / 60)} hours ago`;
        details.push('Market conditions may have changed significantly');
    } else if (minutesSinceSync > 30) {
        shouldSync = true;
        urgency = 'medium';
        reason = `Last sync was ${minutesSinceSync} minutes ago`;
        details.push('Consider syncing for latest prices');
    } else if (minutesSinceSync > 5) {
        reason = 'Recently synced';
        details.push(`Synced ${minutesSinceSync} minutes ago`);
    }

    if (failures >= 3) {
        shouldSync = true;
        urgency = 'critical';
        reason = 'Multiple sync failures detected';
        details.push('Check your Robinhood connection');
    }

    return { should_sync: shouldSync, urgency, reason, details };
};

export const useSyncStore = create<SyncState>()(
    persist(
        (set, get) => ({
            // Initial state
            stage: 'idle',
            progress: 0,
            currentAction: 'Ready to sync',
            lastSuccessfulSyncAt: null,
            syncHistory: [],
            connectionHealth: 'unknown',
            consecutiveFailures: 0,
            recommendation: null,
            totalSyncs: 0,
            successfulSyncs: 0,
            averageSyncDuration: 0,

            // Actions
            startSync: () => set({
                stage: 'connecting',
                progress: 10,
                currentAction: 'Connecting to Robinhood...',
            }),

            updateStage: (stage, progress, action) => set({
                stage,
                progress,
                currentAction: action || SYNC_STAGES[stage].label,
            }),

            completeSync: (result) => {
                const state = get();
                const syncResult: SyncResult = {
                    ...result,
                    id: generateId(),
                    timestamp: new Date().toISOString(),
                };

                const newSuccessfulSyncs = state.successfulSyncs + 1;
                const newAvgDuration =
                    (state.averageSyncDuration * (newSuccessfulSyncs - 1) + result.duration_ms) / newSuccessfulSyncs;

                set({
                    stage: 'success',
                    progress: 100,
                    currentAction: 'Sync complete!',
                    lastSuccessfulSyncAt: new Date().toISOString(),
                    consecutiveFailures: 0,
                    connectionHealth: 'healthy',
                    syncHistory: [syncResult, ...state.syncHistory].slice(0, 10),
                    totalSyncs: state.totalSyncs + 1,
                    successfulSyncs: newSuccessfulSyncs,
                    averageSyncDuration: newAvgDuration,
                    recommendation: calculateRecommendation(new Date().toISOString(), 0),
                });

                // Reset to idle after 3 seconds
                setTimeout(() => {
                    set({ stage: 'idle', progress: 0, currentAction: 'Ready to sync' });
                }, 3000);
            },

            failSync: (error) => {
                const state = get();
                const newFailures = state.consecutiveFailures + 1;

                set({
                    stage: 'error',
                    progress: 0,
                    currentAction: error,
                    consecutiveFailures: newFailures,
                    totalSyncs: state.totalSyncs + 1,
                    connectionHealth: newFailures >= 3 ? 'offline' : 'degraded',
                    recommendation: calculateRecommendation(state.lastSuccessfulSyncAt, newFailures),
                });
            },

            updateHealth: (health) => set({ connectionHealth: health }),

            reset: () => set({
                stage: 'idle',
                progress: 0,
                currentAction: 'Ready to sync',
                lastSuccessfulSyncAt: null,
                syncHistory: [],
                consecutiveFailures: 0,
                connectionHealth: 'unknown',
                recommendation: null,
                totalSyncs: 0,
                successfulSyncs: 0,
                averageSyncDuration: 0,
            }),
        }),
        {
            name: 'magnus-sync-store',
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({
                lastSuccessfulSyncAt: state.lastSuccessfulSyncAt,
                syncHistory: state.syncHistory,
                totalSyncs: state.totalSyncs,
                successfulSyncs: state.successfulSyncs,
                averageSyncDuration: state.averageSyncDuration,
            }),
        }
    )
);

// Optimized selector hooks with shallow equality comparison
// These prevent unnecessary re-renders when unrelated state changes

export const useSyncStatus = () => useSyncStore(
    useShallow((state) => ({
        stage: state.stage,
        progress: state.progress,
        currentAction: state.currentAction,
        isSyncing: !['idle', 'success', 'error'].includes(state.stage),
    }))
);

export const useSyncHealth = () => useSyncStore(
    useShallow((state) => ({
        connectionHealth: state.connectionHealth,
        consecutiveFailures: state.consecutiveFailures,
        recommendation: state.recommendation,
    }))
);

export const useSyncStats = () => useSyncStore(
    useShallow((state) => ({
        totalSyncs: state.totalSyncs,
        successfulSyncs: state.successfulSyncs,
        successRate: state.totalSyncs > 0 ? Math.round((state.successfulSyncs / state.totalSyncs) * 100) : 100,
        averageSyncDuration: state.averageSyncDuration,
        lastSuccessfulSyncAt: state.lastSuccessfulSyncAt,
    }))
);

// Individual atomic selectors for maximum performance
export const useSyncStage = () => useSyncStore((state) => state.stage);
export const useSyncProgress = () => useSyncStore((state) => state.progress);
export const useIsSyncing = () => useSyncStore((state) => !['idle', 'success', 'error'].includes(state.stage));
export const useConnectionHealth = () => useSyncStore((state) => state.connectionHealth);
export const useLastSync = () => useSyncStore((state) => state.lastSuccessfulSyncAt);

// Action selectors (never cause re-renders)
export const useSyncActions = () => useSyncStore(
    useShallow((state) => ({
        startSync: state.startSync,
        updateStage: state.updateStage,
        completeSync: state.completeSync,
        failSync: state.failSync,
        updateHealth: state.updateHealth,
        reset: state.reset,
    }))
);
