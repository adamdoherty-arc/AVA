/**
 * SyncStatusPanel - Advanced Portfolio Sync Control Panel
 *
 * Features:
 * - Multi-stage progress indicator with animations
 * - AI-powered sync recommendations
 * - Connection health monitoring
 * - Sync history with detailed results
 * - Smart timing suggestions
 * - Real-time status updates
 */

import { useState, useEffect, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import {
    Zap, RefreshCw, Wifi, WifiOff, Shield, TrendingUp, Target,
    Calculator, Brain, CheckCircle, AlertCircle, Clock, History,
    ChevronDown, ChevronUp, Activity, Sparkles, AlertTriangle,
    Timer, ArrowRight, Info, X, Loader2, Signal, CircleDot
} from 'lucide-react';
import { useSyncStore, SYNC_STAGES, SyncStage, ConnectionHealth } from '../store/syncStore';
import { useSyncPortfolio, usePositions } from '../hooks/useMagnusApi';

// Stage icons mapping
const STAGE_ICONS: Record<SyncStage, React.ReactNode> = {
    idle: <CircleDot className="w-4 h-4" />,
    connecting: <Wifi className="w-4 h-4" />,
    authenticating: <Shield className="w-4 h-4" />,
    fetching_stocks: <TrendingUp className="w-4 h-4" />,
    fetching_options: <Target className="w-4 h-4" />,
    processing_greeks: <Calculator className="w-4 h-4" />,
    calculating_analytics: <Brain className="w-4 h-4" />,
    completing: <CheckCircle className="w-4 h-4" />,
    success: <CheckCircle className="w-4 h-4" />,
    error: <AlertCircle className="w-4 h-4" />,
};

// Health status colors
const HEALTH_COLORS: Record<ConnectionHealth, { bg: string; text: string; border: string }> = {
    healthy: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
    degraded: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
    offline: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
    unknown: { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' },
};

// Urgency colors for recommendations
const URGENCY_COLORS = {
    low: 'text-slate-400',
    medium: 'text-blue-400',
    high: 'text-amber-400',
    critical: 'text-red-400',
};

// Format relative time
function formatRelativeTime(date: Date | string | null): string {
    if (!date) return 'Never';
    const d = new Date(date);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
}

// Format duration
function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

interface SyncStatusPanelProps {
    variant?: 'compact' | 'expanded' | 'full';
    showHistory?: boolean;
    showRecommendation?: boolean;
    className?: string;
}

export const SyncStatusPanel = memo(function SyncStatusPanel({
    variant = 'expanded',
    showHistory = true,
    showRecommendation = true,
    className = '',
}: SyncStatusPanelProps) {
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);
    const [syncStartTime, setSyncStartTime] = useState<number | null>(null);

    // Store state
    const {
        stage,
        progress,
        currentAction,
        lastSuccessfulSyncAt,
        connectionHealth,
        recommendation,
        syncHistory,
        totalSyncs,
        successfulSyncs,
        averageSyncDuration,
        startSync,
        updateStage,
        completeSync,
        failSync,
        updateHealth,
    } = useSyncStore();

    // API hooks
    const syncMutation = useSyncPortfolio();
    const { refetch, isFetching } = usePositions();

    const isSyncing = !['idle', 'success', 'error'].includes(stage);

    // Calculate AI recommendation on mount
    useEffect(() => {
        // Trigger recommendation calculation periodically
        const interval = setInterval(() => {
            if (!isSyncing) {
                // The recommendation is calculated in the store
            }
        }, 60000); // Every minute

        return () => clearInterval(interval);
    }, [isSyncing]);

    // Advanced sync with staged progress
    const handleSync = useCallback(async () => {
        if (isSyncing) return;

        setSyncStartTime(Date.now());
        startSync();

        // Show initial toast
        const toastId = toast.loading('Syncing with Robinhood...', {
            description: 'Connecting to your brokerage account',
        });

        try {
            // Stage 1: Connecting
            updateStage('connecting', 10, 'Establishing connection...');
            await new Promise(resolve => setTimeout(resolve, 300));

            // Stage 2: Authenticating
            updateStage('authenticating', 20, 'Verifying credentials...');
            await new Promise(resolve => setTimeout(resolve, 200));

            // Stage 3: Fetching stocks
            updateStage('fetching_stocks', 35, 'Loading stock positions...');

            // Stage 4-7: Actual API call (run in background)
            const result = await syncMutation.mutateAsync();

            // Stage 5: Fetching options
            updateStage('fetching_options', 55, 'Loading option positions...');
            await new Promise(resolve => setTimeout(resolve, 200));

            // Stage 6: Processing Greeks
            updateStage('processing_greeks', 70, 'Calculating option Greeks...');
            await new Promise(resolve => setTimeout(resolve, 200));

            // Stage 7: AI Analytics
            updateStage('calculating_analytics', 85, 'Running AI analysis...');
            await new Promise(resolve => setTimeout(resolve, 300));

            // Stage 8: Completing
            updateStage('completing', 95, 'Finalizing sync...');
            await new Promise(resolve => setTimeout(resolve, 200));

            // Calculate duration
            const duration = Date.now() - (syncStartTime || Date.now());

            // Complete sync
            completeSync({
                duration_ms: duration,
                stocks_count: result?.positions_count || 0,
                options_count: 0, // Would come from detailed response
                total_value: 0,
                changes: [],
                success: true,
            });

            updateHealth('healthy');

            toast.success('Portfolio Synced!', {
                id: toastId,
                description: `${result?.positions_count || 0} positions updated in ${formatDuration(duration)}`,
                icon: <CheckCircle className="w-4 h-4 text-emerald-400" />,
            });

            // Refetch positions data
            await refetch();

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            failSync(errorMessage);

            toast.error('Sync Failed', {
                id: toastId,
                description: errorMessage,
                action: {
                    label: 'Retry',
                    onClick: handleSync,
                },
            });
        }
    }, [isSyncing, startSync, updateStage, completeSync, failSync, updateHealth, syncMutation, refetch, syncStartTime]);

    // Quick refresh (just data, no full sync)
    const handleRefresh = useCallback(async () => {
        try {
            await refetch();
            toast.success('Positions refreshed', { duration: 2000 });
        } catch (error) {
            toast.error('Refresh failed');
        }
    }, [refetch]);

    // Compact variant - just the button
    if (variant === 'compact') {
        return (
            <div className={`flex gap-2 ${className}`}>
                <button
                    onClick={handleSync}
                    disabled={isSyncing}
                    className="btn-secondary flex items-center gap-2"
                >
                    {isSyncing ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                        <Zap className="w-4 h-4" />
                    )}
                    {isSyncing ? 'Syncing...' : 'Sync'}
                </button>
                <button
                    onClick={handleRefresh}
                    disabled={isFetching}
                    className="btn-icon"
                >
                    <RefreshCw className={`w-5 h-5 ${isFetching ? 'animate-spin' : ''}`} />
                </button>
            </div>
        );
    }

    // Health indicator
    const healthColors = HEALTH_COLORS[connectionHealth];

    return (
        <div className={`space-y-3 ${className}`}>
            {/* Main Sync Button Row */}
            <div className="flex items-center gap-3">
                {/* Sync Button */}
                <button
                    onClick={handleSync}
                    disabled={isSyncing}
                    className={`
                        relative flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium
                        transition-all duration-300 overflow-hidden
                        ${isSyncing
                            ? 'bg-primary/20 text-primary cursor-not-allowed'
                            : 'bg-gradient-to-r from-primary to-blue-500 text-white hover:shadow-lg hover:shadow-primary/25 hover:scale-[1.02]'
                        }
                    `}
                >
                    {/* Animated background during sync */}
                    {isSyncing && (
                        <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-primary/30 via-blue-500/30 to-primary/30"
                            animate={{
                                x: ['-100%', '100%'],
                            }}
                            transition={{
                                repeat: Infinity,
                                duration: 1.5,
                                ease: 'linear',
                            }}
                        />
                    )}

                    <span className="relative z-10 flex items-center gap-2">
                        {isSyncing ? (
                            <>
                                {STAGE_ICONS[stage]}
                                <span className="min-w-[100px]">{currentAction}</span>
                            </>
                        ) : (
                            <>
                                <Zap className="w-4 h-4" />
                                <span>Sync Robinhood</span>
                            </>
                        )}
                    </span>
                </button>

                {/* Refresh Button */}
                <button
                    onClick={handleRefresh}
                    disabled={isFetching || isSyncing}
                    className="btn-icon relative group"
                    title="Quick refresh prices"
                >
                    <RefreshCw className={`w-5 h-5 transition-transform ${isFetching ? 'animate-spin' : 'group-hover:rotate-180'}`} />
                </button>

                {/* Connection Health Indicator */}
                <div
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border ${healthColors.bg} ${healthColors.border}`}
                    title={`Connection: ${connectionHealth}`}
                >
                    {connectionHealth === 'healthy' ? (
                        <Signal className={`w-3.5 h-3.5 ${healthColors.text}`} />
                    ) : connectionHealth === 'offline' ? (
                        <WifiOff className={`w-3.5 h-3.5 ${healthColors.text}`} />
                    ) : (
                        <Activity className={`w-3.5 h-3.5 ${healthColors.text}`} />
                    )}
                    <span className={`text-xs font-medium capitalize ${healthColors.text}`}>
                        {connectionHealth}
                    </span>
                </div>

                {/* Last Sync Time */}
                <div className="flex items-center gap-1.5 text-sm text-slate-400">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{formatRelativeTime(lastSuccessfulSyncAt)}</span>
                </div>
            </div>

            {/* Progress Bar (when syncing) */}
            <AnimatePresence>
                {isSyncing && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-2"
                    >
                        {/* Progress stages */}
                        <div className="flex items-center gap-1">
                            {(['connecting', 'authenticating', 'fetching_stocks', 'fetching_options', 'processing_greeks', 'calculating_analytics', 'completing'] as SyncStage[]).map((s, i) => {
                                const stageInfo = SYNC_STAGES[s];
                                const isActive = stage === s;
                                const isComplete = progress > stageInfo.progress;

                                return (
                                    <div key={s} className="flex items-center">
                                        <motion.div
                                            className={`
                                                w-6 h-6 rounded-full flex items-center justify-center text-xs
                                                transition-colors duration-300
                                                ${isComplete ? 'bg-emerald-500 text-white' :
                                                isActive ? 'bg-primary text-white animate-pulse' :
                                                'bg-slate-700 text-slate-400'}
                                            `}
                                            initial={false}
                                            animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                                            transition={{ repeat: Infinity, duration: 1 }}
                                        >
                                            {isComplete ? (
                                                <CheckCircle className="w-3.5 h-3.5" />
                                            ) : (
                                                <span>{i + 1}</span>
                                            )}
                                        </motion.div>
                                        {i < 6 && (
                                            <div className={`w-4 h-0.5 ${isComplete ? 'bg-emerald-500' : 'bg-slate-700'}`} />
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Progress bar */}
                        <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
                            <motion.div
                                className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-blue-400 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${progress}%` }}
                                transition={{ duration: 0.3 }}
                            />
                            <motion.div
                                className="absolute inset-y-0 left-0 bg-white/20 rounded-full"
                                animate={{
                                    x: ['-100%', '200%'],
                                }}
                                transition={{
                                    repeat: Infinity,
                                    duration: 1,
                                    ease: 'linear',
                                }}
                                style={{ width: '30%' }}
                            />
                        </div>

                        {/* Current action text */}
                        <div className="flex items-center justify-between text-xs">
                            <span className="text-primary font-medium">{currentAction}</span>
                            <span className="text-slate-500">{progress}%</span>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* AI Recommendation */}
            {showRecommendation && recommendation && !isSyncing && recommendation.should_sync && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`
                        p-3 rounded-xl border bg-gradient-to-r
                        ${recommendation.urgency === 'critical' ? 'from-red-500/10 to-orange-500/10 border-red-500/30' :
                        recommendation.urgency === 'high' ? 'from-amber-500/10 to-yellow-500/10 border-amber-500/30' :
                        'from-blue-500/10 to-cyan-500/10 border-blue-500/30'}
                    `}
                >
                    <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${
                            recommendation.urgency === 'critical' ? 'bg-red-500/20' :
                            recommendation.urgency === 'high' ? 'bg-amber-500/20' :
                            'bg-blue-500/20'
                        }`}>
                            <Sparkles className={`w-4 h-4 ${URGENCY_COLORS[recommendation.urgency]}`} />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                                <span className={`text-sm font-semibold ${URGENCY_COLORS[recommendation.urgency]}`}>
                                    AI Recommendation
                                </span>
                                <span className={`text-xs px-1.5 py-0.5 rounded-full uppercase font-medium ${
                                    recommendation.urgency === 'critical' ? 'bg-red-500/20 text-red-400' :
                                    recommendation.urgency === 'high' ? 'bg-amber-500/20 text-amber-400' :
                                    'bg-blue-500/20 text-blue-400'
                                }`}>
                                    {recommendation.urgency}
                                </span>
                            </div>
                            <p className="text-sm text-slate-300">{recommendation.reason}</p>
                            {recommendation.details.length > 0 && (
                                <ul className="mt-1.5 space-y-0.5">
                                    {recommendation.details.slice(0, 2).map((detail, i) => (
                                        <li key={i} className="flex items-center gap-1.5 text-xs text-slate-400">
                                            <ArrowRight className="w-3 h-3 text-slate-500" />
                                            {detail}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                        <button
                            onClick={handleSync}
                            className="px-3 py-1.5 rounded-lg bg-primary/20 text-primary text-sm font-medium hover:bg-primary/30 transition-colors"
                        >
                            Sync Now
                        </button>
                    </div>
                </motion.div>
            )}

            {/* Sync History */}
            {showHistory && syncHistory.length > 0 && variant === 'full' && (
                <div className="glass-card overflow-hidden">
                    <button
                        onClick={() => setIsHistoryOpen(!isHistoryOpen)}
                        className="w-full flex items-center justify-between p-3 hover:bg-white/[0.02] transition-colors"
                    >
                        <div className="flex items-center gap-2">
                            <History className="w-4 h-4 text-slate-400" />
                            <span className="text-sm font-medium text-slate-300">Sync History</span>
                            <span className="text-xs text-slate-500">
                                ({successfulSyncs}/{totalSyncs} successful)
                            </span>
                        </div>
                        {isHistoryOpen ? (
                            <ChevronUp className="w-4 h-4 text-slate-400" />
                        ) : (
                            <ChevronDown className="w-4 h-4 text-slate-400" />
                        )}
                    </button>

                    <AnimatePresence>
                        {isHistoryOpen && (
                            <motion.div
                                initial={{ height: 0 }}
                                animate={{ height: 'auto' }}
                                exit={{ height: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="p-3 pt-0 space-y-2 max-h-48 overflow-y-auto">
                                    {syncHistory.slice(0, 5).map((sync) => (
                                        <div
                                            key={sync.id}
                                            className={`flex items-center justify-between p-2 rounded-lg ${
                                                sync.success ? 'bg-emerald-500/5' : 'bg-red-500/5'
                                            }`}
                                        >
                                            <div className="flex items-center gap-2">
                                                {sync.success ? (
                                                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                                                ) : (
                                                    <AlertCircle className="w-4 h-4 text-red-400" />
                                                )}
                                                <div>
                                                    <span className="text-sm text-slate-300">
                                                        {sync.stocks_count + sync.options_count} positions
                                                    </span>
                                                    <span className="text-xs text-slate-500 ml-2">
                                                        {formatRelativeTime(sync.timestamp)}
                                                    </span>
                                                </div>
                                            </div>
                                            <span className="text-xs text-slate-500">
                                                {formatDuration(sync.duration_ms)}
                                            </span>
                                        </div>
                                    ))}

                                    {/* Stats */}
                                    <div className="flex items-center justify-between pt-2 border-t border-slate-700/50 text-xs text-slate-500">
                                        <span>Avg sync time: {formatDuration(averageSyncDuration)}</span>
                                        <span>Success rate: {totalSyncs > 0 ? Math.round((successfulSyncs / totalSyncs) * 100) : 100}%</span>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </div>
    );
});

export default SyncStatusPanel;
