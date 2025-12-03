// ============================================
// CENTRALIZED HOOKS EXPORT
// Import all hooks from this single file for convenience
// ============================================

// API Hooks - TanStack Query based
export {
    usePositions,
    usePortfolioSummary,
    useSyncPortfolio,
    useDashboardSummary,
    usePerformanceHistory,
    useResearch,
    useAnalyzeSymbol,
    useRefreshResearch,
    useLiveGames,
    useUpcomingGames,
    useSportsMarkets,
    useBestBets,
    usePredictGame,
    useScanPremiums,
    useOptionsChain,
    useStrategyAnalysis,
    useAgents,
    useInvokeAgent,
    useChat,
    useConversations,
    useWatchlists,
    usePredictionMarkets,
    useHealthCheck,
} from './useMagnusApi'

// AI Streaming Hooks
export {
    useStreamingAI,
    useAIStockAnalysis,
    useAITradeRecommendation,
} from './useStreamingAI'

// Utility Hooks
export {
    // State Management
    useDebounce,
    useThrottle,
    useLocalStorage,
    useSessionStorage,
    usePrevious,
    useToggle,
    useSafeState,

    // Responsive Design
    useMediaQuery,
    useIsMobile,
    useIsTablet,
    useIsDesktop,
    usePrefersDarkMode,
    usePrefersReducedMotion,
    useWindowSize,

    // DOM Interactions
    useClickOutside,
    useScrollPosition,
    useInView,
    useKeyPress,
    useCopyToClipboard,

    // Async Operations
    useAsync,
    useFetch,
    useIsMounted,

    // Timers
    useInterval,
    useTimeout,

    // Document
    useDocumentTitle,
} from './useUtilities'

// Re-export types
export type {
    // Add any types that need to be exported
}
