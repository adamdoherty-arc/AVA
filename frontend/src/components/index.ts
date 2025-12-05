// ============================================
// CENTRALIZED COMPONENTS EXPORT
// Import all components from this single file
// ============================================

// Layout Components
export { Layout } from './Layout'
export { Sidebar } from './Sidebar'

// Error Handling
export { ErrorBoundary, withErrorBoundary } from './ErrorBoundary'
export {
    AIErrorBoundary,
    ErrorReportProvider,
    useErrorReports
} from './AIErrorBoundary'
export { APIErrorDisplay, InlineAPIError } from './APIErrorDisplay'

// Loading States
export {
    Skeleton,
    CardSkeleton,
    TableRowSkeleton,
    TableSkeleton,
    StatsSkeleton,
    ChartSkeleton,
    OptionsChainSkeleton,
    StockAnalysisSkeleton,
    ListItemSkeleton,
    DashboardSkeleton,
    AIResponseSkeleton,
} from './Skeleton'

// AI Components
export { AIAnalysisWidget, QuickAnalysisButton } from './AIAnalysisWidget'
export { AIResearchWidget } from './AIResearchWidget'
export { AvaChatWidget } from './AvaChatWidget'

// AI Notifications
export {
    NotificationProvider,
    useNotifications,
    createAIInsightNotification,
    createMarketAlertNotification
} from './AINotifications'

// Sync Status
export { default as SyncStatusPanel } from './SyncStatusPanel'

// Trading Components
export { PositionsTable } from './PositionsTable'
export { StrategyCard } from './StrategyCard'

// Optimized/Memoized Components
export {
    StatCard,
    DataTable,
    Badge,
    ProgressBar,
    MetricDisplay,
    LoadingButton,
    EmptyState,
    VirtualizedList,
} from './OptimizedComponents'
