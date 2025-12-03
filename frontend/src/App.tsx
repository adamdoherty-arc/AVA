import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from './components/Layout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { AIProvider } from './contexts/AIContext'
import { NotificationProvider } from './contexts/NotificationContext'
import { Loader2 } from 'lucide-react'

// Query client with modern defaults
const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 60_000, // 1 minute
            gcTime: 300_000, // 5 minutes (formerly cacheTime)
            retry: 2,
            refetchOnWindowFocus: false
        }
    }
})

// Loading fallback component
function PageLoader() {
    return (
        <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
                <p className="text-slate-400 text-sm">Loading...</p>
            </div>
        </div>
    )
}

// Lazy load pages for code splitting - better initial load performance
// Dashboard & Core
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Positions = lazy(() => import('./pages/Positions'))
const Settings = lazy(() => import('./pages/Settings'))

// Trading
const PremiumScanner = lazy(() => import('./pages/PremiumScanner'))
const EarningsCalendar = lazy(() => import('./pages/EarningsCalendar'))
const XTradesWatchlists = lazy(() => import('./pages/XTradesWatchlists'))
const Research = lazy(() => import('./pages/Research'))
const Watchlist = lazy(() => import('./pages/Watchlist'))
const DatabaseWatchlist = lazy(() => import('./pages/DatabaseWatchlist'))
const TradingViewWatchlist = lazy(() => import('./pages/TradingViewWatchlist'))

// Options
const OptionsTradingHub = lazy(() => import('./pages/OptionsTradingHub'))
const AIOptionsAgent = lazy(() => import('./pages/AIOptionsAgent'))
const OptionsAnalysisHub = lazy(() => import('./pages/OptionsAnalysisHub'))
const OptionsAnalysis = lazy(() => import('./pages/OptionsAnalysis'))
const TechnicalIndicators = lazy(() => import('./pages/TechnicalIndicators'))
const CalendarSpreads = lazy(() => import('./pages/CalendarSpreads'))
const OptionsFlow = lazy(() => import('./pages/OptionsFlow'))
const PositionSizing = lazy(() => import('./pages/PositionSizing'))
const SupplyDemandZones = lazy(() => import('./pages/SupplyDemandZones'))
const OptionsGreeks = lazy(() => import('./pages/OptionsGreeks'))

// Technical Analysis
const SignalDashboard = lazy(() => import('./pages/SignalDashboard'))
const SmartMoneyConcepts = lazy(() => import('./pages/SmartMoneyConcepts'))
const VolumeAnalysis = lazy(() => import('./pages/VolumeAnalysis'))
const FibonacciAnalysis = lazy(() => import('./pages/FibonacciAnalysis'))
const IchimokuCloud = lazy(() => import('./pages/IchimokuCloud'))

// Sports Betting
const SportsBettingHub = lazy(() => import('./pages/SportsBettingHub'))
const BestBetsUnified = lazy(() => import('./pages/BestBetsUnified'))
const GameCards = lazy(() => import('./pages/GameCards'))
const PredictionMarkets = lazy(() => import('./pages/PredictionMarkets'))
const KalshiMarkets = lazy(() => import('./pages/KalshiMarkets'))

// Analysis
const SectorAnalysis = lazy(() => import('./pages/SectorAnalysis'))
const MarketSentiment = lazy(() => import('./pages/MarketSentiment'))
const MultiAgentResearch = lazy(() => import('./pages/MultiAgentResearch'))
const AnalyticsPerformance = lazy(() => import('./pages/AnalyticsPerformance'))

// Portfolio
const RiskDashboard = lazy(() => import('./pages/RiskDashboard'))
const TradeJournal = lazy(() => import('./pages/TradeJournal'))
const DividendTracker = lazy(() => import('./pages/DividendTracker'))
const TaxLotOptimizer = lazy(() => import('./pages/TaxLotOptimizer'))

// Tools
const Backtesting = lazy(() => import('./pages/Backtesting'))
const AlertManagement = lazy(() => import('./pages/AlertManagement'))

// AI
const Chat = lazy(() => import('./pages/Chat'))
const AgentManagement = lazy(() => import('./pages/AgentManagement'))
const RAGKnowledgeBase = lazy(() => import('./pages/RAGKnowledgeBase'))
const DiscordMessages = lazy(() => import('./pages/DiscordMessages'))

// Development
const EnhancementManager = lazy(() => import('./pages/EnhancementManager'))
const EnhancementAgent = lazy(() => import('./pages/EnhancementAgent'))
const EnhancementQA = lazy(() => import('./pages/EnhancementQA'))

// System
const HealthDashboard = lazy(() => import('./pages/HealthDashboard'))
const SystemMonitoringHub = lazy(() => import('./pages/SystemMonitoringHub'))
const SystemManagementHub = lazy(() => import('./pages/SystemManagementHub'))
const QADashboard = lazy(() => import('./pages/QADashboard'))
const CacheMetrics = lazy(() => import('./pages/CacheMetrics'))
const SubscriptionManagement = lazy(() => import('./pages/SubscriptionManagement'))
const IntegrationTest = lazy(() => import('./pages/IntegrationTest'))

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <AIProvider>
                <NotificationProvider>
                    <ErrorBoundary>
                        <BrowserRouter>
                        <Suspense fallback={<PageLoader />}>
                            <Routes>
                                <Route element={<Layout />}>
                                    {/* Dashboard */}
                                    <Route path="/" element={<Dashboard />} />

                                    {/* Trading */}
                                    <Route path="/positions" element={<Positions />} />
                                    <Route path="/scanner" element={<PremiumScanner />} />
                                    <Route path="/earnings" element={<EarningsCalendar />} />
                                    <Route path="/xtrades" element={<XTradesWatchlists />} />
                                    <Route path="/research" element={<Research />} />
                                    <Route path="/watchlist" element={<Watchlist />} />
                                    <Route path="/db-watchlist" element={<DatabaseWatchlist />} />
                                    <Route path="/tv-watchlist" element={<TradingViewWatchlist />} />

                                    {/* Options */}
                                    <Route path="/options-hub" element={<OptionsTradingHub />} />
                                    <Route path="/ai-options" element={<AIOptionsAgent />} />
                                    <Route path="/options-analysis-hub" element={<OptionsAnalysisHub />} />
                                    <Route path="/options-analysis" element={<OptionsAnalysis />} />
                                    <Route path="/technicals" element={<TechnicalIndicators />} />
                                    <Route path="/calendar-spreads" element={<CalendarSpreads />} />
                                    <Route path="/options-flow" element={<OptionsFlow />} />
                                    <Route path="/position-sizing" element={<PositionSizing />} />
                                    <Route path="/supply-demand" element={<SupplyDemandZones />} />

                                    {/* Technical Analysis */}
                                    <Route path="/signals" element={<SignalDashboard />} />
                                    <Route path="/smart-money" element={<SmartMoneyConcepts />} />
                                    <Route path="/volume-analysis" element={<VolumeAnalysis />} />
                                    <Route path="/options-greeks" element={<OptionsGreeks />} />
                                    <Route path="/fibonacci" element={<FibonacciAnalysis />} />
                                    <Route path="/ichimoku" element={<IchimokuCloud />} />

                                    {/* Sports Betting */}
                                    <Route path="/betting" element={<SportsBettingHub />} />
                                    <Route path="/best-bets" element={<BestBetsUnified />} />
                                    <Route path="/games" element={<GameCards />} />
                                    <Route path="/markets" element={<PredictionMarkets />} />
                                    <Route path="/kalshi" element={<KalshiMarkets />} />

                                    {/* Analysis */}
                                    <Route path="/sectors" element={<SectorAnalysis />} />
                                    <Route path="/sentiment" element={<MarketSentiment />} />
                                    <Route path="/multi-research" element={<MultiAgentResearch />} />
                                    <Route path="/analytics" element={<AnalyticsPerformance />} />

                                    {/* Portfolio */}
                                    <Route path="/risk" element={<RiskDashboard />} />
                                    <Route path="/journal" element={<TradeJournal />} />
                                    <Route path="/dividends" element={<DividendTracker />} />
                                    <Route path="/tax-optimizer" element={<TaxLotOptimizer />} />

                                    {/* Tools */}
                                    <Route path="/backtest" element={<Backtesting />} />
                                    <Route path="/alerts" element={<AlertManagement />} />

                                    {/* AI */}
                                    <Route path="/chat" element={<Chat />} />
                                    <Route path="/agents" element={<AgentManagement />} />
                                    <Route path="/knowledge" element={<RAGKnowledgeBase />} />
                                    <Route path="/discord" element={<DiscordMessages />} />

                                    {/* Development */}
                                    <Route path="/enhancements" element={<EnhancementManager />} />
                                    <Route path="/enhancements/agent" element={<EnhancementAgent />} />
                                    <Route path="/enhancements/qa" element={<EnhancementQA />} />

                                    {/* System */}
                                    <Route path="/health" element={<HealthDashboard />} />
                                    <Route path="/monitoring" element={<SystemMonitoringHub />} />
                                    <Route path="/system" element={<SystemManagementHub />} />
                                    <Route path="/qa" element={<QADashboard />} />
                                    <Route path="/cache" element={<CacheMetrics />} />
                                    <Route path="/subscriptions" element={<SubscriptionManagement />} />
                                    <Route path="/integration-test" element={<IntegrationTest />} />
                                    <Route path="/settings" element={<Settings />} />
                                </Route>
                            </Routes>
                        </Suspense>
                        </BrowserRouter>
                    </ErrorBoundary>
                </NotificationProvider>
            </AIProvider>
        </QueryClientProvider>
    )
}

export default App
