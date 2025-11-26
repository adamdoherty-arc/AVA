import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { GameCards } from './pages/GameCards'
import { PredictionMarkets } from './pages/PredictionMarkets'
import { Chat } from './pages/Chat'
import Positions from './pages/Positions'
import Research from './pages/Research'
import Watchlist from './pages/Watchlist'
import SportsBettingHub from './pages/SportsBettingHub'
import PremiumScanner from './pages/PremiumScanner'
import Settings from './pages/Settings'
import EarningsCalendar from './pages/EarningsCalendar'
import BestBetsUnified from './pages/BestBetsUnified'
import AgentManagement from './pages/AgentManagement'
import XTradesWatchlists from './pages/XTradesWatchlists'
import CalendarSpreads from './pages/CalendarSpreads'
import OptionsFlow from './pages/OptionsFlow'
import PositionSizing from './pages/PositionSizing'
import KalshiMarkets from './pages/KalshiMarkets'
import SectorAnalysis from './pages/SectorAnalysis'
import MarketSentiment from './pages/MarketSentiment'
import MultiAgentResearch from './pages/MultiAgentResearch'
import RiskDashboard from './pages/RiskDashboard'
import TradeJournal from './pages/TradeJournal'
import Backtesting from './pages/Backtesting'
import AlertManagement from './pages/AlertManagement'
import DividendTracker from './pages/DividendTracker'
import TaxLotOptimizer from './pages/TaxLotOptimizer'
// New pages
import AIOptionsAgent from './pages/AIOptionsAgent'
import OptionsTradingHub from './pages/OptionsTradingHub'
import OptionsAnalysisHub from './pages/OptionsAnalysisHub'
import OptionsAnalysis from './pages/OptionsAnalysis'
import TechnicalIndicators from './pages/TechnicalIndicators'
import HealthDashboard from './pages/HealthDashboard'
import SystemMonitoringHub from './pages/SystemMonitoringHub'
import SystemManagementHub from './pages/SystemManagementHub'
import AnalyticsPerformance from './pages/AnalyticsPerformance'
import DiscordMessages from './pages/DiscordMessages'
import RAGKnowledgeBase from './pages/RAGKnowledgeBase'
import SupplyDemandZones from './pages/SupplyDemandZones'
import EnhancementManager from './pages/EnhancementManager'
import EnhancementAgent from './pages/EnhancementAgent'
import EnhancementQA from './pages/EnhancementQA'
import CacheMetrics from './pages/CacheMetrics'
import SubscriptionManagement from './pages/SubscriptionManagement'
import IntegrationTest from './pages/IntegrationTest'
import DatabaseWatchlist from './pages/DatabaseWatchlist'
import TradingViewWatchlist from './pages/TradingViewWatchlist'
// Advanced Technical Analysis pages
import SmartMoneyConcepts from './pages/SmartMoneyConcepts'
import VolumeAnalysis from './pages/VolumeAnalysis'
import OptionsGreeks from './pages/OptionsGreeks'
import FibonacciAnalysis from './pages/FibonacciAnalysis'
import IchimokuCloud from './pages/IchimokuCloud'
import SignalDashboard from './pages/SignalDashboard'

function App() {
  return (
    <BrowserRouter>
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
          <Route path="/cache" element={<CacheMetrics />} />
          <Route path="/subscriptions" element={<SubscriptionManagement />} />
          <Route path="/integration-test" element={<IntegrationTest />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
