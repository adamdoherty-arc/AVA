/**
 * Trading Domain Types
 * ====================
 *
 * Comprehensive TypeScript types for trading functionality.
 * Matches backend Pydantic models for type safety across the stack.
 *
 * @author AVA Trading Platform
 * @updated 2025-11-29
 */

// =============================================================================
// Enums
// =============================================================================

export type OptionType = 'call' | 'put';
export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit';
export type TimeInForce = 'day' | 'gtc' | 'ioc' | 'fok';
export type PositionType = 'stock' | 'option' | 'crypto';
export type Trend = 'bullish' | 'bearish' | 'neutral' | 'mixed';
export type RiskLevel = 'conservative' | 'moderate' | 'aggressive';
export type VolRegime = 'low' | 'normal' | 'elevated' | 'extreme';
export type Recommendation = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';

export type StrategyType =
  | 'cash_secured_put'
  | 'covered_call'
  | 'iron_condor'
  | 'credit_spread'
  | 'debit_spread'
  | 'calendar_spread'
  | 'diagonal_spread'
  | 'straddle'
  | 'strangle'
  | 'butterfly'
  | 'collar';

// =============================================================================
// Market Data Types
// =============================================================================

export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  timestamp: string;
}

export interface OptionContract {
  symbol: string;
  underlying: string;
  optionType: OptionType;
  strike: number;
  expiration: string;
  bid: number;
  ask: number;
  last: number | null;
  volume: number;
  openInterest: number;

  // Greeks
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  iv: number;
}

export interface OptionChain {
  underlying: string;
  underlyingPrice: number;
  expirations: string[];
  calls: OptionContract[];
  puts: OptionContract[];
  timestamp: string;
}

// =============================================================================
// Position Types
// =============================================================================

export interface Position {
  id: string;
  symbol: string;
  positionType: PositionType;
  quantity: number;
  averageCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  dayChange: number;
  dayChangePercent: number;

  // Option-specific
  optionType?: OptionType;
  strike?: number;
  expiration?: string;
  delta?: number;
  theta?: number;
  gamma?: number;
  vega?: number;
}

export interface PortfolioSummary {
  totalValue: number;
  cash: number;
  buyingPower: number;
  dayChange: number;
  dayChangePercent: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;

  // Greeks aggregate
  totalDelta: number;
  totalTheta: number;
  totalGamma: number;
  totalVega: number;

  // Position counts
  stockPositions: number;
  optionPositions: number;

  timestamp: string;
}

// =============================================================================
// AI Score Types
// =============================================================================

export interface StockAIScore {
  symbol: string;
  companyName: string;
  sector: string;

  // Price data
  currentPrice: number;
  dailyChangePct: number;

  // AI Score
  aiScore: number;
  recommendation: Recommendation;
  confidence: number;

  // Components
  predictionScore: number;
  technicalScore: number;
  sentimentScore: number;
  volatilityScore: number;

  // Trend
  trend: Trend;
  trendStrength: number;

  // Technicals
  rsi14: number;
  macdHistogram: number;
  sma20: number;
  sma50: number;

  // Volatility
  ivEstimate: number;
  volRegime: VolRegime;

  // Predictions
  predictedChange1d: number;
  predictedChange5d: number;

  // Levels
  supportPrice: number;
  resistancePrice: number;

  // Meta
  marketCap?: number;
  calculatedAt: string;
}

export interface ScoreComponent {
  name: string;
  rawScore: number;
  weight: number;
  weightedScore: number;
  signals: Record<string, unknown>;
}

// =============================================================================
// Trade Analysis Types
// =============================================================================

export interface TradeRecommendation {
  action: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';
  confidence: number;
  score: number;

  technicalAnalysis: string;
  fundamentalAnalysis: string;
  riskAssessment: string;

  bullishFactors: string[];
  bearishFactors: string[];
  keyRisks: string[];

  entryPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  positionSizePct: number;

  reasoning: string;
}

export interface OptionsStrategy {
  strategyName: string;
  strategyType: StrategyType;
  fitScore: number;

  legs: OptionLeg[];

  maxProfit: number;
  maxLoss: number;
  breakEven: number[];
  probabilityOfProfit: number;

  ivEnvironment: VolRegime;
  idealConditions: string[];
  risks: string[];
  adjustments: string[];

  reasoning: string;
}

export interface OptionLeg {
  optionType: OptionType;
  strike: number;
  expiration: string;
  quantity: number;
  side: OrderSide;
  premium: number;
  delta: number;
}

// =============================================================================
// Premium Scanner Types
// =============================================================================

export interface ScannerResult {
  symbol: string;
  companyName: string;
  sector: string;
  currentPrice: number;

  // Strategy
  strategy: string;
  strike: number;
  expiration: string;
  dte: number;

  // Premium
  premium: number;
  annualizedReturn: number;
  probabilityOfProfit: number;

  // Greeks
  delta: number;
  theta: number;
  iv: number;
  ivRank: number;

  // Scores
  opportunityScore: number;
  liquidityScore: number;

  // Meta
  volume: number;
  openInterest: number;
  bidAskSpread: number;
}

export interface ScannerFilters {
  minDte?: number;
  maxDte?: number;
  minDelta?: number;
  maxDelta?: number;
  minIvRank?: number;
  minPop?: number;
  minAnnualizedReturn?: number;
  sectors?: string[];
  strategies?: StrategyType[];
}

// =============================================================================
// Trade Order Types
// =============================================================================

export interface TradeOrder {
  symbol: string;
  side: OrderSide;
  quantity: number;
  orderType: OrderType;
  timeInForce: TimeInForce;
  limitPrice?: number;
  stopPrice?: number;
}

export interface OrderResponse {
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  filledQuantity: number;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  avgFillPrice?: number;
  createdAt: string;
  updatedAt: string;
}

// =============================================================================
// Trade Journal Types
// =============================================================================

export interface TradeJournalEntry {
  id: string;
  symbol: string;
  positionType: PositionType;
  strategy: string;

  // Entry
  entryDate: string;
  entryPrice: number;
  entryQuantity: number;
  entryCost: number;

  // Exit
  exitDate?: string;
  exitPrice?: number;
  exitQuantity?: number;
  exitProceeds?: number;

  // P&L
  realizedPnl?: number;
  realizedPnlPercent?: number;
  fees: number;

  // Analysis
  setupNotes?: string;
  exitNotes?: string;
  lessonsLearned?: string;
  tags: string[];

  // Meta
  createdAt: string;
  updatedAt: string;
}

// =============================================================================
// Technical Analysis Types
// =============================================================================

export interface TechnicalIndicators {
  symbol: string;
  timestamp: string;

  // Moving Averages
  sma20: number;
  sma50: number;
  sma200: number;
  ema9: number;
  ema21: number;

  // Momentum
  rsi14: number;
  macd: number;
  macdSignal: number;
  macdHistogram: number;
  stochK: number;
  stochD: number;

  // Volatility
  bbUpper: number;
  bbMiddle: number;
  bbLower: number;
  atr14: number;

  // Volume
  volumeSma20: number;
  obv: number;
  vwap: number;

  // Levels
  pivotPoint: number;
  r1: number;
  r2: number;
  s1: number;
  s2: number;
}

export interface SupportResistance {
  symbol: string;
  supports: PriceLevel[];
  resistances: PriceLevel[];
  currentPrice: number;
}

export interface PriceLevel {
  price: number;
  strength: number;
  touches: number;
  type: 'support' | 'resistance';
}

// =============================================================================
// Risk Types
// =============================================================================

export interface RiskMetrics {
  portfolioValue: number;

  // Greeks
  netDelta: number;
  netGamma: number;
  netTheta: number;
  netVega: number;

  // VaR
  var95: number;
  var99: number;
  var95Pct: number;
  var99Pct: number;

  // Concentration
  largestPosition: string;
  largestPositionPct: number;
  sectorConcentration: Record<string, number>;

  // Risk Scores
  overallRiskScore: number;
  concentrationRisk: number;
  greeksRisk: number;
  volatilityRisk: number;

  warnings: string[];
  recommendations: string[];
}

// =============================================================================
// Watchlist Types
// =============================================================================

export interface Watchlist {
  id: string;
  name: string;
  symbols: string[];
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface WatchlistItem {
  symbol: string;
  addedAt: string;
  notes?: string;
  alerts?: WatchlistAlert[];
}

export interface WatchlistAlert {
  id: string;
  type: 'price_above' | 'price_below' | 'pct_change' | 'volume_spike';
  threshold: number;
  triggered: boolean;
  triggeredAt?: string;
}

// =============================================================================
// API Response Types
// =============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface APISuccess<T> {
  success: true;
  data: T;
  timestamp: string;
}

export interface APIError {
  success: false;
  error: {
    code: number;
    name: string;
    message: string;
    details?: Record<string, unknown>;
  };
  timestamp: string;
}

export type APIResponse<T> = APISuccess<T> | APIError;

// =============================================================================
// Utility Types
// =============================================================================

/** Make specific properties optional */
export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/** Make specific properties required */
export type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

/** Extract array element type */
export type ArrayElement<T> = T extends (infer E)[] ? E : never;

/** Create loading state wrapper */
export interface LoadingState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
}
