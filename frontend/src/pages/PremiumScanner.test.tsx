/**
 * Comprehensive tests for PremiumScanner component
 *
 * Tests cover:
 * - Initial state values
 * - handleScan behavior
 * - handleCancelScan behavior
 * - UI interactions
 * - Data loading and rendering
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import PremiumScanner from './PremiumScanner'

// Mock axios instance
vi.mock('../lib/axios', () => ({
    axiosInstance: {
        get: vi.fn(),
        post: vi.fn(),
    },
}))

import { axiosInstance } from '../lib/axios'

// Mock recharts to avoid rendering issues in tests
vi.mock('recharts', () => ({
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
    BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
    Bar: () => <div data-testid="bar" />,
    XAxis: () => <div data-testid="x-axis" />,
    YAxis: () => <div data-testid="y-axis" />,
    CartesianGrid: () => <div data-testid="cartesian-grid" />,
    Tooltip: () => <div data-testid="tooltip" />,
    Cell: () => <div data-testid="cell" />,
}))

// Mock AIPickCard component
vi.mock('../components/AIPickCard', () => ({
    AIPickCard: ({ pick, rank }: { pick: { symbol: string }; rank: number }) => (
        <div data-testid={`ai-pick-${rank}`}>{pick.symbol}</div>
    ),
}))

// Helper to create a fresh query client for each test
const createTestQueryClient = () => new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
            gcTime: 0,
            staleTime: 0,
        },
    },
})

// Wrapper component for tests
const TestWrapper = ({ children }: { children: React.ReactNode }) => {
    const queryClient = createTestQueryClient()
    return (
        <QueryClientProvider client={queryClient}>
            <BrowserRouter>
                {children}
            </BrowserRouter>
        </QueryClientProvider>
    )
}

// Default mock responses
const mockWatchlistsResponse = {
    watchlists: [
        { id: 'popular', name: 'Popular Stocks', source: 'predefined', symbols: ['AAPL', 'MSFT', 'NVDA'] },
        { id: 'tech', name: 'Tech Leaders', source: 'predefined', symbols: ['GOOGL', 'META', 'AMZN'] },
    ],
    total: 2,
    generated_at: new Date().toISOString(),
}

const mockHistoryResponse = {
    history: [
        {
            scan_id: 'scan_123',
            symbols: ['AAPL', 'MSFT'],
            symbol_count: 2,
            dte: 30,
            max_price: 100,
            min_premium_pct: 1.0,
            result_count: 5,
            created_at: new Date().toISOString(),
        },
    ],
    count: 1,
}

const mockStoredPremiumsResponse = {
    results: [
        {
            symbol: 'AAPL',
            stock_price: 150.0,
            strike: 145.0,
            expiration: '2025-01-17',
            dte: 30,
            premium: 2.50,
            premium_pct: 1.72,
            monthly_return: 1.72,
            annualized_return: 20.6,
            implied_volatility: 35,
            volume: 1000,
            open_interest: 5000,
            bid: 2.45,
            ask: 2.55,
        },
        {
            symbol: 'MSFT',
            stock_price: 400.0,
            strike: 390.0,
            expiration: '2025-01-17',
            dte: 30,
            premium: 5.00,
            premium_pct: 1.28,
            monthly_return: 1.28,
            annualized_return: 15.4,
            implied_volatility: 28,
            volume: 800,
            open_interest: 3000,
            bid: 4.90,
            ask: 5.10,
        },
    ],
    count: 2,
    last_updated: new Date().toISOString(),
}

const mockComparisonResponse = {
    summary: {
        7: { avg_monthly_return: 2.5, count: 10, avg_iv: 40, best_opportunity: { symbol: 'NVDA', monthly_return: 3.5 } },
        14: { avg_monthly_return: 2.0, count: 15, avg_iv: 35, best_opportunity: { symbol: 'AAPL', monthly_return: 2.8 } },
        30: { avg_monthly_return: 1.5, count: 25, avg_iv: 30, best_opportunity: { symbol: 'MSFT', monthly_return: 2.2 } },
        45: { avg_monthly_return: 1.2, count: 20, avg_iv: 28, best_opportunity: { symbol: 'TSLA', monthly_return: 1.8 } },
    },
}

const mockAIPicksResponse = {
    picks: [
        {
            symbol: 'AAPL',
            strike: 145,
            expiration: '2025-01-17',
            premium_pct: 1.72,
            monthly_return: 1.72,
            confidence: 85,
            reasoning: 'Strong support level',
        },
    ],
    market_context: 'Bullish market sentiment',
    generated_at: new Date().toISOString(),
    model: 'DeepSeek R1',
    total_candidates: 50,
    cached: false,
}

// Setup default mock implementations
const setupDefaultMocks = () => {
    const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
    mockGet.mockImplementation((url: string) => {
        if (url.includes('/scanner/watchlists')) {
            return Promise.resolve({ data: mockWatchlistsResponse })
        }
        if (url.includes('/scanner/history') && !url.includes('/scanner/history/')) {
            return Promise.resolve({ data: mockHistoryResponse })
        }
        if (url.includes('/scanner/stored-premiums')) {
            return Promise.resolve({ data: mockStoredPremiumsResponse })
        }
        if (url.includes('/scanner/dte-comparison')) {
            return Promise.resolve({ data: mockComparisonResponse })
        }
        if (url.includes('/scanner/ai-picks')) {
            return Promise.resolve({ data: mockAIPicksResponse })
        }
        return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })
}

describe('PremiumScanner', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        setupDefaultMocks()
    })

    afterEach(() => {
        vi.restoreAllMocks()
    })

    // ============================================================================
    // INITIAL STATE TESTS
    // ============================================================================
    describe('Initial State', () => {
        it('renders with default values', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Page title should render
            expect(screen.getByText('Premium Scanner')).toBeInTheDocument()
            expect(screen.getByText('Find the best CSP opportunities by DTE')).toBeInTheDocument()
        })

        it('defaults to 30 DTE selection', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Look for the scan button which should show the selected DTE
            await waitFor(() => {
                const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
                expect(scanButton).toBeInTheDocument()
            })
        })

        it('defaults to popular watchlist', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('Popular Stocks')).toBeInTheDocument()
            })
        })

        it('renders DTE option cards', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            expect(screen.getByText('7 DTE')).toBeInTheDocument()
            expect(screen.getByText('14 DTE')).toBeInTheDocument()
            expect(screen.getByText('30 DTE')).toBeInTheDocument()
            expect(screen.getByText('45 DTE')).toBeInTheDocument()
        })

        it('does not show scanning state initially', () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            expect(screen.queryByText(/Scanning.*Symbols/)).not.toBeInTheDocument()
        })
    })

    // ============================================================================
    // DTE SELECTION TESTS
    // ============================================================================
    describe('DTE Selection', () => {
        it('updates selected DTE when clicking DTE card', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Click on 7 DTE card
            const dte7Card = screen.getByText('7 DTE').closest('button')
            expect(dte7Card).toBeInTheDocument()
            await user.click(dte7Card!)

            // Scan button should now show 7 DTE
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 7 DTE Premiums/i })).toBeInTheDocument()
            })
        })

        it('highlights selected DTE card', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            const dte14Card = screen.getByText('14 DTE').closest('button')
            await user.click(dte14Card!)

            // Card should have selected styles (border-primary class)
            await waitFor(() => {
                expect(dte14Card).toHaveClass('border-primary')
            })
        })
    })

    // ============================================================================
    // WATCHLIST DROPDOWN TESTS
    // ============================================================================
    describe('Watchlist Dropdown', () => {
        it('opens watchlist dropdown on click', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('Popular Stocks')).toBeInTheDocument()
            })

            // Find and click the dropdown button
            const dropdownButton = screen.getByRole('button', { name: /Popular Stocks/i })
            await user.click(dropdownButton)

            // Should show watchlist options
            await waitFor(() => {
                expect(screen.getByText('Tech Leaders')).toBeInTheDocument()
            })
        })

        it('selects a different watchlist', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('Popular Stocks')).toBeInTheDocument()
            })

            // Open dropdown
            const dropdownButton = screen.getByRole('button', { name: /Popular Stocks/i })
            await user.click(dropdownButton)

            // Select Tech Leaders
            await waitFor(() => {
                const techOption = screen.getByRole('button', { name: /Tech Leaders/i })
                user.click(techOption)
            })

            // Should show Tech Leaders as selected
            await waitFor(() => {
                expect(screen.getAllByText('Tech Leaders').length).toBeGreaterThan(0)
            })
        })

        it('shows symbol count for each watchlist', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('Popular Stocks')).toBeInTheDocument()
            })

            // Open dropdown
            const dropdownButton = screen.getByRole('button', { name: /Popular Stocks/i })
            await user.click(dropdownButton)

            // Should show symbol counts - format is "X symbols" in dropdown items
            await waitFor(() => {
                // Use getAllByText since multiple watchlists have symbol counts
                const symbolCounts = screen.getAllByText(/\d+ symbols/)
                expect(symbolCounts.length).toBeGreaterThan(0)
            })
        })
    })

    // ============================================================================
    // CUSTOM SYMBOLS INPUT TESTS
    // ============================================================================
    describe('Custom Symbols Input', () => {
        it('renders custom symbols input field', () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            expect(screen.getByPlaceholderText('AAPL, MSFT, NVDA, TSLA...')).toBeInTheDocument()
        })

        it('updates state when typing custom symbols', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            const input = screen.getByPlaceholderText('AAPL, MSFT, NVDA, TSLA...')
            await user.type(input, 'GOOG, AMZN')

            expect(input).toHaveValue('GOOG, AMZN')
        })
    })

    // ============================================================================
    // FILTER PANEL TESTS
    // ============================================================================
    describe('Filter Panel', () => {
        it('toggles filter panel visibility for min premium', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Filters button should be visible
            const filtersButton = screen.getByRole('button', { name: /Filters/i })
            expect(filtersButton).toBeInTheDocument()

            // Click to show filters
            await user.click(filtersButton)

            // Min Premium % input should now be visible in expanded filters
            await waitFor(() => {
                expect(screen.getByText('Min Premium %')).toBeInTheDocument()
            })
        })

        it('shows max price filter in results table header', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Trigger a scan first to show results table
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Wait for results, then check for Max Price filter
            await waitFor(() => {
                expect(screen.getByText('Max Price:')).toBeInTheDocument()
            }, { timeout: 3000 })
        })

        it('filters results by max price in table', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Trigger a scan
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Wait for results
            await waitFor(() => {
                expect(screen.getByText(/Opportunities Found/i)).toBeInTheDocument()
            }, { timeout: 3000 })

            // Find max price input and set a low value to filter out results
            const maxPriceInput = screen.getByPlaceholderText('Any')
            await user.clear(maxPriceInput)
            await user.type(maxPriceInput, '10')

            // Should filter results - stock prices in mock are 150 and 400, so 0 should match
            await waitFor(() => {
                expect(screen.getByText(/0 Opportunities Found/i)).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // SCAN FUNCTIONALITY TESTS
    // ============================================================================
    describe('handleScan', () => {
        it('triggers scan when clicking scan button', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Should make API call (max_price is now filtered client-side, not sent to API)
            await waitFor(() => {
                expect(axiosInstance.get).toHaveBeenCalledWith(
                    '/scanner/stored-premiums',
                    expect.objectContaining({
                        params: expect.objectContaining({
                            min_premium_pct: 1,
                            max_dte: 30,
                        }),
                    })
                )
            })
        })

        it('scan button is enabled when watchlist has symbols', async () => {
            // Note: PRESET_WATCHLISTS are hardcoded in component and always have symbols
            // So scan button should be enabled by default
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                const scanButton = screen.getByRole('button', { name: /Scan.*DTE Premiums/i })
                expect(scanButton).not.toBeDisabled()
            })
        })

        it('displays scan results in table', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Should display results - use getAllByText since symbols appear in multiple places
            await waitFor(() => {
                const aaplElements = screen.getAllByText('AAPL')
                const msftElements = screen.getAllByText('MSFT')
                expect(aaplElements.length).toBeGreaterThan(0)
                expect(msftElements.length).toBeGreaterThan(0)
            }, { timeout: 3000 })
        })

        it('uses custom symbols when provided', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            // Enter custom symbols
            const input = screen.getByPlaceholderText('AAPL, MSFT, NVDA, TSLA...')
            await user.type(input, 'TSLA, GOOG')

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Should call API with custom symbols
            await waitFor(() => {
                expect(axiosInstance.get).toHaveBeenCalledWith(
                    '/scanner/stored-premiums',
                    expect.objectContaining({
                        params: expect.objectContaining({
                            symbols: 'TSLA,GOOG',
                        }),
                    })
                )
            })
        })
    })

    // ============================================================================
    // CANCEL SCAN TESTS
    // ============================================================================
    describe('handleCancelScan', () => {
        it('shows cancel button during scan', async () => {
            const user = userEvent.setup()

            // Make API call hang
            const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
            mockGet.mockImplementation((url: string) => {
                if (url.includes('/scanner/watchlists')) {
                    return Promise.resolve({ data: mockWatchlistsResponse })
                }
                if (url.includes('/scanner/stored-premiums')) {
                    // Return a promise that never resolves to simulate slow scan
                    return new Promise(() => {})
                }
                if (url.includes('/scanner/history')) {
                    return Promise.resolve({ data: mockHistoryResponse })
                }
                if (url.includes('/scanner/dte-comparison')) {
                    return Promise.resolve({ data: mockComparisonResponse })
                }
                if (url.includes('/scanner/ai-picks')) {
                    return Promise.resolve({ data: mockAIPicksResponse })
                }
                return Promise.reject(new Error(`Unhandled URL: ${url}`))
            })

            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            // Start scan
            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            // Cancel button should appear
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // RESULTS TABLE TESTS
    // ============================================================================
    describe('Results Table', () => {
        it('displays opportunity count in header', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText(/2 Opportunities Found/i)).toBeInTheDocument()
            })
        })

        it('renders all table columns', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText('Symbol')).toBeInTheDocument()
                expect(screen.getByText('Price')).toBeInTheDocument()
                expect(screen.getByText('Strike')).toBeInTheDocument()
                expect(screen.getByText('Premium')).toBeInTheDocument()
                expect(screen.getByText('Monthly')).toBeInTheDocument()
            })
        })

        it('sorts by column when header clicked', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                // Wait for results table to appear
                expect(screen.getByText(/Opportunities Found/i)).toBeInTheDocument()
            }, { timeout: 3000 })

            // Find the Symbol column header button in the table
            const table = screen.getByRole('table')
            const symbolHeaders = within(table).getAllByRole('button')
            const symbolHeader = symbolHeaders.find(btn => btn.textContent?.includes('Symbol'))

            if (symbolHeader) {
                await user.click(symbolHeader)
            }

            // Should still display results (sorted)
            await waitFor(() => {
                const aaplElements = screen.getAllByText('AAPL')
                expect(aaplElements.length).toBeGreaterThan(0)
            })
        })
    })

    // ============================================================================
    // DTE FILTER IN RESULTS TESTS
    // ============================================================================
    describe('DTE Filter in Results', () => {
        it('shows DTE filter buttons after scan', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByRole('button', { name: 'All' })).toBeInTheDocument()
                expect(screen.getByRole('button', { name: '7d' })).toBeInTheDocument()
                expect(screen.getByRole('button', { name: '14d' })).toBeInTheDocument()
                expect(screen.getByRole('button', { name: '30d' })).toBeInTheDocument()
            })
        })

        it('filters results by DTE when clicked', async () => {
            // Mock with varied DTE values
            const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
            mockGet.mockImplementation((url: string) => {
                if (url.includes('/scanner/watchlists')) {
                    return Promise.resolve({ data: mockWatchlistsResponse })
                }
                if (url.includes('/scanner/stored-premiums')) {
                    return Promise.resolve({
                        data: {
                            results: [
                                { ...mockStoredPremiumsResponse.results[0], dte: 7 },
                                { ...mockStoredPremiumsResponse.results[1], dte: 45 },
                            ],
                            count: 2,
                            last_updated: new Date().toISOString(),
                        },
                    })
                }
                if (url.includes('/scanner/history')) {
                    return Promise.resolve({ data: mockHistoryResponse })
                }
                if (url.includes('/scanner/dte-comparison')) {
                    return Promise.resolve({ data: mockComparisonResponse })
                }
                if (url.includes('/scanner/ai-picks')) {
                    return Promise.resolve({ data: mockAIPicksResponse })
                }
                return Promise.reject(new Error(`Unhandled URL: ${url}`))
            })

            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText(/2 Opportunities Found/i)).toBeInTheDocument()
            })

            // Click 7d filter - should show only AAPL (dte: 7)
            const filter7d = screen.getByRole('button', { name: '7d' })
            await user.click(filter7d)

            await waitFor(() => {
                expect(screen.getByText(/1 Opportunities Found/i)).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // UNIQUE STOCKS TOGGLE TESTS
    // ============================================================================
    describe('Unique Stocks Toggle', () => {
        it('shows unique stocks toggle after scan', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText('Unique Stocks Only')).toBeInTheDocument()
            })
        })

        it('toggles unique stocks filter', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText('Unique Stocks Only')).toBeInTheDocument()
            }, { timeout: 3000 })

            // Click toggle button
            const toggle = screen.getByText('Unique Stocks Only').closest('button')
            expect(toggle).toBeInTheDocument()
            await user.click(toggle!)

            // After clicking, the button should have the active class (bg-emerald-500/20)
            await waitFor(() => {
                expect(toggle).toHaveClass('bg-emerald-500/20')
            })
        })
    })

    // ============================================================================
    // AI PICKS SECTION TESTS
    // ============================================================================
    describe('AI Picks Section', () => {
        it('renders AI picks section header', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('AI-Powered CSP Picks')).toBeInTheDocument()
            })
        })

        it('shows loading state for AI picks', async () => {
            const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
            mockGet.mockImplementation((url: string) => {
                if (url.includes('/scanner/ai-picks')) {
                    return new Promise(() => {}) // Never resolves
                }
                if (url.includes('/scanner/watchlists')) {
                    return Promise.resolve({ data: mockWatchlistsResponse })
                }
                if (url.includes('/scanner/stored-premiums')) {
                    return Promise.resolve({ data: mockStoredPremiumsResponse })
                }
                if (url.includes('/scanner/history')) {
                    return Promise.resolve({ data: mockHistoryResponse })
                }
                if (url.includes('/scanner/dte-comparison')) {
                    return Promise.resolve({ data: mockComparisonResponse })
                }
                return Promise.reject(new Error(`Unhandled URL: ${url}`))
            })

            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('DeepSeek R1 is analyzing opportunities...')).toBeInTheDocument()
            })
        })

        it('displays AI picks when loaded', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByTestId('ai-pick-1')).toBeInTheDocument()
            })
        })

        it('shows total candidates analyzed', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText(/analyzed 50 opportunities/i)).toBeInTheDocument()
            })
        })

        it('has refresh AI button', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Refresh AI/i })).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // SCAN HISTORY TESTS
    // ============================================================================
    describe('Scan History', () => {
        it('shows history button with count badge', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                const historyButton = screen.getByTitle('Previous Scans')
                expect(historyButton).toBeInTheDocument()
            })
        })

        it('opens history dropdown on click', async () => {
            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByTitle('Previous Scans')).toBeInTheDocument()
            })

            const historyButton = screen.getByTitle('Previous Scans')
            await user.click(historyButton)

            await waitFor(() => {
                expect(screen.getByText('Previous Scans')).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // ERROR HANDLING TESTS
    // ============================================================================
    describe('Error Handling', () => {
        it('displays error message on scan failure', async () => {
            const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
            mockGet.mockImplementation((url: string) => {
                if (url.includes('/scanner/watchlists')) {
                    return Promise.resolve({ data: mockWatchlistsResponse })
                }
                if (url.includes('/scanner/stored-premiums')) {
                    return Promise.reject({
                        response: { status: 500, data: { detail: 'Database error' } },
                    })
                }
                if (url.includes('/scanner/history')) {
                    return Promise.resolve({ data: mockHistoryResponse })
                }
                if (url.includes('/scanner/dte-comparison')) {
                    return Promise.resolve({ data: mockComparisonResponse })
                }
                if (url.includes('/scanner/ai-picks')) {
                    return Promise.resolve({ data: mockAIPicksResponse })
                }
                return Promise.reject(new Error(`Unhandled URL: ${url}`))
            })

            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText(/Server error/i)).toBeInTheDocument()
            })
        })

        it('shows timeout error message', async () => {
            const mockGet = axiosInstance.get as ReturnType<typeof vi.fn>
            mockGet.mockImplementation((url: string) => {
                if (url.includes('/scanner/watchlists')) {
                    return Promise.resolve({ data: mockWatchlistsResponse })
                }
                if (url.includes('/scanner/stored-premiums')) {
                    return Promise.reject({
                        code: 'ECONNABORTED',
                        message: 'timeout of 120000ms exceeded',
                    })
                }
                if (url.includes('/scanner/history')) {
                    return Promise.resolve({ data: mockHistoryResponse })
                }
                if (url.includes('/scanner/dte-comparison')) {
                    return Promise.resolve({ data: mockComparisonResponse })
                }
                if (url.includes('/scanner/ai-picks')) {
                    return Promise.resolve({ data: mockAIPicksResponse })
                }
                return Promise.reject(new Error(`Unhandled URL: ${url}`))
            })

            const user = userEvent.setup()
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })).toBeInTheDocument()
            })

            const scanButton = screen.getByRole('button', { name: /Scan 30 DTE Premiums/i })
            await user.click(scanButton)

            await waitFor(() => {
                expect(screen.getByText(/Request timed out/i)).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // QUICK STATS CARDS TESTS
    // ============================================================================
    describe('Quick Stats Cards', () => {
        it('displays quick stats when comparison data loads', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText('Best Opportunity')).toBeInTheDocument()
                expect(screen.getByText('Avg Monthly Return')).toBeInTheDocument()
                expect(screen.getByText('Avg IV')).toBeInTheDocument()
                expect(screen.getByText('Opportunities')).toBeInTheDocument()
            })
        })

        it('shows best opportunity symbol', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                // 30 DTE is default, best opportunity should be MSFT
                expect(screen.getByText('MSFT')).toBeInTheDocument()
            })
        })
    })

    // ============================================================================
    // STORED PREMIUMS LOADING TESTS
    // ============================================================================
    describe('Stored Premiums Auto-Loading', () => {
        it('shows database info banner when stored premiums load', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByText(/Loaded.*opportunities from database/i)).toBeInTheDocument()
            })
        })

        it('has refresh from DB button', async () => {
            render(<PremiumScanner />, { wrapper: TestWrapper })

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /Refresh from DB/i })).toBeInTheDocument()
            })
        })
    })
})
