import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    CheckCircle2, XCircle, RefreshCw, Database, Cloud,
    Wifi, Clock, AlertTriangle, Activity, Server,
    Zap, Brain, MessageSquare, TrendingUp, PlayCircle
} from 'lucide-react'

interface TestResult {
    name: string
    status: 'pass' | 'fail'
    response_time_ms: number
    message: string
    data_sample?: string
    error_type?: string
}

interface TestSummary {
    total_tests: number
    passed: number
    failed: number
    pass_rate: number
    total_time_ms: number
}

interface AllTestsResponse {
    summary: TestSummary
    results: TestResult[]
    timestamp: string
}

const categoryIcons: Record<string, React.ReactNode> = {
    'PostgreSQL Database': <Database className="w-4 h-4" />,
    'Robinhood Auth': <Cloud className="w-4 h-4" />,
    'Robinhood Positions': <TrendingUp className="w-4 h-4" />,
    'TradingView Watchlists': <Activity className="w-4 h-4" />,
    'TradingView Symbols': <Activity className="w-4 h-4" />,
    'XTrades Profiles': <Server className="w-4 h-4" />,
    'XTrades Trades': <Server className="w-4 h-4" />,
    'Earnings Database': <Database className="w-4 h-4" />,
    'YFinance Market Data': <TrendingUp className="w-4 h-4" />,
    'YFinance Options Chain': <TrendingUp className="w-4 h-4" />,
    'Premium Scanner': <Zap className="w-4 h-4" />,
    'NFL Database': <Activity className="w-4 h-4" />,
    'NFL Predictor': <Brain className="w-4 h-4" />,
    'Kalshi Database': <Database className="w-4 h-4" />,
    'Ollama Local LLM': <Brain className="w-4 h-4" />,
    'Groq API Key': <Brain className="w-4 h-4" />,
    'Discord Bot Token': <MessageSquare className="w-4 h-4" />
}

export default function IntegrationTest() {
    const [isRunning, setIsRunning] = useState(false)

    const { data: testResults, refetch, isFetching } = useQuery<AllTestsResponse>({
        queryKey: ['integration-tests'],
        queryFn: async () => {
            setIsRunning(true)
            const { data } = await axiosInstance.get('/test/all')
            setIsRunning(false)
            return data
        },
        enabled: false,
        staleTime: 0
    })

    const runTests = async () => {
        setIsRunning(true)
        await refetch()
    }

    const getStatusColor = (status: string) => {
        return status === 'pass' ? 'text-emerald-400' : 'text-red-400'
    }

    const getStatusBg = (status: string) => {
        return status === 'pass' ? 'bg-emerald-500/10' : 'bg-red-500/10'
    }

    // Group results by category
    const groupedResults = testResults?.results.reduce((acc, result) => {
        const category = result.name.includes('Robinhood') ? 'Robinhood' :
                        result.name.includes('TradingView') ? 'TradingView' :
                        result.name.includes('XTrades') ? 'XTrades' :
                        result.name.includes('YFinance') || result.name.includes('Scanner') ? 'Options Data' :
                        result.name.includes('NFL') || result.name.includes('Kalshi') ? 'Sports & Predictions' :
                        result.name.includes('Ollama') || result.name.includes('Groq') || result.name.includes('LLM') ? 'LLM Providers' :
                        result.name.includes('Discord') ? 'Discord' :
                        'Database'

        if (!acc[category]) acc[category] = []
        acc[category].push(result)
        return acc
    }, {} as Record<string, TestResult[]>) || {}

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg">
                            <Wifi className="w-5 h-5 text-white" />
                        </div>
                        Integration Test Suite
                    </h1>
                    <p className="page-subtitle">Verify all API connections and integrations</p>
                </div>
                <button
                    onClick={runTests}
                    disabled={isRunning || isFetching}
                    className="btn-primary flex items-center gap-2"
                >
                    {isRunning || isFetching ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            Running Tests...
                        </>
                    ) : (
                        <>
                            <PlayCircle className="w-5 h-5" />
                            Run All Tests
                        </>
                    )}
                </button>
            </header>

            {/* Summary Cards */}
            {testResults?.summary && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                <Activity className="w-5 h-5 text-blue-400" />
                            </div>
                            <div>
                                <p className="text-sm text-slate-400">Total Tests</p>
                                <p className="text-2xl font-bold text-white">{testResults.summary.total_tests}</p>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                            </div>
                            <div>
                                <p className="text-sm text-slate-400">Passed</p>
                                <p className="text-2xl font-bold text-emerald-400">{testResults.summary.passed}</p>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                                <XCircle className="w-5 h-5 text-red-400" />
                            </div>
                            <div>
                                <p className="text-sm text-slate-400">Failed</p>
                                <p className="text-2xl font-bold text-red-400">{testResults.summary.failed}</p>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                                <TrendingUp className="w-5 h-5 text-purple-400" />
                            </div>
                            <div>
                                <p className="text-sm text-slate-400">Pass Rate</p>
                                <p className={`text-2xl font-bold ${testResults.summary.pass_rate >= 80 ? 'text-emerald-400' : testResults.summary.pass_rate >= 50 ? 'text-amber-400' : 'text-red-400'}`}>
                                    {testResults.summary.pass_rate}%
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="glass-card p-4">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                                <Clock className="w-5 h-5 text-amber-400" />
                            </div>
                            <div>
                                <p className="text-sm text-slate-400">Total Time</p>
                                <p className="text-2xl font-bold text-white">{((testResults.summary?.total_time_ms ?? 0) / 1000).toFixed(1)}s</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Test Results by Category */}
            {testResults && Object.entries(groupedResults).map(([category, results]) => (
                <div key={category} className="glass-card overflow-hidden">
                    <div className="p-4 border-b border-slate-700/50 bg-slate-900/30">
                        <h3 className="font-semibold text-white flex items-center gap-2">
                            {category === 'Database' && <Database className="w-5 h-5 text-blue-400" />}
                            {category === 'Robinhood' && <Cloud className="w-5 h-5 text-green-400" />}
                            {category === 'TradingView' && <Activity className="w-5 h-5 text-purple-400" />}
                            {category === 'XTrades' && <Server className="w-5 h-5 text-cyan-400" />}
                            {category === 'Options Data' && <TrendingUp className="w-5 h-5 text-amber-400" />}
                            {category === 'Sports & Predictions' && <Zap className="w-5 h-5 text-orange-400" />}
                            {category === 'LLM Providers' && <Brain className="w-5 h-5 text-pink-400" />}
                            {category === 'Discord' && <MessageSquare className="w-5 h-5 text-indigo-400" />}
                            {category}
                            <span className="ml-2 text-sm text-slate-400">
                                ({results.filter(r => r.status === 'pass').length}/{results.length} passed)
                            </span>
                        </h3>
                    </div>

                    <div className="divide-y divide-slate-700/50">
                        {results.map((result, idx) => (
                            <div
                                key={idx}
                                className={`p-4 flex items-center justify-between ${getStatusBg(result.status)}`}
                            >
                                <div className="flex items-center gap-3">
                                    {result.status === 'pass' ? (
                                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                                    ) : (
                                        <XCircle className="w-5 h-5 text-red-400" />
                                    )}
                                    <div>
                                        <p className="font-medium text-white flex items-center gap-2">
                                            {categoryIcons[result.name]}
                                            {result.name}
                                        </p>
                                        <p className={`text-sm ${getStatusColor(result.status)}`}>
                                            {result.message}
                                            {result.data_sample && result.data_sample !== 'Data retrieved' && (
                                                <span className="text-slate-400 ml-2">- {result.data_sample}</span>
                                            )}
                                        </p>
                                        {result.error_type && (
                                            <p className="text-xs text-red-400/70 mt-1">
                                                Error type: {result.error_type}
                                            </p>
                                        )}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                                        result.status === 'pass'
                                            ? 'bg-emerald-500/20 text-emerald-400'
                                            : 'bg-red-500/20 text-red-400'
                                    }`}>
                                        {result.status.toUpperCase()}
                                    </span>
                                    <p className="text-xs text-slate-500 mt-1">
                                        {(result.response_time_ms ?? 0).toFixed(0)}ms
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            {/* Empty State */}
            {!testResults && !isRunning && (
                <div className="glass-card p-12 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4">
                        <Wifi className="w-8 h-8 text-slate-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">No Tests Run Yet</h3>
                    <p className="text-slate-400 mb-6">Click "Run All Tests" to verify all integrations</p>
                    <button onClick={runTests} className="btn-primary">
                        <PlayCircle className="w-5 h-5 mr-2" />
                        Run All Tests
                    </button>
                </div>
            )}

            {/* Running State */}
            {isRunning && !testResults && (
                <div className="glass-card p-12 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-blue-500/20 flex items-center justify-center mx-auto mb-4">
                        <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">Running Integration Tests...</h3>
                    <p className="text-slate-400">Testing all connections and APIs</p>
                </div>
            )}

            {/* Timestamp */}
            {testResults?.timestamp && (
                <p className="text-center text-sm text-slate-500">
                    Last run: {new Date(testResults.timestamp).toLocaleString()}
                </p>
            )}
        </div>
    )
}
