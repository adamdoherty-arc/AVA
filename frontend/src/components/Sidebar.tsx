import {
    LayoutDashboard, TrendingUp, Activity, Settings, MessageSquare,
    Search, List, Target, Gamepad2, ChevronLeft, Sparkles, Wallet,
    Calendar, Trophy, Clock, Bot, Smartphone, Layers, BarChart3,
    Calculator, PieChart, Brain, Shield, BookOpen, DollarSign,
    Bell, History, Percent, LineChart, Cpu, Monitor, Database,
    TestTube, CreditCard, Zap, HeartPulse, Cloud, GitBranch, Gauge
} from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'
import clsx from 'clsx'

const navSections = [
    {
        title: 'Trading',
        items: [
            { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
            { to: '/positions', icon: Wallet, label: 'Positions' },
            { to: '/stocks-hub', icon: Brain, label: 'Stocks Hub', highlight: true },
            { to: '/scanner', icon: Search, label: 'Premium Scanner' },
            { to: '/earnings', icon: Calendar, label: 'Earnings Calendar' },
            { to: '/xtrades', icon: Smartphone, label: 'XTrades Watchlists' },
            { to: '/research', icon: TrendingUp, label: 'Research' },
            { to: '/watchlist', icon: List, label: 'Strategy Analyzer' },
            { to: '/db-watchlist', icon: Database, label: 'DB Watchlists' },
            { to: '/tv-watchlist', icon: LineChart, label: 'TradingView Lists' },
        ]
    },
    {
        title: 'Options',
        items: [
            { to: '/options-hub', icon: Layers, label: 'Options Hub' },
            { to: '/ai-options', icon: Bot, label: 'AI Options Agent' },
            { to: '/options-analysis', icon: LineChart, label: 'Options Analysis' },
            { to: '/technicals', icon: Activity, label: 'Technical Indicators' },
            { to: '/calendar-spreads', icon: Calendar, label: 'Calendar Spreads' },
            { to: '/options-flow', icon: BarChart3, label: 'Options Flow' },
            { to: '/position-sizing', icon: Calculator, label: 'Position Sizing' },
            { to: '/supply-demand', icon: Layers, label: 'Supply/Demand Zones' },
        ]
    },
    {
        title: 'Technical Analysis',
        items: [
            { to: '/signals', icon: Zap, label: 'Signal Dashboard', highlight: true },
            { to: '/smart-money', icon: TrendingUp, label: 'Smart Money (ICT)' },
            { to: '/volume-analysis', icon: BarChart3, label: 'Volume Profile' },
            { to: '/options-greeks', icon: Gauge, label: 'Options Greeks' },
            { to: '/fibonacci', icon: GitBranch, label: 'Fibonacci Analysis' },
            { to: '/ichimoku', icon: Cloud, label: 'Ichimoku Cloud' },
        ]
    },
    {
        title: 'Sports',
        items: [
            { to: '/betting', icon: Target, label: 'Betting Hub' },
            { to: '/best-bets', icon: Trophy, label: 'Best Bets' },
            { to: '/games', icon: Gamepad2, label: 'Live Games' },
            { to: '/markets', icon: Activity, label: 'Predictions' },
            { to: '/kalshi', icon: Percent, label: 'Kalshi Markets' },
        ]
    },
    {
        title: 'Analysis',
        items: [
            { to: '/sectors', icon: PieChart, label: 'Sector Analysis' },
            { to: '/sentiment', icon: LineChart, label: 'Market Sentiment' },
            { to: '/multi-research', icon: Brain, label: 'Multi-Agent Research' },
            { to: '/analytics', icon: BarChart3, label: 'Analytics Performance' },
        ]
    },
    {
        title: 'Portfolio',
        items: [
            { to: '/risk', icon: Shield, label: 'Risk Dashboard' },
            { to: '/journal', icon: BookOpen, label: 'Trade Journal' },
            { to: '/dividends', icon: DollarSign, label: 'Dividend Tracker' },
            { to: '/tax-optimizer', icon: Calculator, label: 'Tax Optimizer' },
        ]
    },
    {
        title: 'Tools',
        items: [
            { to: '/backtest', icon: History, label: 'Backtesting' },
            { to: '/alerts', icon: Bell, label: 'Alert Management' },
        ]
    },
    {
        title: 'AI Assistant',
        items: [
            { to: '/chat', icon: MessageSquare, label: 'AVA Chat', highlight: true },
            { to: '/agents', icon: Bot, label: 'Agent Management' },
            { to: '/knowledge', icon: Brain, label: 'Knowledge Base' },
            { to: '/discord', icon: MessageSquare, label: 'Discord Messages' },
        ]
    },
    {
        title: 'Development',
        items: [
            { to: '/enhancements', icon: Sparkles, label: 'Enhancements' },
            { to: '/enhancements/agent', icon: Bot, label: 'Enhancement Agent' },
            { to: '/enhancements/qa', icon: TestTube, label: 'QA Testing' },
            { to: '/automations', icon: Cpu, label: 'Automations' },
        ]
    },
    {
        title: 'System',
        items: [
            { to: '/health', icon: HeartPulse, label: 'Health Dashboard' },
            { to: '/monitoring', icon: Monitor, label: 'System Monitoring' },
            { to: '/system', icon: Settings, label: 'System Management' },
            { to: '/cache', icon: Database, label: 'Cache Metrics' },
            { to: '/subscriptions', icon: CreditCard, label: 'Subscriptions' },
            { to: '/settings', icon: Settings, label: 'Settings' },
        ]
    }
]

export function Sidebar() {
    const [isOpen, setIsOpen] = useState(true)
    const location = useLocation()

    return (
        <aside
            className={clsx(
                "relative flex flex-col transition-all duration-300 ease-out z-40",
                "bg-gradient-to-b from-slate-900/95 to-slate-950/95 backdrop-blur-xl",
                "border-r border-slate-800/50",
                isOpen ? "w-64" : "w-20"
            )}
        >
            {/* Logo Header */}
            <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800/50">
                <div className={clsx(
                    "flex items-center gap-3 transition-opacity duration-200",
                    !isOpen && "opacity-0"
                )}>
                    <div className="relative">
                        <div className="w-9 h-9 rounded-xl overflow-hidden shadow-lg shadow-primary/20 ring-2 ring-primary/30">
                            <img src="/ava/NancyFace.jpg" alt="AVA" className="w-full h-full object-cover" />
                        </div>
                        <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-slate-900" />
                    </div>
                    <div>
                        <h1 className="text-lg font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                            AVA
                        </h1>
                        <p className="text-[10px] text-slate-500 -mt-0.5 font-medium tracking-wider">TRADING AI</p>
                    </div>
                </div>
                <button
                    onClick={() => setIsOpen(!isOpen)}
                    className={clsx(
                        "p-2 rounded-lg transition-all duration-200",
                        "hover:bg-white/5 text-slate-400 hover:text-white",
                        !isOpen && "mx-auto"
                    )}
                >
                    <ChevronLeft className={clsx(
                        "w-5 h-5 transition-transform duration-300",
                        !isOpen && "rotate-180"
                    )} />
                </button>
            </div>

            {/* Navigation */}
            <nav className="flex-1 overflow-y-auto py-4 px-3">
                {navSections.map((section, sectionIdx) => (
                    <div key={section.title} className={clsx(sectionIdx > 0 && "mt-4")}>
                        {isOpen && (
                            <div className="px-3 mb-2">
                                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                                    {section.title}
                                </span>
                            </div>
                        )}
                        <div className="space-y-0.5">
                            {section.items.map((item) => (
                                <NavItem
                                    key={item.to}
                                    to={item.to}
                                    icon={item.icon}
                                    label={item.label}
                                    isOpen={isOpen}
                                    isActive={location.pathname === item.to}
                                    highlight={item.highlight}
                                />
                            ))}
                        </div>
                    </div>
                ))}
            </nav>

            {/* Bottom Status */}
            <div className={clsx(
                "p-4 border-t border-slate-800/50",
                !isOpen && "flex justify-center"
            )}>
                {isOpen ? (
                    <div className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="text-xs font-medium text-emerald-400">Systems Online</span>
                        </div>
                        <p className="text-[10px] text-slate-500 mt-1">48 AI agents ready</p>
                    </div>
                ) : (
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                )}
            </div>
        </aside>
    )
}

interface NavItemProps {
    to: string
    icon: React.ComponentType<{ className?: string }>
    label: string
    isOpen: boolean
    isActive: boolean
    highlight?: boolean
}

function NavItem({ to, icon: Icon, label, isOpen, isActive, highlight }: NavItemProps) {
    return (
        <NavLink
            to={to}
            className={clsx(
                "group relative flex items-center gap-3 px-3 py-2 rounded-xl transition-all duration-200",
                isActive
                    ? "bg-primary/15 text-white"
                    : "text-slate-400 hover:text-white hover:bg-white/5",
                !isOpen && "justify-center px-0"
            )}
        >
            {/* Active indicator */}
            {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-primary rounded-r-full" />
            )}

            {/* Icon */}
            <div className={clsx(
                "relative flex items-center justify-center w-8 h-8 rounded-lg transition-all duration-200",
                isActive
                    ? "bg-primary/20"
                    : "bg-transparent group-hover:bg-white/5",
                highlight && !isActive && "bg-gradient-to-br from-primary/10 to-secondary/10"
            )}>
                <Icon className={clsx(
                    "w-4 h-4 transition-colors",
                    isActive ? "text-primary" : "text-current"
                )} />
                {highlight && (
                    <div className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full animate-pulse" />
                )}
            </div>

            {/* Label */}
            {isOpen && (
                <span className={clsx(
                    "text-sm font-medium transition-colors",
                    isActive && "text-white"
                )}>
                    {label}
                </span>
            )}

            {/* Tooltip for collapsed state */}
            {!isOpen && (
                <div className="absolute left-full ml-2 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 shadow-xl">
                    {label}
                </div>
            )}
        </NavLink>
    )
}
