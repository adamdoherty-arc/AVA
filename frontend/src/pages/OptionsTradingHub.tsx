import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
    Layers, Bot, Calendar, Search, Clock, TrendingUp,
    BarChart3, Calculator, Sparkles, ArrowRight, Zap
} from 'lucide-react'

const HUB_SECTIONS = [
    {
        id: 'ai-agent',
        title: 'AI Options Agent',
        description: 'Multi-criteria analysis with LLM reasoning',
        icon: Bot,
        color: 'from-purple-500 to-indigo-600',
        link: '/ai-options',
        stats: { label: 'MCDM + AI', value: 'Scoring' }
    },
    {
        id: 'premium-scanner',
        title: 'Premium Scanner',
        description: 'Find the best CSP opportunities by DTE',
        icon: Search,
        color: 'from-amber-500 to-orange-600',
        link: '/scanner',
        stats: { label: 'DTE Options', value: '7/14/30/45' }
    },
    {
        id: 'dte-scanner',
        title: '0-7 DTE Scanner',
        description: 'Short-term theta capture opportunities',
        icon: Clock,
        color: 'from-red-500 to-pink-600',
        link: '/dte-scanner',
        stats: { label: 'Short Term', value: 'Theta' }
    },
    {
        id: 'calendar-spreads',
        title: 'Calendar Spreads',
        description: 'Time decay profit finder',
        icon: Calendar,
        color: 'from-blue-500 to-cyan-600',
        link: '/calendar-spreads',
        stats: { label: 'IV Skew', value: 'Analysis' }
    },
    {
        id: 'earnings',
        title: 'Earnings Calendar',
        description: 'High-quality earnings plays',
        icon: TrendingUp,
        color: 'from-emerald-500 to-teal-600',
        link: '/earnings',
        stats: { label: 'IV Plays', value: 'Earnings' }
    },
    {
        id: 'options-flow',
        title: 'Options Flow',
        description: 'Unusual options activity tracking',
        icon: BarChart3,
        color: 'from-violet-500 to-purple-600',
        link: '/options-flow',
        stats: { label: 'Unusual', value: 'Activity' }
    },
    {
        id: 'technicals',
        title: 'Technical Indicators',
        description: 'Comprehensive technical analysis',
        icon: Sparkles,
        color: 'from-cyan-500 to-blue-600',
        link: '/technicals',
        stats: { label: 'RSI/MACD', value: 'Signals' }
    },
    {
        id: 'position-sizing',
        title: 'Position Sizing',
        description: 'Calculate optimal position sizes',
        icon: Calculator,
        color: 'from-slate-500 to-slate-700',
        link: '/position-sizing',
        stats: { label: 'Risk', value: 'Management' }
    }
]

export default function OptionsTradingHub() {
    const [hoveredSection, setHoveredSection] = useState<string | null>(null)

    return (
        <div className="space-y-6">
            {/* Header */}
            <header>
                <h1 className="page-title flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
                        <Layers className="w-5 h-5 text-white" />
                    </div>
                    Options Trading Hub
                </h1>
                <p className="page-subtitle">Your central command center for options trading</p>
            </header>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="stat-card">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-xs text-slate-400 uppercase tracking-wide">Available Tools</p>
                            <p className="text-2xl font-bold text-white">{HUB_SECTIONS.length}</p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center text-blue-400">
                            <Layers className="w-5 h-5" />
                        </div>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-xs text-slate-400 uppercase tracking-wide">AI-Powered</p>
                            <p className="text-2xl font-bold text-purple-400">MCDM + LLM</p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400">
                            <Bot className="w-5 h-5" />
                        </div>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-xs text-slate-400 uppercase tracking-wide">DTE Options</p>
                            <p className="text-2xl font-bold text-amber-400">7-45 Days</p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center text-amber-400">
                            <Clock className="w-5 h-5" />
                        </div>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-xs text-slate-400 uppercase tracking-wide">Strategy Focus</p>
                            <p className="text-2xl font-bold text-emerald-400">CSP / Wheel</p>
                        </div>
                        <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                            <TrendingUp className="w-5 h-5" />
                        </div>
                    </div>
                </div>
            </div>

            {/* Tool Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {HUB_SECTIONS.map((section) => {
                    const Icon = section.icon
                    const isHovered = hoveredSection === section.id

                    return (
                        <Link
                            key={section.id}
                            to={section.link}
                            className="glass-card p-5 group hover:border-primary/50 transition-all duration-300"
                            onMouseEnter={() => setHoveredSection(section.id)}
                            onMouseLeave={() => setHoveredSection(null)}
                        >
                            <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${section.color} flex items-center justify-center shadow-lg mb-4 group-hover:scale-110 transition-transform`}>
                                <Icon className="w-6 h-6 text-white" />
                            </div>

                            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                                {section.title}
                                <ArrowRight className={`w-4 h-4 transition-all ${isHovered ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2'}`} />
                            </h3>

                            <p className="text-sm text-slate-400 mb-4">
                                {section.description}
                            </p>

                            <div className="flex items-center justify-between pt-3 border-t border-slate-700/50">
                                <span className="text-xs text-slate-500">{section.stats.label}</span>
                                <span className="text-sm font-medium text-primary">{section.stats.value}</span>
                            </div>
                        </Link>
                    )
                })}
            </div>

            {/* Quick Actions */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-400" />
                    Quick Actions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Link
                        to="/ai-options"
                        className="flex items-center gap-4 p-4 bg-purple-500/10 rounded-xl border border-purple-500/30 hover:bg-purple-500/20 transition-colors"
                    >
                        <Bot className="w-8 h-8 text-purple-400" />
                        <div>
                            <div className="font-semibold text-white">Run AI Analysis</div>
                            <div className="text-sm text-slate-400">Scan stocks with MCDM + LLM</div>
                        </div>
                    </Link>
                    <Link
                        to="/scanner"
                        className="flex items-center gap-4 p-4 bg-amber-500/10 rounded-xl border border-amber-500/30 hover:bg-amber-500/20 transition-colors"
                    >
                        <Search className="w-8 h-8 text-amber-400" />
                        <div>
                            <div className="font-semibold text-white">Quick Premium Scan</div>
                            <div className="text-sm text-slate-400">Find best CSP opportunities</div>
                        </div>
                    </Link>
                    <Link
                        to="/earnings"
                        className="flex items-center gap-4 p-4 bg-emerald-500/10 rounded-xl border border-emerald-500/30 hover:bg-emerald-500/20 transition-colors"
                    >
                        <Calendar className="w-8 h-8 text-emerald-400" />
                        <div>
                            <div className="font-semibold text-white">Check Earnings</div>
                            <div className="text-sm text-slate-400">Upcoming earnings plays</div>
                        </div>
                    </Link>
                </div>
            </div>

            {/* Workflow Guide */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Recommended Workflow</h3>
                <div className="flex items-center gap-4 overflow-x-auto pb-2">
                    {[
                        { step: 1, label: 'AI Options Agent', desc: 'Screen stocks' },
                        { step: 2, label: 'Premium Scanner', desc: 'Find best DTE' },
                        { step: 3, label: 'Technical Analysis', desc: 'Confirm signals' },
                        { step: 4, label: 'Position Sizing', desc: 'Calculate size' },
                        { step: 5, label: 'Execute Trade', desc: 'Place order' }
                    ].map((item, idx, arr) => (
                        <div key={item.step} className="flex items-center gap-4">
                            <div className="flex-shrink-0 w-32">
                                <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold mb-2">
                                    {item.step}
                                </div>
                                <div className="text-sm font-medium text-white">{item.label}</div>
                                <div className="text-xs text-slate-400">{item.desc}</div>
                            </div>
                            {idx < arr.length - 1 && (
                                <ArrowRight className="w-5 h-5 text-slate-600 flex-shrink-0" />
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
