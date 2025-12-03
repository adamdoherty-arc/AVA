import { useState, useRef, useEffect } from 'react'
import { api } from '../services/api'
import { Send, Bot, User, Loader2, Sparkles, Zap, TrendingUp, PieChart, Activity, Brain, MessageSquare } from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Message {
    role: 'user' | 'assistant'
    content: string
    sources?: any[]
    confidence?: number
    agent_used?: string
    intent?: string
}

export function Chat() {
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: "Hello! I'm **AVA**, your AI trading assistant powered by 35 specialized agents.\n\n" +
                "I can help you with:\n" +
                "- **Portfolio Analysis** - Review your holdings and performance\n" +
                "- **Technical Analysis** - Chart patterns, indicators, and signals\n" +
                "- **Options Strategies** - Generate custom strategies for any symbol\n" +
                "- **Sports Betting** - AI-powered game predictions and best bets\n\n" +
                "What would you like to explore today?"
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSubmit = async (e: React.FormEvent, overrideInput?: string) => {
        e.preventDefault()
        const userMessage = overrideInput || input.trim()
        if (!userMessage || loading) return

        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setLoading(true)

        try {
            const response = await api.chat(userMessage)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                sources: response.sources,
                confidence: response.confidence,
                agent_used: response.agent_used,
                intent: response.intent
            }])
        } catch (error) {
            console.error("Chat error:", error)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "I apologize, but I encountered an error processing your request. Please try again."
            }])
        } finally {
            setLoading(false)
        }
    }

    const handleQuickAction = (action: string) => {
        let message = ""
        switch (action) {
            case 'portfolio': message = "Analyze my portfolio performance"; break;
            case 'opportunities': message = "Find the best trading opportunities today"; break;
            case 'strategies': message = "Generate options strategies for AAPL with moderate risk"; break;
            case 'sports': message = "What are the best NFL bets this week?"; break;
        }
        handleSubmit({ preventDefault: () => { } } as React.FormEvent, message)
    }

    return (
        <div className="flex h-[calc(100vh-8rem)] gap-6">
            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col glass-card overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-slate-800/50 bg-slate-900/30">
                    <div className="flex items-center gap-4">
                        <div className="relative">
                            <div className="w-12 h-12 rounded-xl overflow-hidden shadow-lg shadow-primary/20 ring-2 ring-primary/30">
                                <img src="/ava/NancyFace.jpg" alt="AVA" className="w-full h-full object-cover" />
                            </div>
                            <div className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 bg-emerald-500 rounded-full border-2 border-slate-900" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <h2 className="text-lg font-bold text-white">AVA</h2>
                                <span className="badge-info text-[10px]">v3.0</span>
                            </div>
                            <p className="text-xs text-slate-400">35 Specialized AI Agents Ready</p>
                        </div>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={clsx(
                            "flex gap-4",
                            msg.role === 'user' ? "justify-end" : "justify-start"
                        )}>
                            {msg.role === 'assistant' && (
                                <div className="w-9 h-9 rounded-xl overflow-hidden flex-shrink-0 shadow-lg ring-2 ring-primary/30">
                                    <img src="/ava/NancyFace.jpg" alt="AVA" className="w-full h-full object-cover" />
                                </div>
                            )}

                            <div className={clsx(
                                "max-w-[75%] space-y-2",
                                msg.role === 'user' && "order-first"
                            )}>
                                <div className={clsx(
                                    "p-4 rounded-2xl text-sm leading-relaxed",
                                    msg.role === 'user'
                                        ? "bg-primary text-white rounded-br-md"
                                        : "bg-slate-800/60 text-slate-200 rounded-bl-md border border-slate-700/50"
                                )}>
                                    {msg.role === 'user' ? (
                                        <div className="whitespace-pre-wrap">{msg.content}</div>
                                    ) : (
                                        <div className="prose prose-invert prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-li:my-0.5">
                                            <ReactMarkdown
                                                remarkPlugins={[remarkGfm]}
                                                components={{
                                                    a: ({ ...props }) => <a {...props} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer" />,
                                                    code: ({ ...props }) => <code {...props} className="bg-slate-900/50 rounded px-1.5 py-0.5 text-emerald-400" />,
                                                    pre: ({ ...props }) => <pre {...props} className="bg-slate-900/50 rounded-lg p-3 overflow-x-auto" />,
                                                    strong: ({ ...props }) => <strong {...props} className="text-white font-semibold" />,
                                                }}
                                            >
                                                {msg.content}
                                            </ReactMarkdown>
                                        </div>
                                    )}
                                </div>

                                {/* Agent Info */}
                                {msg.role === 'assistant' && msg.agent_used && (
                                    <div className="flex items-center gap-2 text-xs">
                                        <span className="badge-success">
                                            <Bot className="w-3 h-3 mr-1" />
                                            {msg.agent_used}
                                        </span>
                                        {msg.sources && msg.sources.length > 0 && (
                                            <span className="badge-neutral">
                                                <Sparkles className="w-3 h-3 mr-1" />
                                                {msg.sources.length} sources
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>

                            {msg.role === 'user' && (
                                <div className="w-9 h-9 rounded-xl bg-slate-700 flex items-center justify-center flex-shrink-0">
                                    <User className="w-5 h-5 text-slate-300" />
                                </div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="flex gap-4 justify-start">
                            <div className="w-9 h-9 rounded-xl overflow-hidden flex-shrink-0 shadow-lg ring-2 ring-primary/30">
                                <img src="/ava/NancyFace.jpg" alt="AVA" className="w-full h-full object-cover" />
                            </div>
                            <div className="bg-slate-800/60 p-4 rounded-2xl rounded-bl-md border border-slate-700/50 flex items-center gap-3">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce"></span>
                                </div>
                                <span className="text-sm text-slate-400">Thinking...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-slate-800/50 bg-slate-900/30">
                    <form onSubmit={(e) => handleSubmit(e)} className="flex gap-3">
                        <div className="relative flex-1">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about markets, strategies, sports predictions..."
                                className="input-field pr-4"
                                disabled={loading}
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={!input.trim() || loading}
                            className="btn-primary px-5 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </form>
                </div>
            </div>

            {/* Right Sidebar */}
            <div className="w-80 flex flex-col gap-5 flex-shrink-0">
                {/* Avatar Card */}
                <div className="glass-card p-6 text-center">
                    <div className="relative w-24 h-24 mx-auto mb-4">
                        <div className="w-full h-full rounded-2xl bg-gradient-to-br from-primary to-secondary p-0.5 shadow-xl shadow-primary/20">
                            <div className="w-full h-full rounded-2xl overflow-hidden">
                                <img src="/ava/NancyFace.jpg" alt="AVA" className="w-full h-full object-cover" />
                            </div>
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-emerald-500 rounded-full border-4 border-slate-900 flex items-center justify-center">
                            <span className="w-2 h-2 bg-white rounded-full"></span>
                        </div>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-1">AVA</h3>
                    <p className="text-sm text-slate-400 mb-4">Advanced Virtual Analyst</p>
                    <div className="flex justify-center gap-2">
                        <span className="badge-success">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5" />
                            Online
                        </span>
                        <span className="badge-info">35 Agents</span>
                    </div>
                </div>

                {/* Capabilities */}
                <div className="glass-card p-5 flex-1">
                    <div className="flex items-center gap-2 mb-4">
                        <Zap className="w-4 h-4 text-amber-400" />
                        <h3 className="text-sm font-semibold text-white">Quick Actions</h3>
                    </div>
                    <div className="space-y-2">
                        <QuickActionButton
                            icon={<PieChart className="w-4 h-4" />}
                            label="Portfolio Analysis"
                            description="Review performance"
                            onClick={() => handleQuickAction('portfolio')}
                            color="emerald"
                        />
                        <QuickActionButton
                            icon={<TrendingUp className="w-4 h-4" />}
                            label="Find Opportunities"
                            description="Trading signals"
                            onClick={() => handleQuickAction('opportunities')}
                            color="blue"
                        />
                        <QuickActionButton
                            icon={<Activity className="w-4 h-4" />}
                            label="Options Strategies"
                            description="Custom generation"
                            onClick={() => handleQuickAction('strategies')}
                            color="purple"
                        />
                        <QuickActionButton
                            icon={<Sparkles className="w-4 h-4" />}
                            label="Sports Predictions"
                            description="AI-powered bets"
                            onClick={() => handleQuickAction('sports')}
                            color="amber"
                        />
                    </div>
                </div>

                {/* Agents Info */}
                <div className="glass-card p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <MessageSquare className="w-4 h-4 text-primary" />
                        <h3 className="text-sm font-semibold text-white">Active Capabilities</h3>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                        {['Trading', 'Options', 'Technical', 'Sports', 'Research', 'Portfolio'].map(cap => (
                            <span key={cap} className="text-xs px-2 py-1 rounded-md bg-slate-800/50 text-slate-400 border border-slate-700/50">
                                {cap}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

interface QuickActionButtonProps {
    icon: React.ReactNode
    label: string
    description: string
    onClick: () => void
    color: 'emerald' | 'blue' | 'purple' | 'amber'
}

function QuickActionButton({ icon, label, description, onClick, color }: QuickActionButtonProps) {
    const colorMap = {
        emerald: 'hover:border-emerald-500/40 group-hover:text-emerald-400 group-hover:bg-emerald-500/10',
        blue: 'hover:border-blue-500/40 group-hover:text-blue-400 group-hover:bg-blue-500/10',
        purple: 'hover:border-purple-500/40 group-hover:text-purple-400 group-hover:bg-purple-500/10',
        amber: 'hover:border-amber-500/40 group-hover:text-amber-400 group-hover:bg-amber-500/10',
    }

    return (
        <button
            onClick={onClick}
            className={`group w-full flex items-center gap-3 p-3 rounded-xl bg-slate-800/30 border border-slate-700/50 transition-all duration-200 text-left ${colorMap[color]}`}
        >
            <div className={`w-9 h-9 rounded-lg bg-slate-800/50 flex items-center justify-center text-slate-400 transition-all ${colorMap[color]}`}>
                {icon}
            </div>
            <div>
                <span className="text-sm font-medium text-white block">{label}</span>
                <span className="text-xs text-slate-500">{description}</span>
            </div>
        </button>
    )
}
