import { useState, useRef, useEffect } from 'react'
import { api } from '../services/api'
import {
    Send, Bot, User, ChevronLeft, ChevronRight,
    Zap, PieChart, TrendingUp, Activity, Settings2, ChevronDown, Sparkles
} from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Message {
    role: 'user' | 'assistant'
    content: string
    agent_used?: string
    model_used?: string
}

interface ModelInfo {
    name: string
    description: string
    speed: string
    tier?: 'auto' | 'free' | 'cheap' | 'standard' | 'premium'
}

// AVA avatar image - static NancyFace
const AVA_IMAGE = '/ava/NancyFace.jpg'

// Status labels for loading states
const LOADING_LABELS = ['Thinking', 'Analyzing', 'Processing', 'Computing']

export function AvaChatWidget() {
    // AVA sidebar panel - collapsible
    const [isCollapsed, setIsCollapsed] = useState(false)
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: "Hi! I'm **AVA**, your AI trading assistant. I can help with portfolio analysis, options strategies, and sports predictions. What would you like to explore?"
        }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [selectedModel, setSelectedModel] = useState('auto')
    const [showModelSelect, setShowModelSelect] = useState(false)
    const [models, setModels] = useState<Record<string, ModelInfo>>({
        auto: { name: 'Auto Select', description: 'Intelligently route based on query', speed: 'varies' },
        'groq-llama70b': { name: 'Llama 3.3 70B (Groq)', description: 'Best free model', speed: '~300 tok/s' },
        'hf-llama8b': { name: 'Llama 3.1 8B (HF)', description: 'Free tier', speed: '~50 tok/s' },
        'deepseek-chat': { name: 'DeepSeek Chat', description: '$0.14/1M tokens', speed: '~80 tok/s' },
        'claude-sonnet': { name: 'Claude Sonnet 4.5', description: 'Best reasoning', speed: '~50 tok/s' },
    })
    const [labelIndex, setLabelIndex] = useState(0)
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLInputElement>(null)
    const modelSelectRef = useRef<HTMLDivElement>(null)

    // Rotate loading labels when thinking
    useEffect(() => {
        if (loading) {
            const interval = setInterval(() => {
                setLabelIndex((prev) => (prev + 1) % LOADING_LABELS.length)
            }, 1200)
            return () => clearInterval(interval)
        }
    }, [loading])

    // Load available models on mount
    useEffect(() => {
        api.getChatModels().then(data => {
            if (data.models) {
                setModels(data.models)
            }
        }).catch(() => {
            // Use defaults on error
        })
    }, [])

    // Close model select on outside click
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (modelSelectRef.current && !modelSelectRef.current.contains(event.target as Node)) {
                setShowModelSelect(false)
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        const userMessage = input.trim()
        if (!userMessage || loading) return

        setInput('')
        setMessages(prev => [...prev, { role: 'user', content: userMessage }])
        setLoading(true)

        try {
            const response = await api.chat(userMessage, [], selectedModel)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                agent_used: response.agent_used,
                model_used: selectedModel !== 'auto' ? selectedModel : undefined
            }])
        } catch (error) {
            console.error("Chat error:", error)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "Sorry, I encountered an error. Please try again."
            }])
        } finally {
            setLoading(false)
        }
    }

    const handleQuickAction = (message: string) => {
        setMessages(prev => [...prev, { role: 'user', content: message }])
        setLoading(true)

        api.chat(message, [], selectedModel).then(response => {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                agent_used: response.agent_used
            }])
        }).catch(() => {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "Sorry, I encountered an error. Please try again."
            }])
        }).finally(() => {
            setLoading(false)
        })
    }

    if (isCollapsed) {
        return (
            <div className="w-16 flex-shrink-0 border-l border-slate-800/50 bg-slate-900/50 flex flex-col items-center py-4">
                <button
                    onClick={() => setIsCollapsed(false)}
                    className="w-14 h-14 rounded-xl overflow-hidden shadow-lg shadow-primary/20 hover:scale-105 transition-transform ring-2 ring-primary/50"
                >
                    <img
                        src={AVA_IMAGE}
                        alt="AVA"
                        className="w-full h-full object-cover"
                    />
                </button>
                <div className="mt-2 w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <button
                    onClick={() => setIsCollapsed(false)}
                    className="mt-4 p-2 rounded-lg hover:bg-slate-800/50 text-slate-400 hover:text-white transition-colors"
                >
                    <ChevronLeft size={16} />
                </button>
            </div>
        )
    }

    return (
        <div className="w-80 flex-shrink-0 border-l border-slate-800/50 bg-slate-900/30 flex flex-col">
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-800/50 flex items-center justify-between bg-slate-900/50">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="w-12 h-12 rounded-xl overflow-hidden shadow-lg ring-2 ring-primary/30">
                            <img
                                src={AVA_IMAGE}
                                alt="AVA"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className={clsx(
                            "absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-slate-900",
                            loading ? "bg-amber-500 animate-pulse" : "bg-emerald-500"
                        )} />
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="font-bold text-white text-sm">AVA</h3>
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary font-medium">AI</span>
                        </div>
                        <p className="text-[11px] text-slate-400">
                            {loading ? LOADING_LABELS[labelIndex] + '...' : '35 Agents Ready'}
                        </p>
                    </div>
                </div>
                <button
                    onClick={() => setIsCollapsed(true)}
                    className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                >
                    <ChevronRight size={16} />
                </button>
            </div>

            {/* Avatar Section with Rotating Images */}
            <div className="px-4 py-5 border-b border-slate-800/50 text-center bg-slate-900/30">
                <div className="relative w-36 h-36 mx-auto mb-4">
                    {/* Outer ring */}
                    <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary to-violet-600 p-[3px]">
                        <div className="w-full h-full rounded-2xl bg-slate-900" />
                    </div>

                    {/* Avatar image */}
                    <div className="absolute inset-1 rounded-xl overflow-hidden">
                        <img
                            src={AVA_IMAGE}
                            alt="AVA"
                            className="w-full h-full object-cover"
                        />
                    </div>

                    {/* Status indicator */}
                    <div className={clsx(
                        "absolute -bottom-1 -right-1 w-6 h-6 rounded-full border-3 border-slate-900 flex items-center justify-center",
                        loading ? "bg-amber-500" : "bg-emerald-500"
                    )}>
                        <span className="w-2.5 h-2.5 bg-white rounded-full" />
                    </div>
                </div>
                <h4 className="font-bold text-white text-sm">Advanced Virtual Analyst</h4>
                <p className="text-xs text-slate-400 mt-1">Powered by 35 specialized AI agents</p>

                {/* Model Selector */}
                <div className="mt-3 relative" ref={modelSelectRef}>
                    <button
                        onClick={() => setShowModelSelect(!showModelSelect)}
                        className="flex items-center justify-center gap-2 mx-auto px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50 text-xs text-slate-300 hover:bg-slate-700/50 hover:text-white transition-colors"
                    >
                        <Settings2 size={12} />
                        <span>{models[selectedModel]?.name || 'Auto Select'}</span>
                        <ChevronDown size={12} className={clsx(
                            "transition-transform",
                            showModelSelect && "rotate-180"
                        )} />
                    </button>

                    {showModelSelect && (
                        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 bg-slate-800 border border-slate-700 rounded-xl shadow-xl z-50 overflow-hidden max-h-80 overflow-y-auto">
                            <div className="p-2 border-b border-slate-700 bg-slate-800/80 sticky top-0">
                                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Select Model</span>
                            </div>
                            {Object.entries(models).map(([key, model]) => {
                                // Use API-provided tier or determine from key
                                const tier = model.tier || (
                                    key === 'auto' ? 'auto'
                                    : key.startsWith('groq-') || key.startsWith('hf-') ? 'free'
                                    : key.startsWith('deepseek') || key.startsWith('gemini-flash') ? 'cheap'
                                    : key.includes('claude-sonnet') || key.includes('gpt-4o') && !key.includes('mini') ? 'premium'
                                    : 'standard'
                                )

                                return (
                                    <button
                                        key={key}
                                        onClick={() => {
                                            setSelectedModel(key)
                                            setShowModelSelect(false)
                                        }}
                                        className={clsx(
                                            "w-full px-3 py-2 text-left hover:bg-slate-700/50 transition-colors flex items-center justify-between gap-2",
                                            selectedModel === key && "bg-primary/10"
                                        )}
                                    >
                                        <div className="flex-1 min-w-0">
                                            <div className="text-xs font-medium text-white truncate">{model.name}</div>
                                            <div className="text-[10px] text-slate-400 truncate">{model.description}</div>
                                        </div>
                                        <div className="flex flex-col items-end gap-0.5 flex-shrink-0">
                                            <span className={clsx(
                                                "text-[8px] px-1.5 py-0.5 rounded font-semibold uppercase",
                                                tier === 'free' && "bg-emerald-500/20 text-emerald-400",
                                                tier === 'cheap' && "bg-cyan-500/20 text-cyan-400",
                                                tier === 'standard' && "bg-amber-500/20 text-amber-400",
                                                tier === 'premium' && "bg-violet-500/20 text-violet-400",
                                                tier === 'auto' && "bg-primary/20 text-primary",
                                            )}>
                                                {tier}
                                            </span>
                                            <span className="text-[9px] text-slate-500">{model.speed}</span>
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Actions */}
            <div className="px-3 py-3 border-b border-slate-800/50">
                <div className="flex items-center gap-1.5 mb-2">
                    <Zap className="w-3 h-3 text-amber-400" />
                    <span className="text-xs font-medium text-slate-400">Quick Actions</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                    <QuickBtn icon={<PieChart size={12} />} label="Portfolio" onClick={() => handleQuickAction("Analyze my portfolio")} />
                    <QuickBtn icon={<TrendingUp size={12} />} label="Opportunities" onClick={() => handleQuickAction("Find best trading opportunities")} />
                    <QuickBtn icon={<Activity size={12} />} label="Options" onClick={() => handleQuickAction("Generate options strategies for AAPL")} />
                    <QuickBtn icon={<Sparkles size={12} />} label="NFL Bets" onClick={() => handleQuickAction("What are the best NFL bets?")} />
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
                {messages.map((msg, idx) => (
                    <div key={idx} className={clsx(
                        "flex gap-2",
                        msg.role === 'user' ? "justify-end" : "justify-start"
                    )}>
                        {msg.role === 'assistant' && (
                            <div className="w-6 h-6 rounded-lg overflow-hidden flex-shrink-0 ring-1 ring-primary/30">
                                <img src={AVA_IMAGE} alt="AVA" className="w-full h-full object-cover" />
                            </div>
                        )}

                        <div className={clsx(
                            "max-w-[85%]",
                            msg.role === 'user' && "order-first"
                        )}>
                            <div className={clsx(
                                "px-3 py-2 rounded-xl text-xs leading-relaxed",
                                msg.role === 'user'
                                    ? "bg-primary text-white rounded-br-sm"
                                    : "bg-slate-800/80 text-slate-200 rounded-bl-sm border border-slate-700/50"
                            )}>
                                {msg.role === 'user' ? (
                                    <span>{msg.content}</span>
                                ) : (
                                    <div className="prose prose-invert prose-xs max-w-none">
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm]}
                                            components={{
                                                a: ({ ...props }) => <a {...props} className="text-primary hover:underline" />,
                                                strong: ({ ...props }) => <strong {...props} className="text-white font-semibold" />,
                                                p: ({ ...props }) => <p {...props} className="my-1" />,
                                                ul: ({ ...props }) => <ul {...props} className="my-1 ml-3 list-disc" />,
                                                li: ({ ...props }) => <li {...props} className="my-0.5" />,
                                            }}
                                        >
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>
                                )}
                            </div>

                            {(msg.agent_used || msg.model_used) && (
                                <span className="text-[10px] text-slate-500 flex items-center gap-1 mt-1 px-1">
                                    <Bot size={10} />
                                    {msg.agent_used}
                                    {msg.model_used && <span className="text-slate-600">• {msg.model_used}</span>}
                                </span>
                            )}
                        </div>

                        {msg.role === 'user' && (
                            <div className="w-6 h-6 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
                                <User className="w-3.5 h-3.5 text-slate-300" />
                            </div>
                        )}
                    </div>
                ))}

                {loading && (
                    <div className="flex gap-2 justify-start">
                        <div className="w-6 h-6 rounded-lg overflow-hidden flex-shrink-0 ring-1 ring-primary/30">
                            <img
                                src={AVA_IMAGE}
                                alt="AVA"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="bg-slate-800/80 px-3 py-2 rounded-xl rounded-bl-sm border border-slate-700/50 flex items-center gap-2">
                            <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                <span className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"></span>
                            </div>
                            <span className="text-[10px] text-slate-500">{LOADING_LABELS[labelIndex]}...</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-3 border-t border-slate-800/50 bg-slate-900/50">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask AVA anything..."
                        className="flex-1 px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all"
                        disabled={loading}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || loading}
                        className="px-3 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:shadow-lg hover:shadow-primary/20"
                    >
                        <Send size={14} />
                    </button>
                </form>
                <div className="mt-2 flex items-center justify-center gap-2 text-[10px] text-slate-500">
                    <span>Model:</span>
                    <span className="text-slate-400">{models[selectedModel]?.name}</span>
                </div>
            </div>
        </div>
    )
}

function QuickBtn({ icon, label, onClick }: { icon: React.ReactNode, label: string, onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-slate-800/50 text-slate-300 hover:bg-primary/20 hover:text-primary border border-slate-700/50 hover:border-primary/30 transition-all text-xs group"
        >
            <span className="group-hover:scale-110 transition-transform">{icon}</span>
            {label}
        </button>
    )
}
