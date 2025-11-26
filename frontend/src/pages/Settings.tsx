import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Settings as SettingsIcon, Key, Server, Bell, Palette, Save,
    CheckCircle, AlertCircle, RefreshCw, Eye, EyeOff, Zap, Database,
    Bot, Shield, Cloud, Cpu
} from 'lucide-react'
import clsx from 'clsx'

export default function Settings() {
    const queryClient = useQueryClient()
    const [activeTab, setActiveTab] = useState('connections')
    const [showSecrets, setShowSecrets] = useState(false)

    // Form states
    const [robinhoodUsername, setRobinhoodUsername] = useState('')
    const [robinhoodPassword, setRobinhoodPassword] = useState('')
    const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434')
    const [defaultModel, setDefaultModel] = useState('qwen2.5:32b')

    // Health check query
    const { data: healthData, isLoading: healthLoading, refetch: refetchHealth } = useQuery({
        queryKey: ['system-health'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/health')
            return data
        },
        refetchInterval: 30000
    })

    // Save settings mutation
    const saveMutation = useMutation({
        mutationFn: async (settings: Record<string, any>) => {
            const { data } = await axiosInstance.post('/settings', settings)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['system-health'] })
        }
    })

    const handleSaveConnections = () => {
        saveMutation.mutate({
            robinhood_username: robinhoodUsername,
            robinhood_password: robinhoodPassword
        })
    }

    const handleSaveAI = () => {
        saveMutation.mutate({
            ollama_url: ollamaUrl,
            default_model: defaultModel
        })
    }

    const tabs = [
        { id: 'connections', label: 'Connections', icon: <Server size={18} /> },
        { id: 'ai', label: 'AI Settings', icon: <Zap size={18} /> },
        { id: 'notifications', label: 'Notifications', icon: <Bell size={18} /> },
        { id: 'appearance', label: 'Appearance', icon: <Palette size={18} /> },
        { id: 'system', label: 'System', icon: <Database size={18} /> }
    ]

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-slate-500 to-slate-700 flex items-center justify-center shadow-lg">
                        <SettingsIcon className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Settings</h1>
                        <p className="text-sm text-slate-400">Configure AVA platform settings</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className={clsx(
                        "badge-neutral flex items-center gap-2",
                        healthData?.status === 'healthy' && "badge-success"
                    )}>
                        <span className={clsx(
                            "w-2 h-2 rounded-full",
                            healthData?.status === 'healthy' ? "bg-emerald-400 animate-pulse" : "bg-slate-400"
                        )}></span>
                        {healthData?.status === 'healthy' ? 'System Healthy' : 'Checking...'}
                    </div>
                    <button
                        onClick={() => refetchHealth()}
                        className="btn-secondary p-2"
                        disabled={healthLoading}
                    >
                        <RefreshCw size={16} className={healthLoading ? 'animate-spin' : ''} />
                    </button>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* Tabs Sidebar */}
                <div className="lg:col-span-1">
                    <div className="glass-card p-2 space-y-1">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={clsx(
                                    "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 text-left",
                                    activeTab === tab.id
                                        ? "bg-primary/20 text-primary border border-primary/30"
                                        : "text-slate-400 hover:bg-slate-800/50 hover:text-white border border-transparent"
                                )}
                            >
                                {tab.icon}
                                <span className="font-medium">{tab.label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content Area */}
                <div className="lg:col-span-4">
                    <div className="glass-card p-6">
                        {/* Connections Tab */}
                        {activeTab === 'connections' && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                        <Cloud className="w-5 h-5 text-emerald-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">Broker Connections</h3>
                                        <p className="text-sm text-slate-400">Connect your trading accounts</p>
                                    </div>
                                </div>

                                {/* Robinhood */}
                                <div className="glass-card p-6 border border-emerald-500/20">
                                    <div className="flex items-center justify-between mb-5">
                                        <div className="flex items-center gap-4">
                                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                                                <span className="text-white font-bold text-lg">RH</span>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold text-white">Robinhood</h4>
                                                <p className="text-sm text-slate-400">Stock & Options Trading</p>
                                            </div>
                                        </div>
                                        <span className="badge-warning">Not Connected</span>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4 mb-5">
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Username</label>
                                            <input
                                                type="text"
                                                value={robinhoodUsername}
                                                onChange={(e) => setRobinhoodUsername(e.target.value)}
                                                className="input-field"
                                                placeholder="Email or Username"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Password</label>
                                            <div className="relative">
                                                <input
                                                    type={showSecrets ? 'text' : 'password'}
                                                    value={robinhoodPassword}
                                                    onChange={(e) => setRobinhoodPassword(e.target.value)}
                                                    className="input-field pr-12"
                                                    placeholder="Password"
                                                />
                                                <button
                                                    type="button"
                                                    onClick={() => setShowSecrets(!showSecrets)}
                                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
                                                >
                                                    {showSecrets ? <EyeOff size={18} /> : <Eye size={18} />}
                                                </button>
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleSaveConnections}
                                        disabled={saveMutation.isPending}
                                        className="btn-primary flex items-center gap-2"
                                    >
                                        <Save size={16} />
                                        {saveMutation.isPending ? 'Connecting...' : 'Connect'}
                                    </button>
                                </div>

                                {/* TradingView */}
                                <div className="glass-card p-6 opacity-60">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-4">
                                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                                                <span className="text-white font-bold text-lg">TV</span>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold text-white">TradingView</h4>
                                                <p className="text-sm text-slate-400">Charts & Watchlists</p>
                                            </div>
                                        </div>
                                        <span className="badge-neutral">Coming Soon</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* AI Settings Tab */}
                        {activeTab === 'ai' && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center">
                                        <Cpu className="w-5 h-5 text-violet-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">AI Configuration</h3>
                                        <p className="text-sm text-slate-400">Configure local LLM settings</p>
                                    </div>
                                </div>

                                {/* Ollama Settings */}
                                <div className="glass-card p-6 border border-violet-500/20">
                                    <div className="flex items-center gap-3 mb-5">
                                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
                                            <Bot className="w-5 h-5 text-white" />
                                        </div>
                                        <div>
                                            <h4 className="font-semibold text-white">Local LLM (Ollama)</h4>
                                            <p className="text-sm text-slate-400">Power AVA with local AI models</p>
                                        </div>
                                    </div>

                                    <div className="space-y-4 mb-5">
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Ollama Server URL</label>
                                            <input
                                                type="text"
                                                value={ollamaUrl}
                                                onChange={(e) => setOllamaUrl(e.target.value)}
                                                className="input-field"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Default Model</label>
                                            <select
                                                value={defaultModel}
                                                onChange={(e) => setDefaultModel(e.target.value)}
                                                className="input-field"
                                            >
                                                <option value="qwen2.5:32b">Qwen 2.5 32B (Recommended)</option>
                                                <option value="llama3.3:70b">Llama 3.3 70B (Slowest, Best)</option>
                                                <option value="qwen2.5:14b">Qwen 2.5 14B (Fast)</option>
                                                <option value="mistral:7b">Mistral 7B (Fastest)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleSaveAI}
                                        disabled={saveMutation.isPending}
                                        className="btn-primary flex items-center gap-2"
                                    >
                                        <Save size={16} />
                                        Save AI Settings
                                    </button>
                                </div>

                                {/* Agent Status */}
                                <div className="glass-card p-6">
                                    <div className="flex items-center justify-between mb-5">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center">
                                                <Zap className="w-5 h-5 text-emerald-400" />
                                            </div>
                                            <div>
                                                <h4 className="font-semibold text-white">Active Agents</h4>
                                                <p className="text-sm text-slate-400">35 specialized AI agents ready</p>
                                            </div>
                                        </div>
                                        <span className="badge-success">All Online</span>
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        {['Portfolio', 'Options', 'Research', 'NFL', 'NBA', 'RAG', 'Technical', 'Fundamental'].map(agent => (
                                            <div key={agent} className="flex items-center gap-2 px-4 py-3 bg-slate-800/30 rounded-xl border border-slate-700/50">
                                                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                                                <span className="text-sm text-white">{agent}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Notifications Tab */}
                        {activeTab === 'notifications' && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center">
                                        <Bell className="w-5 h-5 text-amber-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">Notification Settings</h3>
                                        <p className="text-sm text-slate-400">Manage your alert preferences</p>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    {[
                                        { label: 'Price Alerts', desc: 'Get notified when stocks hit target prices', icon: '📊' },
                                        { label: 'Options Expiration', desc: 'Alerts for upcoming expirations', icon: '⏰' },
                                        { label: 'Sports Predictions', desc: 'Game predictions and best bets', icon: '🏈' },
                                        { label: 'System Alerts', desc: 'Important system notifications', icon: '🔔' }
                                    ].map(item => (
                                        <div key={item.label} className="glass-card flex items-center justify-between p-5">
                                            <div className="flex items-center gap-4">
                                                <span className="text-2xl">{item.icon}</span>
                                                <div>
                                                    <h4 className="font-medium text-white">{item.label}</h4>
                                                    <p className="text-sm text-slate-400">{item.desc}</p>
                                                </div>
                                            </div>
                                            <label className="relative inline-flex items-center cursor-pointer">
                                                <input type="checkbox" className="sr-only peer" defaultChecked />
                                                <div className="w-12 h-6 bg-slate-700 peer-focus:ring-2 peer-focus:ring-primary/50 rounded-full peer peer-checked:after:translate-x-6 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                                            </label>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Appearance Tab */}
                        {activeTab === 'appearance' && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-pink-500/20 flex items-center justify-center">
                                        <Palette className="w-5 h-5 text-pink-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">Appearance</h3>
                                        <p className="text-sm text-slate-400">Customize the look and feel</p>
                                    </div>
                                </div>

                                <div className="glass-card p-6">
                                    <h4 className="font-semibold text-white mb-4">Theme</h4>
                                    <div className="grid grid-cols-3 gap-4">
                                        <button className="p-5 rounded-xl border-2 border-primary bg-slate-900 text-center group transition-all">
                                            <div className="w-10 h-10 rounded-xl bg-slate-800 mx-auto mb-3 border border-slate-700"></div>
                                            <span className="text-sm font-medium text-white">Dark</span>
                                            <p className="text-xs text-primary mt-1">Active</p>
                                        </button>
                                        <button className="p-5 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center opacity-50 cursor-not-allowed">
                                            <div className="w-10 h-10 rounded-xl bg-white mx-auto mb-3"></div>
                                            <span className="text-sm font-medium text-slate-400">Light</span>
                                            <p className="text-xs text-slate-500 mt-1">Coming Soon</p>
                                        </button>
                                        <button className="p-5 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center opacity-50 cursor-not-allowed">
                                            <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-slate-800 to-white mx-auto mb-3"></div>
                                            <span className="text-sm font-medium text-slate-400">System</span>
                                            <p className="text-xs text-slate-500 mt-1">Coming Soon</p>
                                        </button>
                                    </div>
                                </div>

                                <div className="glass-card p-6">
                                    <h4 className="font-semibold text-white mb-4">Accent Color</h4>
                                    <div className="flex gap-3">
                                        {[
                                            { color: 'bg-blue-500', name: 'Blue', active: true },
                                            { color: 'bg-violet-500', name: 'Violet', active: false },
                                            { color: 'bg-emerald-500', name: 'Emerald', active: false },
                                            { color: 'bg-amber-500', name: 'Amber', active: false },
                                            { color: 'bg-pink-500', name: 'Pink', active: false },
                                        ].map((c) => (
                                            <button
                                                key={c.name}
                                                className={clsx(
                                                    "w-10 h-10 rounded-xl transition-all",
                                                    c.color,
                                                    c.active ? "ring-2 ring-white ring-offset-2 ring-offset-slate-900" : "hover:scale-110"
                                                )}
                                                title={c.name}
                                            ></button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* System Tab */}
                        {activeTab === 'system' && (
                            <div className="space-y-6 animate-fade-in">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center">
                                        <Shield className="w-5 h-5 text-cyan-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-white">System Information</h3>
                                        <p className="text-sm text-slate-400">Platform status and maintenance</p>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="glass-card p-5">
                                        <div className="text-sm text-slate-400 mb-1">Version</div>
                                        <div className="text-xl font-bold text-white">AVA v3.0.0</div>
                                    </div>
                                    <div className="glass-card p-5">
                                        <div className="text-sm text-slate-400 mb-1">Backend Status</div>
                                        <div className="text-xl font-bold text-emerald-400 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                                            Online
                                        </div>
                                    </div>
                                    <div className="glass-card p-5">
                                        <div className="text-sm text-slate-400 mb-1">Database</div>
                                        <div className="text-xl font-bold text-white">PostgreSQL 15</div>
                                    </div>
                                    <div className="glass-card p-5">
                                        <div className="text-sm text-slate-400 mb-1">AI Engine</div>
                                        <div className="text-xl font-bold text-white">Ollama + Qwen</div>
                                    </div>
                                </div>

                                <div className="glass-card p-6">
                                    <h4 className="font-semibold text-white mb-4">Cache Management</h4>
                                    <div className="flex gap-4">
                                        <button className="btn-secondary flex items-center gap-2">
                                            <RefreshCw size={16} />
                                            Clear Query Cache
                                        </button>
                                        <button className="btn-secondary flex items-center gap-2">
                                            <Database size={16} />
                                            Refresh Data
                                        </button>
                                    </div>
                                </div>

                                <div className="glass-card p-6 border border-red-500/20">
                                    <h4 className="font-semibold text-red-400 mb-4">Danger Zone</h4>
                                    <p className="text-sm text-slate-400 mb-4">These actions are irreversible. Please proceed with caution.</p>
                                    <button className="px-4 py-2 bg-red-500/10 text-red-400 rounded-lg border border-red-500/30 hover:bg-red-500/20 transition-colors">
                                        Reset All Settings
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
