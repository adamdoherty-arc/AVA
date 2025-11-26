import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    MessageCircle, RefreshCw, Search, Filter, Hash,
    User, Clock, ChevronDown, ExternalLink, Copy
} from 'lucide-react'

interface DiscordMessage {
    id: string
    channel: string
    author: string
    content: string
    timestamp: string
    attachments: string[]
    mentions: string[]
    reactions: { emoji: string; count: number }[]
}

interface Channel {
    id: string
    name: string
    type: string
    message_count: number
}

export default function DiscordMessages() {
    const [selectedChannel, setSelectedChannel] = useState<string>('all')
    const [searchQuery, setSearchQuery] = useState('')
    const [limit, setLimit] = useState(50)

    const { data: channels } = useQuery<Channel[]>({
        queryKey: ['discord-channels'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/discord/channels')
            return data?.channels || []
        }
    })

    const { data: messages, isLoading, refetch } = useQuery<DiscordMessage[]>({
        queryKey: ['discord-messages', selectedChannel, searchQuery, limit],
        queryFn: async () => {
            const params = new URLSearchParams()
            if (selectedChannel !== 'all') params.append('channel', selectedChannel)
            if (searchQuery) params.append('search', searchQuery)
            params.append('limit', limit.toString())
            const { data } = await axiosInstance.get(`/discord/messages?${params}`)
            return data?.messages || []
        },
        staleTime: 30000
    })

    const copyMessage = (content: string) => {
        navigator.clipboard.writeText(content)
    }

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp)
        const now = new Date()
        const diff = now.getTime() - date.getTime()
        const hours = diff / (1000 * 60 * 60)

        if (hours < 1) return `${Math.floor(diff / (1000 * 60))}m ago`
        if (hours < 24) return `${Math.floor(hours)}h ago`
        return date.toLocaleDateString()
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <MessageCircle className="w-5 h-5 text-white" />
                        </div>
                        Discord Messages
                    </h1>
                    <p className="page-subtitle">XTrades Discord message history and alerts</p>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Filters */}
            <div className="glass-card p-5">
                <div className="flex flex-wrap items-center gap-4">
                    {/* Search */}
                    <div className="relative flex-1 min-w-[200px]">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search messages..."
                            className="input-field pl-10"
                        />
                    </div>

                    {/* Channel Filter */}
                    <div className="relative">
                        <Hash className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <select
                            value={selectedChannel}
                            onChange={(e) => setSelectedChannel(e.target.value)}
                            className="input-field pl-10 pr-8 appearance-none"
                        >
                            <option value="all">All Channels</option>
                            {channels?.map(channel => (
                                <option key={channel.id} value={channel.id}>
                                    #{channel.name} ({channel.message_count})
                                </option>
                            ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                    </div>

                    {/* Limit */}
                    <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-400">Show:</span>
                        {[25, 50, 100, 200].map(n => (
                            <button
                                key={n}
                                onClick={() => setLimit(n)}
                                className={`px-3 py-1.5 text-sm rounded-lg ${
                                    limit === n
                                        ? 'bg-primary text-white'
                                        : 'bg-slate-800/60 text-slate-400 hover:text-white'
                                }`}
                            >
                                {n}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Channel Stats */}
            {channels && channels.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    {channels.slice(0, 6).map(channel => (
                        <button
                            key={channel.id}
                            onClick={() => setSelectedChannel(channel.id)}
                            className={`glass-card p-4 text-left hover:border-primary/50 transition-colors ${
                                selectedChannel === channel.id ? 'border-primary' : ''
                            }`}
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <Hash className="w-4 h-4 text-slate-400" />
                                <span className="font-medium text-white truncate">{channel.name}</span>
                            </div>
                            <div className="text-sm text-slate-400">{channel.message_count} messages</div>
                        </button>
                    ))}
                </div>
            )}

            {/* Messages */}
            <div className="glass-card overflow-hidden">
                <div className="p-5 border-b border-slate-700/50 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">
                        {messages?.length || 0} Messages
                    </h3>
                    <span className="text-sm text-slate-400">
                        {selectedChannel === 'all' ? 'All channels' : `#${channels?.find(c => c.id === selectedChannel)?.name || selectedChannel}`}
                    </span>
                </div>

                <div className="divide-y divide-slate-700/50 max-h-[600px] overflow-y-auto">
                    {messages && messages.length > 0 ? (
                        messages.map((message) => (
                            <div key={message.id} className="p-4 hover:bg-slate-800/30 transition-colors">
                                <div className="flex items-start gap-4">
                                    <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                                        <User className="w-5 h-5 text-primary" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-3 mb-1">
                                            <span className="font-semibold text-white">{message.author}</span>
                                            <span className="text-xs text-slate-500 flex items-center gap-1">
                                                <Hash className="w-3 h-3" />
                                                {message.channel}
                                            </span>
                                            <span className="text-xs text-slate-500 flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                {formatTime(message.timestamp)}
                                            </span>
                                        </div>
                                        <p className="text-slate-300 whitespace-pre-wrap break-words">
                                            {message.content}
                                        </p>

                                        {/* Mentions */}
                                        {message.mentions.length > 0 && (
                                            <div className="flex flex-wrap gap-2 mt-2">
                                                {message.mentions.map((mention, idx) => (
                                                    <span key={idx} className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">
                                                        @{mention}
                                                    </span>
                                                ))}
                                            </div>
                                        )}

                                        {/* Attachments */}
                                        {message.attachments.length > 0 && (
                                            <div className="flex flex-wrap gap-2 mt-2">
                                                {message.attachments.map((attachment, idx) => (
                                                    <a
                                                        key={idx}
                                                        href={attachment}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex items-center gap-1 text-xs text-primary hover:underline"
                                                    >
                                                        <ExternalLink className="w-3 h-3" />
                                                        Attachment {idx + 1}
                                                    </a>
                                                ))}
                                            </div>
                                        )}

                                        {/* Reactions */}
                                        {message.reactions.length > 0 && (
                                            <div className="flex flex-wrap gap-2 mt-2">
                                                {message.reactions.map((reaction, idx) => (
                                                    <span key={idx} className="px-2 py-0.5 bg-slate-700/50 rounded text-sm">
                                                        {reaction.emoji} {reaction.count}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {/* Actions */}
                                    <button
                                        onClick={() => copyMessage(message.content)}
                                        className="p-2 text-slate-500 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors"
                                        title="Copy message"
                                    >
                                        <Copy className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            {isLoading ? 'Loading messages...' : 'No messages found'}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
