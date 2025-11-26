import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Brain, RefreshCw, Search, Upload, Trash2, FileText,
    Database, Clock, CheckCircle, AlertCircle, Loader2,
    MessageSquare, Zap, BookOpen
} from 'lucide-react'

interface KnowledgeDocument {
    id: string
    title: string
    source: string
    type: 'pdf' | 'txt' | 'md' | 'url'
    chunks: number
    tokens: number
    created_at: string
    last_accessed: string
}

interface RAGStats {
    total_documents: number
    total_chunks: number
    total_tokens: number
    embedding_model: string
    vector_db: string
    last_indexed: string
}

interface QueryResult {
    answer: string
    sources: { document: string; chunk: string; score: number }[]
    tokens_used: number
    latency_ms: number
}

export default function RAGKnowledgeBase() {
    const queryClient = useQueryClient()
    const [searchQuery, setSearchQuery] = useState('')
    const [queryResult, setQueryResult] = useState<QueryResult | null>(null)

    const { data: stats, isLoading: statsLoading } = useQuery<RAGStats>({
        queryKey: ['rag-stats'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/knowledge/stats')
            return data
        }
    })

    const { data: documents, isLoading: docsLoading, refetch } = useQuery<KnowledgeDocument[]>({
        queryKey: ['knowledge-documents'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/knowledge/documents')
            return data?.documents || []
        }
    })

    const queryMutation = useMutation({
        mutationFn: async (query: string) => {
            const { data } = await axiosInstance.post('/knowledge/query', { query })
            return data
        },
        onSuccess: (data) => {
            setQueryResult(data)
        }
    })

    const deleteDocMutation = useMutation({
        mutationFn: async (docId: string) => {
            const { data } = await axiosInstance.delete(`/knowledge/documents/${docId}`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-documents'] })
            queryClient.invalidateQueries({ queryKey: ['rag-stats'] })
        }
    })

    const reindexMutation = useMutation({
        mutationFn: async () => {
            const { data } = await axiosInstance.post('/knowledge/reindex')
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['rag-stats'] })
        }
    })

    const handleQuery = () => {
        if (searchQuery.trim()) {
            queryMutation.mutate(searchQuery)
        }
    }

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'pdf': return '📄'
            case 'txt': return '📝'
            case 'md': return '📋'
            case 'url': return '🌐'
            default: return '📁'
        }
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center shadow-lg">
                            <Brain className="w-5 h-5 text-white" />
                        </div>
                        RAG Knowledge Base
                    </h1>
                    <p className="page-subtitle">Manage documents and query your knowledge base</p>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => reindexMutation.mutate()}
                        disabled={reindexMutation.isPending}
                        className="btn-secondary flex items-center gap-2"
                    >
                        <Zap className={`w-4 h-4 ${reindexMutation.isPending ? 'animate-pulse' : ''}`} />
                        Reindex
                    </button>
                    <button onClick={() => refetch()} className="btn-icon">
                        <RefreshCw className={`w-5 h-5 ${docsLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </header>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <FileText className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Documents</span>
                    </div>
                    <div className="text-2xl font-bold text-white">{stats?.total_documents ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Database className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Chunks</span>
                    </div>
                    <div className="text-2xl font-bold text-white">{stats?.total_chunks?.toLocaleString() ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Zap className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Tokens</span>
                    </div>
                    <div className="text-2xl font-bold text-white">{stats?.total_tokens?.toLocaleString() ?? 0}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Embedding Model</span>
                    </div>
                    <div className="text-lg font-bold text-purple-400 truncate">{stats?.embedding_model ?? 'N/A'}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Database className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Vector DB</span>
                    </div>
                    <div className="text-lg font-bold text-blue-400">{stats?.vector_db ?? 'N/A'}</div>
                </div>
                <div className="glass-card p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <Clock className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-400">Last Indexed</span>
                    </div>
                    <div className="text-sm font-medium text-white">
                        {stats?.last_indexed ? new Date(stats.last_indexed).toLocaleString() : 'Never'}
                    </div>
                </div>
            </div>

            {/* Query Interface */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <MessageSquare className="w-5 h-5 text-purple-400" />
                    Query Knowledge Base
                </h3>
                <div className="flex items-center gap-4 mb-4">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                            placeholder="Ask a question about your documents..."
                            className="input-field pl-10"
                        />
                    </div>
                    <button
                        onClick={handleQuery}
                        disabled={queryMutation.isPending}
                        className="btn-primary px-6 flex items-center gap-2"
                    >
                        {queryMutation.isPending ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Brain className="w-4 h-4" />
                        )}
                        Query
                    </button>
                </div>

                {queryResult && (
                    <div className="space-y-4">
                        <div className="bg-slate-800/40 rounded-xl p-4">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-slate-400">Answer</span>
                                <span className="text-xs text-slate-500">
                                    {queryResult.tokens_used} tokens • {queryResult.latency_ms}ms
                                </span>
                            </div>
                            <p className="text-white whitespace-pre-wrap">{queryResult.answer}</p>
                        </div>

                        {queryResult.sources.length > 0 && (
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-2">Sources</h4>
                                <div className="space-y-2">
                                    {queryResult.sources.map((source, idx) => (
                                        <div key={idx} className="bg-slate-800/40 rounded-lg p-3">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-sm font-medium text-white">{source.document}</span>
                                                <span className="text-xs text-emerald-400">
                                                    {(source.score * 100).toFixed(0)}% match
                                                </span>
                                            </div>
                                            <p className="text-sm text-slate-400 line-clamp-2">{source.chunk}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Documents List */}
            <div className="glass-card overflow-hidden">
                <div className="p-5 border-b border-slate-700/50 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <BookOpen className="w-5 h-5 text-blue-400" />
                        Documents ({documents?.length ?? 0})
                    </h3>
                    <button className="btn-primary flex items-center gap-2">
                        <Upload className="w-4 h-4" />
                        Upload Document
                    </button>
                </div>

                <div className="divide-y divide-slate-700/50 max-h-[500px] overflow-y-auto">
                    {documents && documents.length > 0 ? (
                        documents.map((doc) => (
                            <div key={doc.id} className="p-4 hover:bg-slate-800/30 transition-colors">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <span className="text-2xl">{getTypeIcon(doc.type)}</span>
                                        <div>
                                            <div className="font-medium text-white">{doc.title}</div>
                                            <div className="text-sm text-slate-400">{doc.source}</div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        <div className="text-right">
                                            <div className="text-sm text-slate-400">{doc.chunks} chunks</div>
                                            <div className="text-xs text-slate-500">{doc.tokens.toLocaleString()} tokens</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm text-slate-400">Created</div>
                                            <div className="text-xs text-slate-500">
                                                {new Date(doc.created_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => deleteDocMutation.mutate(doc.id)}
                                            disabled={deleteDocMutation.isPending}
                                            className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            {docsLoading ? (
                                <div className="flex items-center justify-center gap-2">
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Loading documents...
                                </div>
                            ) : (
                                <>
                                    <BookOpen className="w-12 h-12 mx-auto mb-4 text-slate-600" />
                                    <p>No documents in knowledge base</p>
                                    <p className="text-sm mt-1">Upload documents to get started</p>
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Tips */}
            <div className="glass-card p-5">
                <h3 className="text-lg font-semibold text-white mb-4">Quick Tips</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                            <span className="font-medium text-white">Supported Formats</span>
                        </div>
                        <p className="text-sm text-slate-400">PDF, TXT, MD, and web URLs can be indexed</p>
                    </div>
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Zap className="w-4 h-4 text-amber-400" />
                            <span className="font-medium text-white">Chunking</span>
                        </div>
                        <p className="text-sm text-slate-400">Documents are split into ~500 token chunks for better retrieval</p>
                    </div>
                    <div className="bg-slate-800/40 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Brain className="w-4 h-4 text-purple-400" />
                            <span className="font-medium text-white">Embeddings</span>
                        </div>
                        <p className="text-sm text-slate-400">Uses semantic search for relevant document retrieval</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
