import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    TestTube, RefreshCw, CheckCircle, XCircle, AlertTriangle,
    Play, Clock, FileText, Bug, ThumbsUp, ThumbsDown, ArrowLeft,
    MessageSquare, Eye, Code
} from 'lucide-react'
import { Link } from 'react-router-dom'

interface QAItem {
    id: string
    enhancement_id: string
    enhancement_title: string
    status: 'pending_review' | 'testing' | 'approved' | 'rejected' | 'needs_changes'
    submitted_at: string
    reviewed_at?: string
    reviewer?: string
    test_results: {
        unit_tests: { passed: number; failed: number; skipped: number }
        integration_tests: { passed: number; failed: number; skipped: number }
        coverage: number
    }
    checklist: {
        item: string
        checked: boolean
    }[]
    comments: {
        author: string
        content: string
        timestamp: string
    }[]
    files_changed: string[]
}

export default function EnhancementQA() {
    const queryClient = useQueryClient()
    const [selectedItem, setSelectedItem] = useState<string | null>(null)
    const [newComment, setNewComment] = useState('')

    const { data: qaItems, isLoading, refetch } = useQuery<QAItem[]>({
        queryKey: ['qa-items'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/enhancements/qa')
            return data?.items || []
        }
    })

    const updateStatusMutation = useMutation({
        mutationFn: async ({ id, status }: { id: string; status: string }) => {
            const { data } = await axiosInstance.patch(`/enhancements/qa/${id}`, { status })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['qa-items'] })
        }
    })

    const addCommentMutation = useMutation({
        mutationFn: async ({ id, content }: { id: string; content: string }) => {
            const { data } = await axiosInstance.post(`/enhancements/qa/${id}/comments`, { content })
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['qa-items'] })
            setNewComment('')
        }
    })

    const runTestsMutation = useMutation({
        mutationFn: async (id: string) => {
            const { data } = await axiosInstance.post(`/enhancements/qa/${id}/run-tests`)
            return data
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['qa-items'] })
        }
    })

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'approved': return 'text-emerald-400 bg-emerald-500/20'
            case 'rejected': return 'text-red-400 bg-red-500/20'
            case 'testing': return 'text-blue-400 bg-blue-500/20'
            case 'pending_review': return 'text-amber-400 bg-amber-500/20'
            case 'needs_changes': return 'text-purple-400 bg-purple-500/20'
            default: return 'text-slate-400 bg-slate-500/20'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'approved': return <CheckCircle className="w-4 h-4" />
            case 'rejected': return <XCircle className="w-4 h-4" />
            case 'testing': return <TestTube className="w-4 h-4" />
            case 'pending_review': return <Clock className="w-4 h-4" />
            case 'needs_changes': return <AlertTriangle className="w-4 h-4" />
            default: return <Clock className="w-4 h-4" />
        }
    }

    const selectedQAItem = qaItems?.find(item => item.id === selectedItem)

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link to="/enhancements" className="btn-icon">
                        <ArrowLeft className="w-5 h-5" />
                    </Link>
                    <div>
                        <h1 className="page-title flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg">
                                <TestTube className="w-5 h-5 text-white" />
                            </div>
                            Enhancement QA
                        </h1>
                        <p className="page-subtitle">Review, test, and approve enhancements</p>
                    </div>
                </div>
                <button onClick={() => refetch()} className="btn-icon">
                    <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </header>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Pending Review</div>
                    <div className="text-2xl font-bold text-amber-400">
                        {qaItems?.filter(i => i.status === 'pending_review').length ?? 0}
                    </div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Testing</div>
                    <div className="text-2xl font-bold text-blue-400">
                        {qaItems?.filter(i => i.status === 'testing').length ?? 0}
                    </div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Approved</div>
                    <div className="text-2xl font-bold text-emerald-400">
                        {qaItems?.filter(i => i.status === 'approved').length ?? 0}
                    </div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Needs Changes</div>
                    <div className="text-2xl font-bold text-purple-400">
                        {qaItems?.filter(i => i.status === 'needs_changes').length ?? 0}
                    </div>
                </div>
                <div className="glass-card p-4">
                    <div className="text-sm text-slate-400 mb-1">Rejected</div>
                    <div className="text-2xl font-bold text-red-400">
                        {qaItems?.filter(i => i.status === 'rejected').length ?? 0}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* QA Items List */}
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50">
                        <h3 className="text-lg font-semibold text-white">QA Queue</h3>
                    </div>
                    <div className="divide-y divide-slate-700/50 max-h-[600px] overflow-y-auto">
                        {qaItems && qaItems.length > 0 ? (
                            qaItems.map((item) => (
                                <div
                                    key={item.id}
                                    className={`p-4 hover:bg-slate-800/30 transition-colors cursor-pointer ${
                                        selectedItem === item.id ? 'bg-slate-800/50' : ''
                                    }`}
                                    onClick={() => setSelectedItem(item.id)}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <h4 className="font-medium text-white">{item.enhancement_title}</h4>
                                        <span className={`px-2 py-0.5 rounded-lg text-xs font-medium flex items-center gap-1 ${getStatusColor(item.status)}`}>
                                            {getStatusIcon(item.status)}
                                            {item.status.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-4 text-xs text-slate-500">
                                        <span className="flex items-center gap-1">
                                            <Code className="w-3 h-3" />
                                            {item.files_changed.length} files
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <MessageSquare className="w-3 h-3" />
                                            {item.comments.length} comments
                                        </span>
                                        <span>{new Date(item.submitted_at).toLocaleDateString()}</span>
                                    </div>
                                    {/* Test Results Summary */}
                                    <div className="flex items-center gap-3 mt-2">
                                        <span className="text-xs text-emerald-400">
                                            ✓ {item.test_results.unit_tests.passed + item.test_results.integration_tests.passed}
                                        </span>
                                        <span className="text-xs text-red-400">
                                            ✗ {item.test_results.unit_tests.failed + item.test_results.integration_tests.failed}
                                        </span>
                                        <span className="text-xs text-slate-400">
                                            {item.test_results.coverage}% coverage
                                        </span>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="p-12 text-center text-slate-400">
                                {isLoading ? 'Loading QA items...' : 'No items in QA queue'}
                            </div>
                        )}
                    </div>
                </div>

                {/* QA Item Details */}
                <div className="glass-card overflow-hidden">
                    <div className="p-5 border-b border-slate-700/50">
                        <h3 className="text-lg font-semibold text-white">Review Details</h3>
                    </div>
                    {selectedQAItem ? (
                        <div className="p-5 space-y-4 max-h-[600px] overflow-y-auto">
                            {/* Test Results */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Test Results</h4>
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="bg-slate-800/40 rounded-lg p-3">
                                        <div className="text-slate-400 text-xs mb-1">Unit Tests</div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-emerald-400">{selectedQAItem.test_results.unit_tests.passed}✓</span>
                                            <span className="text-red-400">{selectedQAItem.test_results.unit_tests.failed}✗</span>
                                            <span className="text-slate-500">{selectedQAItem.test_results.unit_tests.skipped}⊘</span>
                                        </div>
                                    </div>
                                    <div className="bg-slate-800/40 rounded-lg p-3">
                                        <div className="text-slate-400 text-xs mb-1">Integration Tests</div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-emerald-400">{selectedQAItem.test_results.integration_tests.passed}✓</span>
                                            <span className="text-red-400">{selectedQAItem.test_results.integration_tests.failed}✗</span>
                                            <span className="text-slate-500">{selectedQAItem.test_results.integration_tests.skipped}⊘</span>
                                        </div>
                                    </div>
                                    <div className="bg-slate-800/40 rounded-lg p-3">
                                        <div className="text-slate-400 text-xs mb-1">Coverage</div>
                                        <div className={`text-lg font-bold ${
                                            selectedQAItem.test_results.coverage >= 80 ? 'text-emerald-400' :
                                            selectedQAItem.test_results.coverage >= 60 ? 'text-amber-400' : 'text-red-400'
                                        }`}>
                                            {selectedQAItem.test_results.coverage}%
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={() => runTestsMutation.mutate(selectedQAItem.id)}
                                    disabled={runTestsMutation.isPending}
                                    className="btn-secondary w-full mt-3 flex items-center justify-center gap-2"
                                >
                                    <Play className="w-4 h-4" />
                                    {runTestsMutation.isPending ? 'Running Tests...' : 'Run Tests'}
                                </button>
                            </div>

                            {/* Checklist */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Review Checklist</h4>
                                <div className="space-y-2">
                                    {selectedQAItem.checklist.map((item, idx) => (
                                        <label key={idx} className="flex items-center gap-3 p-2 bg-slate-800/40 rounded-lg cursor-pointer hover:bg-slate-800/60">
                                            <input
                                                type="checkbox"
                                                checked={item.checked}
                                                readOnly
                                                className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-primary"
                                            />
                                            <span className={`text-sm ${item.checked ? 'text-white' : 'text-slate-400'}`}>
                                                {item.item}
                                            </span>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            {/* Files Changed */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Files Changed ({selectedQAItem.files_changed.length})</h4>
                                <div className="space-y-1 max-h-32 overflow-y-auto">
                                    {selectedQAItem.files_changed.map((file, idx) => (
                                        <div key={idx} className="flex items-center gap-2 text-xs text-slate-300 p-2 bg-slate-800/40 rounded">
                                            <Code className="w-3 h-3" />
                                            {file}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Comments */}
                            <div>
                                <h4 className="text-sm font-medium text-slate-400 mb-3">Comments</h4>
                                <div className="space-y-3 max-h-40 overflow-y-auto mb-3">
                                    {selectedQAItem.comments.map((comment, idx) => (
                                        <div key={idx} className="bg-slate-800/40 rounded-lg p-3">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-sm font-medium text-white">{comment.author}</span>
                                                <span className="text-xs text-slate-500">{new Date(comment.timestamp).toLocaleString()}</span>
                                            </div>
                                            <p className="text-sm text-slate-400">{comment.content}</p>
                                        </div>
                                    ))}
                                </div>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={newComment}
                                        onChange={(e) => setNewComment(e.target.value)}
                                        placeholder="Add a comment..."
                                        className="input-field flex-1"
                                    />
                                    <button
                                        onClick={() => addCommentMutation.mutate({ id: selectedQAItem.id, content: newComment })}
                                        disabled={!newComment.trim() || addCommentMutation.isPending}
                                        className="btn-primary"
                                    >
                                        Send
                                    </button>
                                </div>
                            </div>

                            {/* Actions */}
                            <div className="flex gap-2 pt-4 border-t border-slate-700/50">
                                <button
                                    onClick={() => updateStatusMutation.mutate({ id: selectedQAItem.id, status: 'approved' })}
                                    className="flex-1 btn-primary bg-emerald-600 hover:bg-emerald-700 flex items-center justify-center gap-2"
                                >
                                    <ThumbsUp className="w-4 h-4" />
                                    Approve
                                </button>
                                <button
                                    onClick={() => updateStatusMutation.mutate({ id: selectedQAItem.id, status: 'needs_changes' })}
                                    className="flex-1 btn-secondary flex items-center justify-center gap-2"
                                >
                                    <Bug className="w-4 h-4" />
                                    Request Changes
                                </button>
                                <button
                                    onClick={() => updateStatusMutation.mutate({ id: selectedQAItem.id, status: 'rejected' })}
                                    className="flex-1 btn-secondary text-red-400 hover:text-red-300 flex items-center justify-center gap-2"
                                >
                                    <ThumbsDown className="w-4 h-4" />
                                    Reject
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="p-12 text-center text-slate-400">
                            <TestTube className="w-12 h-12 mx-auto mb-4 text-slate-600" />
                            <p>Select an item to review</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
