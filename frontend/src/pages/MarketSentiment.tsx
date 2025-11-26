import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { axiosInstance } from '../lib/axios'
import {
    Brain, RefreshCw, TrendingUp, TrendingDown, Activity,
    MessageSquare, Newspaper, Twitter, BarChart3, Gauge
} from 'lucide-react'
import clsx from 'clsx'

interface SentimentData {
    overall_score: number
    fear_greed_index: number
    social_sentiment: number
    news_sentiment: number
    options_sentiment: number
    put_call_ratio: number
    vix: number
    trending_topics: { topic: string; sentiment: number; mentions: number }[]
    news_headlines: { title: string; source: string; sentiment: 'positive' | 'negative' | 'neutral' }[]
}

export default function MarketSentiment() {
    const { data, isLoading, refetch } = useQuery<SentimentData>({
        queryKey: ['market-sentiment'],
        queryFn: async () => {
            const { data } = await axiosInstance.get('/research/sentiment')
            return data
        },
        staleTime: 60000,
    })

    const getSentimentLabel = (score: number) => {
        if (score >= 80) return { label: 'Extreme Greed', color: 'text-emerald-400' }
        if (score >= 60) return { label: 'Greed', color: 'text-emerald-400' }
        if (score >= 40) return { label: 'Neutral', color: 'text-amber-400' }
        if (score >= 20) return { label: 'Fear', color: 'text-rose-400' }
        return { label: 'Extreme Fear', color: 'text-rose-400' }
    }

    const sentiment = data ? getSentimentLabel(data.fear_greed_index) : { label: 'Loading', color: 'text-slate-400' }

    return (
        <div className="space-y-6">
            {/* Header */}
            <header className="flex items-center justify-between">
                <div>
                    <h1 className="page-title flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center shadow-lg">
                            <Brain className="w-5 h-5 text-white" />
                        </div>
                        Market Sentiment
                    </h1>
                    <p className="page-subtitle">AI-powered sentiment analysis from news, social, and options flow</p>
                </div>
                <button onClick={() => refetch()} disabled={isLoading} className="btn-icon">
                    <RefreshCw className={clsx("w-5 h-5", isLoading && "animate-spin")} />
                </button>
            </header>

            {isLoading ? (
                <div className="card p-8 flex items-center justify-center">
                    <RefreshCw className="w-6 h-6 text-primary animate-spin" />
                    <span className="ml-2 text-slate-400">Analyzing sentiment...</span>
                </div>
            ) : data ? (
                <>
                    {/* Fear & Greed Gauge */}
                    <div className="card p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <Gauge className="w-5 h-5 text-primary" />
                                Fear & Greed Index
                            </h3>
                            <span className={clsx("text-2xl font-bold", sentiment.color)}>
                                {sentiment.label}
                            </span>
                        </div>

                        <div className="relative h-8 bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 rounded-full overflow-hidden">
                            <div
                                className="absolute top-0 w-1 h-full bg-white shadow-lg transition-all duration-500"
                                style={{ left: `${data.fear_greed_index}%` }}
                            />
                        </div>
                        <div className="flex justify-between text-xs text-slate-400 mt-2">
                            <span>Extreme Fear</span>
                            <span>Neutral</span>
                            <span>Extreme Greed</span>
                        </div>

                        <div className="text-center mt-4">
                            <p className="text-6xl font-bold text-primary">{data.fear_greed_index}</p>
                            <p className="text-slate-400">out of 100</p>
                        </div>
                    </div>

                    {/* Sentiment Breakdown */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <MessageSquare className="w-4 h-4" />
                                <span className="text-sm">Social</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                data.social_sentiment >= 50 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                {data.social_sentiment}%
                            </p>
                            <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={clsx(
                                        "h-full rounded-full",
                                        data.social_sentiment >= 50 ? "bg-emerald-500" : "bg-rose-500"
                                    )}
                                    style={{ width: `${data.social_sentiment}%` }}
                                />
                            </div>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <Newspaper className="w-4 h-4" />
                                <span className="text-sm">News</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                data.news_sentiment >= 50 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                {data.news_sentiment}%
                            </p>
                            <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={clsx(
                                        "h-full rounded-full",
                                        data.news_sentiment >= 50 ? "bg-emerald-500" : "bg-rose-500"
                                    )}
                                    style={{ width: `${data.news_sentiment}%` }}
                                />
                            </div>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <Activity className="w-4 h-4" />
                                <span className="text-sm">Options Flow</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                data.options_sentiment >= 50 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                {data.options_sentiment}%
                            </p>
                            <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={clsx(
                                        "h-full rounded-full",
                                        data.options_sentiment >= 50 ? "bg-emerald-500" : "bg-rose-500"
                                    )}
                                    style={{ width: `${data.options_sentiment}%` }}
                                />
                            </div>
                        </div>
                        <div className="card p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-2">
                                <BarChart3 className="w-4 h-4" />
                                <span className="text-sm">Put/Call Ratio</span>
                            </div>
                            <p className={clsx(
                                "text-2xl font-bold",
                                data.put_call_ratio < 1 ? "text-emerald-400" : "text-rose-400"
                            )}>
                                {data.put_call_ratio.toFixed(2)}
                            </p>
                            <p className="text-xs text-slate-500 mt-1">
                                VIX: {data.vix.toFixed(1)}
                            </p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Trending Topics */}
                        <div className="card p-4">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Twitter className="w-5 h-5 text-blue-400" />
                                Trending Topics
                            </h3>
                            <div className="space-y-3">
                                {data.trending_topics.map((topic, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
                                        <div>
                                            <p className="font-medium">{topic.topic}</p>
                                            <p className="text-xs text-slate-400">{topic.mentions.toLocaleString()} mentions</p>
                                        </div>
                                        <div className={clsx(
                                            "px-2 py-1 rounded text-sm font-bold",
                                            topic.sentiment >= 60 ? "bg-emerald-500/20 text-emerald-400" :
                                            topic.sentiment >= 40 ? "bg-amber-500/20 text-amber-400" :
                                            "bg-rose-500/20 text-rose-400"
                                        )}>
                                            {topic.sentiment}%
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Recent Headlines */}
                        <div className="card p-4">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Newspaper className="w-5 h-5 text-amber-400" />
                                Recent Headlines
                            </h3>
                            <div className="space-y-3">
                                {data.news_headlines.map((news, idx) => (
                                    <div key={idx} className="p-2 bg-slate-800/50 rounded-lg">
                                        <div className="flex items-start gap-2">
                                            <div className={clsx(
                                                "w-2 h-2 rounded-full mt-2 flex-shrink-0",
                                                news.sentiment === 'positive' ? "bg-emerald-500" :
                                                news.sentiment === 'negative' ? "bg-rose-500" :
                                                "bg-slate-500"
                                            )} />
                                            <div>
                                                <p className="text-sm">{news.title}</p>
                                                <p className="text-xs text-slate-400 mt-1">{news.source}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </>
            ) : null}
        </div>
    )
}
