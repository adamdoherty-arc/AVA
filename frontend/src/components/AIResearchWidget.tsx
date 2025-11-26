import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

interface AIResearchWidgetProps {
    symbol: string;
    data: any; // Replace with proper type from API
    isLoading: boolean;
    onRefresh: () => void;
}

export function AIResearchWidget({ symbol, data, isLoading, onRefresh }: AIResearchWidgetProps) {
    if (isLoading) {
        return (
            <Card className="w-full h-full flex items-center justify-center min-h-[300px]">
                <Loader2 className="h-8 w-8 animate-spin" />
            </Card>
        );
    }

    if (!data) {
        return (
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>AI Research: {symbol}</CardTitle>
                </CardHeader>
                <CardContent>
                    <p>No data available. Click refresh to analyze.</p>
                    <button onClick={onRefresh} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded">
                        Analyze {symbol}
                    </button>
                </CardContent>
            </Card>
        );
    }

    const getScoreColor = (score: number) => {
        if (score >= 70) return "bg-green-500";
        if (score >= 40) return "bg-yellow-500";
        return "bg-red-500";
    };

    return (
        <Card className="w-full">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-xl font-bold">AI Research: {symbol}</CardTitle>
                <Badge className={`${getScoreColor(data.overall_score)} text-white`}>
                    Score: {data.overall_score}/100
                </Badge>
            </CardHeader>
            <CardContent>
                <Tabs defaultValue="summary" className="w-full">
                    <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger value="summary">Summary</TabsTrigger>
                        <TabsTrigger value="fundamental">Fundamental</TabsTrigger>
                        <TabsTrigger value="technical">Technical</TabsTrigger>
                        <TabsTrigger value="sentiment">Sentiment</TabsTrigger>
                    </TabsList>

                    <TabsContent value="summary" className="space-y-4">
                        <div className="mt-4">
                            <h3 className="font-semibold mb-2">Executive Summary</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-300">
                                {data.summary || "No summary available."}
                            </p>
                        </div>
                        <div className="grid grid-cols-3 gap-4 mt-4">
                            <div className="p-4 border rounded-lg text-center">
                                <div className="text-sm text-gray-500">Fundamental</div>
                                <div className="text-lg font-bold">{data.fundamental?.score || 0}</div>
                            </div>
                            <div className="p-4 border rounded-lg text-center">
                                <div className="text-sm text-gray-500">Technical</div>
                                <div className="text-lg font-bold">{data.technical?.score || 0}</div>
                            </div>
                            <div className="p-4 border rounded-lg text-center">
                                <div className="text-sm text-gray-500">Sentiment</div>
                                <div className="text-lg font-bold">{data.sentiment?.score || 0}</div>
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="fundamental">
                        <div className="mt-4 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <span className="text-sm font-medium">P/E Ratio:</span>
                                    <span className="ml-2">{data.fundamental?.metrics?.pe_ratio || 'N/A'}</span>
                                </div>
                                <div>
                                    <span className="text-sm font-medium">Market Cap:</span>
                                    <span className="ml-2">{data.fundamental?.metrics?.market_cap || 'N/A'}</span>
                                </div>
                            </div>
                            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
                                <p className="text-sm">{data.fundamental?.analysis || "No analysis details."}</p>
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="technical">
                        <div className="mt-4 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <span className="text-sm font-medium">RSI:</span>
                                    <span className="ml-2">{data.technical?.indicators?.rsi || 'N/A'}</span>
                                </div>
                                <div>
                                    <span className="text-sm font-medium">Trend:</span>
                                    <span className="ml-2">{data.technical?.trend || 'N/A'}</span>
                                </div>
                            </div>
                            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
                                <p className="text-sm">{data.technical?.analysis || "No technical analysis."}</p>
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="sentiment">
                        <div className="mt-4 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <span className="text-sm font-medium">Social Score:</span>
                                    <span className="ml-2">{data.sentiment?.social_score || 'N/A'}</span>
                                </div>
                                <div>
                                    <span className="text-sm font-medium">News Sentiment:</span>
                                    <span className="ml-2">{data.sentiment?.news_sentiment || 'N/A'}</span>
                                </div>
                            </div>
                            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
                                <p className="text-sm">{data.sentiment?.analysis || "No sentiment analysis."}</p>
                            </div>
                        </div>
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Card>
    );
}
