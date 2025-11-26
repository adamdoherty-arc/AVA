import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

interface StrategyAnalysis {
    ticker: string;
    strategy_type: string;
    profit_score: number;
    trade_details: string;
    recommendation: 'BUY' | 'HOLD' | 'AVOID';
    max_profit: number;
    max_loss: number;
    probability_profit: number;
    breakeven: number;
    technical_score: number;
    earnings_safe: boolean;
}

interface StrategyCardProps {
    strategy: StrategyAnalysis;
}

export function StrategyCard({ strategy }: StrategyCardProps) {
    const getRecColor = (rec: string) => {
        switch (rec) {
            case 'BUY': return 'bg-green-500';
            case 'HOLD': return 'bg-yellow-500';
            case 'AVOID': return 'bg-red-500';
            default: return 'bg-gray-500';
        }
    };

    const getScoreIcon = (score: number) => {
        if (score >= 70) return <CheckCircle className="h-5 w-5 text-green-500" />;
        if (score >= 50) return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
        return <XCircle className="h-5 w-5 text-red-500" />;
    };

    return (
        <Card className="w-full hover:shadow-lg transition-shadow">
            <CardHeader>
                <div className="flex justify-between items-start">
                    <div>
                        <CardTitle className="text-xl">{strategy.ticker}</CardTitle>
                        <CardDescription>{strategy.strategy_type}</CardDescription>
                    </div>
                    <Badge className={`${getRecColor(strategy.recommendation)} text-white`}>
                        {strategy.recommendation}
                    </Badge>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md font-mono text-sm">
                    {strategy.trade_details}
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                        <span className="text-gray-500">Score:</span>
                        <span className="font-bold flex items-center gap-1">
                            {strategy.profit_score.toFixed(1)}
                            {getScoreIcon(strategy.profit_score)}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-500">Prob. Profit:</span>
                        <span className="font-bold">{strategy.probability_profit.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-500">Max Profit:</span>
                        <span className="text-green-600 font-bold">${strategy.max_profit.toFixed(0)}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-500">Max Loss:</span>
                        <span className="text-red-600 font-bold">${strategy.max_loss.toFixed(0)}</span>
                    </div>
                </div>
            </CardContent>
            <CardFooter>
                <Button className="w-full" variant="outline">
                    Analyze Further
                </Button>
            </CardFooter>
        </Card>
    );
}
