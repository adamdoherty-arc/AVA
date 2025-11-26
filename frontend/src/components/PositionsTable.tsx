import React from 'react';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { ArrowUpRight, ArrowDownRight, ExternalLink } from "lucide-react";

interface Position {
    symbol: string;
    quantity: number;
    avg_buy_price: number;
    current_price: number;
    current_value: number;
    pl: number;
    pl_pct: number;
    type: 'stock' | 'option';
    option_type?: string;
    strike?: number;
    expiration?: string;
}

interface PositionsTableProps {
    positions: Position[];
    type: 'stock' | 'option';
}

export function PositionsTable({ positions, type }: PositionsTableProps) {
    const formatCurrency = (val: number) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);

    const formatPercent = (val: number) =>
        `${val > 0 ? '+' : ''}${val.toFixed(2)}%`;

    return (
        <div className="rounded-md border">
            <Table>
                <TableHeader>
                    <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Qty</TableHead>
                        <TableHead>Avg Price</TableHead>
                        <TableHead>Current</TableHead>
                        <TableHead>Value</TableHead>
                        <TableHead>P/L ($)</TableHead>
                        <TableHead>P/L (%)</TableHead>
                        <TableHead>Action</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    {positions.map((pos, idx) => (
                        <TableRow key={`${pos.symbol}-${idx}`}>
                            <TableCell className="font-medium">
                                <div className="flex flex-col">
                                    <span>{pos.symbol}</span>
                                    {type === 'option' && (
                                        <span className="text-xs text-gray-500">
                                            {pos.expiration} ${pos.strike} {pos.option_type}
                                        </span>
                                    )}
                                </div>
                            </TableCell>
                            <TableCell>{pos.quantity}</TableCell>
                            <TableCell>{formatCurrency(pos.avg_buy_price)}</TableCell>
                            <TableCell>{formatCurrency(pos.current_price)}</TableCell>
                            <TableCell>{formatCurrency(pos.current_value)}</TableCell>
                            <TableCell className={pos.pl >= 0 ? "text-green-600" : "text-red-600"}>
                                {formatCurrency(pos.pl)}
                            </TableCell>
                            <TableCell>
                                <Badge variant={pos.pl_pct >= 0 ? "default" : "destructive"} className="flex w-fit items-center gap-1">
                                    {pos.pl_pct >= 0 ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                                    {formatPercent(pos.pl_pct)}
                                </Badge>
                            </TableCell>
                            <TableCell>
                                <a
                                    href={`https://www.tradingview.com/chart/?symbol=${pos.symbol}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-500 hover:text-blue-700"
                                >
                                    <ExternalLink className="h-4 w-4" />
                                </a>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </div>
    );
}
