import React, { useState } from 'react';

interface Bet {
    id: string;
    team: string;
    type: string;
    line: string;
    odds: number;
}

interface BetSlipProps {
    bets: Bet[];
    onRemoveBet: (id: string) => void;
    onPlaceBet: (amount: number) => void;
    onClearSlip: () => void;
}

const BetSlip: React.FC<BetSlipProps> = ({ bets, onRemoveBet, onPlaceBet, onClearSlip }) => {
    const [wager, setWager] = useState<number>(10);

    const totalOdds = bets.reduce((acc, bet) => acc * (bet.odds > 0 ? (bet.odds / 100) + 1 : (100 / Math.abs(bet.odds)) + 1), 1);
    const potentialPayout = wager * totalOdds;

    return (
        <div className="bg-white border-l border-gray-200 h-full flex flex-col shadow-lg w-80 fixed right-0 top-0 pt-16 z-10">
            <div className="p-4 bg-gray-50 border-b border-gray-200 flex justify-between items-center">
                <h3 className="font-bold text-gray-800">🎫 Bet Slip</h3>
                <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2 py-0.5 rounded-full">
                    {bets.length}
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {bets.length === 0 ? (
                    <div className="text-center text-gray-500 mt-10">
                        <p>Your bet slip is empty.</p>
                        <p className="text-sm mt-2">Click on odds to add bets.</p>
                    </div>
                ) : (
                    bets.map((bet) => (
                        <div key={bet.id} className="bg-gray-50 p-3 rounded border border-gray-200 relative group">
                            <button
                                onClick={() => onRemoveBet(bet.id)}
                                className="absolute top-1 right-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                                ×
                            </button>
                            <div className="font-bold text-sm text-gray-800">{bet.team}</div>
                            <div className="text-xs text-gray-600">{bet.type}</div>
                            <div className="flex justify-between items-center mt-2 text-sm">
                                <span className="font-medium bg-white px-2 py-0.5 rounded border border-gray-100">{bet.line}</span>
                                <span className="font-bold text-blue-600">{bet.odds > 0 ? `+${bet.odds}` : bet.odds}</span>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {bets.length > 0 && (
                <div className="p-4 bg-gray-50 border-t border-gray-200">
                    <div className="mb-4">
                        <label className="block text-xs font-medium text-gray-700 mb-1">Wager Amount</label>
                        <div className="relative rounded-md shadow-sm">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <span className="text-gray-500 sm:text-sm">$</span>
                            </div>
                            <input
                                type="number"
                                min="1"
                                value={wager}
                                onChange={(e) => setWager(Number(e.target.value))}
                                className="focus:ring-blue-500 focus:border-blue-500 block w-full pl-7 pr-12 sm:text-sm border-gray-300 rounded-md py-2"
                            />
                        </div>
                    </div>

                    <div className="flex justify-between items-center mb-4 text-sm">
                        <span className="text-gray-600">Est. Payout:</span>
                        <span className="font-bold text-green-600">${potentialPayout.toFixed(2)}</span>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={onClearSlip}
                            className="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
                        >
                            Clear
                        </button>
                        <button
                            onClick={() => onPlaceBet(wager)}
                            className="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none"
                        >
                            Place Bet
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default BetSlip;
