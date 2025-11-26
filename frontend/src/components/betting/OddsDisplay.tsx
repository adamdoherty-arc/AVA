import React from 'react';

interface OddsData {
    spread_home?: number;
    spread_home_odds?: number;
    total?: number;
    moneyline_home?: number;
}

interface OddsDisplayProps {
    odds: OddsData;
    gameId: string;
    onBetClick?: (type: string, line: string, odds: number) => void;
}

const OddsDisplay: React.FC<OddsDisplayProps> = ({ odds, gameId, onBetClick }) => {
    const fmtOdds = (val?: number) => {
        if (val === undefined || val === null) return "-";
        return val > 0 ? `+${val}` : `${val}`;
    };

    const handleBet = (type: string, line: string, oddsVal?: number) => {
        if (onBetClick && oddsVal !== undefined) {
            onBetClick(type, line, oddsVal);
        }
    };

    return (
        <div className="grid grid-cols-3 gap-2 mt-2">
            <div className="flex flex-col">
                <span className="text-center text-xs text-gray-500 mb-1">Spread</span>
                <button
                    className="bg-gray-50 hover:bg-blue-50 border border-gray-200 text-gray-800 font-medium py-2 px-1 rounded text-sm transition-colors"
                    onClick={() => handleBet('Spread', `${odds.spread_home}`, odds.spread_home_odds)}
                >
                    <div className="flex flex-col items-center leading-tight">
                        <span>{odds.spread_home}</span>
                        <span className="text-xs text-gray-500">{fmtOdds(odds.spread_home_odds)}</span>
                    </div>
                </button>
            </div>

            <div className="flex flex-col">
                <span className="text-center text-xs text-gray-500 mb-1">Total</span>
                <button
                    className="bg-gray-50 hover:bg-blue-50 border border-gray-200 text-gray-800 font-medium py-2 px-1 rounded text-sm transition-colors"
                    onClick={() => handleBet('Total', `O/U ${odds.total}`, -110)}
                >
                    <div className="flex flex-col items-center leading-tight">
                        <span>O/U</span>
                        <span className="text-xs text-gray-500">{odds.total}</span>
                    </div>
                </button>
            </div>

            <div className="flex flex-col">
                <span className="text-center text-xs text-gray-500 mb-1">Moneyline</span>
                <button
                    className="bg-gray-50 hover:bg-blue-50 border border-gray-200 text-gray-800 font-medium py-2 px-1 rounded text-sm transition-colors"
                    onClick={() => handleBet('Moneyline', 'ML', odds.moneyline_home)}
                >
                    <div className="flex flex-col items-center leading-tight">
                        <span>{fmtOdds(odds.moneyline_home)}</span>
                    </div>
                </button>
            </div>
        </div>
    );
};

export default OddsDisplay;
