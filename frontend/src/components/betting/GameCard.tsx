import React from 'react';

interface GameCardProps {
    game: {
        id: string;
        league: string;
        home_team: string;
        away_team: string;
        home_score: number;
        away_score: number;
        status: string;
        is_live: boolean;
        game_time: string;
        away_record?: string;
        home_record?: string;
    };
    onClick?: () => void;
}

const GameCard: React.FC<GameCardProps> = ({ game, onClick }) => {
    return (
        <div
            className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow cursor-pointer mb-3"
            onClick={onClick}
        >
            <div className="flex justify-between items-center mb-2">
                <span className="font-bold text-gray-600 text-sm">{game.league}</span>
                <span className={`px-2 py-0.5 rounded text-xs font-semibold ${game.is_live ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                    }`}>
                    {game.status}
                </span>
            </div>

            <div className="flex justify-between items-center">
                <div className="flex-1">
                    <div className="text-lg font-bold text-gray-900">{game.away_team}</div>
                    <div className="text-xs text-gray-500">{game.away_record}</div>
                </div>
                <div className="text-2xl font-bold px-4 text-gray-800">
                    {game.away_score}
                </div>
            </div>

            <div className="flex justify-between items-center mt-2">
                <div className="flex-1">
                    <div className="text-lg font-bold text-gray-900">{game.home_team}</div>
                    <div className="text-xs text-gray-500">{game.home_record}</div>
                </div>
                <div className="text-2xl font-bold px-4 text-gray-800">
                    {game.home_score}
                </div>
            </div>

            <div className="mt-3 text-right text-xs text-gray-500 font-medium">
                {game.game_time}
            </div>
        </div>
    );
};

export default GameCard;
