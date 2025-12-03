"""
LLM Earnings Analyzer Service
Provides AI-powered analysis of earnings reports and transcripts using Local LLM
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.magnus_local_llm import get_magnus_llm, TaskComplexity

logger = logging.getLogger(__name__)

class LLMEarningsAnalyzer:
    """
    Analyzes earnings reports, guidance, and transcripts
    """

    def __init__(self) -> None:
        self.llm = get_magnus_llm()

    def analyze_earnings_report(self, 
                               symbol: str, 
                               report_text: str,
                               estimates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze earnings report text against estimates
        
        Args:
            symbol: Stock symbol
            report_text: Raw text of the earnings release
            estimates: Dictionary with EPS and Revenue estimates
            
        Returns:
            Dictionary with sentiment, key takeaways, and guidance analysis
        """
        try:
            # Truncate report text if too long (simple truncation)
            # In production, use smarter chunking or RAG
            max_len = 3000
            truncated_text = report_text[:max_len] + "..." if len(report_text) > max_len else report_text
            
            prompt = f"""Analyze this earnings report for {symbol}:

ESTIMATES:
- EPS Estimate: {estimates.get('eps_est', 'N/A')}
- Revenue Estimate: {estimates.get('rev_est', 'N/A')}

REPORT TEXT:
{truncated_text}

Please provide:
1. Did they beat/miss EPS and Revenue?
2. Sentiment Analysis (Positive/Negative/Neutral)
3. Key Takeaways (Bullet points)
4. Guidance Update (Raised/Lowered/Reaffirmed)
5. Any red flags or major risks mentioned?

Format the response as JSON with keys: 'beat_miss_status', 'sentiment', 'takeaways', 'guidance', 'risks'.
"""
            
            response = self.llm.query(
                prompt=prompt,
                complexity=TaskComplexity.ANALYTICAL,
                use_trading_context=True,
                max_tokens=1000
            )
            
            # Parse JSON response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                elif "{" in response and "}" in response:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                    return json.loads(json_str)
                else:
                    return {
                        'sentiment': 'Unknown',
                        'takeaways': [response],
                        'beat_miss_status': 'Unknown'
                    }
            except json.JSONDecodeError:
                return {
                    'sentiment': 'Error parsing',
                    'takeaways': [response],
                    'beat_miss_status': 'Error'
                }
                
        except Exception as e:
            logger.error(f"Error analyzing earnings for {symbol}: {e}")
            return {'error': str(e)}

    def analyze_transcript_sentiment(self, transcript_segments: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of earnings call transcript segments"""
        # Implementation for transcript analysis
        pass
