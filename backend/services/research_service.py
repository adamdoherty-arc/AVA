import logging
from typing import Dict, Any, Optional
from backend.config import get_settings
from src.agents.ai_research.orchestrator import ResearchOrchestrator
from src.agents.ai_research.models import ResearchRequest, ResearchReport

logger = logging.getLogger(__name__)

class ResearchService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.orchestrator = ResearchOrchestrator(
            llm_provider=self.settings.LLM_PROVIDER,
            model_name=self.settings.LLM_MODEL
        )

    async def analyze_symbol(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze a stock symbol using the multi-agent orchestrator.
        """
        try:
            request = ResearchRequest(
                symbol=symbol,
                force_refresh=force_refresh,
                include_sections=['fundamental', 'technical', 'sentiment', 'options']
            )
            
            report: ResearchReport = await self.orchestrator.analyze(request)
            return report.to_dict()
            
        except Exception as e:
            logger.error(f"Error in ResearchService.analyze_symbol: {str(e)}")
            raise

# Singleton instance
_research_service = None

def get_research_service() -> ResearchService:
    global _research_service
    if _research_service is None:
        _research_service = ResearchService()
    return _research_service
