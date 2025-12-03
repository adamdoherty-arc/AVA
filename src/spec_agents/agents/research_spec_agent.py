"""
Research SpecAgent - Tier 3

Deeply understands the Research feature.
Tests: Research queries, RAG retrieval, document indexing.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('research')
class ResearchSpecAgent(BaseSpecAgent):
    """
    SpecAgent for Research feature.

    Validates:
    - Research API endpoints
    - RAG query accuracy
    - Document retrieval
    - Knowledge base health
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='research',
            description='Research and knowledge base queries',
            enable_browser=True,
            enable_database=True,
        )
        self._research_data = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all research API endpoints"""
        results = []

        # Test /api/research/query
        result = await self.test_endpoint(
            method='POST',
            path='/research/query',
            expected_status=200,
            body={'question': 'What is the wheel strategy?'},
            validate_response=self._validate_research_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_post('/research/query', json={'question': 'What is the wheel strategy?'})
            if response.status_code == 200:
                self._research_data = response.json()

        # Test /api/research/documents
        result = await self.test_endpoint(
            method='GET',
            path='/research/documents',
            expected_status=200,
        )
        results.append(result)

        # Test /api/research/status
        result = await self.test_endpoint(
            method='GET',
            path='/research/status',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_research_response(self, response) -> List[Issue]:
        """Custom validation for research response"""
        issues = []
        data = response.json()

        # Check for answer
        answer = data.get('answer', data.get('response', ''))
        if not answer:
            issues.append(Issue(
                title="Empty research answer",
                description="Research query returned no answer",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))

        # Check for sources/citations
        sources = data.get('sources', data.get('citations', data.get('documents', [])))
        if isinstance(sources, list) and len(sources) == 0:
            issues.append(Issue(
                title="No sources cited",
                description="Research answer should include source documents",
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="rag",
            ))

        # Check confidence score if present
        confidence = data.get('confidence', data.get('relevance_score', None))
        if confidence is not None:
            if isinstance(confidence, (int, float)) and (confidence < 0 or confidence > 1):
                issues.append(Issue(
                    title="Invalid confidence score",
                    description=f"Confidence {confidence} should be 0-1",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="data_quality",
                ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            success = await self.navigate_to('/research')
            if not success:
                success = await self.navigate_to('/knowledge')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to research page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to research page",
                        description="Could not load /research route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test search input
            results.append(await self.test_element_exists(
                "[data-testid='research-input'], input[type='text'], textarea",
                "Research Input"
            ))

            # Test search button
            results.append(await self.test_button_clickable(
                "[data-testid='search-button'], button:has-text('Search'), button[type='submit']",
                "Search Button"
            ))

            # Test results area
            results.append(await self.test_element_exists(
                "[data-testid='results'], .research-results, .answer-container",
                "Results Area"
            ))

            await self.take_screenshot("research_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="Research UI Tests",
                passed=False,
                issues=[Issue(
                    title="UI testing exception",
                    description=str(e),
                    severity=IssueSeverity.HIGH,
                    feature=self.feature_name,
                    component="ui",
                )],
            ))

        return results

    async def test_business_logic(self) -> List[TestResult]:
        """Test research/RAG logic"""
        results = []
        issues = []

        # Test various query types
        test_queries = [
            {'query': 'What is a covered call?', 'expect_topic': 'options'},
            {'query': 'How does theta decay work?', 'expect_topic': 'greeks'},
            {'query': 'What is the wheel strategy?', 'expect_topic': 'wheel'},
        ]

        for test in test_queries:
            try:
                response = await self.http_post('/research/query', json={'question': test['query']})
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('answer', data.get('response', '')).lower()

                    # Check if answer relates to expected topic
                    if test['expect_topic'] not in answer:
                        issues.append(Issue(
                            title=f"Answer may not match query",
                            description=f"Query about '{test['expect_topic']}' got answer not mentioning that topic",
                            severity=IssueSeverity.LOW,
                            feature=self.feature_name,
                            component="rag",
                        ))
            except Exception as e:
                logger.debug(f"Query test failed: {e}")

        results.append(TestResult(
            test_name="Research RAG Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test research data consistency"""
        results = []
        issues = []

        # Check knowledge base status
        try:
            status_response = await self.http_get('/research/status')
            if status_response.status_code == 200:
                status = status_response.json()

                # Check document count
                doc_count = status.get('document_count', status.get('indexed_documents', 0))
                if doc_count == 0:
                    issues.append(Issue(
                        title="Empty knowledge base",
                        description="No documents indexed for research",
                        severity=IssueSeverity.HIGH,
                        feature=self.feature_name,
                        component="indexing",
                    ))

                # Check last update
                last_update = status.get('last_indexed', status.get('last_update'))
                if last_update:
                    from datetime import datetime, timedelta
                    try:
                        if isinstance(last_update, str):
                            lu = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        else:
                            lu = datetime.fromtimestamp(last_update)

                        now = datetime.now(lu.tzinfo) if lu.tzinfo else datetime.now()
                        age = now - lu

                        if age > timedelta(days=7):
                            issues.append(Issue(
                                title="Stale knowledge base",
                                description=f"Last indexed {age.days} days ago",
                                severity=IssueSeverity.MEDIUM,
                                feature=self.feature_name,
                                component="data_freshness",
                            ))
                    except Exception as e:
                        logger.debug(f"Could not parse last_update: {e}")

        except Exception as e:
            logger.debug(f"Could not check research status: {e}")

        results.append(TestResult(
            test_name="Research Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
