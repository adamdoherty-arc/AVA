"""
AVA Chatbot SpecAgent - Tier 2

Deeply understands the AVA chatbot feature.
Tests: Chat endpoints, NLP processing, agent routing.
"""

import logging
from typing import List, Dict, Any

from ..base_spec_agent import BaseSpecAgent, Issue, IssueSeverity, TestResult
from ..spec_agent_registry import register_spec_agent

logger = logging.getLogger(__name__)


@register_spec_agent('ava-chatbot')
class AVAChatbotSpecAgent(BaseSpecAgent):
    """
    SpecAgent for AVA Chatbot feature.

    Validates:
    - Chat API endpoints
    - NLP processing
    - Agent routing
    - Response generation
    """

    def __init__(self) -> None:
        super().__init__(
            feature_name='ava-chatbot',
            description='AVA intelligent chatbot assistant',
            enable_browser=True,
            enable_database=True,
        )
        self._chat_response = None

    async def test_api_endpoints(self) -> List[TestResult]:
        """Test all chatbot-related API endpoints"""
        results = []

        # Test /api/chat endpoint
        result = await self.test_endpoint(
            method='POST',
            path='/chat',
            expected_status=200,
            body={
                'message': 'Hello',
                'user_id': 'test_user'
            },
            validate_response=self._validate_chat_response,
        )
        results.append(result)

        if result.passed:
            response = await self.http_post('/chat', json={
                'message': 'Hello',
                'user_id': 'test_user'
            })
            if response.status_code == 200:
                self._chat_response = response.json()

        # Test /api/chat/history
        result = await self.test_endpoint(
            method='GET',
            path='/chat/history?user_id=test_user',
            expected_status=200,
        )
        results.append(result)

        # Test health endpoint
        result = await self.test_endpoint(
            method='GET',
            path='/health',
            expected_status=200,
        )
        results.append(result)

        return results

    def _validate_chat_response(self, response) -> List[Issue]:
        """Custom validation for chat response"""
        issues = []
        data = response.json()

        # Check for response content
        response_text = data.get('response', data.get('message', ''))
        if not response_text:
            issues.append(Issue(
                title="Empty chat response",
                description="Chat API returned no response text",
                severity=IssueSeverity.HIGH,
                feature=self.feature_name,
                component="api",
            ))

        # Check for agent used
        agent_used = data.get('agent', data.get('routed_to', ''))
        if not agent_used:
            issues.append(Issue(
                title="No agent routing info",
                description="Chat response should indicate which agent handled the request",
                severity=IssueSeverity.LOW,
                feature=self.feature_name,
                component="api",
            ))

        # Check response time
        response_time = data.get('response_time_ms', 0)
        if response_time > 5000:  # 5 seconds
            issues.append(Issue(
                title="Slow chat response",
                description=f"Response took {response_time}ms (>5s threshold)",
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="performance",
            ))

        return issues

    async def test_ui_components(self) -> List[TestResult]:
        """Test UI components using Playwright"""
        results = []

        try:
            # Navigate to chat page
            success = await self.navigate_to('/chat')
            if not success:
                success = await self.navigate_to('/ava')

            if not success:
                results.append(TestResult(
                    test_name="Navigate to chat page",
                    passed=False,
                    issues=[Issue(
                        title="Failed to navigate to chat page",
                        description="Could not load /chat or /ava route",
                        severity=IssueSeverity.CRITICAL,
                        feature=self.feature_name,
                        component="ui",
                    )],
                ))
                return results

            # Test chat input exists
            results.append(await self.test_element_exists(
                "[data-testid='chat-input'], input[type='text'], textarea",
                "Chat Input"
            ))

            # Test send button
            results.append(await self.test_button_clickable(
                "[data-testid='send-button'], button:has-text('Send'), button[type='submit']",
                "Send Button"
            ))

            # Test message container
            results.append(await self.test_element_exists(
                "[data-testid='messages'], .messages, .chat-history",
                "Message Container"
            ))

            # Take screenshot
            await self.take_screenshot("ava_chatbot_page")

        except Exception as e:
            logger.error(f"UI testing failed: {e}")
            results.append(TestResult(
                test_name="AVA Chatbot UI Tests",
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
        """Test chatbot business logic"""
        results = []
        issues = []

        # Test various query types
        test_queries = [
            {'query': 'What is my portfolio value?', 'expect_agent': 'portfolio'},
            {'query': 'Show me options for AAPL', 'expect_agent': 'options'},
            {'query': 'What are the best wheel opportunities?', 'expect_agent': 'strategy'},
        ]

        for test in test_queries:
            try:
                response = await self.http_post('/chat', json={
                    'message': test['query'],
                    'user_id': 'test_user'
                })
                if response.status_code == 200:
                    data = response.json()
                    agent_used = data.get('agent', '').lower()

                    # Check if appropriate agent was used
                    if test['expect_agent'] not in agent_used:
                        issues.append(Issue(
                            title=f"Unexpected agent routing",
                            description=f"Query '{test['query']}' routed to '{agent_used}' instead of '{test['expect_agent']}'",
                            severity=IssueSeverity.MEDIUM,
                            feature=self.feature_name,
                            component="routing",
                        ))
            except Exception as e:
                issues.append(Issue(
                    title=f"Query test failed",
                    description=f"Error testing query '{test['query']}': {str(e)}",
                    severity=IssueSeverity.MEDIUM,
                    feature=self.feature_name,
                    component="api",
                ))

        results.append(TestResult(
            test_name="Agent Routing Logic",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results

    async def test_data_consistency(self) -> List[TestResult]:
        """Test chatbot data consistency"""
        results = []
        issues = []

        # Test conversation persistence
        test_message = "test message for consistency check"

        try:
            # Send a message
            await self.http_post('/chat', json={
                'message': test_message,
                'user_id': 'consistency_test_user'
            })

            # Check history contains the message
            history_response = await self.http_get('/chat/history?user_id=consistency_test_user')
            if history_response.status_code == 200:
                history = history_response.json()
                messages = history.get('messages', history) if isinstance(history, dict) else history

                if isinstance(messages, list):
                    found = any(test_message in str(m) for m in messages)
                    if not found:
                        issues.append(Issue(
                            title="Message not persisted",
                            description="Sent message not found in conversation history",
                            severity=IssueSeverity.HIGH,
                            feature=self.feature_name,
                            component="persistence",
                        ))
        except Exception as e:
            issues.append(Issue(
                title="History check failed",
                description=str(e),
                severity=IssueSeverity.MEDIUM,
                feature=self.feature_name,
                component="consistency",
            ))

        results.append(TestResult(
            test_name="Chat Data Consistency",
            passed=len(issues) == 0,
            issues=issues,
        ))

        return results
