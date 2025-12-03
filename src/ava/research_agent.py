"""
AI-Native Research Agent
Continuously discovers improvements from external sources

Updated: 2025-11-29 - Now uses FREE LLM providers (Groq/Ollama) by default
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

import praw
from github import Github
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from src.ava.core.llm_engine import LLMClient, LLMProvider
from src.ava.core.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class ResearchAgent:
    """
    Autonomous research agent that discovers improvements from multiple sources
    Now uses FREE LLM providers (Groq/Ollama) by default!
    """

    def __init__(self, db_config: Dict[str, str], use_local_llm: bool = True):
        """
        Initialize the research agent

        Args:
            db_config: Database connection configuration
            use_local_llm: DEPRECATED - now uses centralized config for LLM provider
        """
        self.db_config = db_config

        # Initialize similarity model (runs locally, no API costs)
        logger.info("Loading sentence transformer model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize API clients
        self.reddit = None
        self.github = None
        self.llm = None

        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize external API clients"""
        # Reddit API
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='Magnus Research Agent v1.0'
            )
            logger.info("Reddit client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit client: {e}")

        # GitHub API
        try:
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                self.github = Github(github_token)
                logger.info("GitHub client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GitHub client: {e}")

        # LLM Client - Use centralized config for FREE provider
        try:
            config = get_config()
            provider_map = {
                "ollama": LLMProvider.OLLAMA,
                "groq": LLMProvider.GROQ,
                "huggingface": LLMProvider.HUGGINGFACE,
                "anthropic": LLMProvider.ANTHROPIC,
                "openai": LLMProvider.OPENAI,
            }
            provider = provider_map.get(config.ai.provider.lower(), LLMProvider.GROQ)

            self.llm = LLMClient(
                provider=provider,
                model=config.ai.default_model,
                cache_enabled=True,
                cache_ttl=600  # 10 minute cache for research
            )
            logger.info(f"LLM client initialized with FREE provider: {provider.value}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    def get_db_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of normalized content for deduplication"""
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def check_duplicate(self, content_hash: str) -> bool:
        """Check if content already exists in database"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM ci_research_findings WHERE content_hash = %s",
                    (content_hash,)
                )
                return cur.fetchone() is not None

    def find_similar_findings(self, content: str, threshold: float = 0.8) -> List[int]:
        """Find similar existing findings using semantic similarity"""
        embedding = self.similarity_model.encode(content)

        # Get recent findings to compare against (last 30 days)
        with self.get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, content
                    FROM ci_research_findings
                    WHERE discovered_at > NOW() - INTERVAL '30 days'
                    ORDER BY discovered_at DESC
                    LIMIT 1000
                """)
                recent_findings = cur.fetchall()

        similar_ids = []
        for finding in recent_findings:
            existing_embedding = self.similarity_model.encode(finding['content'])
            similarity = self.similarity_model.similarity(embedding, existing_embedding).item()

            if similarity > threshold:
                similar_ids.append(finding['id'])

        return similar_ids

    async def scan_reddit(self, subreddit_name: str, search_queries: List[str],
                          time_filter: str = 'day') -> List[Dict[str, Any]]:
        """
        Scan Reddit for relevant posts

        Args:
            subreddit_name: Subreddit to scan (e.g., 'options', 'algotrading')
            search_queries: List of search terms
            time_filter: Time filter ('hour', 'day', 'week', 'month')

        Returns:
            List of findings
        """
        if not self.reddit:
            logger.warning("Reddit client not initialized")
            return []

        findings = []
        subreddit = self.reddit.subreddit(subreddit_name)

        logger.info(f"Scanning r/{subreddit_name}...")

        for query in search_queries:
            try:
                # Search subreddit
                for submission in subreddit.search(query, time_filter=time_filter, limit=20):
                    # Combine title and selftext
                    content = f"{submission.title}\n\n{submission.selftext}"

                    # Check for duplicates
                    content_hash = self.calculate_content_hash(content)
                    if self.check_duplicate(content_hash):
                        continue

                    findings.append({
                        'source_type': 'reddit',
                        'source_url': f"https://reddit.com{submission.permalink}",
                        'source_title': submission.title,
                        'source_author': str(submission.author),
                        'source_date': datetime.fromtimestamp(submission.created_utc),
                        'content': content,
                        'content_hash': content_hash,
                        'upvotes': submission.score,
                        'comment_count': submission.num_comments,
                        'metadata': {
                            'subreddit': subreddit_name,
                            'query': query,
                            'flair': submission.link_flair_text
                        }
                    })

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error scanning Reddit query '{query}': {e}")

        logger.info(f"Found {len(findings)} Reddit findings")
        return findings

    async def scan_github(self, search_query: str, language: str = 'python') -> List[Dict[str, Any]]:
        """
        Scan GitHub for relevant repositories and issues

        Args:
            search_query: GitHub search query
            language: Programming language filter

        Returns:
            List of findings
        """
        if not self.github:
            logger.warning("GitHub client not initialized")
            return []

        findings = []

        logger.info(f"Scanning GitHub for '{search_query}'...")

        try:
            # Search repositories
            repos = self.github.search_repositories(
                query=f"{search_query} language:{language}",
                sort='updated',
                order='desc'
            )

            for repo in repos[:10]:  # Limit to top 10
                content = f"{repo.name}\n\n{repo.description}\n\nREADME:\n{repo.get_readme().decoded_content.decode()[:1000]}"

                content_hash = self.calculate_content_hash(content)
                if self.check_duplicate(content_hash):
                    continue

                findings.append({
                    'source_type': 'github',
                    'source_url': repo.html_url,
                    'source_title': repo.name,
                    'source_author': repo.owner.login,
                    'source_date': repo.updated_at,
                    'content': content,
                    'content_hash': content_hash,
                    'upvotes': repo.stargazers_count,
                    'metadata': {
                        'stars': repo.stargazers_count,
                        'forks': repo.forks_count,
                        'language': repo.language,
                        'topics': repo.get_topics()
                    }
                })

                await asyncio.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error scanning GitHub: {e}")

        logger.info(f"Found {len(findings)} GitHub findings")
        return findings

    def analyze_relevance(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze relevance and score the finding (FREE via Groq/Ollama)

        Args:
            finding: Finding dictionary

        Returns:
            Analysis results with scores
        """
        prompt = f"""Evaluate this research finding for the Magnus Trading Platform.

Finding:
Title: {finding['source_title']}
Source: {finding['source_type']}
Content: {finding['content'][:2000]}

Provide scores (0-100) and analysis as JSON:
{{
  "relevance_score": 85,
  "applicability_score": 70,
  "novelty_score": 60,
  "implementation_difficulty": 40,
  "summary": "2-3 sentence summary",
  "keywords": ["options", "backtesting"],
  "technologies": ["pandas", "scikit-learn"],
  "recommendation": "yes/no/maybe"
}}
"""

        try:
            if not self.llm:
                raise ValueError("LLM client not initialized")

            # Use async LLM call
            async def get_analysis():
                return await self.llm.generate(
                    system="You are an AI analyst evaluating research findings "
                           "for a Python options trading platform. Return valid JSON only.",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.3,
                    max_tokens=500
                )

            # Run async
            response = asyncio.run(get_analysis())
            analysis_text = response.content

            # Parse JSON response - extract from code blocks if present
            if '```json' in analysis_text:
                analysis_text = analysis_text.split('```json')[1].split('```')[0].strip()
            elif '```' in analysis_text:
                analysis_text = analysis_text.split('```')[1].split('```')[0].strip()

            analysis = json.loads(analysis_text)

            return {
                'relevance_score': analysis.get('relevance_score', 50),
                'applicability_score': analysis.get('applicability_score', 50),
                'novelty_score': analysis.get('novelty_score', 50),
                'implementation_difficulty': analysis.get('implementation_difficulty', 50),
                'content_summary': analysis.get('summary', ''),
                'keywords': analysis.get('keywords', []),
                'extracted_technologies': {
                    'technologies': analysis.get('technologies', [])
                },
                'ai_analysis': analysis_text,
                'recommendation': analysis.get('recommendation', 'maybe')
            }

        except Exception as e:
            logger.error(f"Error analyzing relevance: {e}")
            return {
                'relevance_score': 50,
                'applicability_score': 50,
                'novelty_score': 50,
                'implementation_difficulty': 50,
                'content_summary': '',
                'keywords': [],
                'extracted_technologies': {},
                'ai_analysis': f"Error: {str(e)}",
                'recommendation': 'maybe'
            }

    def save_finding(self, finding: Dict[str, Any], analysis: Dict[str, Any]) -> int:
        """
        Save finding to database

        Args:
            finding: Finding data
            analysis: AI analysis results

        Returns:
            Finding ID
        """
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ci_research_findings (
                        source_type, source_url, source_title, source_author, source_date,
                        content, content_hash, content_summary,
                        relevance_score, applicability_score, novelty_score, implementation_difficulty,
                        ai_analysis, keywords, extracted_technologies,
                        upvotes, downvotes, comment_count,
                        status
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s
                    ) RETURNING id
                """, (
                    finding['source_type'],
                    finding['source_url'],
                    finding['source_title'],
                    finding.get('source_author'),
                    finding.get('source_date'),
                    finding['content'],
                    finding['content_hash'],
                    analysis['content_summary'],
                    analysis['relevance_score'],
                    analysis['applicability_score'],
                    analysis['novelty_score'],
                    analysis['implementation_difficulty'],
                    analysis['ai_analysis'],
                    analysis['keywords'],
                    json.dumps(analysis['extracted_technologies']),
                    finding.get('upvotes', 0),
                    finding.get('downvotes', 0),
                    finding.get('comment_count', 0),
                    'analyzed' if analysis['relevance_score'] >= 60 else 'rejected'
                ))

                finding_id = cur.fetchone()[0]
                conn.commit()

                return finding_id

    async def run_scan(self, source_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a complete research scan

        Args:
            source_configs: List of source configurations

        Returns:
            Scan results summary
        """
        logger.info("Starting research scan...")
        start_time = datetime.now()

        all_findings = []

        for config in source_configs:
            source_type = config['source_type']
            logger.info(f"Scanning {source_type}...")

            try:
                if source_type == 'reddit':
                    findings = await self.scan_reddit(
                        config['subreddit'],
                        config['search_queries'],
                        config.get('time_filter', 'day')
                    )
                    all_findings.extend(findings)

                elif source_type == 'github':
                    for query in config['search_queries']:
                        findings = await self.scan_github(query)
                        all_findings.extend(findings)

            except Exception as e:
                logger.error(f"Error scanning {source_type}: {e}")

        # Analyze and save findings
        saved_count = 0
        high_relevance_count = 0

        for finding in all_findings:
            try:
                # Analyze relevance
                analysis = self.analyze_relevance(finding)

                # Save to database
                finding_id = self.save_finding(finding, analysis)
                saved_count += 1

                if analysis['relevance_score'] >= 80:
                    high_relevance_count += 1
                    logger.info(f"High relevance finding: {finding['source_title']} (score: {analysis['relevance_score']})")

            except Exception as e:
                logger.error(f"Error processing finding: {e}")

        duration = (datetime.now() - start_time).total_seconds()

        results = {
            'findings_discovered': len(all_findings),
            'findings_saved': saved_count,
            'high_relevance_count': high_relevance_count,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Scan complete: {results}")
        return results


async def main():
    """Example usage"""
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'magnus'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    # Initialize agent
    agent = ResearchAgent(db_config, use_local_llm=False)

    # Define sources to scan
    sources = [
        {
            'source_type': 'reddit',
            'subreddit': 'options',
            'search_queries': ['trading platform', 'options scanner', 'portfolio tracker'],
            'time_filter': 'week'
        },
        {
            'source_type': 'reddit',
            'subreddit': 'algotrading',
            'search_queries': ['algorithmic trading', 'backtesting', 'market data'],
            'time_filter': 'week'
        },
        {
            'source_type': 'github',
            'search_queries': ['options trading python', 'trading bot python']
        }
    ]

    # Run scan
    results = await agent.run_scan(sources)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
