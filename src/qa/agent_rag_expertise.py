"""
Agent RAG Expertise System
==========================

Each QA agent has their own specialized knowledge base stored in a vector database.
This enables agents to perform expert reviews based on their domain knowledge.

Features:
- Agent-specific vector collections
- Semantic search for relevant expertise
- Learning from historical reviews
- Expertise confidence scoring
- Multi-source knowledge integration
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Database
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExpertiseDocument:
    """Represents a piece of expertise knowledge"""
    title: str
    content: str
    doc_type: str  # 'best_practice', 'pattern', 'anti_pattern', 'guideline', 'checklist'
    domain: str
    tags: List[str]
    source: Optional[str] = None
    relevance_score: float = 1.0


@dataclass
class RelevantExpertise:
    """Search result from expertise base"""
    document: ExpertiseDocument
    similarity_score: float
    referenced_count: int


class AgentRAGExpertise:
    """
    RAG-based expertise system for QA agents.

    Each agent has their own vector collection containing domain-specific
    knowledge that they use to perform expert reviews.
    """

    def __init__(
        self,
        agent_name: str,
        embedding_model: str = "all-mpnet-base-v2",
        chroma_path: str = "./chroma_qa_agents"
    ):
        """
        Initialize agent expertise system.

        Args:
            agent_name: Name of the QA agent
            embedding_model: Sentence transformer model
            chroma_path: Path to ChromaDB storage
        """
        self.agent_name = agent_name
        self.collection_name = f"qa_{agent_name}_expertise"

        # Database config
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        # Initialize embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            raise ImportError("sentence-transformers required")

        # Initialize ChromaDB
        if HAS_CHROMADB:
            logger.info(f"Initializing ChromaDB for agent: {agent_name}")
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"agent_name": agent_name, "description": f"Expertise for {agent_name}"}
            )
            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
        else:
            raise ImportError("chromadb required")

    def add_expertise(
        self,
        document: ExpertiseDocument,
        persist_to_db: bool = True
    ) -> str:
        """
        Add expertise document to agent's knowledge base.

        Args:
            document: Expertise document to add
            persist_to_db: Also persist to PostgreSQL

        Returns:
            Document ID
        """
        # Generate embedding
        combined_text = f"{document.title}\n\n{document.content}"
        embedding = self.embedding_model.encode(combined_text)

        # Create unique ID
        doc_id = f"{self.agent_name}_{document.domain}_{datetime.now().timestamp()}"

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[combined_text],
            metadatas=[{
                "agent_name": self.agent_name,
                "title": document.title,
                "doc_type": document.doc_type,
                "domain": document.domain,
                "tags": ",".join(document.tags),
                "source": document.source or "",
                "relevance_score": document.relevance_score
            }]
        )

        # Persist to PostgreSQL if requested
        if persist_to_db:
            self._persist_to_database(doc_id, document, embedding.tolist())

        logger.info(f"Added expertise: {document.title} to {self.agent_name}")
        return doc_id

    def _persist_to_database(
        self,
        doc_id: str,
        document: ExpertiseDocument,
        embedding: List[float]
    ):
        """Persist expertise to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Convert embedding to pgvector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            query = """
                INSERT INTO qa_agent_expertise (
                    agent_name, document_title, document_content, document_type,
                    expertise_domain, relevance_tags, embedding, embedding_model,
                    source_reference, relevance_score
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """

            cur.execute(query, (
                self.agent_name,
                document.title,
                document.content,
                document.doc_type,
                document.domain,
                document.tags,
                embedding_str,
                "all-mpnet-base-v2",
                document.source,
                document.relevance_score
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error persisting to database: {e}")

    def search_expertise(
        self,
        query: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None
    ) -> List[RelevantExpertise]:
        """
        Search agent's expertise base for relevant knowledge.

        Args:
            query: Search query
            n_results: Number of results
            domain_filter: Filter by expertise domain
            doc_type_filter: Filter by document type

        Returns:
            List of relevant expertise documents
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Build filter
        where_filter = {"agent_name": self.agent_name}
        if domain_filter:
            where_filter["domain"] = domain_filter
        if doc_type_filter:
            where_filter["doc_type"] = doc_type_filter

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter if len(where_filter) > 1 else None
        )

        # Format results
        relevant_docs = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]

            doc = ExpertiseDocument(
                title=metadata['title'],
                content=results['documents'][0][i],
                doc_type=metadata['doc_type'],
                domain=metadata['domain'],
                tags=metadata['tags'].split(',') if metadata['tags'] else [],
                source=metadata.get('source'),
                relevance_score=float(metadata.get('relevance_score', 1.0))
            )

            relevant = RelevantExpertise(
                document=doc,
                similarity_score=1 - results['distances'][0][i] if 'distances' in results else 1.0,
                referenced_count=0  # TODO: Track usage
            )

            relevant_docs.append(relevant)

        logger.info(f"Found {len(relevant_docs)} relevant expertise documents for: {query[:50]}...")
        return relevant_docs

    def get_review_context(
        self,
        task_description: str,
        code_snippet: Optional[str] = None,
        max_context_length: int = 3000
    ) -> str:
        """
        Get relevant expertise context for reviewing a task.

        Args:
            task_description: Description of task to review
            code_snippet: Optional code to review
            max_context_length: Maximum context length

        Returns:
            Formatted context for review
        """
        # Search for relevant expertise
        query = f"{task_description}\n{code_snippet if code_snippet else ''}"
        relevant = self.search_expertise(query, n_results=5)

        # Build context
        context_parts = [
            f"# {self.agent_name} Expertise Context\n",
            f"## Task to Review\n{task_description}\n"
        ]

        if code_snippet:
            context_parts.append(f"## Code Snippet\n```\n{code_snippet[:500]}...\n```\n")

        context_parts.append("\n## Relevant Expertise\n")

        total_length = sum(len(p) for p in context_parts)

        for i, rel in enumerate(relevant, 1):
            doc_text = f"\n### {i}. {rel.document.title} ({rel.document.domain})\n"
            doc_text += f"**Type:** {rel.document.doc_type}\n"
            doc_text += f"**Relevance:** {rel.similarity_score:.2f}\n\n"
            doc_text += f"{rel.document.content}\n"

            if total_length + len(doc_text) > max_context_length:
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n".join(context_parts)

    def get_checklist_for_review(
        self,
        task_type: str,
        feature_area: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get review checklist based on task type and feature area.

        Args:
            task_type: Type of task (feature, bug_fix, etc.)
            feature_area: Feature area being modified

        Returns:
            Checklist items
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = """
                SELECT checklist_items
                FROM qa_review_checklist
                WHERE agent_name = %s
                AND task_type = %s
                AND (feature_area = %s OR feature_area IS NULL)
                AND is_active = true
                ORDER BY version DESC
                LIMIT 1;
            """

            cur.execute(query, (self.agent_name, task_type, feature_area))
            result = cur.fetchone()

            cur.close()
            conn.close()

            if result and result['checklist_items']:
                return result['checklist_items']
            else:
                return self._get_default_checklist(task_type)

        except Exception as e:
            logger.error(f"Error getting checklist: {e}")
            return self._get_default_checklist(task_type)

    def _get_default_checklist(self, task_type: str) -> List[Dict[str, Any]]:
        """Get default checklist for agent"""
        # Agent-specific default checklists
        checklists = {
            'code-reviewer': [
                {"item": "Code follows project coding standards", "category": "standards", "critical": True},
                {"item": "No code duplication or copy-paste", "category": "quality", "critical": False},
                {"item": "Functions are properly documented", "category": "documentation", "critical": True},
                {"item": "Error handling is comprehensive", "category": "robustness", "critical": True},
                {"item": "No console.log or debug statements", "category": "cleanup", "critical": False}
            ],
            'security-auditor': [
                {"item": "No SQL injection vulnerabilities", "category": "security", "critical": True},
                {"item": "Input validation is present", "category": "security", "critical": True},
                {"item": "No hardcoded credentials", "category": "security", "critical": True},
                {"item": "Proper authentication checks", "category": "security", "critical": True},
                {"item": "Sensitive data is encrypted", "category": "security", "critical": True}
            ],
            'performance-engineer': [
                {"item": "No N+1 query problems", "category": "performance", "critical": True},
                {"item": "Database queries are optimized", "category": "performance", "critical": True},
                {"item": "Appropriate caching in place", "category": "performance", "critical": False},
                {"item": "No memory leaks", "category": "performance", "critical": True},
                {"item": "Response time is acceptable", "category": "performance", "critical": False}
            ],
            'test-automator': [
                {"item": "Unit tests cover new code", "category": "testing", "critical": True},
                {"item": "Integration tests exist", "category": "testing", "critical": False},
                {"item": "Edge cases are tested", "category": "testing", "critical": True},
                {"item": "Tests are independent", "category": "testing", "critical": False},
                {"item": "Test coverage >= 80%", "category": "coverage", "critical": False}
            ]
        }

        return checklists.get(self.agent_name, [
            {"item": "Code review completed", "category": "general", "critical": True}
        ])

    def update_expertise_from_review(
        self,
        task_id: int,
        sign_off_id: int,
        learnings: List[str]
    ):
        """
        Update expertise based on learnings from a review.

        Args:
            task_id: Task that was reviewed
            sign_off_id: Sign-off ID
            learnings: List of learnings/patterns discovered
        """
        for learning in learnings:
            doc = ExpertiseDocument(
                title=f"Learning from Task #{task_id}",
                content=learning,
                doc_type="pattern",
                domain="learned_from_reviews",
                tags=["learned", f"task-{task_id}"],
                source=f"sign_off_{sign_off_id}"
            )

            self.add_expertise(doc)

        logger.info(f"Added {len(learnings)} learnings from task #{task_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get expertise base statistics"""
        return {
            "agent_name": self.agent_name,
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "embedding_model": self.embedding_model.__class__.__name__,
            "embedding_dim": self.embedding_dim
        }


class QAAgentExpertiseRegistry:
    """
    Registry managing all QA agent expertise systems.

    Provides centralized access to all agent knowledge bases.
    """

    def __init__(self) -> None:
        self.agents: Dict[str, AgentRAGExpertise] = {}
        self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        """Initialize expertise systems for default QA agents"""
        default_agents = [
            'code-reviewer',
            'security-auditor',
            'performance-engineer',
            'database-optimizer',
            'test-automator',
            'api-architect',
            'frontend-developer',
            'backend-architect'
        ]

        for agent_name in default_agents:
            try:
                self.agents[agent_name] = AgentRAGExpertise(agent_name)
                logger.info(f"Initialized expertise for: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {agent_name}: {e}")

    def get_agent_expertise(self, agent_name: str) -> Optional[AgentRAGExpertise]:
        """Get expertise system for an agent"""
        if agent_name not in self.agents:
            try:
                self.agents[agent_name] = AgentRAGExpertise(agent_name)
            except Exception as e:
                logger.error(f"Failed to create expertise for {agent_name}: {e}")
                return None

        return self.agents.get(agent_name)

    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents"""
        return {
            agent_name: expertise.get_stats()
            for agent_name, expertise in self.agents.items()
        }

    def seed_default_expertise(self) -> None:
        """Seed default expertise for all agents"""
        # Code Reviewer expertise
        code_reviewer = self.get_agent_expertise('code-reviewer')
        if code_reviewer:
            code_reviewer.add_expertise(ExpertiseDocument(
                title="DRY Principle - Don't Repeat Yourself",
                content="""
                Avoid code duplication by extracting common logic into reusable functions.
                - If you see the same code in 3+ places, refactor it
                - Use inheritance or composition to share behavior
                - Extract magic numbers into named constants
                - Create utility functions for repeated operations
                """,
                doc_type="best_practice",
                domain="code_quality",
                tags=["dry", "refactoring", "code-smell"]
            ))

            code_reviewer.add_expertise(ExpertiseDocument(
                title="Anti-Pattern: God Object",
                content="""
                DO NOT create classes/modules that know too much or do too much.
                - Single Responsibility Principle: One class, one purpose
                - If a class has >10 methods, consider splitting
                - Avoid classes with >500 lines of code
                - Prefer composition over complex inheritance
                """,
                doc_type="anti_pattern",
                domain="code_quality",
                tags=["god-object", "srp", "architecture"]
            ))

        # Security Auditor expertise
        security = self.get_agent_expertise('security-auditor')
        if security:
            security.add_expertise(ExpertiseDocument(
                title="SQL Injection Prevention",
                content="""
                ALWAYS use parameterized queries or ORMs to prevent SQL injection.
                - NEVER concatenate user input into SQL strings
                - Use placeholders: cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                - Validate and sanitize ALL user inputs
                - Use prepared statements for all database queries
                - Review OWASP Top 10 regularly
                """,
                doc_type="best_practice",
                domain="security",
                tags=["sql-injection", "owasp", "database"]
            ))

            security.add_expertise(ExpertiseDocument(
                title="Sensitive Data Exposure",
                content="""
                Protect sensitive data in transit and at rest.
                - Use HTTPS/TLS for all API communications
                - Encrypt passwords with bcrypt (not MD5/SHA1)
                - Never log passwords or API keys
                - Store secrets in environment variables, not code
                - Use .env files (and add to .gitignore)
                - Implement proper key rotation policies
                """,
                doc_type="best_practice",
                domain="security",
                tags=["encryption", "secrets", "data-protection"]
            ))

        # Performance Engineer expertise
        perf = self.get_agent_expertise('performance-engineer')
        if perf:
            perf.add_expertise(ExpertiseDocument(
                title="N+1 Query Problem",
                content="""
                Avoid the N+1 query anti-pattern where you fetch records in a loop.
                - Use JOIN queries or eager loading instead
                - Example BAD: for user in users: user.get_posts() (N+1)
                - Example GOOD: SELECT users JOIN posts ON... (1 query)
                - Use Django's select_related() and prefetch_related()
                - Monitor query counts in development
                """,
                doc_type="anti_pattern",
                domain="performance",
                tags=["n+1", "database", "optimization"]
            ))

        logger.info("Seeded default expertise for all agents")


# Global registry instance
_global_expertise_registry = None


def get_expertise_registry() -> QAAgentExpertiseRegistry:
    """Get global expertise registry"""
    global _global_expertise_registry
    if _global_expertise_registry is None:
        _global_expertise_registry = QAAgentExpertiseRegistry()
    return _global_expertise_registry


if __name__ == "__main__":
    # Test the system
    print("Testing QA Agent RAG Expertise System\n")

    # Get registry
    registry = get_expertise_registry()

    # Seed default expertise
    print("Seeding default expertise...")
    registry.seed_default_expertise()

    # Test code reviewer
    print("\n--- Testing Code Reviewer ---")
    code_reviewer = registry.get_agent_expertise('code-reviewer')
    if code_reviewer:
        stats = code_reviewer.get_stats()
        print(f"Agent: {stats['agent_name']}")
        print(f"Documents: {stats['document_count']}")

        # Search for expertise
        results = code_reviewer.search_expertise("code duplication refactoring")
        print(f"\nSearch results for 'code duplication':")
        for i, rel in enumerate(results, 1):
            print(f"{i}. {rel.document.title} (similarity: {rel.similarity_score:.3f})")

        # Get review context
        context = code_reviewer.get_review_context(
            "Add new feature with 200 lines of duplicated code",
            "def process_data():\n    # ... 200 lines ...\ndef handle_data():\n    # ... same 200 lines ..."
        )
        print(f"\nReview context length: {len(context)} characters")

    # Test security auditor
    print("\n--- Testing Security Auditor ---")
    security = registry.get_agent_expertise('security-auditor')
    if security:
        results = security.search_expertise("sql injection parameterized queries")
        print(f"Search results for 'sql injection':")
        for i, rel in enumerate(results, 1):
            print(f"{i}. {rel.document.title} (similarity: {rel.similarity_score:.3f})")

    # All agent stats
    print("\n--- All Agent Statistics ---")
    all_stats = registry.get_all_agent_stats()
    for agent_name, stats in all_stats.items():
        print(f"{agent_name}: {stats['document_count']} documents")

    print("\n✅ QA Agent RAG Expertise System test complete!")
