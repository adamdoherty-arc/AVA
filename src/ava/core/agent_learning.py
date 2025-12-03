"""
Agent Learning System - Memory, Performance Tracking, and Auto-Updates
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_response_time: float
    last_execution: Optional[datetime]
    success_rate: float
    user_satisfaction: float
    last_updated: datetime


@dataclass
class AgentMemory:
    """Agent memory entry"""
    agent_name: str
    memory_key: str
    memory_value: Any
    context: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int


@dataclass
class AgentFeedback:
    """User feedback for agent"""
    agent_name: str
    feedback_type: str  # 'positive', 'negative', 'suggestion'
    feedback_text: str
    user_id: str
    timestamp: datetime
    resolved: bool


class AgentLearningSystem:
    """
    Agent Learning System
    
    Features:
    - Performance tracking
    - Memory persistence
    - Feedback collection
    - Auto-update mechanisms
    - Strategy learning
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """Initialize learning system"""
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        # In-memory caches
        self._performance_cache: Dict[str, AgentPerformance] = {}
        self._memory_cache: Dict[str, List[AgentMemory]] = {}
        
        # Initialize database tables
        self._initialize_database()
        
        logger.info("AgentLearningSystem initialized")
    
    def _initialize_database(self) -> None:
        """Initialize database tables for learning system"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Agent performance table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    agent_name VARCHAR(100) PRIMARY KEY,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    average_response_time FLOAT DEFAULT 0.0,
                    last_execution TIMESTAMP,
                    success_rate FLOAT DEFAULT 0.0,
                    user_satisfaction FLOAT DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Agent memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id SERIAL PRIMARY KEY,
                    agent_name VARCHAR(100) NOT NULL,
                    memory_key VARCHAR(255) NOT NULL,
                    memory_value JSONB,
                    context JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_accessed TIMESTAMP DEFAULT NOW(),
                    access_count INTEGER DEFAULT 0,
                    UNIQUE(agent_name, memory_key)
                )
            """)
            
            # Agent feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_feedback (
                    id SERIAL PRIMARY KEY,
                    agent_name VARCHAR(100) NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    feedback_text TEXT,
                    user_id VARCHAR(100),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Agent execution log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_execution_log (
                    id SERIAL PRIMARY KEY,
                    agent_name VARCHAR(100) NOT NULL,
                    execution_id VARCHAR(100) NOT NULL,
                    input_text TEXT,
                    result JSONB,
                    error TEXT,
                    response_time_ms FLOAT,
                    user_id VARCHAR(100),
                    platform VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory_name ON agent_memory(agent_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_feedback_name ON agent_feedback(agent_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_execution_name ON agent_execution_log(agent_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_execution_time ON agent_execution_log(timestamp)")
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Agent learning database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing learning database: {e}")
    
    def log_execution(
        self,
        agent_name: str,
        execution_id: str,
        input_text: str,
        result: Dict[str, Any],
        error: Optional[str] = None,
        response_time_ms: float = 0.0,
        user_id: Optional[str] = None,
        platform: str = "web"
    ):
        """Log agent execution"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO agent_execution_log 
                (agent_name, execution_id, input_text, result, error, response_time_ms, user_id, platform)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                agent_name,
                execution_id,
                input_text,
                json.dumps(result),
                error,
                response_time_ms,
                user_id,
                platform
            ))
            
            # Update performance metrics
            success = error is None
            cur.execute("""
                INSERT INTO agent_performance (agent_name, total_executions, successful_executions, failed_executions, last_execution)
                VALUES (%s, 1, %s, %s, NOW())
                ON CONFLICT (agent_name) DO UPDATE SET
                    total_executions = agent_performance.total_executions + 1,
                    successful_executions = agent_performance.successful_executions + CASE WHEN %s THEN 1 ELSE 0 END,
                    failed_executions = agent_performance.failed_executions + CASE WHEN %s THEN 0 ELSE 1 END,
                    average_response_time = (
                        (agent_performance.average_response_time * agent_performance.total_executions + %s) /
                        (agent_performance.total_executions + 1)
                    ),
                    last_execution = NOW(),
                    success_rate = (
                        (agent_performance.successful_executions + CASE WHEN %s THEN 1 ELSE 0 END)::FLOAT /
                        (agent_performance.total_executions + 1)::FLOAT
                    ),
                    last_updated = NOW()
            """, (agent_name, 1 if success else 0, 0 if success else 1, success, success, response_time_ms, success))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Update cache
            self._update_performance_cache(agent_name)
            
        except Exception as e:
            logger.error(f"Error logging execution: {e}")
    
    def get_performance(self, agent_name: str) -> Optional[AgentPerformance]:
        """Get agent performance metrics"""
        # Check cache first
        if agent_name in self._performance_cache:
            return self._performance_cache[agent_name]
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT * FROM agent_performance WHERE agent_name = %s
            """, (agent_name,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                perf = AgentPerformance(
                    agent_name=row['agent_name'],
                    total_executions=row['total_executions'],
                    successful_executions=row['successful_executions'],
                    failed_executions=row['failed_executions'],
                    average_response_time=row['average_response_time'],
                    last_execution=row['last_execution'],
                    success_rate=row['success_rate'],
                    user_satisfaction=row['user_satisfaction'],
                    last_updated=row['last_updated']
                )
                self._performance_cache[agent_name] = perf
                return perf
            
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
        
        return None
    
    def store_memory(
        self,
        agent_name: str,
        memory_key: str,
        memory_value: Any,
        context: Optional[Dict] = None
    ):
        """Store agent memory"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO agent_memory (agent_name, memory_key, memory_value, context, last_accessed, access_count)
                VALUES (%s, %s, %s, %s, NOW(), 1)
                ON CONFLICT (agent_name, memory_key) DO UPDATE SET
                    memory_value = EXCLUDED.memory_value,
                    context = EXCLUDED.context,
                    last_accessed = NOW(),
                    access_count = agent_memory.access_count + 1
            """, (
                agent_name,
                memory_key,
                json.dumps(memory_value),
                json.dumps(context or {})
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def get_memory(self, agent_name: str, memory_key: str) -> Optional[Any]:
        """Retrieve agent memory"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                UPDATE agent_memory 
                SET last_accessed = NOW(), access_count = access_count + 1
                WHERE agent_name = %s AND memory_key = %s
                RETURNING memory_value
            """, (agent_name, memory_key))
            
            row = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
            
            if row:
                return json.loads(row['memory_value'])
            
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
        
        return None
    
    def add_feedback(
        self,
        agent_name: str,
        feedback_type: str,
        feedback_text: str,
        user_id: Optional[str] = None
    ):
        """Add user feedback"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO agent_feedback (agent_name, feedback_type, feedback_text, user_id)
                VALUES (%s, %s, %s, %s)
            """, (agent_name, feedback_type, feedback_text, user_id))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
    
    def get_all_performance(self) -> List[AgentPerformance]:
        """Get performance for all agents"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("SELECT * FROM agent_performance ORDER BY total_executions DESC")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            return [
                AgentPerformance(
                    agent_name=row['agent_name'],
                    total_executions=row['total_executions'],
                    successful_executions=row['successful_executions'],
                    failed_executions=row['failed_executions'],
                    average_response_time=row['average_response_time'],
                    last_execution=row['last_execution'],
                    success_rate=row['success_rate'],
                    user_satisfaction=row['user_satisfaction'],
                    last_updated=row['last_updated']
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error getting all performance: {e}")
            return []
    
    def _update_performance_cache(self, agent_name: str):
        """Update performance cache"""
        perf = self.get_performance(agent_name)
        if perf:
            self._performance_cache[agent_name] = perf

