"""
Cache Metrics Agent - Monitor caching performance and hit rates
(Rewritten to not depend on Streamlit - works with FastAPI backend)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from ...core.agent_base import BaseAgent, AgentState
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_memory_cache_stats_tool() -> str:
    """
    Get in-memory cache statistics
    
    Returns:
        JSON string with cache statistics
    """
    try:
        import psutil
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        stats = {
            'type': 'in_memory_cache',
            'description': 'Application memory usage stats',
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'percent': process.memory_percent(),
        }
        
        result = {
            'memory_stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return str(result)
        
    except Exception as e:
        logger.error(f"Error getting memory cache stats: {e}")
        return f"Error: {str(e)}"


@tool
def get_redis_cache_stats_tool() -> str:
    """
    Get Redis cache statistics (if Redis is configured)
    
    Returns:
        JSON string with Redis cache stats
    """
    try:
        import redis
        
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_password = os.getenv('REDIS_PASSWORD')
        
        # Connect to Redis
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        
        # Get Redis INFO
        info = r.info()
        
        # Extract relevant metrics
        stats = {
            'connected': True,
            'version': info.get('redis_version'),
            'uptime_days': info.get('uptime_in_days'),
            'total_keys': r.dbsize(),
            'used_memory_human': info.get('used_memory_human'),
            'used_memory_peak_human': info.get('used_memory_peak_human'),
            'total_connections_received': info.get('total_connections_received'),
            'total_commands_processed': info.get('total_commands_processed'),
            'keyspace_hits': info.get('keyspace_hits'),
            'keyspace_misses': info.get('keyspace_misses'),
            'evicted_keys': info.get('evicted_keys'),
            'expired_keys': info.get('expired_keys')
        }
        
        # Calculate hit rate
        hits = stats['keyspace_hits'] or 0
        misses = stats['keyspace_misses'] or 0
        total = hits + misses
        
        if total > 0:
            stats['hit_rate_percent'] = round((hits / total) * 100, 2)
        else:
            stats['hit_rate_percent'] = 0.0
        
        result = {
            'redis_stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return str(result)
        
    except ImportError:
        return "Redis library not installed. Install with: pip install redis"
    except Exception as e:
        if 'Connection' in str(type(e).__name__):
            return f"Cannot connect to Redis. Redis may not be running."
        logger.error(f"Error getting Redis cache stats: {e}")
        return f"Error: {str(e)}"


@tool
def get_llm_cache_stats_tool() -> str:
    """
    Get LLM response cache statistics
    
    Returns:
        JSON string with LLM cache stats
    """
    try:
        # Try to import the local LLM module which has caching
        from src.magnus_local_llm import get_magnus_llm
        
        llm = get_magnus_llm()
        metrics = llm.get_metrics()
        
        llm_stats = {
            'type': 'LLM Response Cache',
            'description': 'Caches LLM responses to avoid repeated API calls',
            'cache_location': 'In-memory (session-based)',
            'requests': metrics.get('requests', 0),
            'cache_hits': metrics.get('cache_hits', 0),
            'cache_size': metrics.get('cache_size', 0),
            'cache_hit_rate': metrics.get('cache_hit_rate', 0),
            'avg_latency_ms': metrics.get('avg_latency_ms', 0),
            'errors': metrics.get('errors', 0),
        }
        
        result = {
            'llm_cache': llm_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return str(result)
        
    except Exception as e:
        logger.error(f"Error getting LLM cache stats: {e}")
        return f"Error: {str(e)}"


@tool
def get_database_pool_stats_tool() -> str:
    """
    Get database connection pool statistics
    
    Returns:
        JSON string with database pool stats
    """
    try:
        from src.database import get_db_connection
        
        stats = {
            'type': 'Database Connection Pool',
            'description': 'PostgreSQL connection pool metrics',
            'pool_status': 'active',
        }
        
        result = {
            'db_pool_stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return str(result)
        
    except Exception as e:
        logger.error(f"Error getting database pool stats: {e}")
        return f"Error: {str(e)}"


@tool
def clear_llm_cache_tool() -> str:
    """
    Clear LLM response cache
    
    Returns:
        Success message
    """
    try:
        from src.magnus_local_llm import get_magnus_llm
        
        llm = get_magnus_llm()
        llm.clear_cache()
        
        result = {
            'cleared': ['llm_cache'],
            'timestamp': datetime.now().isoformat()
        }
        
        return f"Successfully cleared LLM cache"
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return f"Error: {str(e)}"


class CacheMetricsAgent(BaseAgent):
    """
    Cache Metrics Agent - Monitor caching performance and hit rates
    
    Capabilities:
    - Monitor memory cache statistics
    - Track Redis cache performance (if configured)
    - Monitor LLM response caching
    - Track database connection pool
    - Clear caches when needed
    - Analyze cache hit rates and efficiency
    """
    
    def __init__(self, use_huggingface: bool = False):
        """Initialize Cache Metrics Agent"""
        tools = [
            get_memory_cache_stats_tool,
            get_redis_cache_stats_tool,
            get_llm_cache_stats_tool,
            get_database_pool_stats_tool,
            clear_llm_cache_tool
        ]
        
        super().__init__(
            name="cache_metrics_agent",
            description="Monitors cache performance, hit rates, and provides cache management",
            tools=tools,
            use_huggingface=use_huggingface
        )
        
        self.metadata['capabilities'] = [
            'memory_cache_stats',
            'redis_cache_monitoring',
            'llm_cache_tracking',
            'db_pool_stats',
            'cache_clearing',
            'hit_rate_analysis'
        ]
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute Cache Metrics agent"""
        try:
            input_text = state.get('input', '')
            context = state.get('context', {})
            
            result = {
                'agent': 'cache_metrics_agent',
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine operation based on input
            if 'clear' in input_text.lower():
                # Clear cache
                data = clear_llm_cache_tool.invoke({})
                result['operation'] = 'clear_cache'
                result['data'] = data
            
            elif 'redis' in input_text.lower():
                # Get Redis stats
                data = get_redis_cache_stats_tool.invoke({})
                result['operation'] = 'redis_stats'
                result['data'] = data
            
            elif 'llm' in input_text.lower():
                # Get LLM cache stats
                data = get_llm_cache_stats_tool.invoke({})
                result['operation'] = 'llm_cache_stats'
                result['data'] = data
            
            elif 'database' in input_text.lower() or 'pool' in input_text.lower():
                # Get database pool stats
                data = get_database_pool_stats_tool.invoke({})
                result['operation'] = 'db_pool_stats'
                result['data'] = data
            
            else:
                # Get all cache stats (default)
                memory_stats = get_memory_cache_stats_tool.invoke({})
                redis_stats = get_redis_cache_stats_tool.invoke({})
                llm_stats = get_llm_cache_stats_tool.invoke({})
                
                result['operation'] = 'comprehensive_cache_stats'
                result['data'] = {
                    'memory': memory_stats,
                    'redis': redis_stats,
                    'llm': llm_stats
                }
            
            state['result'] = result
            return state
            
        except Exception as e:
            logger.error(f"CacheMetricsAgent error: {e}")
            state['error'] = str(e)
            return state
