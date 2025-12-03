"""
AVA Telegram Webhook Server
============================

FastAPI-based webhook server for receiving Telegram updates via HTTPS.
Replaces long-polling with webhook mode for better performance and lower latency.

Features:
- HTTPS webhook endpoint with signature validation
- Message enqueueing to Redis Queue
- Health check and metrics endpoints
- Request logging with correlation IDs
- Graceful shutdown handling

Usage:
    uvicorn src.ava.webhook_server:app --host 0.0.0.0 --port 8000

Requirements:
    - Public HTTPS endpoint (use Ngrok for dev, Caddy/Nginx for prod)
    - TELEGRAM_WEBHOOK_URL environment variable set
    - Redis server running
"""

import os
import sys
import hashlib
import hmac
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from rq import Queue
from dotenv import load_dotenv
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AVA Telegram Webhook Server",
    description="Webhook endpoint for AVA Telegram Bot",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_WEBHOOKS = os.getenv("TELEGRAM_USE_WEBHOOKS", "true").lower() == "true"

# Redis connection
redis_client: Optional[redis.Redis] = None

# Queue names
QUEUE_HIGH = "ava:messages:high"
QUEUE_NORMAL = "ava:messages:normal"
QUEUE_LOW = "ava:messages:low"


# Pydantic models
class TelegramUpdate(BaseModel):
    """Telegram Update object"""
    update_id: int
    message: Optional[Dict[str, Any]] = None
    edited_message: Optional[Dict[str, Any]] = None
    callback_query: Optional[Dict[str, Any]] = None
    inline_query: Optional[Dict[str, Any]] = None


class WebhookResponse(BaseModel):
    """Response model for webhook endpoint"""
    ok: bool
    message_id: Optional[str] = None
    queued_at: Optional[str] = None
    error: Optional[str] = None


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global redis_client

    logger.info("Starting AVA Telegram Webhook Server...")

    # Connect to Redis
    try:
        redis_client = await redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
        await redis_client.ping()
        logger.info(f"✅ Connected to Redis: {REDIS_URL}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        redis_client = None

    # Check if webhook mode is enabled
    if not USE_WEBHOOKS:
        logger.warning("⚠️ Webhook mode is disabled in config (USE_WEBHOOKS=false)")

    logger.info("✅ Webhook server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    global redis_client

    logger.info("Shutting down AVA Telegram Webhook Server...")

    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

    logger.info("✅ Shutdown complete")


# Dependency injection
async def get_redis() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")
    return redis_client


def verify_telegram_signature(
    body: bytes,
    signature: Optional[str]
) -> bool:
    """
    Verify Telegram webhook signature

    Args:
        body: Raw request body
        signature: X-Telegram-Bot-Api-Secret-Token header

    Returns:
        True if signature is valid
    """
    if not TELEGRAM_WEBHOOK_SECRET:
        logger.warning("TELEGRAM_WEBHOOK_SECRET not set, skipping validation")
        return True

    if not signature:
        return False

    # Telegram uses the secret token directly, not HMAC
    return signature == TELEGRAM_WEBHOOK_SECRET


def determine_priority(update: TelegramUpdate) -> str:
    """
    Determine message priority based on content

    Args:
        update: Telegram update object

    Returns:
        Queue name (high/normal/low)
    """
    # High priority: Commands
    if update.message and update.message.get("text", "").startswith("/"):
        command = update.message["text"].split()[0].lower()
        if command in ["/start", "/help", "/status", "/portfolio"]:
            return QUEUE_HIGH

    # High priority: Callback queries (button presses)
    if update.callback_query:
        return QUEUE_HIGH

    # Normal priority: Text and voice messages
    if update.message:
        return QUEUE_NORMAL

    # Low priority: Everything else
    return QUEUE_LOW


def create_message_payload(update: TelegramUpdate) -> Dict[str, Any]:
    """
    Create standardized message payload for queue

    Args:
        update: Telegram update object

    Returns:
        Message payload dictionary
    """
    message = update.message or update.edited_message or {}
    callback_query = update.callback_query or {}

    # Extract chat and user IDs
    chat_id = None
    user_id = None
    message_type = "unknown"
    content = None

    if message:
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")

        if message.get("text"):
            message_type = "text"
            content = message["text"]
        elif message.get("voice"):
            message_type = "voice"
            content = message["voice"]["file_id"]
        elif message.get("photo"):
            message_type = "photo"
            content = message["photo"][-1]["file_id"]

    if callback_query:
        chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")
        message_type = "callback_query"
        content = callback_query.get("data")

    return {
        "message_id": f"msg_{update.update_id}",
        "update_id": update.update_id,
        "chat_id": chat_id,
        "user_id": user_id,
        "message_type": message_type,
        "content": content,
        "raw_update": update.{},
        "timestamp": datetime.utcnow().isoformat(),
        "retry_count": 0
    }


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AVA Telegram Webhook Server",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check(redis_conn: redis.Redis = Depends(get_redis)):
    """
    Health check endpoint

    Returns:
        Health status of all components
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    # Check Redis
    try:
        await redis_conn.ping()
        health["components"]["redis"] = {"status": "up"}
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["redis"] = {"status": "down", "error": str(e)}

    # Check queue lengths
    try:
        high_len = await redis_conn.llen(QUEUE_HIGH)
        normal_len = await redis_conn.llen(QUEUE_NORMAL)
        low_len = await redis_conn.llen(QUEUE_LOW)

        health["components"]["queues"] = {
            "status": "up",
            "high": high_len,
            "normal": normal_len,
            "low": low_len,
            "total": high_len + normal_len + low_len
        }
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["queues"] = {"status": "error", "error": str(e)}

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)


@app.post("/webhook/{bot_token}")
async def telegram_webhook(
    bot_token: str,
    request: Request,
    redis_conn: redis.Redis = Depends(get_redis),
    x_telegram_bot_api_secret_token: Optional[str] = Header(None)
):
    """
    Telegram webhook endpoint

    Args:
        bot_token: Bot token from URL path
        request: FastAPI request object
        redis_conn: Redis connection
        x_telegram_bot_api_secret_token: Telegram signature header

    Returns:
        Webhook response
    """
    # Validate bot token
    if bot_token != TELEGRAM_BOT_TOKEN:
        logger.warning(f"Invalid bot token received: {bot_token[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid bot token")

    # Read and verify request body
    body = await request.body()

    if not verify_telegram_signature(body, x_telegram_bot_api_secret_token):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse update
    try:
        update_data = await request.json()
        update = TelegramUpdate(**update_data)
    except Exception as e:
        logger.error(f"Failed to parse update: {e}")
        raise HTTPException(status_code=400, detail="Invalid update format")

    # Create message payload
    payload = create_message_payload(update)

    # Determine priority and queue
    queue_name = determine_priority(update)

    # Enqueue message
    try:
        await redis_conn.rpush(queue_name, json.dumps(payload))

        logger.info(
            f"Message enqueued: {payload['message_id']} "
            f"(type={payload['message_type']}, queue={queue_name})"
        )

        return WebhookResponse(
            ok=True,
            message_id=payload["message_id"],
            queued_at=payload["timestamp"]
        )

    except Exception as e:
        logger.error(f"Failed to enqueue message: {e}")
        raise HTTPException(status_code=500, detail="Failed to enqueue message")


@app.get("/metrics")
async def metrics(redis_conn: redis.Redis = Depends(get_redis)):
    """
    Prometheus-compatible metrics endpoint

    Returns:
        Text-based metrics
    """
    try:
        # Queue lengths
        high_len = await redis_conn.llen(QUEUE_HIGH)
        normal_len = await redis_conn.llen(QUEUE_NORMAL)
        low_len = await redis_conn.llen(QUEUE_LOW)

        # Build Prometheus metrics
        metrics_text = f"""# HELP ava_queue_length Current length of message queues
# TYPE ava_queue_length gauge
ava_queue_length{{priority="high"}} {high_len}
ava_queue_length{{priority="normal"}} {normal_len}
ava_queue_length{{priority="low"}} {low_len}

# HELP ava_webhook_server_up Webhook server status (1=up, 0=down)
# TYPE ava_webhook_server_up gauge
ava_webhook_server_up 1
"""

        return metrics_text

    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return "# Error generating metrics\n"


@app.post("/admin/clear-queues")
async def clear_queues(redis_conn: redis.Redis = Depends(get_redis)):
    """
    Admin endpoint to clear all queues

    WARNING: This deletes all pending messages!

    Returns:
        Number of messages deleted
    """
    try:
        high_deleted = await redis_conn.delete(QUEUE_HIGH)
        normal_deleted = await redis_conn.delete(QUEUE_NORMAL)
        low_deleted = await redis_conn.delete(QUEUE_LOW)

        total_deleted = high_deleted + normal_deleted + low_deleted

        logger.warning(f"Admin action: Cleared all queues ({total_deleted} messages)")

        return {
            "ok": True,
            "messages_deleted": total_deleted,
            "queues_cleared": ["high", "normal", "low"]
        }

    except Exception as e:
        logger.error(f"Failed to clear queues: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear queues")


# Run server (for development)
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting webhook server in development mode...")
    logger.info(f"Listening on http://0.0.0.0:8000")
    logger.info(f"Webhook URL: POST /webhook/{TELEGRAM_BOT_TOKEN}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
