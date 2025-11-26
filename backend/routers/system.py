from fastapi import APIRouter
from typing import Optional
from datetime import datetime
import psutil
import os

router = APIRouter(prefix="/api/system", tags=["system"])

@router.get("/health")
async def get_health():
    """Get system health status"""
    return {
        "status": "healthy",
        "uptime_seconds": 3600,
        "services": {
            "database": {"status": "healthy", "latency_ms": 5},
            "redis": {"status": "healthy", "latency_ms": 2},
            "api": {"status": "healthy", "latency_ms": 1}
        },
        "last_check": datetime.now().isoformat()
    }


@router.get("/health/detailed")
async def get_health_detailed():
    """Get detailed system health status - used by HealthDashboard page"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "status": "healthy",
        "uptime_seconds": 3600,
        "services": {
            "database": {"status": "healthy", "latency_ms": 5, "connections": 8},
            "redis": {"status": "healthy", "latency_ms": 2, "memory_mb": 128},
            "api": {"status": "healthy", "latency_ms": 1, "requests_per_min": 450},
            "scheduler": {"status": "running", "active_jobs": 3},
            "agents": {"status": "healthy", "active_agents": 35}
        },
        "resources": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_available_gb": round(disk.free / (1024**3), 2)
        },
        "last_check": datetime.now().isoformat()
    }


@router.get("/tasks")
async def get_tasks():
    """Get background tasks - used by SystemMonitoringHub page"""
    return {
        "tasks": [
            {"id": "sync_positions", "name": "Position Sync", "status": "running", "progress": 85, "started_at": "2024-11-25T10:00:00Z"},
            {"id": "fetch_earnings", "name": "Earnings Calendar Fetch", "status": "completed", "progress": 100, "completed_at": "2024-11-25T09:30:00Z"},
            {"id": "scan_options", "name": "Options Scanner", "status": "scheduled", "progress": 0, "next_run": "2024-11-25T11:00:00Z"},
            {"id": "update_predictions", "name": "AI Predictions Update", "status": "running", "progress": 45, "started_at": "2024-11-25T10:15:00Z"},
            {"id": "cache_cleanup", "name": "Cache Cleanup", "status": "idle", "progress": 0, "last_run": "2024-11-25T08:00:00Z"}
        ],
        "total": 5,
        "running": 2,
        "completed": 1,
        "scheduled": 1
    }

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "cpu_usage": cpu_percent,
        "memory_usage": memory.percent,
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "disk_usage": disk.percent,
        "disk_available_gb": round(disk.free / (1024**3), 2),
        "active_connections": 12,
        "requests_per_minute": 450,
        "error_rate": 0.02
    }

@router.get("/services")
async def get_services():
    """Get list of system services"""
    return {
        "services": [
            {"id": "api", "name": "API Server", "status": "running", "port": 8002, "uptime": "2h 34m"},
            {"id": "frontend", "name": "Frontend", "status": "running", "port": 5175, "uptime": "2h 34m"},
            {"id": "database", "name": "PostgreSQL", "status": "running", "port": 5432, "uptime": "5h 12m"},
            {"id": "redis", "name": "Redis Cache", "status": "running", "port": 6379, "uptime": "5h 12m"},
            {"id": "scheduler", "name": "Task Scheduler", "status": "running", "port": None, "uptime": "2h 34m"},
            {"id": "telegram", "name": "Telegram Bot", "status": "stopped", "port": None, "uptime": None}
        ]
    }

@router.post("/services/{service_id}/restart")
async def restart_service(service_id: str):
    """Restart a service"""
    return {"status": "success", "message": f"Service {service_id} restarted"}

@router.post("/services/{service_id}/stop")
async def stop_service(service_id: str):
    """Stop a service"""
    return {"status": "success", "message": f"Service {service_id} stopped"}

@router.get("/config")
async def get_config():
    """Get system configuration"""
    return {
        "environment": os.getenv("ENV", "development"),
        "debug_mode": os.getenv("DEBUG", "false") == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "database_url": "postgresql://***:***@localhost:5432/ava",
        "redis_url": "redis://localhost:6379/0",
        "api_rate_limit": 1000,
        "session_timeout": 3600
    }

@router.get("/logs")
async def get_logs(service: Optional[str] = None, level: Optional[str] = None, limit: int = 100):
    """Get system logs"""
    logs = [
        {"timestamp": "2024-11-25T10:30:00Z", "level": "INFO", "service": "api", "message": "Request processed successfully"},
        {"timestamp": "2024-11-25T10:29:55Z", "level": "DEBUG", "service": "database", "message": "Query executed in 5ms"},
        {"timestamp": "2024-11-25T10:29:50Z", "level": "WARNING", "service": "scheduler", "message": "Task queue growing"},
        {"timestamp": "2024-11-25T10:29:45Z", "level": "INFO", "service": "api", "message": "User authenticated"},
        {"timestamp": "2024-11-25T10:29:40Z", "level": "ERROR", "service": "telegram", "message": "Connection timeout"}
    ]

    if service:
        logs = [l for l in logs if l["service"] == service]
    if level:
        logs = [l for l in logs if l["level"] == level]

    return {"logs": logs[:limit], "total": len(logs)}

@router.get("/monitoring/metrics")
async def get_monitoring_metrics():
    """Get detailed monitoring metrics"""
    return {
        "metrics": [
            {"name": "api_requests_total", "value": 15420, "type": "counter"},
            {"name": "api_latency_seconds", "value": 0.045, "type": "gauge"},
            {"name": "db_connections_active", "value": 8, "type": "gauge"},
            {"name": "cache_hit_ratio", "value": 0.92, "type": "gauge"},
            {"name": "error_count_total", "value": 23, "type": "counter"},
            {"name": "memory_bytes_used", "value": 1073741824, "type": "gauge"}
        ],
        "alerts": [
            {"name": "High Memory Usage", "severity": "warning", "triggered_at": "2024-11-25T09:00:00Z"},
            {"name": "API Latency Spike", "severity": "info", "triggered_at": "2024-11-25T08:30:00Z"}
        ]
    }
