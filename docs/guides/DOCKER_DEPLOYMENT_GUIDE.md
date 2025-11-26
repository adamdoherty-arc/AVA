# Magnus Trading Platform - Docker Deployment Guide

Complete guide for deploying Magnus using Docker and Docker Compose.

---

## Quick Start

```bash
# 1. Clone repository (if not already done)
git clone https://github.com/your-org/magnus.git
cd magnus

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your API keys
nano .env

# 4. Build and start all services
docker-compose up -d

# 5. View logs
docker-compose logs -f app

# 6. Access the application
# Web UI: http://localhost:8501
# Flower (Celery monitoring): http://localhost:5555
```

---

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- At least 4GB RAM available
- At least 10GB disk space

### Install Docker

**Windows:**
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
winget install Docker.DockerDesktop
```

**Mac:**
```bash
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

---

## Architecture

The Docker Compose setup includes 7 services:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Magnus Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Nginx     │  │   Streamlit  │  │    Flower    │          │
│  │ Reverse Proxy│  │   Frontend   │  │  Monitoring  │          │
│  │   :80/443    │  │    :8501     │  │    :5555     │          │
│  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘          │
│          │                │                  │                   │
│          └────────────────┴──────────────────┘                   │
│                           │                                      │
│     ┌─────────────────────┴─────────────────────────┐           │
│     │                                                 │           │
│  ┌──▼──────────┐  ┌──────────────┐  ┌──────────────┐│           │
│  │   Celery    │  │    Celery    │  │    Redis     ││           │
│  │   Worker    │  │     Beat     │  │    Cache &   ││           │
│  │             │  │   Scheduler  │  │    Broker    ││           │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘│           │
│         │                │                  │         │           │
│         └────────────────┴──────────────────┘         │           │
│                          │                            │           │
│                   ┌──────▼────────┐                   │           │
│                   │  PostgreSQL   │                   │           │
│                   │   Database    │                   │           │
│                   │     :5432     │                   │           │
│                   └───────────────┘                   │           │
└─────────────────────────────────────────────────────────────────┘
```

### Service Descriptions

| Service | Purpose | Port | Required |
|---------|---------|------|----------|
| **app** | Streamlit web application | 8501 | ✅ Yes |
| **postgres** | PostgreSQL database | 5432 | ✅ Yes |
| **redis** | Cache & message broker | 6379 | ✅ Yes |
| **celery-worker** | Background task processing | - | ✅ Yes |
| **celery-beat** | Scheduled task scheduling | - | ✅ Yes |
| **flower** | Celery monitoring UI | 5555 | ⚠️ Recommended |
| **nginx** | Reverse proxy (production) | 80/443 | ❌ Optional |

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database
POSTGRES_DB=magnus
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_PORT=5432

# Redis
REDIS_PASSWORD=your_redis_password_here
REDIS_PORT=6379

# Application
APP_PORT=8501
FLOWER_PORT=5555

# API Keys
ROBINHOOD_USERNAME=your_robinhood_username
ROBINHOOD_PASSWORD=your_robinhood_password
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=...
KALSHI_API_KEY=...
KALSHI_API_SECRET=...

# Optional
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=...
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
runOnSave = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#667eea"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

---

## Deployment Steps

### 1. Development Deployment

For local development with hot-reloading:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Restart specific service
docker-compose restart app

# Stop all services
docker-compose down
```

### 2. Production Deployment

For production with Nginx reverse proxy:

```bash
# Build production images
docker-compose build --no-cache

# Start with production profile
docker-compose --profile production up -d

# View all running containers
docker ps

# Check health status
docker-compose ps
```

### 3. Scaling Workers

Scale Celery workers for high load:

```bash
# Scale to 4 worker instances
docker-compose up -d --scale celery-worker=4

# Monitor workers in Flower
open http://localhost:5555
```

---

## Database Management

### Initialize Database

```bash
# Run migrations (first time setup)
docker-compose exec app python -c "from src.database.init_db import initialize; initialize()"

# Apply performance indexes
docker-compose exec postgres psql -U postgres -d magnus -f /docker-entrypoint-initdb.d/02-indexes.sql

# Verify tables created
docker-compose exec postgres psql -U postgres -d magnus -c "\dt"
```

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres magnus > backup_$(date +%Y%m%d).sql

# Restore from backup
docker-compose exec -T postgres psql -U postgres magnus < backup_20250121.sql
```

### Access Database Shell

```bash
# PostgreSQL shell
docker-compose exec postgres psql -U postgres -d magnus

# Redis CLI
docker-compose exec redis redis-cli -a your_redis_password_here
```

---

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f celery-worker

# Last 100 lines
docker-compose logs --tail=100 app

# Follow new logs only
docker-compose logs -f --since 5m app
```

### Health Checks

```bash
# Check all service health
docker-compose ps

# Test application endpoint
curl http://localhost:8501/_stcore/health

# Check database connection
docker-compose exec postgres pg_isready -U postgres

# Check Redis connection
docker-compose exec redis redis-cli -a your_redis_password_here ping
```

### Celery Monitoring (Flower)

Access Flower UI at: http://localhost:5555

Features:
- Active tasks
- Worker status
- Task history
- Success/failure rates
- Task routing

---

## Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild images
docker-compose build --no-cache app

# Restart with zero downtime
docker-compose up -d --no-deps --build app
```

### Clear Caches

```bash
# Clear Redis cache
docker-compose exec redis redis-cli -a your_redis_password_here FLUSHALL

# Clear Docker build cache
docker builder prune -a

# Remove unused volumes
docker volume prune
```

### Resource Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove everything including images
docker-compose down -v --rmi all
```

---

## Troubleshooting

### App Won't Start

```bash
# Check logs for errors
docker-compose logs app

# Common issues:
# 1. Database not ready - wait 30s and retry
# 2. Missing API keys in .env file
# 3. Port 8501 already in use
```

### Database Connection Failed

```bash
# Verify database is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres pg_isready -U postgres

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d postgres
```

### Celery Tasks Not Running

```bash
# Check worker status
docker-compose logs celery-worker

# Check Redis connection
docker-compose exec redis redis-cli -a password ping

# Restart workers
docker-compose restart celery-worker celery-beat

# View tasks in Flower
open http://localhost:5555
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory: 8GB

# Or limit service memory in docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 2G
```

### Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Create required directories
mkdir -p logs data models

# Fix Docker socket permissions (Linux)
sudo chmod 666 /var/run/docker.sock
```

---

## Performance Tuning

### Optimize PostgreSQL

Add to `docker-compose.yml` postgres service:

```yaml
command:
  - postgres
  - -c
  - shared_buffers=256MB
  - -c
  - max_connections=200
  - -c
  - work_mem=16MB
  - -c
  - maintenance_work_mem=64MB
```

### Optimize Redis

Add to `docker-compose.yml` redis service:

```yaml
command: >
  redis-server
  --appendonly yes
  --requirepass ${REDIS_PASSWORD}
  --maxmemory 1gb
  --maxmemory-policy allkeys-lru
```

### Optimize Celery

Adjust worker concurrency:

```yaml
# In celery-worker service:
command: celery -A src.services.celery_app worker --loglevel=info --concurrency=8 --max-tasks-per-child=1000
```

---

## Security Best Practices

### 1. Change Default Passwords

```bash
# Generate secure passwords
openssl rand -base64 32

# Update .env file with strong passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
```

### 2. Use Docker Secrets (Production)

```yaml
# docker-compose.yml
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

### 3. Enable HTTPS (Nginx)

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/nginx.key \
  -out nginx/ssl/nginx.crt
```

### 4. Restrict Network Access

```yaml
# Expose only required ports
# Remove postgres/redis port mappings in production
```

---

## Useful Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose stop

# Restart service
docker-compose restart app

# View logs
docker-compose logs -f app

# Execute command in container
docker-compose exec app python manage.py

# Access container shell
docker-compose exec app bash

# Remove everything
docker-compose down -v --rmi all

# Check resource usage
docker stats

# Inspect container
docker inspect magnus-app

# View network
docker network ls
docker network inspect magnus_magnus-network

# Prune unused resources
docker system prune -a
```

---

## Production Checklist

Before deploying to production:

- [ ] Set strong passwords in `.env`
- [ ] Enable HTTPS with SSL certificates
- [ ] Set up automated backups (database + volumes)
- [ ] Configure log rotation
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure firewall rules
- [ ] Enable Docker health checks
- [ ] Set resource limits (CPU/memory)
- [ ] Configure auto-restart policies
- [ ] Set up CI/CD pipeline
- [ ] Test disaster recovery procedures
- [ ] Document deployment process
- [ ] Set up alerting (PagerDuty/Slack)

---

## Support & Resources

- **Docker Documentation:** https://docs.docker.com/
- **Docker Compose Reference:** https://docs.docker.com/compose/
- **PostgreSQL Docker:** https://hub.docker.com/_/postgres
- **Redis Docker:** https://hub.docker.com/_/redis
- **Celery Documentation:** https://docs.celeryq.dev/
- **Flower Documentation:** https://flower.readthedocs.io/

---

**Magnus Trading Platform** • Docker Deployment Guide v1.0
