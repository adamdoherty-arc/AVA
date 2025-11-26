# Quick Start Guide - AI Research API

Get the Research API running in 5 minutes!

## Prerequisites

- Python 3.9+
- Redis (for caching)
- Ollama (for local LLM) OR OpenAI API key

## Step 1: Install Dependencies

```bash
cd /c/Code/WheelStrategy
pip install -r src/api/requirements.txt
```

This installs:
- FastAPI, uvicorn
- Redis client
- CrewAI, LangChain
- yfinance, pandas
- All required dependencies

## Step 2: Start Redis

### Option A: Docker (Easiest)
```bash
docker run -d -p 6379:6379 redis:latest
```

### Option B: Native Installation

**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**Verify Redis:**
```bash
redis-cli ping
# Should return: PONG
```

## Step 3: Install LLM (Choose One)

### Option A: Ollama (Recommended - Free & Local)

1. **Install Ollama:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

2. **Pull Model:**
```bash
ollama pull llama3.2
```

3. **Verify:**
```bash
ollama run llama3.2
# Type "hello" and press Enter
# Press Ctrl+D to exit
```

### Option B: OpenAI GPT

1. **Get API Key:** https://platform.openai.com/api-keys

2. **Set Environment Variable:**
```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows
set OPENAI_API_KEY=sk-...
```

3. **Configure:**
Create `.env` file:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

## Step 4: Start the API

### Method 1: Using Startup Script (Recommended)

```bash
cd /c/Code/WheelStrategy
python src/api/start_api.py
```

This will:
- Check all prerequisites
- Verify Redis connection
- Verify LLM availability
- Start the API with auto-reload

### Method 2: Direct uvicorn

```bash
cd /c/Code/WheelStrategy
uvicorn src.api.research_endpoints:app --reload --host 0.0.0.0 --port 8000
```

### Success!

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Step 5: Test the API

### Interactive Docs
Open browser: **http://localhost:8000/docs**

Click "Try it out" on any endpoint!

### cURL Test
```bash
# Health check
curl http://localhost:8000/health

# Get research for AAPL
curl http://localhost:8000/api/research/AAPL | jq

# Check cache status
curl http://localhost:8000/api/research/AAPL/status | jq
```

### Python Test
```bash
cd /c/Code/WheelStrategy
python src/api/example_usage.py
```

## Step 6: Use the API

### Python Example
```python
import requests

# Get research
response = requests.get('http://localhost:8000/api/research/AAPL')
data = response.json()

print(f"Rating: {data['overall_rating']}/5.0")
print(f"Action: {data['analysis']['recommendation']['action']}")
print(f"Summary: {data['quick_summary']}")
```

### JavaScript Example
```javascript
const getResearch = async (symbol) => {
  const response = await fetch(`http://localhost:8000/api/research/${symbol}`);
  const data = await response.json();
  return data;
};

const report = await getResearch('AAPL');
console.log(report.overall_rating);
```

## Common Issues

### Issue: "Redis connection failed"
**Solution:**
```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
docker run -d -p 6379:6379 redis:latest
```

### Issue: "Ollama not found"
**Solution:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Or use OpenAI instead
export OPENAI_API_KEY="sk-..."
```

### Issue: "Port 8000 already in use"
**Solution:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn src.api.research_endpoints:app --port 8001
```

### Issue: "Module not found"
**Solution:**
```bash
# Reinstall dependencies
pip install -r src/api/requirements.txt
```

## What's Next?

### 1. Read Full Documentation
See: `src/api/README.md` for complete API reference

### 2. Explore Examples
Run: `python src/api/example_usage.py`

### 3. Customize
- Adjust rate limits in `src/api/research_endpoints.py`
- Change cache TTL in `RedisCache` initialization
- Switch LLM models in orchestrator

### 4. Deploy to Production
- See deployment section in `src/api/README.md`
- Use Docker Compose for easy deployment
- Add authentication for production use

## API Endpoints Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research/{symbol}` | GET | Get research (cached) |
| `/api/research/{symbol}/refresh` | GET | Force fresh analysis |
| `/api/research/{symbol}/status` | GET | Check cache status |
| `/api/research/{symbol}/cache` | DELETE | Clear cache |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |

## Performance Tips

1. **Use Cache**: Don't force refresh unnecessarily
2. **Selective Sections**: Only request needed analysis
   ```
   GET /api/research/AAPL?include_fundamental=true&include_options=false
   ```
3. **Batch Processing**: Analyze multiple symbols during off-hours
4. **Monitor Rate Limits**: Stay under 10 req/min (default)

## Configuration Options

Edit `.env` file:
```bash
# LLM
LLM_PROVIDER=ollama      # or 'openai'
LLM_MODEL=llama3.2       # or 'gpt-4'

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60

# API Keys (optional)
OPENAI_API_KEY=sk-...
ALPHA_VANTAGE_API_KEY=your_key
```

## Support

For detailed documentation:
- API Reference: `src/api/README.md`
- Implementation Details: `API_IMPLEMENTATION_SUMMARY.md`
- Code Examples: `src/api/example_usage.py`

---

**That's it! You're ready to use the AI Research API!**

Happy Trading! 📈
