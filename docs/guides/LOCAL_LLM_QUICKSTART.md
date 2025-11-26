# Magnus Local LLM Quick Start Guide
## Get AVA Running on Your RTX 4090 in 15 Minutes

**Last Updated:** 2025-01-20

---

## Why Local LLM?

✅ **Privacy:** All data stays on your machine
✅ **Cost:** Save $400-500/month on API costs
✅ **Speed:** No network latency, instant responses
✅ **Control:** Full control over models and behavior
✅ **Reliability:** No API rate limits or downtime

---

## Step 1: Install Ollama (5 minutes)

1. Download Ollama for Windows:
   ```
   https://ollama.ai/download/windows
   ```

2. Run the installer (double-click the `.exe`)

3. Verify installation in PowerShell:
   ```powershell
   ollama --version
   ```

   You should see something like: `ollama version 0.1.x`

---

## Step 2: Download Models (10-30 minutes)

Run the provided installation script:

```bash
install_local_llm.bat
```

Or manually download models:

```powershell
# Primary model (recommended)
ollama pull qwen2.5:32b-instruct-q4_K_M

# Fast model (optional but recommended)
ollama pull qwen2.5:14b-instruct-q4_K_M

# Complex model (optional, for advanced analysis)
ollama pull llama3.3:70b-instruct-q4_K_M
```

**Download sizes:**
- Qwen 2.5 32B: ~20GB
- Qwen 2.5 14B: ~9GB
- Llama 3.3 70B: ~40GB

---

## Step 3: Test the Installation

Run the test script:

```bash
python test_local_llm.py
```

You should see:
- ✓ Model availability check
- ✓ Simple query test
- ✓ Trading analysis test
- ✓ Performance benchmark

---

## Step 4: Start Using Local LLM

### Option A: Use in Python Code

```python
from src.magnus_local_llm import get_magnus_llm, TaskComplexity

# Get the LLM service
llm = get_magnus_llm()

# Simple query
response = llm.query("Explain covered calls")

# Trading analysis
analysis = llm.analyze_trade(
    symbol="AAPL",
    analysis_type="technical",
    context={"price": 185.50, "volume": 50_000_000}
)

print(analysis)
```

### Option B: Use in Dashboard

The Magnus dashboard will automatically use local LLM when available!

Just start the dashboard normally:

```bash
streamlit run dashboard.py
```

All AI features (AVA chatbot, trade analysis, etc.) will use your local models.

---

## Performance Expectations

### Qwen 2.5 32B (Primary Model)
- **Speed:** 40-50 tokens/second
- **Quality:** Excellent for trading analysis
- **VRAM Usage:** ~20GB
- **Best for:** Most trading tasks, research, strategy

### Qwen 2.5 14B (Fast Model)
- **Speed:** 80-100 tokens/second
- **Quality:** Good for simple queries
- **VRAM Usage:** ~9GB
- **Best for:** Chat, quick questions, price checks

### Llama 3.3 70B (Complex Model)
- **Speed:** 8-11 tokens/second
- **Quality:** Best reasoning and analysis
- **VRAM Usage:** 14GB + 25GB RAM
- **Best for:** Deep research, complex modeling

---

## Troubleshooting

### "ollama: command not found"
- Restart your terminal after installing Ollama
- Or add Ollama to PATH manually

### "Model not found"
- Make sure you pulled the models: `ollama list`
- Re-run: `ollama pull qwen2.5:32b-instruct-q4_K_M`

### Slow performance
- Close other GPU-intensive applications
- Check GPU usage: `nvidia-smi`
- Ensure latest NVIDIA drivers are installed
- Try the 14B model for faster responses

### Out of memory errors
- Close other applications
- Use only one model at a time
- Start with the 14B model
- Don't run 70B model if you have < 64GB RAM

---

## Advanced Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Magnus LLM Configuration
MAGNUS_DEFAULT_MODEL=balanced  # fast, balanced, or complex
MAGNUS_ENABLE_CACHING=true
MAGNUS_TEMPERATURE=0.7
```

### Model Selection Strategy

The service automatically selects models based on task complexity:

| Task Type | Model Used | Reasoning |
|-----------|------------|-----------|
| Chat, simple queries | Qwen 14B | Speed |
| Trade analysis, research | Qwen 32B | Balance |
| Deep research, modeling | Llama 70B | Quality |

You can override this with the `complexity` parameter:

```python
response = llm.query(
    "Complex financial analysis...",
    complexity=TaskComplexity.COMPLEX  # Force 70B model
)
```

---

## Monitoring & Metrics

Check performance metrics:

```python
llm = get_magnus_llm()
metrics = llm.get_metrics()

print(f"Total requests: {metrics['requests']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
print(f"Avg latency: {metrics['avg_latency_ms']:.0f}ms")
```

Monitor GPU usage in real-time:

```bash
# Windows
nvidia-smi -l 1

# Or use GPU-Z for detailed monitoring
```

---

## Switching Back to Cloud APIs

If you want to use cloud APIs instead:

1. Set `MAGNUS_USE_LOCAL_LLM=false` in `.env`
2. Or modify the orchestrator to use OpenAI/Anthropic

The system is designed to support both local and cloud models!

---

## Next Steps

1. ✅ Install Ollama and models
2. ✅ Run test script
3. ✅ Test with simple queries
4. ✅ Integrate with your trading workflow
5. ✅ Monitor performance and adjust

For detailed integration plans, see: [LOCAL_LLM_INTEGRATION_PLAN.md](LOCAL_LLM_INTEGRATION_PLAN.md)

---

## Support & Resources

- **Ollama Docs:** https://ollama.ai/docs
- **Qwen 2.5 Model Card:** https://huggingface.co/Qwen/Qwen2.5-32B
- **Llama 3.3:** https://ai.meta.com/blog/llama-3-3/
- **Magnus Docs:** See `features/` directory

**Need help?** Check the troubleshooting section or open an issue on GitHub.

---

**Happy Trading with Local AI! 🚀📈**
