# Ollama Integration - Setup Guide

## Quick Start

### 1. Start Ollama
Ollama is installed but not running. Start it with:

**Option A: From Start Menu**
- Press Windows key
- Type "Ollama"
- Click "Ollama" app

**Option B: From Command Line**
```bash
ollama serve
```

---

### 2. Check Available Models
```bash
python get_ollama_models.py
```

This will show all installed models and recommend the best one.

---

### 3. Refresh Game Hub
Once Ollama is running:
1. Restart Streamlit (Ctrl+C, then run `streamlit run dashboard.py`)
2. Go to Game Hub
3. The dropdown will now show your Ollama models!

---

## Recommended Models for Sports Analysis

### Best Options (in order):
1. **qwen2.5-coder** (if installed) - Excellent for technical analysis
2. **qwen2.5** - Great general purpose
3. **llama3.1** - Solid all-arounder
4. **deepseek-coder** - Good for data analysis

### If you don't have these, install one:
```bash
# Best for sports analysis
ollama pull qwen2.5:latest

# Or for larger context (if you have RAM)
ollama pull qwen2.5:14b
```

---

## What Changed in the UI

The dropdown now shows:
```
[AI Model Dropdown ▼]
├─ Ollama: qwen2.5:latest  ← Your local models (when running)
├─ Ollama: llama3.1:latest
├─ Ollama: ...
├─ Groq Cloud             ← Cloud options
└─ DeepSeek Cloud
```

**Default Selection:**
- If Ollama running: Best available local model (qwen2.5-coder > qwen2.5 > llama > first model)
- If Ollama not running: Falls back to "Local AI (Basic)"

---

## Benefits of Ollama Models

### vs Basic Local AI:
- ✅ **Much better analysis** - Advanced reasoning
- ✅ **Better predictions** - More accurate game insights
- ✅ **Faster** - Optimized inference
- ✅ **Free** - No API costs
- ✅ **Private** - Data stays local

### vs Cloud (Groq/DeepSeek):
- ✅ **No cost** - Unlimited usage
- ✅ **No rate limits** - No throttling
- ✅ **Works offline** - No internet needed
- ✅ **Privacy** - Your data never leaves your machine
- ⚠️ **Slower** - ~2-5 seconds vs cloud <1s
- ⚠️ **Needs RAM** - 8GB+ recommended

---

## How It Works

1. **Detection**: Game Hub checks `http://localhost:11434/api/tags` for models
2. **Auto-select**: Picks best available model
3. **Analysis**: Uses selected model for all AI predictions
4. **Fallback**: If Ollama stops, falls back to basic mode

---

## Troubleshooting

### "Ollama server not running"
```bash
# Check if it's running
tasklist | findstr ollama

# Start it
ollama serve

# Or open Ollama app from Start menu
```

### Models not showing in dropdown
1. Make sure Ollama is running
2. Verify models installed: `ollama list`
3. Restart Streamlit

### Slow performance
- Use smaller models (7B instead of 70B)
- Close other applications
- Upgrade RAM if possible

---

## Testing

**Run this to test:**
```bash
python get_ollama_models.py
```

**Should show:**
```
Available Ollama Models:
============================================================
  - qwen2.5:latest (4.7 GB)
  - llama3.1:latest (4.7 GB)

Total: 2 models

============================================================
RECOMMENDATIONS:
============================================================
BEST: qwen2.5:latest - Great general purpose
```

---

## Next Steps

1. **Start Ollama** (from Start menu or `ollama serve`)
2. **Restart Streamlit** (`Ctrl+C` then `streamlit run dashboard.py`)
3. **Refresh browser** (F5)
4. **Check dropdown** - should show your models!
5. **Test analysis** - Subscribe to a game and see AI predictions

The Game Hub will automatically use your best local model! 🚀
