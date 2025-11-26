"""
Get available Ollama models
"""
import requests
import json

try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])

        print("Available Ollama Models:")
        print("="*60)
        for model in models:
            name = model.get('name', 'Unknown')
            size = model.get('size', 0) / (1024**3)  # Convert to GB
            print(f"  - {name} ({size:.1f} GB)")

        print(f"\nTotal: {len(models)} models")

        # Recommend best model
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)

        model_names = [m.get('name', '') for m in models]

        # Best models in order of preference
        if any('qwen2.5-coder' in m for m in model_names):
            best = [m for m in model_names if 'qwen2.5-coder' in m][0]
            print(f"BEST: {best} - Excellent for code and analysis")
        elif any('qwen2.5' in m for m in model_names):
            best = [m for m in model_names if 'qwen2.5' in m][0]
            print(f"BEST: {best} - Great general purpose")
        elif any('llama3.1' in m for m in model_names):
            best = [m for m in model_names if 'llama3.1' in m][0]
            print(f"BEST: {best} - Solid all-rounder")
        elif any('deepseek-coder' in m for m in model_names):
            best = [m for m in model_names if 'deepseek-coder' in m][0]
            print(f"BEST: {best} - Good for technical analysis")
        else:
            best = model_names[0] if model_names else None
            print(f"BEST: {best} - First available model")

    else:
        print(f"Error: HTTP {response.status_code}")

except requests.exceptions.ConnectionError:
    print("ERROR: Ollama server not running")
    print("\nStart Ollama with:")
    print("  - Open Ollama app from Start menu")
    print("  - Or run: ollama serve")

except Exception as e:
    print(f"ERROR: {e}")
