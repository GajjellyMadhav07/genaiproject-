## Code Generation Assistant with Multi-Model Architecture

This project is a production-ready, multi-modal code generation assistant that integrates:
- Code generation via Hugging Face-hosted code LLMs (e.g., CodeLlama/StarCoder) using LoRA-finetuned models when available
- Architecture diagram generation via Stable Diffusion
- NLP-based code analysis (tokenization, complexity, pattern detection)
- SQLite-backed chat history storage
- Gradio UI for Hugging Face Spaces and a Flask API for programmatic access

### Key Features
- Generate, debug, and document code; produce unit/integration tests
- Automatically create architecture diagrams from prompts or generated code
- Analyze code with NLP techniques
- Persist chat sessions (prompt, code, diagram, analysis) in SQLite
- Monitoring hooks for quality and latency metrics

### Running Locally
1. Create a Python 3.10+ virtual environment
2. Install requirements:
```
pip install -r requirements.txt
```
3. Export your Hugging Face token (read access to models used):
```
set HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
```
4. Start the Gradio app:
```
python app.py
```

Gradio will launch at http://localhost:7860.

### API (Flask) - Local
Optionally run the Flask API server:
```
python -m app.api.server
```
This exposes endpoints like `/api/generate` and `/api/history` on http://localhost:8000.

### Hugging Face Spaces
- This project is optimized for HF Spaces free-tier. The Gradio app is the primary entry point (`app.py`).
- SQLite database is stored at `./data/app.db`. Make sure the Space has persistent storage enabled.
- Configure environment variable `HF_TOKEN` in the Space secrets.

### Configuration
Environment variables (optional):
- `HF_TOKEN`: Hugging Face token
- `CODE_MODEL_ID`: default code model id (e.g., `codellama/CodeLlama-7b-Instruct-hf` or `bigcode/starcoder2-3b`)
- `DIFFUSION_MODEL_ID`: default SD model id (e.g., `stabilityai/stable-diffusion-2-1`)

### Tests
Run unit and integration tests:
```
pytest -q --maxfail=1 --disable-warnings
```

### Docker
Build and run:
```
docker build -t gen-ai-app .
docker run -p 7860:7860 -e HF_TOKEN=%HF_TOKEN% gen-ai-app
```



