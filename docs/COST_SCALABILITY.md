# Cost, Throughput, and Scalability Notes

## Deployment Assumptions
- HF Spaces free-tier CPU for Gradio frontend
- HF Inference API for code models and Stable Diffusion
- SQLite for persistence

## Cost Drivers
- Model inference calls (code + diffusion)
- Network egress for images
- Storage size of SQLite DB and images (base64)

## Tips
- Reduce `max_new_tokens` for codegen
- Cache responses per prompt/session
- Downscale image size or limit diffusion calls per request
- Batch history reads and paginate

## Throughput
- Aim for <2s codegen latency (depends on model)
- Diagram generation may dominate latency; consider async and progressive rendering

## Scalability
- Move from SQLite to a hosted DB if QPS > ~5
- Add background workers for image generation
- Use CDN for serving diagrams
