# Video Upscaling on Mac (mps/mlx)
It uses actual Real-ESRGAN (RRDBNet architecture) for mps, but currently needs custom models to take advantage of mlx functionality.

How it works:
- Checks resolution; skips if already ≥720p (unless --force).
- Auto-picks x2plus (fast) or x4plus (higher scale) based on how much upscaling is actually needed.
- Downloads official weights once, caches at ~/.cache/real_esrgan_mps/.
- Runs tiled inference on MPS (important — full-frame inference on MPS will OOM on anything HD-sized; tiling keeps memory bounded).
- Final Lanczos resize locks the output to an exact 720p size.
- Video: frame-by-frame through OpenCV, then ffmpeg re-attaches the original audio track.
```bash
pip install torch torchvision opencv-python pillow numpy
python mps_upscaler.py your_movie.mp4
python mps_upscaler.py photo.png --model x4plus
```

Tuning knobs worth knowing about:
--tile 192 is a safe default for 8–16GB M1s - bump to 384 or --tile 0 on M1 Pro/Max/Ultra with more unified memory for real speed gains.

For a full-length movie, expect this to be slow (plain PyTorch loop, no FP16/batching) — it's correct and it's real Real-ESRGAN
