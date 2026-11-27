#!/usr/bin/env bash
# Rebuild every showcase artefact from a clean clone (synthetic data, fixed seeds, CPU only).
#
# Requirements:
#   - Python >= 3.11 with this repo installed:  pip install ".[serve]"
#   - vhs >= 0.10 (charmbracelet/vhs) + ttyd + ffmpeg   (only for demo.gif / demo.mp4)
#
# Outputs (showcase/):
#   results/episode.json     task 2 — one synthetic episode, end to end
#   results/benchmark.json   task 3 — 250 seeded synthetic episodes
#   poster.png               1280x720 timeline of the episode
#   demo.mp4, demo.gif       terminal run of the episode, then the chart
#
# Usage: bash showcase/make_showcase.sh [--no-video]
set -euo pipefail
cd "$(dirname "$0")/.."

export EWAT_MODEL="${EWAT_MODEL:-showcase/.cache/synthetic.pt}"
export PYTHONHASHSEED=0
SCENARIO=fail_slow_latency
SEED=0

echo "== 1/4 train the typer on synthetic episodes (seeds 10000-10029 per scenario)"
ewat train

echo "== 2/4 one synthetic episode end to end ($SCENARIO, seed $SEED)"
ewat demo --scenario "$SCENARIO" --seed "$SEED" \
  --out showcase/results/episode.json --plot showcase/poster.png

echo "== 3/4 synthetic benchmark (50 episodes x 5 scenarios, seeds 0-49)"
ewat benchmark --episodes 50 --seed0 0 --out showcase/results/benchmark.json

if [[ "${1:-}" == "--no-video" ]]; then exit 0; fi

echo "== 4/4 terminal demo (vhs) + chart reveal (ffmpeg)"
for tool in vhs ttyd ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing $tool (see header)"; exit 1; }
done
tmp=showcase/.cache
mkdir -p "$tmp"
vhs showcase/demo.tape                       # → showcase/.cache/terminal.mp4
# terminal run, then the poster for 6 s with a fade-in
ffmpeg -y -loglevel error -i "$tmp/terminal.mp4" -loop 1 -t 6 -i showcase/poster.png \
  -filter_complex "[0:v]scale=1280:720,setsar=1,fps=25[a];[1:v]scale=1280:720,setsar=1,fps=25,fade=in:0:12[b];[a][b]concat=n=2:v=1[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart showcase/demo.mp4
ffmpeg -y -loglevel error -i showcase/demo.mp4 \
  -vf "fps=10,scale=1024:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" \
  showcase/demo.gif
ls -la showcase/demo.gif showcase/demo.mp4 showcase/poster.png
