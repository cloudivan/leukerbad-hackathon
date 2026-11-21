#!/usr/bin/env bash
# End-to-end MuJoCo Sim -> Google Compute Engine (GCE) GPU -> 8-DoF SmolVLA checkpoint pipeline.
#
# Usage:
#   # Mode A: You already have a GCE GPU VM reachable via SSH (or gcloud compute ssh):
#   GCE_SSH="ubuntu@YOUR_GCE_IP" ./gce_train.sh
#
#   # Mode B: Provision a new GCE L4 GPU VM via gcloud, train, download checkpoint:
#   GCP_PROJECT="your-gcp-project-id" ./gce_train.sh --create-vm
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NORMA_CORE="${NORMA_CORE:-$HOME/Developer/norma-core}"          # ADJUST: norma-core checkout
SMOLVLA_DIR="$NORMA_CORE/software/ai/smolvla_py"
PARQUET_OUT="$HERE/datasets/sim_pick_place_8dof.parquet"

EPISODES="${EPISODES:-50}"                                      # ADJUST: MuJoCo episodes to generate
STEPS="${STEPS:-5000}"                                          # ADJUST: training steps on GCE GPU
BATCH_SIZE="${BATCH_SIZE:-32}"                                  # ADJUST: 32 fits on 24GB L4, 64 on A100/L40S
GCP_ZONE="${GCP_ZONE:-europe-west4-a}"                          # ADJUST: GCE GPU zone (europe-west4-a / us-central1-a)
VM_NAME="${VM_NAME:-elrobot-gpu-trainer}"                       # ADJUST: GCE instance name
GCE_SSH="${GCE_SSH:-}"                                          # ADJUST: user@ip if skipping gcloud

if [[ "${1:-}" == "--create-vm" ]]; then
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "ERROR: gcloud CLI not found. Install with: brew install --cask google-cloud-sdk" >&2
    exit 1
  fi
  : "${GCP_PROJECT:?Set GCP_PROJECT=your-project-id before using --create-vm}"
  echo "[1/4] Creating GCE L4 GPU VM ($VM_NAME in $GCP_ZONE)..."
  gcloud compute instances create "$VM_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu124-ubuntu-2204-py310 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"
fi

echo "[1/4] Generating $EPISODES MuJoCo 8-DoF pick-and-place episodes..."
cd "$HERE"
uv run python mujoco_sim.py --episodes "$EPISODES" --out "$PARQUET_OUT"

echo "[2/4] Building self-contained SmolVLA training bundle..."
cd "$SMOLVLA_DIR"
./scripts/build_bundle.sh "$PARQUET_OUT"
BUNDLE_ZIP="$SMOLVLA_DIR/smolvla-bundle.zip"
echo "Bundle ready: $BUNDLE_ZIP"

if [[ -z "$GCE_SSH" ]] && ! command -v gcloud >/dev/null 2>&1; then
  echo ""
  echo "Local bundle is ready at: $BUNDLE_ZIP"
  echo "To run on your GCE GPU VM once you have the IP:"
  echo "  GCE_SSH=ubuntu@YOUR_GCE_IP ./gce_train.sh"
  exit 0
fi

echo "[3/4] Uploading bundle and launching 8-DoF training on GCE GPU..."
REMOTE_CMD="curl -LsSf https://astral.sh/uv/install.sh | sh && \
  export PATH=\"\$HOME/.local/bin:\$PATH\" && \
  rm -rf ~/smolvla-bundle && python3 -m zipfile -e ~/smolvla-bundle.zip ~/ && \
  cd ~/smolvla-bundle && uv sync && \
  uv run python scripts/train.py --steps $STEPS --batch-size $BATCH_SIZE \
    --parquets datasets/$(basename "$PARQUET_OUT") --output ~/checkpoints/elrobot_8dof"

if [[ -n "$GCE_SSH" ]]; then
  scp "$BUNDLE_ZIP" "$GCE_SSH:~/smolvla-bundle.zip"
  ssh "$GCE_SSH" "$REMOTE_CMD"
  echo "[4/4] Downloading trained checkpoint to $HERE/checkpoints/elrobot_8dof/final ..."
  mkdir -p "$HERE/checkpoints/elrobot_8dof"
  scp -r "$GCE_SSH:~/checkpoints/elrobot_8dof/final" "$HERE/checkpoints/elrobot_8dof/"
else
  gcloud compute scp --zone="$GCP_ZONE" "$BUNDLE_ZIP" "$VM_NAME:~/smolvla-bundle.zip"
  gcloud compute ssh --zone="$GCP_ZONE" "$VM_NAME" --command="$REMOTE_CMD"
  echo "[4/4] Downloading trained checkpoint to $HERE/checkpoints/elrobot_8dof/final ..."
  mkdir -p "$HERE/checkpoints/elrobot_8dof"
  gcloud compute scp --recurse --zone="$GCP_ZONE" "$VM_NAME:~/checkpoints/elrobot_8dof/final" "$HERE/checkpoints/elrobot_8dof/"
fi

echo "Done! Checkpoint ready at: $HERE/checkpoints/elrobot_8dof/final"
