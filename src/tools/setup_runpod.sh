#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: src/tools/setup_runpod.sh [options]

Set up this repository on a fresh RunPod Linux instance.

Options:
  --skip-apt              Do not install system packages with apt-get.
  --skip-assets           Do not download Kaggle official assets or the C++ engine.
  --skip-kaggle-check     Do not require Kaggle credentials before asset download.
  -h, --help              Show this help.

Environment:
  UV_SYNC_ARGS            Extra arguments appended to "uv sync --locked --dev".
  CXX                     C++ compiler used by build_cpp_engine.py, default is g++.
  HF_TOKEN                Optional Hugging Face token used by the hf CLI.
  HF_MODEL_REPO           Hugging Face model repo, default kaba777/ptcg-mega-lucario-models.

Kaggle credentials:
  Provide KAGGLE_API_TOKEN to skip interactive login automatically.
  Alternatively, put kaggle.json at ~/.kaggle/kaggle.json, or provide KAGGLE_USERNAME and
  KAGGLE_KEY in the environment. If credentials are missing, the script starts
  "uv run kaggle auth login" so you can complete OAuth in a browser.
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

skip_apt=0
skip_assets=0
skip_kaggle_check=0

while (($#)); do
  case "$1" in
    --skip-apt)
      skip_apt=1
      ;;
    --skip-assets)
      skip_assets=1
      ;;
    --skip-kaggle-check)
      skip_kaggle_check=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" != "Linux" ]]; then
  die "RunPod setup expects Linux; current system is $(uname -s)"
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  die "RunPod setup expects x86_64; current machine is $(uname -m)"
fi

if [[ "$skip_apt" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
  missing_packages=()
  command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
  command -v git >/dev/null 2>&1 || missing_packages+=(git)
  command -v g++ >/dev/null 2>&1 || missing_packages+=(g++)
  if ((${#missing_packages[@]})); then
    log "Installing system packages: ${missing_packages[*]}"
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates "${missing_packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates "${missing_packages[@]}"
    else
      die "missing packages (${missing_packages[*]}) and sudo is unavailable"
    fi
  fi
fi

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v git >/dev/null 2>&1 || die "git is required"
command -v "${CXX:-g++}" >/dev/null 2>&1 || die "C++ compiler not found: ${CXX:-g++}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation did not put uv on PATH"

log "Syncing Python dependencies"
# shellcheck disable=SC2086
uv sync --locked --dev ${UV_SYNC_ARGS:-}

log "Checking PyTorch and CUDA visibility"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

log "Checking Hugging Face CLI"
uv run hf --version
hf_authenticated=0
if uv run hf auth whoami; then
  hf_authenticated=1
  log "Hugging Face authentication is available"
else
  log "Hugging Face authentication is not configured; starting interactive login"
  uv run hf auth login
  if uv run hf auth whoami; then
    hf_authenticated=1
    log "Hugging Face authentication is available"
  fi
fi

if [[ "$hf_authenticated" -eq 0 ]]; then
  die "Hugging Face authentication is required to download weights."
fi
hf_model_repo="${HF_MODEL_REPO:-kaba777/ptcg-mega-lucario-models}"
hf_tmp="$(mktemp -d "${TMPDIR:-/tmp}/ptcg-hf-weights.XXXXXX")"
trap 'rm -rf "$hf_tmp"' EXIT
log "Downloading Transformer and GBDT weights from ${hf_model_repo}"
uv run hf download "$hf_model_repo" \
  mega_lucario/transformer/model.pt \
  mega_lucario/transformer/model.bin \
  mega_lucario/gbdt/model.gbdt \
  --repo-type model \
  --local-dir "$hf_tmp"
mkdir -p \
  outputs/mega_lucario/transformer_training/model \
  models/mega_lucario
cp "$hf_tmp/mega_lucario/transformer/model.pt" \
  outputs/mega_lucario/transformer_training/model/model.pt
cp "$hf_tmp/mega_lucario/transformer/model.bin" \
  outputs/mega_lucario/transformer_training/model/model.bin
cp "$hf_tmp/mega_lucario/gbdt/model.gbdt" \
  models/mega_lucario/model.gbdt
log "Downloading latest PPO cycle from ${hf_model_repo}"
uv run python src/tools/install_latest_ppo_cycle.py "$hf_model_repo" \
  --repo-type model \
  --remote-root mega_lucario/ppo \
  --output-dir outputs/mega_lucario/ppo_training \
  --alias latest

if [[ "$skip_assets" -eq 0 ]]; then
  if [[ "$skip_kaggle_check" -eq 0 ]]; then
    if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
      log "Kaggle API token environment variable is available"
    elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
      chmod 600 "$HOME/.kaggle/kaggle.json"
      log "Kaggle API token file is available"
    elif [[ -f "$HOME/.kaggle/credentials.json" ]]; then
      chmod 600 "$HOME/.kaggle/credentials.json"
      log "Kaggle OAuth credentials are available"
    elif {
      [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]
    }; then
      log "Kaggle authentication is not configured; starting interactive login"
      if ! uv run kaggle auth login; then
        uv run kaggle auth print-access-token >/dev/null || die "Kaggle authentication failed."
      fi
    else
      log "Kaggle environment credentials are available"
    fi
  fi
  log "Downloading official assets"
  uv run python src/tools/fetch_official_assets.py
  log "Downloading C++ engine"
  uv run python src/tools/fetch_cg_engine.py
fi

log "Building C++ arena"
uv run python src/tools/build_cpp_engine.py

log "RunPod setup complete"
