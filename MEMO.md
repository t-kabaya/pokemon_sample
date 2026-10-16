以下を実行中。GBDTの学習まで終わった。

実際の学習順
1. ルール方策で教師データを集める
collect_replays.py
uv run python src/agents/mega_lucario/scripts/collect_replays.py \
  --output-dir outputs/mega_lucario/rule_replays \
  --teacher mega_lucario
2. GBDTを学習する
extract.py
train.py
uv run python src/agents/mega_lucario/gbdt/scripts/extract.py \
  --replay-dir outputs/mega_lucario/rule_replays \
  --output-dir outputs/mega_lucario/gbdt_features

uv run python src/agents/mega_lucario/gbdt/scripts/train.py \
  --feature-dir outputs/mega_lucario/gbdt_features \
  --output-dir models/mega_lucario
3. Transformerを学習する
train.py
uv run python src/agents/mega_lucario/transformer/scripts/train.py \
  --feature-dir outputs/mega_lucario/distillation \
  --output-dir outputs/mega_lucario/transformer \
  --device cpu \
  --epochs 5
4. PPOで更新する
collect.py
train.py


# GBDTで学習したルカリオを動かす。
uv run python src/agents/mega_lucario/gbdt/scripts/audit.py \
  --replay-dir outputs/mega_lucario/rule_replays \
  --model-dir models/mega_lucario \
  --output outputs/mega_lucario/gbdt_parity.json \
  --probe build/gbdt_probe

Transformerの学習は以下のように終わった。
op1": 0.9230369552510229, "loss": 0.21506202791358156, "kl": 0.03697953406820614, "action_nll": 0.6359911763259676, "value_loss": 0.45793349911266856, "value_mse": 0.5793305893771475}}
{"selected_epoch": 4, "test": null}
.venv/bin/python src/agents/mega_lucario/transformer/scripts/export.py \
          --checkpoint outputs/mega_lucario/transformer_training/model/model.pt \
          --output outputs/mega_lucario/transformer_training/model/model.bin
Exported outputs/mega_lucario/transformer_training/model/model.bin: 3927739 bytes
kaba@kabayanoMacBook-Air ptcg-population-rl % 

PPOで学習。
tmux new -s ppo で、tmuxを起動
make mega-lucario-ppo-train-next-cycle \
  PPO_DEVICE=cuda \
  PPO_WORKERS=128 \
  PPO_UPDATES=64

# RunPodで、PPO学習を行う方法
v16のcpuを借りる。160GB

## git clone
cd /workspace && git clone https://github.com/t-kabaya/pokemon_sample.git ptcg-population-rl && cd ptcg-population-rl

setup (途中で、hf tokenの入力と、kaggleへのログインの必要あり。)
make runpod-setup


学習が終わったらアップロード
make hf-upload-ppo-weights
hf上のmega_lucario/ppo/cycle00x　に保存される。


# CPUとGPUの使い分け
rollout収集              CPU / C++ arena
rollout extraction       CPU
rollout behavior audit   CPU
PPO update               cuda指定ならGPU
export / validation      主にCPU