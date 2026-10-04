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
