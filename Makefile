PYTHON ?= .venv/bin/python
ARENA ?= build/arena_cpp
MODEL ?= models/mega_lucario/model.gbdt
VIEW_DIR ?= outputs/mega_lucario/rule_vs_gbdt_viewer
VIEW_HTML ?= $(VIEW_DIR)/game-0.html
VIEW_JSON ?= $(VIEW_DIR)/game-0.viewer.json
RATING_DIR ?= outputs/mega_lucario/rule_vs_gbdt_rating
RATING_GAMES ?= 100
RATING_WORKERS ?= 2
TRANSFORMER_DIR ?= outputs/mega_lucario/transformer_training
TRANSFORMER_REPLAYS ?= $(TRANSFORMER_DIR)/gbdt_replays
TRANSFORMER_FEATURES ?= $(TRANSFORMER_DIR)/distillation
TRANSFORMER_OUTPUT ?= $(TRANSFORMER_DIR)/model
TRANSFORMER_BIN ?= $(TRANSFORMER_OUTPUT)/model.bin
TRANSFORMER_GAMES ?= 5000
TRANSFORMER_WORKERS ?= 1
TRANSFORMER_EPOCHS ?= 5
TRANSFORMER_BATCH_SIZE ?= 256
TRANSFORMER_DEVICE ?= cpu

.PHONY: mega-lucario-rule-vs-gbdt-viewer
# Python版の公式シミュレーターでルールベースのメガルカリオとGBDTを1回対戦させ、
# 公式ビューアへ自動送信するHTMLリプレイを生成する。
mega-lucario-rule-vs-gbdt-viewer:
	rm -rf $(VIEW_DIR)
	mkdir -p $(VIEW_DIR)
	$(PYTHON) src/tools/run_mega_lucario_gbdt_viewer.py \
	  --model $(MODEL) \
	  --output-json $(VIEW_JSON) \
	  --output-html $(VIEW_HTML)
	$(PYTHON) -m webbrowser file://$(abspath $(VIEW_HTML))

.PHONY: mega-lucario-rule-vs-gbdt-rating
# C++ arenaでルールベースのメガルカリオとGBDTを席交換込みで対戦させ、勝敗とratingを計測する。
# 結果は $(RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-rule-vs-gbdt-rating: $(ARENA)
	rm -rf $(RATING_DIR)
	mkdir -p $(RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "rule": {"agent": "mega_lucario"},' \
	  '  "gbdt": {"agent": "mega_lucario_gbdt", "model": "$(abspath $(MODEL))"}' \
	  '}' > $(RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(RATING_DIR)/roster.json \
	  --output-dir $(RATING_DIR)/ratings \
	  --games-per-pair $(RATING_GAMES) \
	  --workers $(RATING_WORKERS)

$(ARENA):
	$(PYTHON) src/tools/build_cpp_engine.py --output $(ARENA)

.PHONY: mega-lucario-transformer-train
# GBDTをteacherとして対戦リプレイを収集し、Transformerの蒸留データ抽出・学習・C++用exportまで行う。
# 結果は $(TRANSFORMER_DIR)/ に出力される（gbdt_replays/, distillation/, model/model.pt, model/model.bin）。
mega-lucario-transformer-train: $(ARENA)
	rm -rf $(TRANSFORMER_DIR)
	mkdir -p $(TRANSFORMER_DIR)
	$(PYTHON) src/agents/mega_lucario/scripts/collect_replays.py \
	  --binary $(ARENA) \
	  --output-dir $(TRANSFORMER_REPLAYS) \
	  --games $(TRANSFORMER_GAMES) \
	  --teacher mega_lucario_gbdt \
	  --model $(MODEL) \
	  --workers $(TRANSFORMER_WORKERS)
	$(PYTHON) src/agents/mega_lucario/transformer/scripts/extract.py \
	  --replay-dir $(TRANSFORMER_REPLAYS) \
	  --model $(MODEL) \
	  --output-dir $(TRANSFORMER_FEATURES) \
	  --workers $(TRANSFORMER_WORKERS)
	$(PYTHON) src/agents/mega_lucario/transformer/scripts/train.py \
	  --feature-dir $(TRANSFORMER_FEATURES) \
	  --output-dir $(TRANSFORMER_OUTPUT) \
	  --epochs $(TRANSFORMER_EPOCHS) \
	  --batch-size $(TRANSFORMER_BATCH_SIZE) \
	  --device $(TRANSFORMER_DEVICE)
	$(PYTHON) src/agents/mega_lucario/transformer/scripts/export.py \
	  --checkpoint $(TRANSFORMER_OUTPUT)/model.pt \
	  --output $(TRANSFORMER_BIN)
