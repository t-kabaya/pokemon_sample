PYTHON ?= .venv/bin/python
ARENA ?= build/arena_cpp
MODEL ?= models/mega_lucario/model.gbdt
VIEW_DIR ?= outputs/mega_lucario/rule_vs_gbdt_viewer
VIEW_HTML ?= $(VIEW_DIR)/game-0.html
VIEW_JSON ?= $(VIEW_DIR)/game-0.viewer.json
TRANSFORMER_VIEW_DIR ?= outputs/mega_lucario/rule_vs_transformer_viewer
TRANSFORMER_VIEW_HTML ?= $(TRANSFORMER_VIEW_DIR)/game-0.html
TRANSFORMER_VIEW_JSON ?= $(TRANSFORMER_VIEW_DIR)/game-0.viewer.json
RATING_DIR ?= outputs/mega_lucario/rule_vs_gbdt_rating
RATING_GAMES ?= 100
RATING_WORKERS ?= 2
TRANSFORMER_RATING_DIR ?= outputs/mega_lucario/rule_vs_transformer_rating
TRANSFORMER_RATING_GAMES ?= 100
TRANSFORMER_RATING_WORKERS ?= 2
PPO_RATING_DIR ?= outputs/mega_lucario/rule_vs_ppo_rating
PPO_RATING_GAMES ?= 100
PPO_RATING_WORKERS ?= 2
TRANSFORMER_DIR ?= outputs/mega_lucario/transformer_training
TRANSFORMER_REPLAYS ?= $(TRANSFORMER_DIR)/gbdt_replays
TRANSFORMER_FEATURES ?= $(TRANSFORMER_DIR)/distillation
TRANSFORMER_OUTPUT ?= $(TRANSFORMER_DIR)/model
TRANSFORMER_BIN ?= $(TRANSFORMER_OUTPUT)/model.bin
TRANSFORMER_CHECKPOINT ?= $(TRANSFORMER_OUTPUT)/model.pt
TRANSFORMER_GAMES ?= 5000
TRANSFORMER_GAMES_PER_OPPONENT ?= $(shell expr $(TRANSFORMER_GAMES) / 4)
TRANSFORMER_WORKERS ?= 1
TRANSFORMER_EPOCHS ?= 5
TRANSFORMER_BATCH_SIZE ?= 256
TRANSFORMER_DEVICE ?= cpu
PPO_DIR ?= outputs/mega_lucario/ppo_training
PPO_CYCLES ?= 1
PPO_GAMES ?= 5000
PPO_WORKERS ?= 2
PPO_DEVICE ?= cpu
PPO_BATCH_SIZE ?= 1024
PPO_MICRO_BATCH_SIZE ?= 128
PPO_UPDATES ?= 128
PPO_TEMPERATURE ?= 1.0
PPO_BIN ?= $(PPO_DIR)/cycle-000/models/model.bin
RUNPOD_SETUP_ARGS ?=

.PHONY: runpod-setup
# RunPodの新規Podでuv環境、Kaggle資産、C++ arenaの初期セットアップを行う。
# 初回は make runpod-setup をそのまま実行する。
# 永続Volumeなどで外部資産が取得済みなら RUNPOD_SETUP_ARGS="--skip-assets" を指定する。
# build/arena_cpp がビルド済みなら RUNPOD_SETUP_ARGS="--skip-build" を指定する。
# 依存環境とCUDA確認だけ行う場合は RUNPOD_SETUP_ARGS="--skip-assets --skip-build" を指定する。
runpod-setup:
	bash src/tools/setup_runpod.sh $(RUNPOD_SETUP_ARGS)

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

.PHONY: mega-lucario-rule-vs-transformer-viewer
# Python版の公式シミュレーターでルールベースのメガルカリオとTransformerを1回対戦させ、
# Player0をルールベース、Player1をTransformerとして実行し、
# 公式ビューアへ自動送信するHTMLリプレイを生成する。
mega-lucario-rule-vs-transformer-viewer:
	rm -rf $(TRANSFORMER_VIEW_DIR)
	mkdir -p $(TRANSFORMER_VIEW_DIR)
	$(PYTHON) src/tools/run_mega_lucario_transformer_viewer.py \
	  --checkpoint $(TRANSFORMER_CHECKPOINT) \
	  --device $(TRANSFORMER_DEVICE) \
	  --output-json $(TRANSFORMER_VIEW_JSON) \
	  --output-html $(TRANSFORMER_VIEW_HTML)
	$(PYTHON) -m webbrowser file://$(abspath $(TRANSFORMER_VIEW_HTML))

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

.PHONY: mega-lucario-rule-vs-transformer-rating
# C++ arenaでルールベースのメガルカリオとTransformerを席交換込みで100戦させ、勝率とratingを計測する。
# 結果は $(TRANSFORMER_RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-rule-vs-transformer-rating: $(ARENA)
	rm -rf $(TRANSFORMER_RATING_DIR)
	mkdir -p $(TRANSFORMER_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "rule": {"agent": "mega_lucario"},' \
	  '  "transformer": {"agent": "mega_lucario_transformer", "model": "$(abspath $(TRANSFORMER_BIN))"}' \
	  '}' > $(TRANSFORMER_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(TRANSFORMER_RATING_DIR)/roster.json \
	  --output-dir $(TRANSFORMER_RATING_DIR)/ratings \
	  --games-per-pair $(TRANSFORMER_RATING_GAMES) \
	  --workers $(TRANSFORMER_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(TRANSFORMER_RATING_DIR)/ratings/games.jsonl

.PHONY: mega-lucario-rule-vs-ppo-rating
# C++ arenaでルールベースのメガルカリオとPPO後Transformerを席交換込みで100戦させ、勝率とratingを計測する。
# デフォルトでは rule=mega_lucario と ppo=mega_lucario_transformer（$(PPO_BIN)）を対戦させる。
# 結果は $(PPO_RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-rule-vs-ppo-rating: $(ARENA)
	rm -rf $(PPO_RATING_DIR)
	mkdir -p $(PPO_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "rule": {"agent": "mega_lucario"},' \
	  '  "ppo": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_BIN))"}' \
	  '}' > $(PPO_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(PPO_RATING_DIR)/roster.json \
	  --output-dir $(PPO_RATING_DIR)/ratings \
	  --games-per-pair $(PPO_RATING_GAMES) \
	  --workers $(PPO_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(PPO_RATING_DIR)/ratings/games.jsonl

$(ARENA):
	$(PYTHON) src/tools/build_cpp_engine.py --output $(ARENA)

.PHONY: mega-lucario-transformer-train mega-lucario-transformer-train-run
# GBDTをteacherとして対戦リプレイを収集し、Transformerの蒸留データ抽出・学習・C++用exportまで行う。
# 結果は $(TRANSFORMER_DIR)/ に出力される（gbdt_replays/, distillation/, model/model.pt, model/model.bin）。
mega-lucario-transformer-train:
	/usr/bin/time -p $(MAKE) mega-lucario-transformer-train-run

mega-lucario-transformer-train-run: $(ARENA)
	rm -rf $(TRANSFORMER_DIR)
	mkdir -p $(TRANSFORMER_DIR)
	$(PYTHON) src/agents/mega_lucario/scripts/collect_replays.py \
	  --binary $(ARENA) \
	  --output-dir $(TRANSFORMER_REPLAYS) \
	  --games-per-opponent $(TRANSFORMER_GAMES_PER_OPPONENT) \
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

.PHONY: mega-lucario-ppo-train
# 学習済みTransformerを初期方策として、C++ arenaのon-policy rolloutからPPOで改善し、
# cycleごとにmodel.pt/model.binをexportしてnative validationまで行う。
# 例 make mega-lucario-ppo-train PPO_GAMES=1000 PPO_WORKERS=2 PPO_DEVICE=cpu
mega-lucario-ppo-train: $(ARENA)
	rm -rf $(PPO_DIR)
	mkdir -p $(PPO_DIR)
	/usr/bin/time -p $(PYTHON) src/agents/mega_lucario/ppo/scripts/run_cycle.py \
	  --checkpoint $(TRANSFORMER_CHECKPOINT) \
	  --output-dir $(PPO_DIR) \
	  --gbdt-model $(MODEL) \
	  --binary $(ARENA) \
	  --cycles $(PPO_CYCLES) \
	  --games $(PPO_GAMES) \
	  --workers $(PPO_WORKERS) \
	  --temperature $(PPO_TEMPERATURE) \
	  --device $(PPO_DEVICE) \
	  --batch-size $(PPO_BATCH_SIZE) \
	  --micro-batch-size $(PPO_MICRO_BATCH_SIZE) \
	  --updates $(PPO_UPDATES)
