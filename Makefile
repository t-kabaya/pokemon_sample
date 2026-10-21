PYTHON ?= .venv/bin/python
TIME ?=
ARENA ?= build/arena_cpp
MODEL ?= models/mega_lucario/model.gbdt
VIEW_DIR ?= outputs/mega_lucario/rule_vs_gbdt_viewer
VIEW_HTML ?= $(VIEW_DIR)/game-0.html
VIEW_JSON ?= $(VIEW_DIR)/game-0.viewer.json
TRANSFORMER_VIEW_DIR ?= outputs/mega_lucario/rule_vs_transformer_viewer
TRANSFORMER_VIEW_HTML ?= $(TRANSFORMER_VIEW_DIR)/game-0.html
TRANSFORMER_VIEW_JSON ?= $(TRANSFORMER_VIEW_DIR)/game-0.viewer.json
PPO_VIEW_DIR ?= outputs/mega_lucario/rule_vs_ppo_viewer
PPO_VIEW_HTML ?= $(PPO_VIEW_DIR)/game-0.html
PPO_VIEW_JSON ?= $(PPO_VIEW_DIR)/game-0.viewer.json
RATING_DIR ?= outputs/mega_lucario/rule_vs_gbdt_rating
RATING_GAMES ?= 100
RATING_WORKERS ?= 2
TRANSFORMER_RATING_DIR ?= outputs/mega_lucario/rule_vs_transformer_rating
TRANSFORMER_RATING_GAMES ?= 100
TRANSFORMER_RATING_WORKERS ?= 2
PPO_RATING_DIR ?= outputs/mega_lucario/rule_vs_ppo_rating
PPO_RATING_GAMES ?= 100
PPO_RATING_WORKERS ?= 2
PPO_PREVIOUS_VS_LATEST_RATING_DIR ?= outputs/mega_lucario/ppo_previous_vs_latest_rating
PPO_PREVIOUS_VS_LATEST_RATING_GAMES ?= 1000
PPO_PREVIOUS_VS_LATEST_RATING_WORKERS ?= 2
TRANSFORMER_VS_PPO_RATING_DIR ?= outputs/mega_lucario/transformer_vs_ppo_rating
TRANSFORMER_VS_PPO_RATING_GAMES ?= 1000
TRANSFORMER_VS_PPO_RATING_WORKERS ?= 2
TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR ?= outputs/mega_lucario/transformer_vs_ppo_cycle001_rating
TRANSFORMER_VS_PPO_CYCLE001_RATING_GAMES ?= 1000
TRANSFORMER_VS_PPO_CYCLE001_RATING_WORKERS ?= 2
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
PPO_BASE_CYCLE ?= 0
PPO_BASE_CYCLE_ID ?= $(shell printf "%03d" $(PPO_BASE_CYCLE))
PPO_NEXT_CYCLE ?= $(shell expr $(PPO_BASE_CYCLE) + 1)
PPO_CYCLE_IDS ?= 000 001
PPO_GAMES ?= 5000
# Reduced from 64 to 32 after rollout collection timed out on an RTX 4090 instance.
# Reported CPU utilization fell from 99% to 54%; consider retrying 64 workers now
# that the minimum rollout timeout has increased from 300 to 600 seconds.
PPO_WORKERS ?= 32
PPO_EXTRACT_WORKERS ?= 32
PPO_ROLLOUT_CHUNK_GAMES ?= 64
PPO_AUDIT_THREADS ?= 32
PPO_VALIDATION_WORKERS ?= 3
PPO_VALIDATION_THREADS ?= 32
PPO_DEVICE ?= cpu
PPO_BATCH_SIZE ?= 1024
PPO_MICRO_BATCH_SIZE ?= 128
PPO_UPDATES ?= 128
PPO_TEMPERATURE ?= 1.0
PPO_BIN ?= $(PPO_DIR)/cycle-000/models/model.bin
PPO_CHECKPOINT ?= $(PPO_DIR)/cycle-000/models/model.pt
PPO_CYCLE001_BIN ?= $(PPO_DIR)/cycle-001/models/model.bin
PPO_LATEST_BIN ?= $(PPO_DIR)/latest/models/model.bin
PPO_LATEST_CHECKPOINT ?= $(PPO_DIR)/latest/models/model.pt
PPO_PREVIOUS_BIN ?= $(PPO_DIR)/previous/models/model.bin
PPO_BASE_CHECKPOINT ?= $(PPO_DIR)/cycle-$(PPO_BASE_CYCLE_ID)/models/model.pt
RUNPOD_SETUP_ARGS ?=
HF_NAMESPACE ?= kaba777
HF_MODEL_REPO ?= $(HF_NAMESPACE)/ptcg-mega-lucario-models
HF_DATASET_REPO ?= $(HF_NAMESPACE)/ptcg-mega-lucario-runs
HF_REPO_VISIBILITY ?= --private
HF_ARTIFACT_DIR ?= artifacts/hf
HF_PPO_DOWNLOAD_DIR ?= $(HF_ARTIFACT_DIR)/ppo-download
HF_TRANSFORMER_MODEL_DIR ?= mega_lucario/transformer
HF_GBDT_MODEL_DIR ?= mega_lucario/gbdt
HF_PPO_MODEL_DIR ?= mega_lucario/ppo

# -----------------------------------------------------------------------------
# Setup tasks
# -----------------------------------------------------------------------------

.PHONY: runpod-setup
# RunPodの新規Podでuv環境、Kaggle資産、C++ arenaの初期セットアップを行う。
# Hugging FaceからTransformer model.pt/model.binとGBDT model.gbdtを取得し、PPO学習の入力を用意する。
# 初回は make runpod-setup をそのまま実行する。
# 永続Volumeなどで外部資産が取得済みなら RUNPOD_SETUP_ARGS="--skip-assets" を指定する。
# C++ arenaは毎回ビルドする。
runpod-setup:
	bash src/tools/setup_runpod.sh $(RUNPOD_SETUP_ARGS)

.PHONY: hf-login
# Hugging FaceのWrite権限つきtokenでログインする。tokenは https://huggingface.co/settings/tokens で作成する。
hf-login:
	uv run hf auth login

$(ARENA):
	$(PYTHON) src/tools/build_cpp_engine.py --output $(ARENA)

# -----------------------------------------------------------------------------
# Upload tasks
# -----------------------------------------------------------------------------

.PHONY: hf-create-repos
# Hugging Face上にモデル保存用repoを作成する。
# 例 make hf-create-repos HF_NAMESPACE=<user-or-org>
# 公開repoにする場合は HF_REPO_VISIBILITY=--public を指定する。
hf-create-repos:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	uv run hf repos create $(HF_MODEL_REPO) --repo-type model $(HF_REPO_VISIBILITY) --exist-ok

.PHONY: hf-upload-transformer-weights
# Transformerで作成したモデル重みだけをHugging Faceへアップロードする。
hf-upload-transformer-weights:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	uv run hf upload $(HF_MODEL_REPO) $(TRANSFORMER_CHECKPOINT) $(HF_TRANSFORMER_MODEL_DIR)/model.pt --repo-type model --commit-message "Upload Transformer checkpoint"
	uv run hf upload $(HF_MODEL_REPO) $(TRANSFORMER_BIN) $(HF_TRANSFORMER_MODEL_DIR)/model.bin --repo-type model --commit-message "Upload Transformer C++ weights"

.PHONY: hf-upload-gbdt-weights
# GBDTで作成したモデル重みをHugging Faceへアップロードする。
hf-upload-gbdt-weights:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	uv run hf upload $(HF_MODEL_REPO) $(MODEL) $(HF_GBDT_MODEL_DIR)/model.gbdt --repo-type model --commit-message "Upload GBDT weights"

.PHONY: hf-upload-ppo-weights
# ローカルの最新PPO cycleを自動判定し、Hugging Faceの$(HF_PPO_MODEL_DIR)/cycle-xxx/へアップロードする。
hf-upload-ppo-weights:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	uv run python src/tools/upload_latest_ppo_cycle.py $(HF_MODEL_REPO) \
	  --repo-type model \
	  --remote-root $(HF_PPO_MODEL_DIR) \
	  --output-dir $(PPO_DIR)

.PHONY: hf-install-ppo-cycles
# Hugging Faceから指定したPPO cycle群を取得し、$(PPO_DIR)/cycle-xxx/models/へ配置する。
# 例 make hf-install-ppo-cycles PPO_CYCLE_IDS="000 001"
hf-install-ppo-cycles:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	rm -rf $(HF_PPO_DOWNLOAD_DIR)
	mkdir -p $(HF_PPO_DOWNLOAD_DIR)
	set -e; for cycle in $(PPO_CYCLE_IDS); do \
	  uv run hf download $(HF_MODEL_REPO) \
	    $(HF_PPO_MODEL_DIR)/cycle-$$cycle/model.pt \
	    $(HF_PPO_MODEL_DIR)/cycle-$$cycle/model.bin \
	    --repo-type model \
	    --local-dir $(HF_PPO_DOWNLOAD_DIR); \
	  mkdir -p $(PPO_DIR)/cycle-$$cycle/models; \
	  cp -f $(HF_PPO_DOWNLOAD_DIR)/$(HF_PPO_MODEL_DIR)/cycle-$$cycle/model.pt \
	    $(PPO_DIR)/cycle-$$cycle/models/model.pt; \
	  cp -f $(HF_PPO_DOWNLOAD_DIR)/$(HF_PPO_MODEL_DIR)/cycle-$$cycle/model.bin \
	    $(PPO_DIR)/cycle-$$cycle/models/model.bin; \
	done

.PHONY: hf-install-latest-ppo-cycle
# Hugging Face上の$(HF_PPO_MODEL_DIR)/cycle-xxxから一番新しいPPO cycleを自動判定して取得する。
hf-install-latest-ppo-cycle:
	test -n "$(HF_NAMESPACE)" || (echo "Set HF_NAMESPACE=<user-or-org>" >&2; exit 1)
	uv run python src/tools/install_latest_ppo_cycle.py $(HF_MODEL_REPO) \
	  --repo-type model \
	  --remote-root $(HF_PPO_MODEL_DIR) \
	  --output-dir $(PPO_DIR) \
	  --alias latest \
	  --previous-alias previous

.PHONY: hf-upload-training-weights
# Transformerのmodel.pt/model.binとGBDTのmodel.gbdtだけをHugging Faceへアップロードする。
hf-upload-training-weights: hf-upload-transformer-weights hf-upload-gbdt-weights

.PHONY: hf-upload-artifacts
# 互換用alias。保存対象はTransformerとGBDTのモデル重み3ファイルのみ。
hf-upload-artifacts: hf-upload-training-weights

.PHONY: hf-clean-raw-uploads
# 過去に丸ごとアップロードしたoutputs/やmodels/配下をHugging Faceから削除する。
hf-clean-raw-uploads:
	uv run python src/tools/delete_hf_paths.py $(HF_MODEL_REPO) models/ --repo-type model --commit-message "Delete raw model upload"
	uv run python src/tools/delete_hf_paths.py $(HF_DATASET_REPO) outputs/ --repo-type dataset --commit-message "Delete raw output upload"

.PHONY: hf-save-training-weights
# Hugging Face repoを作成し、TransformerとGBDTのモデル重み3ファイルだけをアップロードする。
hf-save-training-weights: hf-create-repos hf-upload-training-weights

.PHONY: hf-save-artifacts
# 互換用alias。保存対象はTransformerとGBDTのモデル重み3ファイルのみ。
hf-save-artifacts: hf-save-training-weights

# -----------------------------------------------------------------------------
# Viewer tasks
# -----------------------------------------------------------------------------

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

.PHONY: mega-lucario-rule-vs-ppo-viewer
# Python版の公式シミュレーターでルールベースのメガルカリオとPPO後Transformerを1回対戦させ、
# Player0をルールベース、Player1をPPO後Transformerとして実行し、
# 可視化上ではPlayer0がルールベース、Player1がPPO。
# 公式ビューアへ自動送信するHTMLリプレイを生成する。
mega-lucario-rule-vs-ppo-viewer:
	rm -rf $(PPO_VIEW_DIR)
	mkdir -p $(PPO_VIEW_DIR)
	$(PYTHON) src/tools/run_mega_lucario_transformer_viewer.py \
	  --checkpoint $(PPO_CHECKPOINT) \
	  --device $(PPO_DEVICE) \
	  --output-json $(PPO_VIEW_JSON) \
	  --output-html $(PPO_VIEW_HTML)
	$(PYTHON) -m webbrowser file://$(abspath $(PPO_VIEW_HTML))

.PHONY: mega-lucario-rule-vs-uploaded-ppo-viewer
# Hugging FaceからPPO重みを取得してから、ルールベースのメガルカリオとPPOを1回対戦・可視化する。
mega-lucario-rule-vs-uploaded-ppo-viewer: hf-install-latest-ppo-cycle
	rm -rf $(PPO_VIEW_DIR)
	mkdir -p $(PPO_VIEW_DIR)
	$(PYTHON) src/tools/run_mega_lucario_transformer_viewer.py \
	  --checkpoint $(PPO_LATEST_CHECKPOINT) \
	  --device $(PPO_DEVICE) \
	  --output-json $(PPO_VIEW_JSON) \
	  --output-html $(PPO_VIEW_HTML)
	$(PYTHON) -m webbrowser file://$(abspath $(PPO_VIEW_HTML))

# -----------------------------------------------------------------------------
# Rating tasks
# -----------------------------------------------------------------------------

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

.PHONY: mega-lucario-rule-vs-uploaded-ppo-rating
# Hugging FaceからPPO重みを取得してから、ルールベースのメガルカリオとPPOを対戦させる。
mega-lucario-rule-vs-uploaded-ppo-rating: hf-install-latest-ppo-cycle $(ARENA)
	rm -rf $(PPO_RATING_DIR)
	mkdir -p $(PPO_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "rule": {"agent": "mega_lucario"},' \
	  '  "ppo": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_LATEST_BIN))"}' \
	  '}' > $(PPO_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(PPO_RATING_DIR)/roster.json \
	  --output-dir $(PPO_RATING_DIR)/ratings \
	  --games-per-pair $(PPO_RATING_GAMES) \
	  --workers $(PPO_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(PPO_RATING_DIR)/ratings/games.jsonl

.PHONY: mega-lucario-previous-ppo-vs-uploaded-ppo-rating
# Hugging Faceから最新PPOと一つ前のPPOを取得し、席交換込みで対戦させ、勝率とratingを計測する。
# previous=一つ前のPPO、latest=最新PPO。
# 結果は $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-previous-ppo-vs-uploaded-ppo-rating: hf-install-latest-ppo-cycle $(ARENA)
	rm -rf $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)
	mkdir -p $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "previous": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_PREVIOUS_BIN))"},' \
	  '  "latest": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_LATEST_BIN))"}' \
	  '}' > $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)/roster.json \
	  --output-dir $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)/ratings \
	  --games-per-pair $(PPO_PREVIOUS_VS_LATEST_RATING_GAMES) \
	  --workers $(PPO_PREVIOUS_VS_LATEST_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(PPO_PREVIOUS_VS_LATEST_RATING_DIR)/ratings/games.jsonl

.PHONY: mega-lucario-transformer-vs-ppo-rating
# C++ arenaで初期TransformerとPPO後Transformerを席交換込みで1000戦させ、勝率とratingを計測する。
# transformer=学習済みTransformer（$(TRANSFORMER_BIN)）、ppo=PPO後Transformer（$(PPO_BIN)）。
# 結果は $(TRANSFORMER_VS_PPO_RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-transformer-vs-ppo-rating: $(ARENA)
	rm -rf $(TRANSFORMER_VS_PPO_RATING_DIR)
	mkdir -p $(TRANSFORMER_VS_PPO_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "transformer": {"agent": "mega_lucario_transformer", "model": "$(abspath $(TRANSFORMER_BIN))"},' \
	  '  "ppo": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_BIN))"}' \
	  '}' > $(TRANSFORMER_VS_PPO_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(TRANSFORMER_VS_PPO_RATING_DIR)/roster.json \
	  --output-dir $(TRANSFORMER_VS_PPO_RATING_DIR)/ratings \
	  --games-per-pair $(TRANSFORMER_VS_PPO_RATING_GAMES) \
	  --workers $(TRANSFORMER_VS_PPO_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(TRANSFORMER_VS_PPO_RATING_DIR)/ratings/games.jsonl

.PHONY: mega-lucario-transformer-vs-ppo-cycle001-rating
# C++ arenaで初期Transformerと取得済みPPO cycle-001を席交換込みで1000戦させ、勝率とratingを計測する。
# transformer=学習済みTransformer（$(TRANSFORMER_BIN)）、ppo-cycle001=PPO cycle-001（$(PPO_CYCLE001_BIN)）。
# 結果は $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)/ratings/ に出力される（games.jsonl, ratings.json, ratings.csv）。
mega-lucario-transformer-vs-ppo-cycle001-rating: $(ARENA)
	rm -rf $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)
	mkdir -p $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)
	printf '%s\n' \
	  '{' \
	  '  "transformer": {"agent": "mega_lucario_transformer", "model": "$(abspath $(TRANSFORMER_BIN))"},' \
	  '  "ppo-cycle001": {"agent": "mega_lucario_transformer", "model": "$(abspath $(PPO_CYCLE001_BIN))"}' \
	  '}' > $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)/roster.json
	$(PYTHON) src/tools/rating_arena.py \
	  --binary $(ARENA) \
	  --roster $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)/roster.json \
	  --output-dir $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)/ratings \
	  --games-per-pair $(TRANSFORMER_VS_PPO_CYCLE001_RATING_GAMES) \
	  --workers $(TRANSFORMER_VS_PPO_CYCLE001_RATING_WORKERS)
	$(PYTHON) src/tools/summarize_match_results.py $(TRANSFORMER_VS_PPO_CYCLE001_RATING_DIR)/ratings/games.jsonl

# -----------------------------------------------------------------------------
# Train tasks
# -----------------------------------------------------------------------------

.PHONY: mega-lucario-transformer-train mega-lucario-transformer-train-run
# GBDTをteacherとして対戦リプレイを収集し、Transformerの蒸留データ抽出・学習・C++用exportまで行う。
# 結果は $(TRANSFORMER_DIR)/ に出力される（gbdt_replays/, distillation/, model/model.pt, model/model.bin）。
mega-lucario-transformer-train:
	$(TIME) $(MAKE) mega-lucario-transformer-train-run

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
# PPO_WORKERSはC++ arenaの同時process数、PPO_ROLLOUT_CHUNK_GAMESは各processのゲーム数。
# PPO_AUDIT_THREADSとPPO_VALIDATION_THREADSはPyTorch CPU推論のスレッド数。
# 例 make mega-lucario-ppo-train PPO_GAMES=1000 PPO_WORKERS=32 PPO_EXTRACT_WORKERS=32 PPO_DEVICE=cuda
mega-lucario-ppo-train: $(ARENA)
	rm -rf $(PPO_DIR)
	mkdir -p $(PPO_DIR)
	$(TIME) $(PYTHON) src/agents/mega_lucario/ppo/scripts/run_cycle.py \
	  --checkpoint $(TRANSFORMER_CHECKPOINT) \
	  --output-dir $(PPO_DIR) \
	  --gbdt-model $(MODEL) \
	  --binary $(ARENA) \
	  --cycles $(PPO_CYCLES) \
	  --games $(PPO_GAMES) \
	  --workers $(PPO_WORKERS) \
	  --extract-workers $(PPO_EXTRACT_WORKERS) \
	  --rollout-chunk-games $(PPO_ROLLOUT_CHUNK_GAMES) \
	  --audit-threads $(PPO_AUDIT_THREADS) \
	  --validation-workers $(PPO_VALIDATION_WORKERS) \
	  --validation-threads $(PPO_VALIDATION_THREADS) \
	  --temperature $(PPO_TEMPERATURE) \
	  --device $(PPO_DEVICE) \
	  --batch-size $(PPO_BATCH_SIZE) \
	  --micro-batch-size $(PPO_MICRO_BATCH_SIZE) \
	  --updates $(PPO_UPDATES)

.PHONY: mega-lucario-ppo-train-next-cycle
# 既存のPPO cycleを初期方策として、次のcycleを$(PPO_DIR)へ追加する。
# 例 make mega-lucario-ppo-train-next-cycle PPO_BASE_CYCLE=0  # cycle-000 -> cycle-001
mega-lucario-ppo-train-next-cycle: $(ARENA)
	test -f $(PPO_BASE_CHECKPOINT) || (echo "Missing PPO_BASE_CHECKPOINT=$(PPO_BASE_CHECKPOINT)" >&2; exit 1)
	mkdir -p $(PPO_DIR)
	$(TIME) $(PYTHON) src/agents/mega_lucario/ppo/scripts/run_cycle.py \
	  --checkpoint $(PPO_BASE_CHECKPOINT) \
	  --output-dir $(PPO_DIR) \
	  --start-cycle $(PPO_NEXT_CYCLE) \
	  --gbdt-model $(MODEL) \
	  --binary $(ARENA) \
	  --cycles $(PPO_CYCLES) \
	  --games $(PPO_GAMES) \
	  --workers $(PPO_WORKERS) \
	  --extract-workers $(PPO_EXTRACT_WORKERS) \
	  --rollout-chunk-games $(PPO_ROLLOUT_CHUNK_GAMES) \
	  --audit-threads $(PPO_AUDIT_THREADS) \
	  --validation-workers $(PPO_VALIDATION_WORKERS) \
	  --validation-threads $(PPO_VALIDATION_THREADS) \
	  --temperature $(PPO_TEMPERATURE) \
	  --device $(PPO_DEVICE) \
	  --batch-size $(PPO_BATCH_SIZE) \
	  --micro-batch-size $(PPO_MICRO_BATCH_SIZE) \
	  --updates $(PPO_UPDATES)
