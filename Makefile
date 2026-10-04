PYTHON ?= .venv/bin/python
ARENA ?= build/arena_cpp
MODEL ?= models/mega_lucario/model.gbdt
VIEW_DIR ?= outputs/mega_lucario/rule_vs_gbdt_viewer
VIEW_HTML ?= $(VIEW_DIR)/game-0.html
VIEW_JSON ?= $(VIEW_DIR)/game-0.viewer.json
RATING_DIR ?= outputs/mega_lucario/rule_vs_gbdt_rating
RATING_GAMES ?= 100
RATING_WORKERS ?= 2

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
