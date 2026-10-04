PYTHON ?= .venv/bin/python
MODEL ?= models/mega_lucario/model.gbdt
VIEW_DIR ?= outputs/mega_lucario/rule_vs_gbdt_viewer
VIEW_HTML ?= $(VIEW_DIR)/game-0.html
VIEW_JSON ?= $(VIEW_DIR)/game-0.viewer.json

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
