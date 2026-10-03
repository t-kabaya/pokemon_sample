# Pokémon TCG AI Battle — Population-based RLのサンプル実装

[English version](README.md)

Kaggleの[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)向けの最小学習パイプラインです。C++対戦環境、GBDTによる模倣学習、Transformer蒸留、PPOを一つにまとめています。

[24th Solution: A Population-Based RL Ecosystem](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/writeups/24th-solution-a-population-based-rl-ecosystem)のサンプル実装として、公式4デッキに対応しています。

## コードの配置

```text
src/
├── agents/
│   ├── <agent名>/
│   │   ├── main.py, deck.csv    # 公式Python方策と同梱デッキ
│   │   ├── cpp/                # ルール方策・factory・学習記録
│   │   ├── scripts/            # 教師対戦収集・動作確認
│   │   ├── gbdt/               # 特徴・木の学習、cpp/、scripts/
│   │   ├── transformer/        # トークン・蒸留、cpp/、scripts/
│   │   └── ppo/                # rollout・GAE・更新、cpp/、scripts/
│   └── cpp/registry.h          # エージェントの登録のみ
├── engine/cpp/                 # 観測・合法手・対戦進行・記録
├── engine/cg/                  # 無変更の外部エンジン、Git対象外
└── tools/                      # 取得・ビルド・公式照合・レーティング
```

重み・リプレイ・特徴データ・ビルド成果物はGit追跡対象外です。

### 対象の4エージェント

| エージェント名 | 公式デッキ | GBDT特徴数 | コード・公式Notebook |
|---|---|---:|---|
| `mega_lucario` | メガルカリオex | 4,926 | [実装](src/agents/mega_lucario/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck) |
| `dragapult` | ドラパルトex | 5,078 | [実装](src/agents/dragapult/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck) |
| `iono` | ナンジャモ | 4,882 | [実装](src/agents/iono/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck) |
| `mega_abomasnow` | メガユキノオーex | 4,714 | [実装](src/agents/mega_abomasnow/)・[Notebook](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck) |

全4種に公式Python/C++ルール方策、GBDT、Transformer、PPOがあります。
C++方策名はルールなら上記の名前、学習済み方策なら `<名前>_gbdt` / `<名前>_transformer` です。PPOもTransformer方策として読み込みます。
学習スクリプトは各方式の `scripts/` にあり、引数は `--help` で確認できます。

## uv環境を用意する

Linux x86-64、Python 3.12以上、C++20対応の `g++` が必要です。GBDTはCPU、Transformer/PPOの更新はCUDA版PyTorchを使います。

uvが未導入なら、[公式インストーラ](https://docs.astral.sh/uv/getting-started/installation/)を実行してターミナルを開き直します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

以降のコマンドはリポジトリ直下で実行します。`uv sync` が `.venv` を作成するため、手動のactivateは不要です。

```bash
uv sync --locked --dev
uv run kaggle --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

lockに記録されたLinux用PyTorchはCUDA版です。CUDAが `False` の場合はGPUとドライバへのアクセスを確認してください。
CPUで学習する場合は `--device cpu` を指定します。[uvとPyTorchの公式説明](https://docs.astral.sh/uv/guides/integration/pytorch/)も参照してください。

Kaggle CLIはdev依存に含めています。利用者自身のKaggle認証とコンペデータへのアクセスを用意し、外部資産を取得・ビルドします。

```bash
uv run python src/tools/fetch_official_assets.py
uv run python src/tools/fetch_cg_engine.py
uv run python src/tools/build_cpp_engine.py
```

`fetch_official_assets.py` は現在の Kaggle 配布の `cg-lib` をダウンロードできるように設計されています。リポジトリ側の古い固定ハッシュを厳密に検証したい場合は `--require-hash-check` を使います。

デッキは各エージェントの `deck.csv` として同梱しています。配布C++エンジンは `src/engine/cg/` に取得します。
配布ソースを変更せずincludeします。C++推論はuv環境に含まれるOpenBLASを利用できます。
取得元・配置変更・オフライン検証は[エンジンの案内](src/engine/README.md)にあります。

## 解法概要

コンペでは、観測と合法手一覧を受け取り、選んだ候補の番号を返すAIを作ります。相手の手札や山札の順序が見えない中で、カードの展開・攻撃・エネルギーの使い方を判断します。ゲームルールはコンペ配布のC++エンジンを使います。

![解法全体：リプレイ、GBDT、Transformer、PPO、対戦評価](docs/assets/Fig1.png)

1. **ルール方策の対戦を記録する。** 選択前の観測と実際の行動をリプレイに保存します。
2. **GBDTで行動を模倣する。** 盤面と合法手候補から特徴量を作り、LightGBM LambdaRankで教師が選ぶ候補を上位にします。MAINとそれ以外の判断に別々の木を使い、単一の `.gbdt` に保存します。
3. **Transformerへ蒸留する。** GBDTを対戦させ、候補スコア・行動・終局結果からPolicyとValueを学びます。
4. **PPOで更新する。** 現在の方策で新しい対戦を収集し、収集時の行動確率とValue、終局報酬からGAEとPPO更新を行います。
5. **C++で再対戦し、評価する。** `.pt` を `.bin` にexportし、Pythonを呼ばずに推論します。固定した比較対象・共通seed・席交換で対戦し、レーティングの推定値と不確かさを保存します。

図は公開Writeupの全体像です。人間・LBエージェントのリプレイ、対面別Expert、母集団の自動選抜・大規模運用はこのサンプルに含めません。図中の試合数や処理時間は、このリポジトリの実測値ではありません。

![Transformerの構造](docs/assets/Fig2.png)

Transformerは全デッキとも **128次元・6層・4 heads・最大120トークン・980,916パラメータ**です。
Global・Board・Zone・Option・STOPをEncoderへ渡し、Pointer Policyと51段階のValue分布を出します。デッキ語彙と固有特徴が異なり、モデル構造は共通です。
追加3デッキでは公開カードの配置、進化組、エネルギー、攻撃IDなどを特徴化しています。メガルカリオ固有の攻撃結果計算は流用せず、未実装の結果チャネルは未知として扱います。

## レーティングを計測する

対戦結果からベイズBradley–Terryモデルでレーティングを推定します。以下は公式4方策を各組2試合ずつ対戦させる例です。

```bash
uv run python src/tools/rating_arena.py \
  --output-dir outputs/rating_example --games-per-pair 2 --workers 2
```

`--games-per-pair` は**席交換の両方向を合わせた偶数の試合数**です。両方向で同じseedを使います。席0/1はエンジン上の先攻/後攻の直接指定ではありません。
同じ出力先に `--resume` を指定すると完了済み対戦を再利用でき、試合数を増やせます。

`ratings.json` / `ratings.csv` に推定値、標準偏差、下側評価値、対面別W/D/Lを保存します。基準は比較集団の平均1,000です。
**公式LBレーティングとは別の内部尺度**であり、比較相手を変えると値も変わります。

学習済みモデルを比較する `--roster`、推定条件、再開手順は[レーティングの案内](src/tools/README.md)を参照してください。

## 開発

RuffでPythonの書式を整え、import・型アノテーション・docstringを確認します。

```bash
uv run ruff format src
uv run ruff check src
```

## ライセンス

このリポジトリのコード・文書・図は [Apache License 2.0](LICENSE) で公開しています。
