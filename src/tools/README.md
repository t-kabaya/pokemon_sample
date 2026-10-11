# Downloads, builds, official-policy validation, and ratings

See the [root README](../../README.md) for environment setup and agent layout.

| Script | Purpose |
|---|---|
| `fetch_official_assets.py` | Download the Python cg API and optional notebooks; verify bundled decks |
| `fetch_cg_engine.py` | Download, extract, and verify the unmodified C++ engine |
| `build_cpp_engine.py` | Build four schemas, the arena, and each deck's GBDT/Transformer probes |
| `setup_runpod.sh` | Set up uv, Python dependencies, Kaggle assets, and the C++ arena on RunPod |
| `validate_cpp_agents.py` | Compare C++ observations/actions with Python; add `--notebook-root` to compare original notebooks |
| `rating_arena.py` | Collect and resume matches for a fixed pool, then save ratings |
| `rating.py` | Bayesian Bradley–Terry estimation |

## Choose the comparison pool

Without `--roster`, the arena compares the four official rule policies. Specify trained models with a JSON roster:

```json
{
  "rule": {"agent": "dragapult"},
  "teacher": {"agent": "dragapult_gbdt", "model": "dragapult/gbdt/model.gbdt"},
  "policy": {"agent": "dragapult_transformer", "model": "dragapult/ppo/model.bin"},
  "opponent": {"agent": "iono"}
}
```

Labels identify entries in results, so different checkpoints can have different labels. Model paths are relative to the roster file.
For `outputs/roster.json`, the paths above resolve from `outputs/`.
Pass `--roster outputs/roster.json --output-dir outputs/ratings --games-per-pair 2` to `rating_arena.py`.
PPO uses the Transformer inference format. Agent factories validate model decks and feature schemas.

## Match conditions and outputs

Every pair of distinct labels plays both seat assignments with common seeds.
`--games-per-pair` is the even total across both assignments. Seats 0/1 are distinct from the game's first/second player.
`--workers` controls concurrent C++ processes; OpenBLAS and OMP each use one thread per process.

- `manifest.json`: SHA-256 hashes of the executable, models, and decks, plus the pool and seed allocation.
- `games.jsonl`: Labels, seeds, results, steps, errors, and responsible seats for each match.
- `ratings.json`: Estimates, standard deviations, 95% intervals, lower bounds, W/D/L, unresolved counts, and covariance.
- `ratings.csv`: Per-agent estimates, standard deviations, lower bounds, match counts, and weights.

Use `--resume` to reuse completed matches and increase `--games-per-pair`. With the same count, it only recomputes ratings.
Use a new output directory when changing executables, models, decks, the pool, or seeds.
Incomplete batches are rerun after interruption. A file lock prevents concurrent collection into the same directory.

## Estimation and interpretation

The model uses `P(A wins) = sigmoid((rA - rB) × ln(10) / 400)` and a Gaussian prior with standard deviation 600. Newton's method finds the MAP estimate.
Ratings are centered at zero and shifted by 1,000 for display. Centered Laplace covariance gives uncertainty for both ratings and rating differences.
The lower bound is `mu - 2 × sigma`; the 95% interval is `mu ± 1.96 × sigma`.

Defaults use 16-match smoothing of win profiles, a similarity bandwidth of 0.12, inverse-density opponent weights, and a cap of 500 effective matches per matchup.
Raw match counts are retained. `--raw-bt` disables profile weighting and the cap; `--prior-sigma` / `--pair-cap` change them individually.
Draws count as half a win. An attributable agent error is a loss; engine errors, unknown attribution, and unfinished matches are excluded from estimation and counted separately.
The resolved comparison graph must be connected to produce a ranking.

If any standard deviation exceeds `--max-sigma`, results are saved and the command exits with code 2.
This uncertainty is conditional on the model and comparison pool; it does not bound official LB rating error.
With the default 500-match cap, adding matches to the same matchup cannot reduce uncertainty indefinitely.
Use a fixed baseline and pool, enough matches, and inspect regressions in individual matchups.
