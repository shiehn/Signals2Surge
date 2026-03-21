# Synth2Surge

> Part of the [Signals & Sorcery](https://signalsandsorcery.com) family of apps.

Translate arbitrary VST synth patches (Serum, Vital, etc.) into sonically similar [Surge XT](https://surge-synthesizer.github.io/) patches — using black-box optimization that gets smarter over time through self-supervised machine learning.

Synth2Surge has two modes:

- **Use** — Give it an audio file, get back a Surge XT preset that sounds like it
- **Train** — Let it run unattended, generating its own training data and learning to make better predictions with each cycle

The system is a closed loop: it randomizes Surge XT parameters, renders audio, then learns to predict which parameters produced that audio. Over time, the ML model learns to warm-start the CMA-ES optimizer with increasingly accurate initial guesses, making optimization faster and more accurate.

## How It Works

### Without ML (classical optimization)

```
Target Audio (WAV)                 Surge XT
┌─────────────────────┐           ┌────────────────────────┐
│  User-provided or   │           │  ~280 active params    │
│  captured from a    │           │  (normalized 0-1)      │
│  source plugin      │           └──────────┬─────────────┘
└─────────┬───────────┘                      │
          │                        ┌─────────▼─────────────┐
          │                        │  CMA-ES Optimizer      │
          ├───────────────────────▶│  (Optuna)              │
          │                        │                        │
          │  compare audio         │  1. Suggest params     │
          │  (MR-STFT loss)        │  2. Render candidate   │
          │                        │  3. Compute loss       │
          │                        │  4. Repeat 800 trials  │
          │                        └─────────┬─────────────┘
          │                                  │
          │                        ┌─────────▼─────────────┐
          │                        │  Best Surge XT Patch   │
          │                        │  (.fxp preset file)    │
          │                        └───────────────────────┘
```

### With ML (self-improving)

```
┌─────────────────────────────────────────────────────────────────┐
│                  SELF-IMPROVING CLOSED LOOP                      │
│                                                                   │
│  TRAINING (runs unattended)          INFERENCE (user-facing)     │
│  ─────────────────────────           ────────────────────────    │
│                                                                   │
│  1. Randomize Surge params           1. User provides audio      │
│  2. Render audio (random MIDI)       2. Extract 512-dim features │
│  3. Store (features, params)         3. ML predicts initial      │
│     as ground truth                     params (warm-start)      │
│  4. Train neural net on              4. CMA-ES refines from      │
│     accumulated data                    predicted starting point │
│  5. Repeat (thousands of times)      5. Output: Surge XT preset │
│                                                                   │
│  Each training cycle makes           No MIDI info needed —       │
│  inference predictions better        just audio in, preset out   │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- **OS:** macOS (Apple Silicon recommended, Intel supported)
- **Python:** 3.11 or later
- **Surge XT:** Installed as VST3 at `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` (free from [surge-synthesizer.github.io](https://surge-synthesizer.github.io/))
- **PyTorch** *(optional)*: Required only for ML training and `--warm-start` inference. The classical CMA-ES optimizer works without it.
- **Source plugin** *(for capture only)*: Any VST3 or Audio Unit instrument (Serum, Vital, Diva, etc.)

## Installation

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/shiehn/signals-to-surge.git
cd signals-to-surge
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. (Optional) Install PyTorch for ML features
uv pip install torch

# 4. Verify
synth2surge --help
uv run --extra dev pytest tests/unit/
```

---

## Usage: Matching Sounds

This is the primary workflow — give Synth2Surge audio and get a Surge XT preset.

### Step 1: Capture a preset (if translating from another synth)

```bash
synth2surge capture \
  --plugin "/path/to/Serum.vst3" \
  --output-dir ./workspace
```

Opens the source plugin's GUI. Select a preset, close the window, and it captures:
- `workspace/target_audio.wav` — rendered audio
- `workspace/target_state.bin` — binary plugin state

| Flag | Default | Description |
|---|---|---|
| `--plugin` | *(required)* | Path to source VST3/AU plugin |
| `--output-dir` | `./workspace` | Output directory |
| `--no-gui` | `false` | Skip editor GUI, use default state |
| `--state-file` | *none* | Load `.fxp` or state file instead of GUI |
| `--probe-mode` | `single` | MIDI probe: `single`, `thorough` (6 probes), or `full` (14 probes) |
| `--duration` | `4.0` | Render duration in seconds |
| `--note` | `60` | MIDI note (60 = C4) |
| `--velocity` | `100` | MIDI velocity (0-127) |

### Step 2: Optimize

```bash
synth2surge optimize \
  --target ./workspace/target_audio.wav \
  --output-dir ./workspace
```

Produces:
- `workspace/best_patch.bin` — Surge XT plugin state
- `workspace/best_patch.fxp` — Surge XT preset file (load in any DAW)
- `workspace/best_audio.wav` — rendered audio of the best match

| Flag | Default | Description |
|---|---|---|
| `--target` | *(required)* | Path to target audio WAV file |
| `--output-dir` | `./workspace` | Output directory |
| `--surge-plugin` | `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` | Path to Surge XT |
| `--trials-t1` | `300` | Trials for tier 1 (structural: osc types, filters, ADSR, levels) |
| `--trials-t2` | `300` | Trials for tier 2 (shaping: osc params, LFOs, pitch, width) |
| `--trials-t3` | `200` | Trials for tier 3 (detail: FX, modulation depths) |
| `--stages` | `1,2,3` | Which stages to run (comma-separated) |
| `--probe-mode` | `single` | MIDI probe mode: `single`, `thorough`, `full` |
| `--warm-start` | `false` | Use ML model to predict initial params (requires PyTorch + trained model) |

**Quick test** (stage 1 only, fewer trials):

```bash
synth2surge optimize --target ./workspace/target_audio.wav --stages 1 --trials-t1 50
```

**With ML warm-start** (after training):

```bash
synth2surge optimize --target ./workspace/target_audio.wav --warm-start
```

### Staged optimization explained

CMA-ES works well up to ~200 dimensions. Surge XT has ~280 active parameters, so the optimizer runs in 3 stages, each tackling a different tier while freezing previous results:

| Stage | Tier | Parameters | What it controls |
|---|---|---|---|
| 1 | Structural | ~45 | Oscillator types, filter types/cutoff/resonance, ADSR envelopes, mixer levels, FM depth |
| 2 | Shaping | ~60 | Oscillator shape params, pitch/octave, LFO rates, filter balance, drift, feedback |
| 3 | Detail | ~175 | Everything else: FX, modulation depths, secondary settings |

---

## ML Training: Self-Improving Mode

The ML system is what makes Synth2Surge unique. It has a **closed training loop** — the system generates its own labeled training data without any human involvement.

### The core idea

1. **Randomize** all ~280 Surge XT parameters to uniform [0, 1]
2. **Render** audio through Surge XT with semi-random MIDI (random notes, velocities, durations)
3. **Extract** a 512-dimensional audio feature vector (mel-spectrogram statistics)
4. **Store** the pair `(audio_features, parameters)` — this IS the ground truth label, no human labeling needed
5. **Train** a neural network to predict parameters from audio features
6. **Use** the trained model to warm-start CMA-ES with a predicted initial guess
7. **Repeat** — better warm-starts lead to faster optimization, generating even more training data

At inference time, a user provides only an audio file. The model extracts audio features and predicts synth parameters. No MIDI information is needed — the model has learned the timbral characteristics of parameter combinations by hearing thousands of randomized patches.

### Step-by-step training

#### 1. Generate training data

```bash
# Fast mode: randomize params, render, store (~2 seconds per patch, no optimization)
synth2surge data generate --mode render-only --count 1000

# Slower mode: also runs CMA-ES to generate trial-level data (~5 min per patch)
synth2surge data generate --mode optimize --count 50 --trials 200

# Use Surge XT factory presets as ground truth (known-good, musically useful patches)
synth2surge data generate --mode factory --count 100
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `render-only` | `render-only` (fast), `optimize` (rich), or `factory` (known presets) |
| `--count` | `100` | Number of patches to generate |
| `--surge-plugin` | `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` | Path to Surge XT |
| `--db-path` | `./workspace/experience.db` | SQLite database path |
| `--trials` | `200` | CMA-ES trials per run (optimize mode only) |
| `--seed` | `42` | Random seed for reproducibility |
| `--resume` | `true` | Skip if store already has enough data |

**Data generation modes compared:**

| Mode | Speed | What it stores | Best for |
|---|---|---|---|
| `render-only` | ~1,800 patches/hour | Ground truth `(features, params)` pairs | Bootstrapping: accumulate lots of data fast |
| `optimize` | ~12 patches/hour | Ground truth + ~200 `(params, loss)` trial rows per run | Richer training signal for the audio encoder |
| `factory` | ~1,800 patches/hour | Factory preset ground truth pairs | High-quality, musically meaningful data |

#### 2. Check data status

```bash
synth2surge data status
```

Shows total runs, trials, loss statistics, and how close you are to model readiness thresholds.

#### 3. Train the model

```bash
synth2surge train run
```

Trains a FeatureMLP predictor on all accumulated data. Saves a versioned checkpoint to `workspace/models/`.

| Flag | Default | Description |
|---|---|---|
| `--db-path` | `./workspace/experience.db` | Experience store path |
| `--models-dir` | `./workspace/models` | Checkpoint output directory |
| `--max-epochs` | `200` | Maximum training epochs |
| `--patience` | `20` | Early stopping patience |
| `--lr` | `0.001` | Learning rate |

#### 4. Check training status

```bash
synth2surge train status
```

Shows data counts, model versions, and whether you've hit the thresholds for MLP (50 runs) or CNN (200 runs) training.

#### 5. Use the trained model

```bash
synth2surge optimize --target audio.wav --warm-start
```

The `--warm-start` flag loads the latest trained model, predicts initial parameters, estimates confidence via MC Dropout, and either:
- **High confidence** (>0.6): Uses the prediction with tight CMA-ES search (sigma=0.15)
- **Medium confidence** (0.3-0.6): Uses the prediction with moderate search (sigma=0.3)
- **Low confidence** (<0.3): Falls back to default random CMA-ES initialization (sigma=0.4)

The system **never performs worse** than classical CMA-ES — low confidence triggers a graceful fallback.

### Autonomous training loop

For hands-off training, run the full cycle in one command:

```bash
synth2surge train loop \
  --cycles 10 \
  --patches-per-cycle 200 \
  --trials 200
```

Each cycle automatically:
1. Generates random patches (90% render-only, 10% with optimization)
2. Trains/retrains the predictor on all accumulated data
3. Reports metrics (loss, model version, data count)

Leave it running overnight or over a weekend to build up training data.

| Flag | Default | Description |
|---|---|---|
| `--cycles` | `10` | Number of generate-then-train cycles |
| `--patches-per-cycle` | `100` | Patches to generate per cycle |
| `--trials` | `200` | CMA-ES trials for the optimization fraction |
| `--seed` | `42` | Random seed |

### Data milestones

| Training examples | What unlocks |
|---|---|
| 10 | Minimum to attempt training (may not be useful yet) |
| 50 | FeatureMLP starts providing useful tier-1 warm-starts |
| 200 | SpectrogramCNN model becomes viable with per-tier prediction heads |
| 1,000+ | Model reliable enough to consider reducing CMA-ES trial budgets |
| 5,000+ | Model could potentially replace CMA-ES for familiar sound categories |

### Storage estimates

| Scale | Database size |
|---|---|
| 1,000 render-only runs | ~5 MB |
| 1,000 runs with 200 trials each | ~1 GB |
| 10,000 render-only runs | ~50 MB |

---

## Other Commands

### Build the FAISS prior index

Pre-render Surge XT factory patch variations for nearest-neighbor warm-starting:

```bash
synth2surge build-prior --max-patches 100 --variations 5
```

### Inspect a patch

```bash
synth2surge inspect --patch "/Library/Application Support/Surge XT/patches_factory/Leads/Acidofil.fxp"
```

### REST API server

```bash
synth2surge serve --host 127.0.0.1 --port 8000
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/capture` | Start a headless capture job |
| `POST` | `/optimize` | Start an optimization job |
| `GET` | `/jobs/{job_id}` | Poll job status and progress |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a running job |
| `GET` | `/prior/status` | Check FAISS prior index status |

---

## Architecture

### System overview

Synth2Surge is a pure Python application built on [Spotify's pedalboard](https://github.com/spotify/pedalboard) for VST3/AU plugin hosting. The ML subsystem is optional — the classical CMA-ES optimizer works standalone.

```
┌─────────────────────────────────────────────────────────────────┐
│                         SYNTH2SURGE                              │
│                                                                   │
│  ┌───────────┐   ┌────────────────┐   ┌──────────────────────┐  │
│  │  CLI      │   │  REST API      │   │  ML Training Loop    │  │
│  │  (typer)  │   │  (FastAPI)     │   │  (autonomous)        │  │
│  └─────┬─────┘   └───────┬────────┘   └──────────┬───────────┘  │
│        │                  │                       │               │
│  ┌─────▼──────────────────▼───────────────────────▼───────────┐  │
│  │                    CORE PIPELINE                             │  │
│  │                                                              │  │
│  │  Capture ──▶ Optimize ──▶ Export                             │  │
│  │              ▲      │                                        │  │
│  │              │      ▼                                        │  │
│  │        ┌─────┴──────────┐    ┌──────────────────────┐       │  │
│  │        │  Loss Function │    │  ML Subsystem         │       │  │
│  │        │  (Enriched     │    │  (PyTorch, optional)  │       │  │
│  │        │   MR-STFT)     │    │                       │       │  │
│  │        └────────────────┘    │  Experience Store     │       │  │
│  │                              │  Predictor (MLP/CNN)  │       │  │
│  │  ┌──────────────────────┐    │  Warm Starter         │       │  │
│  │  │  Plugin Host         │    │  Audio Encoder        │       │  │
│  │  │  (pedalboard)        │    │  Hybrid Loss          │       │  │
│  │  │  Surge XT raw_value  │    └──────────────────────┘       │  │
│  │  └──────────────────────┘                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Module map

```
src/synth2surge/
├── audio/                  # Plugin hosting & rendering
│   ├── engine.py           #   PluginHost: pedalboard wrapper (load, render, state, raw_value API)
│   ├── midi.py             #   MIDI probe generation (single/thorough/full multi-probe system)
│   └── renderer.py         #   High-level render pipeline
│
├── loss/                   # Audio comparison (objective functions)
│   ├── mr_stft.py          #   Multi-Resolution STFT loss (spectral convergence + log-magnitude)
│   ├── enriched.py         #   Enriched loss: MR-STFT + MFCC + envelope + centroid + flux
│   └── features.py         #   512-dim mel feature vectors (+ 128-dim learned encoder features)
│
├── ml/                     # Machine learning subsystem (optional, requires PyTorch)
│   ├── experience_store.py #   SQLite store for optimization runs and trials
│   ├── data_generator.py   #   Autonomous self-play data generation (render-only / optimize / factory)
│   ├── predictor.py        #   FeatureMLP and SpectrogramCNN models (audio → params)
│   ├── trainer.py          #   Training loop: tier-weighted MSE, AdamW, cosine annealing, early stopping
│   ├── warm_start.py       #   CMA-ES warm-start from ML predictions with MC Dropout confidence
│   ├── encoder.py          #   AudioEncoder CNN: log-mel spectrogram → 128-dim L2-normalized embedding
│   ├── hybrid_loss.py      #   Hybrid loss: enriched MR-STFT + learned audio similarity (alpha-blended)
│   ├── triplet_dataset.py  #   Triplet mining from optimization trial data for encoder training
│   └── training_loop.py    #   Autonomous generate → train → evaluate loop
│
├── surge/                  # Surge XT patch management
│   ├── patch.py            #   XML/FXP parser, writer, mutator
│   ├── fxp_export.py       #   FXP header construction (CcnK format)
│   ├── preset_loader.py    #   Auto-discover parameter mapping, load FXP → plugin
│   ├── parameter_space.py  #   Parameter definitions, bounds, 3-tier classification
│   └── factory.py          #   Factory patch discovery
│
├── prior/                  # FAISS nearest-neighbor index
│   ├── generator.py        #   Gaussian patch variation generator
│   └── index.py            #   FAISS IndexFlatIP (brute-force cosine similarity)
│
├── optimizer/              # CMA-ES optimization
│   └── loop.py             #   Multi-stage loop: 3-tier CMA-ES with parameter freezing
│
├── capture/                # Source plugin capture
│   └── workflow.py         #   GUI and headless capture workflows
│
├── api/                    # REST API
│   ├── app.py              #   FastAPI application factory
│   ├── routes.py           #   Endpoints with background job management
│   └── schemas.py          #   Pydantic request/response models
│
├── cli/                    # Command-line interface
│   └── main.py             #   Typer CLI: capture, optimize, data, train, build-prior, inspect, serve
│
├── config.py               #   All configuration: audio, MIDI, loss, ML, enriched loss, learned loss
└── types.py                #   Shared dataclasses: CaptureResult, OptimizationResult, MLPrediction
```

### Key design decisions

#### Plugin hosting via pedalboard's `raw_value` API

Surge XT exposes ~280 active parameters through pedalboard, each with a `raw_value` property normalized to [0, 1]. The optimizer works directly with these raw values, bypassing XML patch manipulation during the optimization loop. The plugin handles internal domain mapping (cutoff in Hz, time in log-seconds, etc.) transparently.

#### Enriched MR-STFT loss

The base loss (MR-STFT) computes spectral convergence and log-magnitude distance at 3 FFT resolutions (2048, 1024, 512). The enriched version adds four complementary metrics that fill perceptual gaps:

| Component | Weight | What it captures | MR-STFT gap it fills |
|---|---|---|---|
| MR-STFT | 0.60 | Spectral magnitude distribution | *(baseline)* |
| MFCC distance | 0.15 | Timbral shape (vocal-tract-like filtering) | Coarse spectral shape |
| Envelope correlation | 0.10 | Amplitude contour (attack, sustain, release) | Temporal dynamics |
| Spectral centroid | 0.10 | Brightness evolution over time | Time-varying character |
| Spectral flux | 0.05 | Rate of spectral change (transients, modulation) | Temporal texture |

All components use librosa (already a dependency). Total overhead: ~5ms per evaluation.

#### Staged CMA-ES optimization

CMA-ES works well up to ~200 dimensions. With ~280 active parameters, the 3-tier staged approach optimizes the most impactful parameters first, then freezes those values and refines secondary parameters, then detail parameters. Each stage runs an independent CMA-ES sampler.

#### ML parameter predictor (two architectures)

**FeatureMLP** (~800K parameters, used when < 200 training examples):
```
Input: 512-dim mel-spectrogram statistics
  → Linear(512, 512) + LayerNorm + GELU + Dropout(0.1)
  → Linear(512, 512) + LayerNorm + GELU + Dropout(0.1)
  → Linear(512, N_params) + Sigmoid
Output: [0,1] parameter predictions for all ~280 Surge XT params
```

**SpectrogramCNN** (~350K parameters, used when >= 200 training examples):
```
Input: Log-mel spectrogram [1, 128, T]
  → 4-layer CNN encoder (32→64→128→256 channels, stride-2, BatchNorm, ReLU)
  → AdaptiveAvgPool → Flatten → 256-dim embedding
  → Per-tier prediction heads:
      Tier 1 (structural, ~45 params): 256 → 128 → N → Sigmoid
      Tier 2 (shaping, ~60 params):    256 → 128 → N → Sigmoid
      Tier 3 (detail, ~175 params):    256 → 64  → N → Sigmoid
```

Training uses **tier-weighted MSE loss** (tier 1: 3x, tier 2: 1.5x, tier 3: 1x) because getting oscillator type and filter cutoff right matters far more than modulation depth.

#### Warm-start confidence via MC Dropout

At inference, the model runs 10 forward passes with dropout enabled. The variance across predictions gives per-parameter uncertainty. Overall confidence = `1 - mean(std)`. This determines how tightly to initialize CMA-ES around the prediction:

| Confidence | Action | CMA-ES sigma |
|---|---|---|
| > 0.6 | Use prediction, tight search | 0.15 |
| 0.3 - 0.6 | Use prediction, moderate search | 0.30 |
| < 0.3 | **Fall back to default** (no warm-start) | 0.40 |

#### Audio encoder and hybrid loss

The AudioEncoder is a separate CNN that maps spectrograms to 128-dim L2-normalized embeddings. Trained via triplet margin ranking loss on optimization trial data — it learns that low-loss candidates should embed closer to the target than high-loss ones.

The hybrid loss blends enriched MR-STFT with learned similarity:
```
hybrid = (1 - alpha) * enriched_loss + alpha * learned_distance
```
Alpha starts at 0 and increases to max 0.3 as the encoder validates against held-out data. The enriched MR-STFT always remains dominant to prevent adversarial gaming.

#### Experience store (SQLite)

All optimization runs and trials are logged to a SQLite database with numpy array BLOBs. Schema:

- **`runs`** table: run_id, target_features (512-dim), best_params, ground_truth_params, best_loss, generation_mode, model_version
- **`trials`** table: run_id, stage, trial_idx, params, loss
- **`model_versions`** table: version_id, training metrics, checkpoint path

This is the foundation everything else builds on — every optimization run generates free training data.

### Dependencies

| Package | Purpose |
|---|---|
| `pedalboard` | VST3/AU plugin hosting, MIDI rendering, parameter access |
| `librosa` | STFT computation, mel spectrograms, MFCC, spectral features |
| `numpy` | Numerical operations |
| `soundfile` | WAV file I/O |
| `optuna` + `cmaes` | CMA-ES optimization framework |
| `faiss-cpu` | Nearest-neighbor similarity search |
| `lxml` | Surge XT XML patch parsing |
| `fastapi` + `uvicorn` | REST API server |
| `typer` + `rich` | CLI framework with progress bars |
| `pydantic` + `pydantic-settings` | Configuration and API schema validation |
| `mido` | MIDI message handling |
| `torch` *(optional)* | Neural networks for parameter prediction and learned audio similarity |

---

## Testing

207 unit tests covering all modules, plus integration, acceptance, and end-to-end tests.

```bash
# All unit tests (no plugins required, ~3 seconds)
uv run --extra dev pytest tests/unit/

# All tests (requires Surge XT for integration/e2e)
uv run --extra dev pytest

# With coverage
uv run --extra dev pytest --cov=synth2surge --cov-report=term-missing

# Lint
uv run --extra dev ruff check src/ tests/
```

Tests that require Surge XT are marked `@pytest.mark.requires_surge` and auto-skip if it's not installed. ML tests auto-skip if PyTorch is not installed.

| Level | What it validates |
|---|---|
| **Unit** | Loss math, XML parsing, parameter normalization, MIDI generation, FAISS indexing, CLI args, API routes, experience store, ML models, training, warm-start |
| **Integration** | Plugin loading, audio rendering, state round-trip, optimizer convergence |
| **Acceptance** | Loss ranking correctness, feature similarity, self-translation quality |
| **E2E** | Full capture → optimize pipeline, CLI workflow |

## License

GPLv3 -- see [LICENSE](LICENSE).
