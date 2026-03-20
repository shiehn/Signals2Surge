# Synth2Surge

> Part of the [Signals & Sorcery](https://signalsandsorcery.com) family of apps.

Translate arbitrary VST synth patches (Serum, Vital, etc.) into sonically similar [Surge XT](https://surge-synthesizer.github.io/) patches via black-box optimization.

Synth2Surge loads a source synthesizer plugin, captures a preset's audio fingerprint, then uses CMA-ES evolutionary optimization to iteratively search Surge XT's parameter space for a patch that sounds as close as possible to the original. The objective function is a Multi-Resolution STFT perceptual loss, which compares spectral content across multiple frequency resolutions to capture both tonal character and fine detail.

## How It Works

```
Source Plugin (e.g. Serum)         Surge XT
┌────────────────────────┐        ┌─────────────────────────┐
│  Load preset           │        │  775 parameters         │
│  Render MIDI probe ────┼──┐     │  (normalized 0-1)       │
│  = target_audio.wav    │  │     └──────────┬──────────────┘
└────────────────────────┘  │                │
                            │     ┌──────────▼──────────────┐
                            │     │  CMA-ES Optimizer       │
                            │     │  (Optuna)               │
                            ├────▶│                         │
                            │     │  Suggest params         │
                            │     │  Render candidate audio │
                            │     │  Compute MR-STFT loss   │
                            │     │  Repeat N trials        │
                            │     └──────────┬──────────────┘
                            │                │
                            │     ┌──────────▼──────────────┐
                            │     │  Best Surge XT Patch    │
                            │     │  (lowest loss)          │
                            │     └─────────────────────────┘
```

1. **Capture** -- Load your source synth plugin, select a preset (via native GUI or state file), render a fixed MIDI probe (C4, 4 seconds) to produce `target_audio.wav`
2. **Match** (optional) -- Query a FAISS index of pre-rendered Surge XT factory patch variations to find the closest starting point
3. **Optimize** -- Run multi-stage CMA-ES optimization across 3 parameter tiers (structural, shaping, detail) to minimize MR-STFT loss between candidate and target audio
4. **Export** -- Save the best-matching Surge XT patch state and rendered audio

## Requirements

- **OS:** macOS (Apple Silicon or Intel)
- **Python:** 3.11 or later
- **Surge XT:** Installed as VST3 at `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` (free download from [surge-synthesizer.github.io](https://surge-synthesizer.github.io/))
- **Source plugin:** Any VST3 or Audio Unit instrument plugin you want to translate from (e.g. Serum, Vital, Diva)

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/shiehn/signals-to-surge.git
cd signals-to-surge
```

### 3. Create virtual environment and install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This installs the `synth2surge` CLI command and all dependencies including test tools.

### 4. Verify installation

```bash
# Check the CLI is available
synth2surge --help

# Run the test suite (requires Surge XT installed for integration tests)
pytest
```

## Usage

### Capture a preset

Open the source plugin's native GUI, select your desired preset, then close the window to capture:

```bash
synth2surge capture \
  --plugin "/path/to/Serum.vst3" \
  --output-dir ./workspace
```

This produces:
- `workspace/target_audio.wav` -- rendered audio of the preset
- `workspace/target_state.bin` -- binary plugin state snapshot

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--plugin` | *(required)* | Path to source VST3/AU plugin |
| `--output-dir` | `./workspace` | Directory for output files |
| `--no-gui` | `false` | Skip the editor GUI, use the plugin's default state |
| `--state-file` | *none* | Load a saved `.vstpreset` or state file instead of opening GUI |
| `--duration` | `4.0` | Total render duration in seconds |
| `--note` | `60` | MIDI note number (60 = C4) |
| `--velocity` | `100` | MIDI velocity (0-127) |

### Optimize a Surge XT patch

Run the CMA-ES optimizer to find a Surge XT patch matching your target audio:

```bash
synth2surge optimize \
  --target ./workspace/target_audio.wav \
  --output-dir ./workspace \
  --trials-t1 300 \
  --trials-t2 300 \
  --trials-t3 200
```

This produces:
- `workspace/best_patch.bin` -- Surge XT plugin state (load via DAW)
- `workspace/best_audio.wav` -- rendered audio of the best match

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--target` | *(required)* | Path to target audio WAV file |
| `--output-dir` | `./workspace` | Directory for output files |
| `--surge-plugin` | `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` | Path to Surge XT VST3 |
| `--trials-t1` | `300` | Trials for tier 1 (oscillator types, filter cutoff, ADSR, levels) |
| `--trials-t2` | `300` | Trials for tier 2 (osc params, LFOs, pitch, width) |
| `--trials-t3` | `200` | Trials for tier 3 (remaining detail parameters) |
| `--stages` | `1,2,3` | Which optimization stages to run (comma-separated) |

**Staged optimization explained:**

The optimizer runs in up to 3 stages. Each stage optimizes a different tier of parameters while freezing the best values found in previous stages:

- **Stage 1 (Structural):** Oscillator types, filter types/cutoff/resonance, amplitude and filter envelope ADSR, mixer levels, FM depth. These have the biggest impact on timbre.
- **Stage 2 (Shaping):** Oscillator shape parameters, pitch/octave, LFO rates and depths, filter balance, drift, feedback. These refine the tonal character.
- **Stage 3 (Detail):** All remaining parameters (FX, modulation depths, secondary settings). Fine-tuning.

For a quick test, run only stage 1 with fewer trials:

```bash
synth2surge optimize --target ./workspace/target_audio.wav --stages 1 --trials-t1 50
```

### Build the FAISS prior index

Pre-render Surge XT factory patch variations to enable warm-start initialization:

```bash
synth2surge build-prior \
  --max-patches 100 \
  --variations 5
```

| Flag | Default | Description |
|---|---|---|
| `--factory-dir` | `/Library/Application Support/Surge XT/patches_factory` | Surge XT factory patches directory |
| `--surge-plugin` | `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` | Path to Surge XT VST3 |
| `--variations` | `5` | Number of Gaussian-mutated variations per factory patch |
| `--max-patches` | `50` | Maximum factory patches to process |
| `--output` | `./workspace/prior_index` | Output directory for the FAISS index |

### Inspect a Surge XT patch

View metadata and parameters of an `.fxp` or XML patch file:

```bash
synth2surge inspect --patch "/Library/Application Support/Surge XT/patches_factory/Leads/Acidofil.fxp"
```

### Start the REST API server

```bash
synth2surge serve --host 127.0.0.1 --port 8000
```

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/capture` | Start a headless capture job |
| `POST` | `/optimize` | Start an optimization job |
| `GET` | `/jobs/{job_id}` | Poll job status and progress |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a running job |
| `GET` | `/prior/status` | Check if FAISS prior index is built |

## Testing

The project has **140 tests** across 4 levels, achieving **84% code coverage**.

### Run all tests

```bash
pytest
```

### Run by test level

```bash
# Unit tests only (no plugins required, fast)
pytest tests/unit/

# Integration tests (requires Surge XT installed)
pytest tests/integration/

# Acceptance tests (quality validation, requires Surge XT)
pytest tests/acceptance/

# End-to-end tests (full pipeline, requires Surge XT)
pytest tests/e2e/
```

### Run with coverage report

```bash
pytest --cov=synth2surge --cov-report=term-missing
```

### Run linting

```bash
ruff check src/ tests/
```

### Test markers

Tests that require Surge XT are marked with `@pytest.mark.requires_surge` and auto-skip if the plugin is not installed. Slow tests (>30s) are marked with `@pytest.mark.slow`.

```bash
# Skip slow tests
pytest -m "not slow"

# Run only tests that don't need Surge XT
pytest -m "not requires_surge"
```

### Test breakdown

| Level | Count | Requires Surge XT | What it validates |
|---|---|---|---|
| **Unit** | 73 | No | Loss math, XML parsing, parameter normalization, MIDI generation, FAISS indexing, CLI args, API routes |
| **Integration** | 32 | Yes | Plugin loading, audio rendering, state round-trip, optimizer convergence, capture workflow |
| **Acceptance** | 4 | Partially | Loss ranking correctness, feature similarity, self-translation quality |
| **E2E** | 3 | Yes | Full capture-to-optimize pipeline, CLI workflow, inspect command |

## Architecture

### System overview

Synth2Surge is a pure Python application. The original design called for a separate C++/JUCE daemon for plugin hosting, but this was replaced with [Spotify's pedalboard](https://github.com/spotify/pedalboard) library, which wraps JUCE internally and exposes VST3/AU plugin hosting directly from Python.

```
src/synth2surge/
├── audio/                  # Plugin hosting & rendering
│   ├── engine.py           #   PluginHost: pedalboard wrapper (load, render, state, raw_value API)
│   ├── midi.py             #   MIDI probe generation (fixed note for consistent comparison)
│   └── renderer.py         #   High-level render pipeline
├── loss/                   # Objective function
│   ├── mr_stft.py          #   Multi-Resolution STFT loss (spectral convergence + log-magnitude)
│   └── features.py         #   512-dim mel feature vectors for FAISS indexing
├── surge/                  # Surge XT patch management
│   ├── patch.py            #   XML/FXP parser, writer, mutator (handles all 637 factory patches)
│   ├── parameter_space.py  #   Parameter definitions, bounds, 3-tier classification
│   └── factory.py          #   Factory patch discovery
├── prior/                  # FAISS nearest-neighbor index
│   ├── generator.py        #   Gaussian patch variation generator
│   └── index.py            #   FAISS IndexFlatIP (brute-force cosine similarity)
├── optimizer/              # CMA-ES optimization
│   ├── loop.py             #   Multi-stage optimization loop (the core)
│   └── strategies.py       #   Parameter injection, tier classification
├── capture/                # Source plugin capture
│   └── workflow.py         #   GUI and headless capture workflows
├── api/                    # REST API
│   ├── app.py              #   FastAPI application factory
│   ├── routes.py           #   Endpoints with background job management
│   └── schemas.py          #   Pydantic request/response models
├── cli/                    # Command-line interface
│   └── main.py             #   Typer CLI (capture, optimize, build-prior, inspect, serve)
├── config.py               #   Pydantic settings (audio, MIDI, loss, paths, optimization)
└── types.py                #   Shared dataclasses (CaptureResult, RenderResult, etc.)
```

### Key design decisions

**Plugin hosting via pedalboard's `raw_value` API:**
Surge XT exposes 775 parameters through pedalboard, each with a `raw_value` property normalized to [0, 1]. The optimizer works directly with these raw values, bypassing XML patch manipulation entirely during the optimization loop. This avoids the problem of mismatched parameter naming between Surge XT's internal XML format and pedalboard's exposed parameter names.

**Multi-Resolution STFT loss:**
The objective function computes spectral convergence and log-magnitude distance at 3 FFT resolutions (2048, 1024, 512). Spectral convergence captures large spectral peaks (the "shape" of the sound), while log-magnitude distance captures quieter details and noise floors. Multiple resolutions ensure both time resolution (small FFT) and frequency resolution (large FFT) are represented.

**Staged CMA-ES optimization:**
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) works well up to ~100-200 dimensions. With 775 total parameters, optimizing everything at once would converge poorly. The 3-tier staged approach optimizes the most impactful parameters first (oscillator/filter types, ADSR, levels), then freezes those values and refines secondary parameters, then detail parameters.

**FAISS prior index:**
Before optimization, the system can query a pre-built FAISS index of Surge XT factory patch audio fingerprints to find the closest starting point. This warm-starts the optimizer near a reasonable solution rather than searching from random parameters. The index uses L2-normalized 512-dim mel-spectrogram statistics with inner-product search (equivalent to cosine similarity).

### Dependencies

| Package | Purpose |
|---|---|
| `pedalboard` | VST3/AU plugin hosting, MIDI rendering, parameter access |
| `librosa` | STFT computation, mel spectrograms |
| `numpy` | Numerical operations |
| `soundfile` | WAV file I/O |
| `optuna` | CMA-ES optimization framework |
| `cmaes` | CMA-ES algorithm backend for Optuna |
| `faiss-cpu` | Nearest-neighbor similarity search |
| `lxml` | Surge XT XML patch parsing |
| `fastapi` / `uvicorn` | REST API server |
| `typer` / `rich` | CLI framework with progress bars |
| `pydantic` / `pydantic-settings` | Configuration and API schema validation |

## License

GPLv3 -- see [LICENSE](LICENSE).
