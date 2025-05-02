# model-runtime

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local LLM runtime with provider abstraction, circuit-breaker routing, and typed failure codes — run models on your own hardware without hard-wiring any backend.

## 🚀 Overview

The flagship AI-infrastructure runtime of the 2025 roadmap. `model-runtime` abstracts *where inference happens* behind a `Provider` protocol: register Ollama, llama.cpp, or a simulated backend, and the **ModelRouter** finds a healthy provider that knows the requested model — skipping providers whose circuit breaker has tripped. Failures are classified by frozen enum codes (`MODEL_NOT_FOUND`, `CONTEXT_OVERFLOW`, `CIRCUIT_OPEN`, …) so callers react precisely instead of parsing strings.

## ✨ Features

- **Provider protocol:** structural typing; any backend with list/generate/is_available plugs in
- **Circuit breaker:** per-provider failure counting, open state after threshold, cooldown-based recovery; success resets
- **Failover routing:** tries providers in order, skips unknown models and open circuits, reports every attempt's reason
- **Typed error codes:** enum-coded exceptions (`RuntimeErrorCode`) under one base class
- **Context guard:** overflow detected before generation starts
- **Deterministic simulation:** `SimulatedProvider` for tests/CI with no hardware needed
- **Zero dependencies**

## 🚧 Structure

```
local-model-runtime/
├── src/model_runtime/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/local-model-runtime.git
cd local-model-runtime
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from model_runtime import GenerateRequest, ModelRouter, ModelSpec, SimulatedProvider

primary = SimulatedProvider([
    ModelSpec(name="mini-7b", backend="simulated",
              context_window=4096, parameters_billion=7.0),
])

router = ModelRouter([primary])
response = router.route(GenerateRequest(model="mini-7b", prompt="سلام دنیا"))
print(response.text, response.latency_ms)
```

### Failover in action

```python
router = ModelRouter([failing_provider, backup_provider])
response = router.route(request)
print(f"served by {response.backend}")
```

## 🔧 Error Handling

```text
RuntimeError_ (code-carrying base)
├── ModelNotFoundError        # no healthy provider knows the model
├── ProviderUnavailableError  # backend down
├── ContextOverflowError      # max_tokens > context window
├── GenerationTimeoutError    # reserved for async paths
└── CircuitOpenError          # breaker tripped, cooldown active
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen specs/requests/responses
- Zero comments — names carry the meaning
- Circuit behavior tested deterministically (threshold + reset + skip)

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
