import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from model_runtime import (
    CircuitBreaker,
    CircuitOpenError,
    ContextOverflowError,
    FailingProvider,
    GenerateRequest,
    GenerateResponse,
    ModelNotFoundError,
    ModelRouter,
    ModelSpec,
    ProviderUnavailableError,
    SimulatedProvider,
)


def make_spec(name: str = "mini-7b", window: int = 4096) -> ModelSpec:
    return ModelSpec(name=name, backend="simulated",
                     context_window=window, parameters_billion=7.0)


def test_spec_rejects_tiny_context():
    with pytest.raises(ValueError):
        make_spec(window=64)


def test_request_temperature_bounds():
    with pytest.raises(ValueError):
        GenerateRequest(model="m", prompt="hi", temperature=3.5)


def test_simulated_provider_generates_echo():
    provider = SimulatedProvider([make_spec()])
    response = provider.generate(GenerateRequest(model="mini-7b", prompt="سلام"))
    assert "echo" in response.text
    assert response.backend == "simulated"


def test_unknown_model_raises():
    provider = SimulatedProvider([make_spec()])
    with pytest.raises(ModelNotFoundError):
        provider.generate(GenerateRequest(model="ghost", prompt="x"))


def test_overflow_detected_before_generation():
    provider = SimulatedProvider([make_spec(window=128)])
    with pytest.raises(ContextOverflowError):
        provider.generate(GenerateRequest(model="mini-7b", prompt="x", max_tokens=500))


def test_failing_provider_reports_unavailable():
    provider = FailingProvider()
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerateRequest(model="anything", prompt="x"))


def test_router_prefers_healthy_provider():
    primary = SimulatedProvider([make_spec("main-model")])
    router = ModelRouter([primary])
    response = router.route(GenerateRequest(model="main-model", prompt="hello"))
    assert response.backend == "simulated"


def test_router_falls_back_on_provider_failure():
    broken = FailingProvider()
    healthy = SimulatedProvider([
        ModelSpec(name="shared-model", backend="failing",
                  context_window=4096, parameters_billion=7.0)
    ])
    router = ModelRouter([broken, healthy])
    response = router.route(GenerateRequest(model="shared-model", prompt="go"))
    assert response.backend == "simulated"


def test_circuit_opens_after_threshold():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60.0)
    def always_fails():
        raise ProviderUnavailableError("down")

    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            breaker.call(always_fails)

    with pytest.raises(CircuitOpenError):
        breaker.call(always_fails)


def test_circuit_success_resets_failures():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60.0)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ProviderUnavailableError("blip")
        return "ok"

    with pytest.raises(ProviderUnavailableError):
        breaker.call(flaky)
    assert breaker.call(flaky) == "ok"
    assert not breaker.is_open


def test_router_skips_open_circuit_and_uses_backup():
    broken = FailingProvider()
    healthy = SimulatedProvider([
        ModelSpec(name="dual", backend="failing",
                  context_window=2048, parameters_billion=3.0),
        ModelSpec(name="dual", backend="simulated",
                  context_window=2048, parameters_billion=3.0),
    ])

    class DualCatalog(SimulatedProvider):
        def list_models(self):
            return [ModelSpec("dual", "simulated", 2048, 3.0)]

    backup = DualCatalog([ModelSpec("dual", "simulated", 2048, 3.0)])
    router = ModelRouter([broken, backup], breaker_threshold=1, cooldown=60.0)
    response = router.route(GenerateRequest(model="dual", prompt="test"))
    assert response.backend == "simulated"


def test_router_unknown_model_everywhere():
    router = ModelRouter([SimulatedProvider([make_spec()])])
    with pytest.raises(ModelNotFoundError):
        router.route(GenerateRequest(model="nonexistent", prompt="q"))
