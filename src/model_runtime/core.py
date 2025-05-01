from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence


class RuntimeErrorCode(str, Enum):
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CONTEXT_OVERFLOW = "context_overflow"
    GENERATION_TIMEOUT = "generation_timeout"
    CIRCUIT_OPEN = "circuit_open"


class RuntimeError_(Exception):
    code: RuntimeErrorCode = RuntimeErrorCode.PROVIDER_UNAVAILABLE


class ModelNotFoundError(RuntimeError_):
    code = RuntimeErrorCode.MODEL_NOT_FOUND


class ProviderUnavailableError(RuntimeError_):
    code = RuntimeErrorCode.PROVIDER_UNAVAILABLE


class ContextOverflowError(RuntimeError_):
    code = RuntimeErrorCode.CONTEXT_OVERFLOW


class GenerationTimeoutError(RuntimeError_):
    code = RuntimeErrorCode.GENERATION_TIMEOUT


class CircuitOpenError(RuntimeError_):
    code = RuntimeErrorCode.CIRCUIT_OPEN


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    context_window: int
    parameters_billion: float

    def __post_init__(self) -> None:
        if self.context_window < 128:
            raise ValueError("context_window must be >= 128")


@dataclass(frozen=True)
class GenerateRequest:
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within [0, 2]")


@dataclass(frozen=True)
class GenerateResponse:
    text: str
    tokens_generated: int
    latency_ms: float
    backend: str


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None
    threshold: int = 3
    cooldown_seconds: float = 30.0

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        return (time.monotonic() - self.opened_at) < self.cooldown_seconds

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


class Provider(Protocol):
    backend_name: str

    def list_models(self) -> Sequence[ModelSpec]: ...

    def generate(self, request: GenerateRequest) -> GenerateResponse: ...

    def is_available(self) -> bool: ...


class SimulatedProvider:
    backend_name = "simulated"

    def __init__(self, models: Sequence[ModelSpec], fail_rate: float = 0.0) -> None:
        self._models = {spec.name: spec for spec in models}
        self._fail_rate = fail_rate

    def list_models(self) -> Sequence[ModelSpec]:
        return tuple(self._models.values())

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        started = time.perf_counter()
        spec = self._models.get(request.model)
        if spec is None:
            raise ModelNotFoundError(request.model)
        if request.max_tokens > spec.context_window:
            raise ContextOverflowError(
                f"max_tokens {request.max_tokens} exceeds window {spec.context_window}"
            )
        output = f"[{spec.name}] echo({len(request.prompt)} chars)"
        latency = (time.perf_counter() - started) * 1000 + 1.0
        return GenerateResponse(
            text=output,
            tokens_generated=min(request.max_tokens, len(output.split())),
            latency_ms=round(latency, 3),
            backend=self.backend_name,
        )


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown_seconds: float = 5.0) -> None:
        self._state = CircuitState(threshold=threshold, cooldown_seconds=cooldown_seconds)

    @property
    def is_open(self) -> bool:
        return self._state.is_open

    def call(self, operation, *args, **kwargs):
        if self._state.is_open:
            raise CircuitOpenError("circuit is open; try later")
        try:
            result = operation(*args, **kwargs)
        except RuntimeError_:
            self._state.record_failure()
            raise
        self._state.record_success()
        return result


class FailingProvider(SimulatedProvider):
    backend_name = "failing"

    def __init__(self) -> None:
        super().__init__(models=[])
        self._attempts = 0

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        self._attempts += 1
        raise ProviderUnavailableError(f"backend down after {self._attempts} tries")


class ModelRouter:
    def __init__(self, providers: Sequence[Provider],
                 breaker_threshold: int = 2, cooldown: float = 1.0) -> None:
        if not providers:
            raise ProviderUnavailableError("no providers registered")
        self._providers = list(providers)
        self._breakers = {
            provider.backend_name: CircuitBreaker(breaker_threshold, cooldown)
            for provider in providers
        }

    def route(self, request: GenerateRequest) -> GenerateResponse:
        errors: list[str] = []
        for provider in self._providers:
            known = {spec.name for spec in provider.list_models()}
            if request.model not in known and provider.backend_name != "fallback":
                continue
            breaker = self._breakers[provider.backend_name]
            try:
                return breaker.call(provider.generate, request)
            except CircuitOpenError as exc:
                errors.append(f"{provider.backend_name}: circuit open")
            except RuntimeError_ as exc:
                errors.append(f"{provider.backend_name}: {exc.code.value}")
        detail = "; ".join(errors) or "model unknown on all providers"
        raise ModelNotFoundError(f"routing failed for {request.model!r}: {detail}")

    def available_backends(self) -> tuple[str, ...]:
        return tuple(p.backend_name for p in self._providers if p.is_available())
