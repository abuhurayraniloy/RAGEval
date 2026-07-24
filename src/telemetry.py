import time
import logging
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("uvicorn.error")

SERVICE_NAME = "rageval"


def setup_tracing() -> None:
    resource = Resource.create({"service.name": SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing intialized (console exporter).")


def get_tracer():
    return trace.get_tracer(SERVICE_NAME)


@contextmanager
def llm_span(span_name: str, model: str):
    tracer = get_tracer()
    start = time.perf_counter()

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("llm.model", model)

        token_data = {"prompt_tokens": None, "completion_tokens": None}

        def set_tokens(
            prompt_tokens: int | None, completion_tokens: int | None
        ) -> None:
            token_data["prompt_tokens"] = prompt_tokens
            token_data["completion_tokens"] = completion_tokens

        try:
            yield set_tokens
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("llm.latency_ms", round(latency_ms, 2))
            if token_data["prompt_tokens"] is not None:
                span.set_attribute("llm.prompt_tokens", token_data["prompt_tokens"])
            if token_data["completion_tokens"] is not None:
                span.set_attribute(
                    "llm.completion_tokens", token_data["completion_tokens"]
                )
                total = (token_data["prompt_tokens"] or 0) + token_data[
                    "completion_tokens"
                ]
                span.set_attribute("llm.total_tokens", total)
