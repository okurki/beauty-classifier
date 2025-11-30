import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.types import ASGIApp

INFO = Gauge("fastapi_app_info", "FastAPI application information.", ["app_name"])
REQUESTS = Counter(
    "fastapi_requests_total",
    "Total count of requests by method and path.",
    ["method", "path", "app_name"],
)
RESPONSES = Counter(
    "fastapi_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path", "status_code", "app_name"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path", "app_name"],
)
EXCEPTIONS = Counter(
    "fastapi_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path", "exception_type", "app_name"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "fastapi_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path", "app_name"],
)

# RL Agent Metrics
RL_CUMULATIVE_REWARD = Gauge(
    "rl_cumulative_reward",
    "Cumulative reward (likes - dislikes) from RL agent",
    ["app_name"],
)
RL_AVERAGE_REWARD = Gauge(
    "rl_average_reward",
    "Average reward per impression from RL agent",
    ["app_name"],
)
RL_EXPLORATION_RATE = Gauge(
    "rl_exploration_rate",
    "Percentage of exploratory actions by RL agent",
    ["app_name"],
)
RL_CTR = Gauge(
    "rl_ctr",
    "Click-through rate (feedback rate) for RL agent",
    ["app_name"],
)
RL_LIKE_RATE = Gauge(
    "rl_like_rate",
    "Percentage of feedback that was positive",
    ["app_name"],
)
RL_TOTAL_IMPRESSIONS = Counter(
    "rl_total_impressions",
    "Total number of celebrity impressions",
    ["app_name"],
)
RL_TOTAL_LIKES = Counter(
    "rl_total_likes",
    "Total number of likes received",
    ["app_name"],
)
RL_TOTAL_DISLIKES = Counter(
    "rl_total_dislikes",
    "Total number of dislikes received",
    ["app_name"],
)
RL_CELEBRITY_ALPHA = Gauge(
    "rl_celebrity_alpha",
    "Beta distribution alpha parameter for celebrity",
    ["celebrity_id", "celebrity_name", "app_name"],
)
RL_CELEBRITY_BETA = Gauge(
    "rl_celebrity_beta",
    "Beta distribution beta parameter for celebrity",
    ["celebrity_id", "celebrity_name", "app_name"],
)
RL_CELEBRITY_MEAN_PROB = Gauge(
    "rl_celebrity_mean_probability",
    "Mean success probability for celebrity",
    ["celebrity_id", "celebrity_name", "app_name"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, app_name: str = "fastapi-app") -> None:
        super().__init__(app)
        self.app_name = app_name
        INFO.labels(app_name=self.app_name).inc()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path, is_handled_path = self.get_path(request)

        if not is_handled_path:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(
            method=method, path=path, app_name=self.app_name
        ).inc()
        REQUESTS.labels(method=method, path=path, app_name=self.app_name).inc()
        before_time = time.perf_counter()
        try:
            response = await call_next(request)
        except BaseException as e:
            status_code = HTTP_500_INTERNAL_SERVER_ERROR
            EXCEPTIONS.labels(
                method=method,
                path=path,
                exception_type=type(e).__name__,
                app_name=self.app_name,
            ).inc()
            raise e from None
        else:
            status_code = response.status_code
            after_time = time.perf_counter()
            # retrieve trace id for exemplar
            span = trace.get_current_span()
            trace_id = trace.format_trace_id(span.get_span_context().trace_id)

            REQUESTS_PROCESSING_TIME.labels(
                method=method, path=path, app_name=self.app_name
            ).observe(after_time - before_time, exemplar={"TraceID": trace_id})
        finally:
            RESPONSES.labels(
                method=method,
                path=path,
                status_code=status_code,
                app_name=self.app_name,
            ).inc()
            REQUESTS_IN_PROGRESS.labels(
                method=method, path=path, app_name=self.app_name
            ).dec()

        return response

    @staticmethod
    def get_path(request: Request) -> tuple[str, bool]:
        for route in request.app.routes:
            match, child_scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True

        return request.url.path, False


def update_rl_metrics(app_name: str = "fastapi-app") -> None:
    """
    Update Prometheus metrics from RL agent state.
    Should be called periodically or after feedback updates.
    """
    from src.infrastructure.ml_models.rl.agent import rl_agent
    
    try:
        stats = rl_agent.get_global_stats()
        
        # Update global metrics
        RL_CUMULATIVE_REWARD.labels(app_name=app_name).set(stats["cumulative_reward"])
        RL_AVERAGE_REWARD.labels(app_name=app_name).set(stats["average_reward"])
        RL_EXPLORATION_RATE.labels(app_name=app_name).set(stats["exploration_rate"])
        RL_CTR.labels(app_name=app_name).set(stats["ctr"])
        RL_LIKE_RATE.labels(app_name=app_name).set(stats["like_rate"])
        
        # Update celebrity-specific metrics for top celebrities
        top_celebrities = rl_agent.get_top_celebrities(top_k=20)
        for celeb in top_celebrities:
            celeb_id = str(celeb["celebrity_id"])
            celeb_name = celeb["celebrity_name"]
            
            celeb_stats = rl_agent.get_celebrity_stats(celeb["celebrity_id"])
            if celeb_stats:
                RL_CELEBRITY_ALPHA.labels(
                    celebrity_id=celeb_id,
                    celebrity_name=celeb_name,
                    app_name=app_name
                ).set(celeb_stats["alpha"])
                
                RL_CELEBRITY_BETA.labels(
                    celebrity_id=celeb_id,
                    celebrity_name=celeb_name,
                    app_name=app_name
                ).set(celeb_stats["beta"])
                
                RL_CELEBRITY_MEAN_PROB.labels(
                    celebrity_id=celeb_id,
                    celebrity_name=celeb_name,
                    app_name=app_name
                ).set(celeb_stats["mean_probability"])
    except Exception as e:
        # Don't fail if metrics update fails
        import logging
        logging.getLogger(__name__).warning(f"Failed to update RL metrics: {e}")


def setting_otlp(
    app: ASGIApp, app_name: str, endpoint: str, log_correlation: bool = True
) -> None:
    # Setting OpenTelemetry
    # set the service name to show in traces
    resource = Resource.create(attributes={"service.name": app_name})

    # set the tracer provider
    tracer = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer)

    tracer.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )

    if log_correlation:
        LoggingInstrumentor().instrument(set_logging_format=True)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer)
