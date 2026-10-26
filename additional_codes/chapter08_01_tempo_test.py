import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# 1. Setup the Resource (identifies your service in Grafana)
resource = Resource(attributes={
    SERVICE_NAME: "tempo-test-python"
})

# 2. Setup the Tracer Provider
provider = TracerProvider(resource=resource)

# 3. Setup the OTLP Exporter (pointing to your Tempo gRPC port)
# 'insecure=True' is used because your local Tempo is not using TLS
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)

# 4. Add the Span Processor to the provider
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)
trace.set_tracer_provider(provider)

# 5. Create a Tracer and a Test Span
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("hello-tempo-span") as span:
    span.set_attribute("custom.message", "Observability is working!")
    print("Sending span to Tempo...")
    time.sleep(0.5)  # Simulate some work
    print("Span sent!")

# Ensure all spans are flushed before the script exits
provider.shutdown()
