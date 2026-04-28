# Stage 1: Builder phase for installing dependencies [cite: 60]
FROM python:3.11-slim as builder

# Set working directory for build
WORKDIR /build

# Copy requirements and install packages without cache to reduce layer size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Stage 2: Runtime phase [cite: 60]
FROM python:3.11-slim

# Create a non-root user to avoid running the application as root
RUN useradd -m -r appuser

WORKDIR /app

# Copy installed packages and binaries from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code [cite: 60]
COPY app/ /app/app/

# Restrict permissions to the non-root user
RUN chown -R appuser:appuser /app

# Switch context to non-root user
USER appuser

# Define the default entrypoint to run the FastAPI application [cite: 60]
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
