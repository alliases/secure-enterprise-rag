# Stage 1: Builder phase for installing dependencies
FROM python:3.13-slim as builder

WORKDIR /build

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy poetry config files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use virtualenvs, then install main dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Stage 2: Runtime phase
FROM python:3.13-slim

# Create a non-root user
RUN useradd -m -r appuser

WORKDIR /app

# Copy installed packages and binaries (Updated paths to python3.13)
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application source code
COPY app/ /app/app/

# Restrict permissions
RUN chown -R appuser:appuser /app

# Switch context
USER appuser

# Define the default entrypoint
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
