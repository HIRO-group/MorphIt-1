# MorphIt FastAPI service — CPU-only.
#
# Stage 1 (builder): installs Python deps and compiles the Cython
# triangle_hash extension. Needs build-essential.
# Stage 2 (runtime): copies installed site-packages and the compiled
# .so into a clean python:3.10-slim — no compilers in the runtime image.

FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first so the dependency resolver does not pull CUDA wheels.
# Pin matches the GPU version used in requirements.txt for parity.
# `--extra-index-url` (not `--index-url`) keeps PyPI as the primary index;
# pip 25+ rejects the torch index's typing-extensions wheel because its
# metadata Name is `typing_extensions` (underscore) while the requirement
# uses a hyphen — letting PyPI serve transitive deps avoids the mismatch.
RUN pip install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY src/ ./src/
# Compiles triangle_hash.cpython-310-*.so in place inside src/.
RUN cd src && python setup.py build_ext --inplace


FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
COPY web/ /app/web/

# Non-root user; tempdirs are created under /tmp at request time.
RUN useradd --create-home --shell /usr/sbin/nologin morphit \
 && chown -R morphit:morphit /app
USER morphit

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "/app/web/api", "--host", "0.0.0.0", "--port", "8000"]
