FROM python:3.9-slim

# Note: This sandbox provides resource isolation (cpu/memory/timeout), a
# non-root user, no network access, and a read-only root filesystem. It does
# NOT provide hardened security against deliberate container escapes or
# zero-days. It is designed to constrain and gate experimental ML scripts,
# not to run untrusted malware safely.

RUN useradd --create-home --uid 1000 sandboxuser
WORKDIR /app

# Installed before COPY so this layer caches across candidate script
# changes - a candidate whose whole premise is ML research needs these to
# do anything beyond pure stdlib.
COPY requirements-sandbox.txt /app/requirements-sandbox.txt
RUN pip install --no-cache-dir -r requirements-sandbox.txt

COPY . /app
RUN chown -R sandboxuser:sandboxuser /app
USER sandboxuser

CMD ["python", "candidate_script.py"]
