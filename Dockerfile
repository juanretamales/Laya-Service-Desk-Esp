FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 10001 appuser
WORKDIR /workspace
COPY requirements/dev.txt /tmp/dev.txt
RUN pip install --no-cache-dir -r /tmp/dev.txt
USER appuser
CMD ["python", "-m", "pytest", "-q"]
