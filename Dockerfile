# SageRec HTTP serving image (ADR-009, optional ADR-010 re-rank).
# Serves a precomputed sagerec_serving_artifact. It does not train,
# compile graph_sampler, or download MovieLens. Mount the artifact at
# /models/embeddings.npz (or set SAGEREC_SERVING_ARTIFACT). Optional
# re-ranking needs SAGEREC_RERANK_ARTIFACT and rerank=true on the request.
FROM python:3.12-slim

WORKDIR /app

COPY python/requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt

COPY python/sagerec_serving.py python/sagerec_rerank.py python/sagerec_api.py /app/

ENV PYTHONPATH=/app \
    SAGEREC_SERVING_ARTIFACT=/models/embeddings.npz

EXPOSE 8000

ENTRYPOINT ["python", "-m", "uvicorn", "sagerec_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
