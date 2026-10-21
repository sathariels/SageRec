# SageRec Phase 9 HTTP serving (ADR-009).
# Export needs PYTHONPATH to include the Release build (graph_sampler) and python/.
# The API process only needs python/ plus a .npz artifact.

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/build:$(CURDIR)/python:$(PYTHONPATH)
ARTIFACT ?= results/serving/embeddings.npz
HOST ?= 127.0.0.1
PORT ?= 8000
IMAGE ?= sagerec-serve

.PHONY: export-embeddings serve loadtest test-serving docker-build docker-run

export-embeddings:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=path/to/sagerec_checkpoint.pt" >&2; exit 1)
	$(PYTHON) scripts/export_embeddings.py \
	  --checkpoint $(CHECKPOINT) \
	  --output $(ARTIFACT) \
	  $(if $(TRAIN_PAIRS),--train-pairs $(TRAIN_PAIRS),) \
	  $(if $(PROCESSED_DIR),--processed-dir $(PROCESSED_DIR),) \
	  $(if $(TITLES),--titles $(TITLES),)

serve:
	$(PYTHON) scripts/serve_api.py --artifact $(ARTIFACT) --host $(HOST) --port $(PORT)

loadtest:
	$(PYTHON) scripts/loadtest_serving.py

test-serving:
	$(PYTHON) -m pytest python/tests/test_serving_api.py -q --tb=short

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 \
	  -v $(abspath $(ARTIFACT)):/models/embeddings.npz:ro \
	  -e SAGEREC_SERVING_ARTIFACT=/models/embeddings.npz \
	  $(IMAGE)
