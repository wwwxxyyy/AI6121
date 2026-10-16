PYTHON ?= .venv/bin/python
export MPLBACKEND = Agg
export MPLCONFIGDIR ?= /tmp/ai6121-sfm-mpl
export OPENBLAS_NUM_THREADS = 1

.PHONY: test demos

# The two real videos are the only test cases; each run has a fresh output folder.
test:
	$(PYTHON) scripts/test_videos.py

demos: test
