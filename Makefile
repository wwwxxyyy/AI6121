PYTHON ?= .venv/bin/python
export MPLBACKEND = Agg
export MPLCONFIGDIR ?= /tmp/ai6121-sfm-mpl
export OPENBLAS_NUM_THREADS = 1

.PHONY: test demos demo-hydrant demo-garden

test:
	$(PYTHON) -m pytest -q

demos: demo-hydrant demo-garden

# Each target requires a new or empty output directory.
demo-hydrant:
	$(PYTHON) run_sfm.py --video demos/hydrant/input.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 3000 --output outputs/hydrant
	$(PYTHON) scripts/verify_reconstruction.py outputs/hydrant
	$(PYTHON) scripts/stereo_cloud.py outputs/hydrant --rectify-alpha -1 --disparities 224
	$(PYTHON) scripts/render_cloud_views.py outputs/hydrant --title Hydrant --stereo

demo-garden:
	$(PYTHON) run_sfm.py --video demos/garden/input.mp4 --fps 2 --max-frames 0 --refine-focal --nfeatures 6000 --output outputs/garden
	$(PYTHON) scripts/verify_reconstruction.py outputs/garden
	$(PYTHON) scripts/stereo_cloud.py outputs/garden
	$(PYTHON) scripts/render_cloud_views.py outputs/garden --title 'Yunnan Garden' --stereo
