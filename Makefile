notebook:
	uv run jupyter notebook --port=10101

nano-train:
	uv run python -m src.scripts.nano.__train__
