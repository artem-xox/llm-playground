notebook:
	uv run jupyter notebook --port=10101

train-nano:
	uv run python -m src.scripts.nano.__train__

inference-nano:
	uv run python -m src.scripts.nano.__inference__
