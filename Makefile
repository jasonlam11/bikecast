.PHONY: setup data stations aggregate weather calendar features backtest eda app test lint format

setup:
	uv sync

data:
	uv run python -m bikecast.data.pipeline $(if $(MONTHS),--months $(MONTHS))
	$(MAKE) stations
	$(MAKE) aggregate
	$(MAKE) weather
	$(MAKE) calendar

stations:
	uv run python -m bikecast.data.stations

aggregate:
	uv run python -m bikecast.data.aggregate

weather:
	uv run python -m bikecast.data.weather

calendar:
	uv run python -m bikecast.data.calendar

features:
	uv run python -m bikecast.features.build

backtest:
	uv run python -m bikecast.evaluation.backtest $(if $(MODELS),--models $(MODELS))
	uv run python -m bikecast.evaluation.report

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .
