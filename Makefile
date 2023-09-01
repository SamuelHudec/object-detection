install::
	pip install --upgrade pip wheel build
	pip install -e .[dev]

black::
	black .

flake::
	flake8 .

isort::
	isort .

test-run::
	python -m ml-detection.detect_objects

lint:: black isort flake
