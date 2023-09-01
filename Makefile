install::
	pip install --upgrade pip wheel build
	pip install -e .[dev]

black::
	black .

flake::
	flake8 .

isort::
	isort .

lint:: black-check isort-check flake