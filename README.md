# ML Object Detection

## Environment setup

To use this library/repo, make sure that you have Python >= `3.9.*` - we recommend using [pyenv][] for managing different
Python versions (see: [how to install pyenv on MacOS][]).

This project defines dependencies in [pyproject.toml](./pyproject.toml) according to [PEP-621](https://peps.python.org/pep-0621/)
For development create virtual env with:
```bash
python -m venv venv
source venv/bin/activate
```
Then run:
```bash
make install
```

[pyenv]: https://github.com/pyenv/pyenv#installationbrew
[how to install pyenv on MacOS]: https://jordanthomasg.medium.com/python-development-on-macos-with-pyenv-2509c694a808

## Usage

Run test. For first time it will require some time to download and cache model.
```bash
make test-run
```
Next runs gonna be much faster than first one (test run). I added challenging image 
just to demonstrate algorithm and see what you can get. 

Results you can find in `results` folder in this repo. As you can see on test output
model doesn't recognize lama instead of sheep (probability is ~0.9). Actually, thats
ok, because this is lightweight model and can be fine-tuned to target task. 

If you want to play with model use command:
```bash
python -m ml-detection.detect_objects --image-path <path-to-image>
```
as path, you can use url or absolute path on your local.