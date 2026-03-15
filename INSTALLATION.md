# Installation Notes

`psutil` and `numba` are Python packages from PyPI, not Ubuntu `apt` package names.

If you are working inside this project's virtual environment, install dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you only want those two packages:

```bash
python -m pip install psutil numba
```

## Why `apt install psutil numba` fails

- `apt` installs system packages from Ubuntu repositories.
- These packages are published primarily for Python via `pip`.
- Inside a virtual environment, `pip` is the correct installer so the packages are isolated to the project environment.
