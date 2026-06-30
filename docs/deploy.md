# Deploying Inventory Twin to Streamlit Community Cloud

This guide deploys the app to [Streamlit Community Cloud](https://streamlit.io/cloud), the
free tier that runs a Streamlit app straight from a public GitHub repository. The app is a
thin presentation layer over a tested compute core, so deployment is little more than
installing the dependencies and pointing the platform at the entry script.

## Prerequisites

- The repository pushed to **GitHub** (public, or visible to your Streamlit Cloud account).
- A free **Streamlit Community Cloud** account, signed in with that GitHub account.
- The committed `requirements.txt` (see [Regenerating requirements.txt](#regenerating-requirementstxt)).

## What the platform reads

Streamlit Community Cloud builds the environment from files already in the repository:

| File | Role |
| --- | --- |
| `requirements.txt` | The Python dependencies, installed with pip. Generated from `uv.lock`. |
| `.python-version` | Records the development Python (`3.11`). Set the same version in the deploy settings (below). |
| `.streamlit/config.toml` | App configuration — dark theme, headless server, usage stats off. |
| `ui/app.py` | The **main file** (entry script) you point the platform at. |

The first line of `requirements.txt` is a bare `.`. This makes the repository install
itself as a package during deployment, which is the most robust way to ensure the sibling
top-level packages (`core`, `analytics`, `ui`, `visualization`, and the rest) are
importable by the pages. Keep it.

There is no `packages.txt` in this repository, and none is currently needed: the app
depends only on Python packages with installable wheels. Add a `packages.txt` only if a
future feature needs a system (apt) package — that file is for external system
dependencies, not Python ones.

## Deploy steps

1. Sign in to Streamlit Community Cloud and choose **Create app → Deploy a public app from GitHub**.
2. Select this **repository** and the **branch** to deploy.
3. Set the **Main file path** to `ui/app.py`.
4. Open **Advanced settings** and set the **Python version to 3.11**, to match
   `.python-version` and `pyproject.toml` (`requires-python = ">=3.11"`). Matching the
   deployment Python to the development Python avoids dependency-resolution surprises.
5. Click **Deploy**. The platform installs `requirements.txt` and launches `ui/app.py`.

The first build takes a few minutes (numpy, pandas, scipy, and pyarrow are the heavy
wheels). Subsequent redeploys reuse the cached environment.

## Local headless run (deployment parity)

The committed `.streamlit/config.toml` already runs the server headless, so the local
command exercises the same configuration as the hosted app:

```bash
uv run streamlit run ui/app.py
```

## Pre-deploy verification

The app's boot is gated by an automated headless smoke test that loads the entry script
and every one of the nine pages through Streamlit's own test runtime, asserting none
raises. Run it before deploying:

```bash
uv run pytest tests/ui/test_app_smoke.py
```

A green run means every page renders its first frame without error in a headless
environment — the same conditions as the deploy target.

## Regenerating requirements.txt

`requirements.txt` is a generated artifact, exported from `uv.lock`. Whenever a dependency
changes, regenerate it with the exact command recorded in the file's own header:

```bash
uv export --frozen --no-dev --no-editable --no-hashes --format requirements-txt -o requirements.txt
```

A test (`tests/unit/test_requirements_sync.py`) guards against drift: it fails if the
committed file no longer matches the export, so a forgotten regeneration is caught by the
test suite rather than at deploy time.

## Troubleshooting

- **`ModuleNotFoundError` for `core`, `analytics`, `ui`, … on the deploy target.** Confirm
  the bare `.` line is still present at the top of `requirements.txt`; it is what installs
  the project so its packages are importable.
- **A dependency is missing or the wrong version.** Regenerate `requirements.txt` (above)
  and redeploy — the committed file may have drifted from `uv.lock`.
- **Python-version mismatch errors.** Ensure the Streamlit Cloud Python version is set to
  3.11, matching `.python-version`.
