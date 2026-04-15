# Contributing

This project accepts contributions that improve correctness, reproducibility, operability, and developer ergonomics. Treat every change as if it will be consumed by someone who did not watch you build it.

## Ground Rules

- Keep changes scoped. One pull request should solve one problem.
- Prefer explicit configs and schema updates over hidden behavior changes.
- If you change an artifact format, update `docs/specs/checkpoint-schema.md`.
- If you change public behavior, update examples and README where relevant.

## Development Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

This repository uses a Python style stack that favors consistency over individual taste.

- `black` for formatting.
- `isort` for import ordering.
- `ruff` for fast lint and correctness checks.
- `mypy` for incremental type-checking.

Run the checks that apply to the current checkout before opening a pull request:

```bash
black .
isort .
ruff check .
mypy speaker_verification
# Optional if tests are present in your checkout:
pytest
```

## Typing and Docstrings

- New public functions, methods, dataclasses, and modules should include PEP 484 type hints.
- Public modules should use Google-style or NumPy-style docstrings consistently.
- Do not add placeholder docstrings that repeat the function name without explaining inputs, outputs, or behavior.

## Testing Requirements

- Every behavior change must include tests or a clear explanation for why automated coverage is not practical.
- Prefer small unit tests over integration-only coverage.
- Tests must not depend on untracked datasets, personal file paths, or network access.
- Use fixtures and synthetic artifacts for dataset-contract tests.

## Pull Request Flow

1. Fork the repository and create a topic branch from `main`.
2. Make the change in the smallest sensible scope.
3. Run formatting, linting, type checks, and tests locally.
4. Update docs, examples, and schemas if the change affects users or artifacts.
5. Open a pull request with a clear title and description.

Good pull requests include:

- the problem statement,
- the implementation summary,
- any schema or API changes,
- validation evidence,
- follow-up work that is intentionally out of scope.

## Commit Message Style

Use clear, imperative commit messages. Conventional Commits are preferred.

Examples:

- `feat: add realtime EEND diarization example`
- `fix: align export wrapper with EENDQueryModel outputs`
- `docs: document Train_Ali synthetic pretraining`
- `refactor: simplify diarization loss configuration`

Avoid vague messages such as `update`, `fix stuff`, or `change readme`.

## Reporting Bugs and Requesting Features

- Use the issue templates in `.github/ISSUE_TEMPLATE/`.
- For bugs, include environment details, reproduction steps, logs, and artifact metadata.
- For features, describe the workload and why current behavior is insufficient.

## Code of Conduct

Be direct, technical, and respectful. Critique code and design decisions, not people. Harassment, dismissive behavior, or low-effort antagonism are not acceptable in issues, pull requests, or discussions.

## Review Expectations

Maintainers will prioritize:

- correctness and regression risk,
- API and checkpoint-schema clarity,
- test quality,
- operational impact,
- documentation completeness.

If a change improves local convenience but makes the project harder to reason about in CI, releases, or downstream integrations, expect pushback.

