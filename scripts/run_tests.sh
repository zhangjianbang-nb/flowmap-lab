#!/usr/bin/env bash
# Run the full test suite from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -t . -v
