#!/bin/bash
cd "$(dirname "$0")"
source ~/.zshrc 2>/dev/null
.venv/bin/python3 src/main.py "$@"
