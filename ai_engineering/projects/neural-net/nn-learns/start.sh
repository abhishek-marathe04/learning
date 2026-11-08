#!/bin/bash

# Start the Neural Network Visualization app
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

./.venv/bin/python -m streamlit run app.py
