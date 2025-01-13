#!/bin/bash
# Script to run the API server

uvicorn src.api.serve_model:app --host 0.0.0.0 --port 8000
