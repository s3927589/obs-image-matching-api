#!/bin/bash

exec python3 ./consumer.py &

uvicorn main:app --reload --host 0.0.0.0 --port $PORT

fg %1
