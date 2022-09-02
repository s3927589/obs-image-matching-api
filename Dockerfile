FROM python:3.9.5 AS base
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.9-slim AS users
WORKDIR /obs
EXPOSE 8000

COPY --from=base /root/.cache /root/.cache
COPY --from=base requirements.txt .
RUN pip install -r requirements.txt && rm -rf /root/.cache

COPY . /obs

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
