FROM python:3.9.5 AS base
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.9-slim AS users
WORKDIR /obs
# EXPOSE 8000

COPY --from=base /root/.cache /root/.cache
COPY --from=base requirements.txt .
RUN apt-get update
RUN apt-get -y install libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install -r requirements.txt && rm -rf /root/.cache

COPY . /obs

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
