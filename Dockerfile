FROM python:3.9.5 AS base
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.9-slim AS users
WORKDIR /obs

COPY --from=base /root/.cache /root/.cache
COPY --from=base requirements.txt .
RUN apt-get update
RUN apt-get -y install libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install -r requirements.txt && rm -rf /root/.cache

# RabbitMQ
RUN apt-get update
RUN apt-get install curl software-properties-common apt-transport-https lsb-release
RUN curl -fsSL https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/erlang.gpg
RUN apt-get update
RUN apt-get install erlang
RUN curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash
RUN apt-get update
RUN apt-get install rabbitmq-server
RUN rabbitmq-plugins enable rabbitmq_management

COPY . /obs

# ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
# ENTRYPOINT ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
# CMD uvicorn main:app --host 0.0.0.0 --port $PORT
RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh"]
