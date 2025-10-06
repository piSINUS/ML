FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y python3.8 python3.8-distutils curl && \
    curl -sS  https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3.8", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]