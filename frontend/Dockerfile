FROM python:3.8-slim
WORKDIR /app

COPY requirements.txt .
COPY wheels/ ./wheels/

RUN pip install --find-links=./wheels -r requirements.txt

COPY . .

CMD ["python", "main.py"]
