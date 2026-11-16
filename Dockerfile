FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p reports

COPY model.py .
COPY test_model.py .

CMD ["python", "model.py"]
