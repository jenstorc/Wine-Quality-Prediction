FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY api/ /api/

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]