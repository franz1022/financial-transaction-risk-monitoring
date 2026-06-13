FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt ./requirements-docker.txt

RUN python -m pip --timeout 180 --retries 10 install --upgrade pip && python -m pip --timeout 180 --retries 10 install -r requirements-docker.txt

COPY app ./app
COPY models ./models
COPY outputs ./outputs
COPY .streamlit ./.streamlit

EXPOSE 8000
EXPOSE 8501

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]