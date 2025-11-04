FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN mkdir -p ./models/ && \
    mkdir -p /app/uploads && \
    chmod 777 /app/uploads

# Add src directory to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]