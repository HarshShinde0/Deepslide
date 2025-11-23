FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 streamlit
RUN chown -R streamlit:streamlit /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN mkdir -p ./models/ && \
    mkdir -p /app/uploads && \
    chmod 777 /app/uploads && \
    chmod 777 ./models/

# Add src directory to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

EXPOSE 8501

# Set up proper permissions
RUN mkdir -p /app/.streamlit && \
    chown -R streamlit:streamlit /app/.streamlit

# Switch to non-root user
USER streamlit

# Configure Streamlit
RUN mkdir -p /app/.streamlit
COPY config.toml /app/.streamlit/config.toml

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]