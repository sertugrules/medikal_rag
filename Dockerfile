# Temel Python image
FROM python:3.12

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

# Dosyaları kopyala
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
COPY .env ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY ./backend/vectorstore /app/vectorstore
# Log ve vektör dizinlerini oluştur
RUN mkdir -p backend/logs backend/vectorstore

# Portları aç
EXPOSE 9999
EXPOSE 8501

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 9999 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
