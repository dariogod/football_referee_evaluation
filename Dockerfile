FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/


ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose port for FastAPI
EXPOSE 4321

# Run FastAPI app with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "4321"]