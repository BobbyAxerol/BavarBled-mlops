FROM python:3.12-slim

WORKDIR /app

# Install optimized CPU-only PyTorch and other requirements to avoid bloated CUDA binaries
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    pandas \
    numpy \
    numba \
    scipy \
    yfinance \
    pyyaml \
    mlflow-skinny \
    fastapi \
    uvicorn \
    pydantic \
    python-dotenv

# Copy the source package
COPY src /app/bavar_bled/src

EXPOSE 8000

CMD ["uvicorn", "bavar_bled.src.scoring.serve:app", "--host", "0.0.0.0", "--port", "8000"]
