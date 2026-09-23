FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch, installed first so requirements.txt does not pull
# the much larger CUDA build. Torch comes from the PyTorch CPU index;
# its normal dependencies still come from PyPI. pip picks the right
# wheel for the machine's Python version and CPU (x86_64 or ARM).
RUN pip install --no-cache-dir --default-timeout=1000 torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]