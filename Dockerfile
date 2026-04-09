# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the handling script is executable
RUN chmod +x run_all.sh

# Set environment variables for better logging
ENV PYTHONUNBUFFERED=1

# Default command: Runs the handler.
# You can override this when running the container
CMD ["./run_all.sh"]
