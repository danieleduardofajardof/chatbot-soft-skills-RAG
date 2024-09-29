# Dockerfile
FROM arm64v8/ubuntu:20.04

# Set environment variable to avoid tzdata interaction
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Optionally set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1



# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the application
CMD ["/usr/local/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
