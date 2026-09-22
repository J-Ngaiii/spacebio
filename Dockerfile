# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
# Now copy the rest of the code
COPY . .

# Optionally install as package
RUN pip install -e .

# Default command (can override with `docker run`)
CMD ["python", "main.py"]
