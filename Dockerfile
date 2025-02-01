# Use an appropriate base image (update as per your needs)
FROM python:3.12  

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the application code (if needed)
COPY . .

# Set default command (optional, update as needed)
CMD ["python", "app.py"]
