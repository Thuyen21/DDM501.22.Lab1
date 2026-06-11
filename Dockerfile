FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# Copy requirements first to leverage build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and the model into the image
COPY . .

# Declare the port the container listens on
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
