# Pull the Python image from Docker Hub
# Use a lightweight Python image
FROM python:3.14-slim

# set the working directory inside container
WORKDIR /app

# Copy dependency list before copying application code
COPY requirement.txt .

# Install the dependencies
RUN pip install -r requirement.txt


# Copy application code and saved model inside /app
COPY . . 

# Document the port used by the FastAPI application
EXPOSE 8000

# Start the API server when container is ready
# 0.0.0.0 - allows connections through the container's network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]docker run --rm -p 8000:8000 mlops-iris-image-cs

