# Use a base image, for example, Python
FROM python:3.8-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy purchases.json and requirements.txt into the container
COPY purchases.json .
COPY requirements.txt requirements.txt

# Install any required dependencies
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy main.py into the container
COPY main.py .

# Expose port 8000
EXPOSE 8000

# Define the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
