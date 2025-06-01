# Dockerfile

# 1. Base Image
FROM python:3.10-slim

# 2. Set Working Directory (this will be the root for our application code)
WORKDIR /service

# 3. Install System Dependencies (if any)
# ...

# 4. Copy requirements file and install Python dependencies
COPY requirements.txt /service/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /service/requirements.txt

# 5. Copy Application Code
# Copy the local 'app' directory to '/service/app' inside the image
COPY ./app /service/app

# 6. Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HOST=0.0.0.0

# 7. Expose Port
EXPOSE ${PORT}

# 8. Define Command to Run Application
# Uvicorn will look for the 'app' package (now at /service/app)
# and then main.py within it, and the 'app' instance in main.py.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
