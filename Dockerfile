# # Use official Python image
# FROM python:3.10

# # Set work directory inside the container
# WORKDIR /app

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # Install FFmpeg (for video/audio processing)
# RUN apt-get update && apt-get install -y ffmpeg

# # Copy the entire project into the container
# COPY . .

# # Expose Flask port
# EXPOSE 5000

# # Run the app
# CMD ["python", "app.py"]



FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean


# Create app directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Upgrade pip & install all dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy rest of the app
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose port
EXPOSE 5000

ENV FFMPEG_LOCATION=/usr/bin


# Run the Flask app
CMD ["python", "app.py"]

