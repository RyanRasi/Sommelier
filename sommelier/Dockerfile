# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update \
    && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

# Django ENV
ENV PYTHONUNBUFFERED=1

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run Django management commands
RUN python manage.py collectstatic --noinput
RUN python manage.py makemigrations
RUN python manage.py migrate

# Command to run the development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]