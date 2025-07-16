#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Apply database migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Optional: Download ML model if large
# wget -O ml_model/model.pkl "YOUR_DOWNLOAD_URL" 