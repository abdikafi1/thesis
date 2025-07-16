#!/bin/bash
# Start Gunicorn
gunicorn fraud_detection.wsgi:application 