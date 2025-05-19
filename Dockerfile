FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt
EXPOSE $PORT 
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app