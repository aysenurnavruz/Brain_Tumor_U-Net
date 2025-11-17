FROM python:3.9-slim

WORKDIR /app

COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web_app/. .

EXPOSE 5000
CMD ["python", "app.py"] 