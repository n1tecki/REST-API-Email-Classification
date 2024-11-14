FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t my-fastapi-app .
# docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns -v $(pwd)/data:/app/data fastapi-app