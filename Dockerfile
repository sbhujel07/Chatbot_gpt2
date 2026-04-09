#Base image 
From python:3.12-slim

#set working dir
WORKDIR /app


#Copy requirements first(for caching)
COPY requirements.txt .

#install python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy api code
COPY app.py .
COPY api/  ./api
COPY src/ ./src
COPY configs/ ./configs

#Expose fastapi port
EXPOSE 8000

#Run Fastapi
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
