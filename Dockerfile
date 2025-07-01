FROM python:3.10-slim

# Avoid interactive prompts and reduce image size
ENV DEBIAN_FRONTEND=noninteractive

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Selectively copy only inference code
COPY main.py ./main.py
COPY scripts/ ./scripts/
COPY models/deep_learning/ ./models/deep_learning/


# Expose port
EXPOSE 8080

#Run app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]