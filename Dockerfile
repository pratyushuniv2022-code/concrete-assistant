# Base image
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "streamlit_concrete_assistant.py", "--server.port=8501", "--server.address=0.0.0.0"]
