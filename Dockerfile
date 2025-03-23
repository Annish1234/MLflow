# Use Python 3.9 as the base image
FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file from `api/` and install dependencies
COPY --chown=user api/requirement.txt .
RUN pip install --no-cache-dir --upgrade -r requirement.txt

# Copy the full project after installing dependencies
COPY --chown=user . .

# Expose the required port
EXPOSE 7860

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
