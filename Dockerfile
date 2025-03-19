# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install basic requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    cython \
    pytest \
    build

WORKDIR /app
COPY . .

# Creates a non-root user with an explicit UID and adds permission to access the app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser --disabled-password --gecos "" --uid 5678 appuser && \
    chown -R appuser:appuser /app
USER appuser

RUN pip install --no-cache-dir -e .

# During debugging, this entry point will be overridden.
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["pytest", "statmoments"]
