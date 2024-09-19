# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install basic requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
# Install requirements for compilation and testing
RUN python -m pip install --no-cache-dir cython Cython pytest

WORKDIR /app
COPY . .

# Creates a non-root user with an explicit UID and adds permission to access the app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

RUN python -m pip install -e .

# During debugging, this entry point will be overridden.
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["pytest", "statmoments"]
