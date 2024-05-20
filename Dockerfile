# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-alpine

RUN apk update
RUN apk add build-base linux-headers libc-dev gcc
RUN apk add python3-dev libffi-dev py-cffi hdf5-dev

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
# Install requirements for compilation
RUN python -m pip install --no-cache-dir cython Cython pytest

WORKDIR /statmoments
COPY . /statmoments

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /statmoments
USER appuser

RUN python -m pip install -e .

# During debugging, this entry point will be overridden.
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "-m", "statmoments.tests"]
