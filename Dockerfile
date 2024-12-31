ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# Install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U "pip>=21.2"
RUN pip install --no-cache-dir -r /tmp/requirements.txt --timeout=120 --retries 5 && rm -f /tmp/requirements.txt

# Add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# Copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

RUN pip install .


# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Set the command to export PROJECT_DIR and run uvicorn
CMD ["sh", "-c", "export PROJECT_DIR=$(pwd) && uvicorn src.image_ml_pod_fastapi.app:app --host 0.0.0.0 --port 8000"]
