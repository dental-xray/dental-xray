#!/bin/bash

# Load environment variables from .env file
source .env

# create tag for docker image with GCP path
docker tag $GAR_IMAGE:dev $GAR_PATH

# push docker image to Google Artifact Registry
docker push $GAR_PATH

# in termal, run "./upload_docker.sh"
