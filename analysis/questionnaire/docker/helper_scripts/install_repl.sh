#!/bin/bash
set -e

# If you want to make sure that the Docker image downloads a certain repository
mkdir -p /home/rstudio/development \
    && cd /home/rstudio/development \
    && sudo chown -R rstudio /home
