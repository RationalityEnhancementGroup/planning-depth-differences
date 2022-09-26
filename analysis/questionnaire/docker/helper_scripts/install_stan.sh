#!/bin/bash
set -e

apt-get update
apt-get -y install apt-utils libnode-dev libxt6 libxml2-dev libglpk-dev\
&& install2.r --error --skipinstalled \
    arrangements \
    testthat \
    brms \
    bayestestR \
    devtools \
    MCMCpack \
    logspline \
    coda \
&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds
