#!/bin/bash
set -e

mkdir -p /home/rstudio/.R/ \
    && echo "CXXFLAGS=-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function -Wno-macro-redefined" >> /home/rstudio/.R/Makevars \
    && echo "CXXFLAGS+=-flto -Wno-unused-local-typedefs" >> /home/rstudio/.R/Makevars \
    && echo "CXXFLAGS += -Wno-ignored-attributes -Wno-deprecated-declarations" >> /home/rstudio/.R/Makevars \
    && echo "CXX14 = g++ -std=c++1y" >> /home/rstudio/.R/Makevars \
    && echo "rstan::rstan_options(auto_write = TRUE)" >> /home/rstudio/.Rprofile \
    && echo "options(mc.cores = parallel::detectCores())" >> /home/rstudio/.Rprofile
