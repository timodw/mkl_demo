#!/bin/bash

icc -Wall -O3 -g -I /opt/lynx/software/Eigen/3.3.4/easybuild/Eigen-3.3.4-easybuild-devel -qopenmp main.cpp
