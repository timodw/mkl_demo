#!/bin/bash

icc -Wall -O3 -g -I eigen-git-mirror -qopenmp -mkl matrix_operations.cpp bench.cpp main.cpp
