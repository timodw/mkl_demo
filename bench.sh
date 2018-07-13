#!/bin/bash

echo 'n_threads;problem_size;base_dur;eigen_dur;mkl_dur' > bench_results.csv

for n_threads in 1 2 4 8 16 32
do
	echo "NOW RUNNING FOR $n_threads THREADS"
	for problem_size in 100 200 400 800 1600 3200 6400 12800 25600
	do
		echo "PROBLEM SIZE: $problem_size"
		OMP_NUM_THREADS=$n_threads numactl --physcpubind=0-35 -- ./a.out $problem_size >> bench_results.csv
	done	
done
