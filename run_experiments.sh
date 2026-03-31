export OMP_DISPLAY_ENV=TRUE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_SCHEDULE=STATIC
export OMP_WAIT_POLICY=ACTIVE

for num_threads in 2 4 6 8 12 16; do
  export OMP_NUM_THREADS=$num_threads
  export OMP_PLACES=$(seq -s ',' -f '{%g}' 0 $((num_threads-1)))
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  OMP_PLACES=$OMP_PLACES"
  python3 run_benchmarks_multithread_solver.py
done