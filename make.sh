nvcc -o "$1.out" -O3 -arch=compute_30 -code=sm_30 $1 -lboost_system -lboost_timer
