nvcc -o vectorAddTest -O3 -arch=compute_30 -code=sm_30 vectorAdd.cu -lboost_system -lboost_timer
