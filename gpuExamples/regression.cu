#include "common/book.h"


const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N + threadsPerBlock-1)/ threadsPerBlock);


// struct for storing matrices
typedef struct {
    int m;
    int n;
    float* elts;
} Matrix;



/*
* kernel to compute dot product between vectors *a and *b. result places in *c.
*/
__device__ void dot( float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while(tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    //set the cache values
    cache[cacheIndex] = temp;
    
    //synchronize threads in this block
    __syncthreads();

    //for reductions, threadsPerBlock must be power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];    
}



/*
* A: matrix (symmetric and positive definite) 
* L: L matrix (lower triangular)
*/
__device__ void cholesky( Matrix A, Matrix L){

} 





/*
* A,B: matrices to multiply
* C: resulting matrix of A*B
*/
__device__ void matrix_multiply( Matrix A, Matrix B, Matrix C){

    //Block row and column
    int row_block = blockIdx.y;
    int col_block = blockIdx.x;

    //each thread block computes submatrix of dimensions BLOCK_SIZE*BLOCK_SIZE
    Matrix sub_c = get_sub_mtx(C, row_block, column_block);

    // each element computes one element of sub matrix sub_c
    // we accumulate the results in val
    float val = 0;

    // thread row and column
    int row = threadIdx.y;
    int col = threadIdx.x;


    // loop the sub matrices of A and B required to compute the sub_c matrix
    // note that this assumes A.n is a multiple of BLOCK_SIZE
    for (int i = 0 ; i < (A.n / BLOCK_SIZE); ++i){
        
        // get sub-mtx sub_a of A and sub_b of B
        Matrix sub_a = get_sub_mtx(A, row_block, i);
        Matrix sub_b = get_sub_mtx(B, i, col_block); 
        
        //TODO complete kernel        
    }
    //TODO finish kerner matrix mult
    
}  


/*
* I_vals: matrix of independent variable values [ [x0,y0,z0...], [x1,y1,z1...], ...]
* b: vector of values corresponsing to elements in I_vals matrix. 
* *r: pointer to result of regression - solution
*/
__global__ void linear_regression( Matrix I_vals, Matrix b, float *r) {
    
}




//TODO linear regression using dot product and inverse/LU/Choelsky




