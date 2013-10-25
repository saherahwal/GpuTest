#include "common/book.h"


const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N + threadsPerBlock-1)/ threadsPerBlock);


// struct for storing matrices
typedef struct {
    int m;
    int n;
    int stride;
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
* S: Matrix to get sub matrix from
* row, column: row and column to start at
* return matrix X which is sub matrix of S
*/
__device__ Matrix get_sub_mtx( const Matrix S, int row, int col){
    Matrix X;
    X.n = BLOCK_SIZE;
    X.m = BLOCK_SIZE;
    X.stride = S.stride;
    X.elts = &S.elts[S.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return X;
}


// Get a matrix element
__device__ float get_elt(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void set_elt(Matrix A, int row, int col, float value)
{
     A.elements[row * A.stride + col] = value;
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
       
        //shared memory to fill sub matrices 
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];       
        
        As[row][col] = get_elt(sub_a, row, col);
        Bs[row][col] = get_elt(sub_b, row, col);
        
	//synchronize the threads
  	__syncthreads();

        // multiply sub matrices
        for(int e = 0; e < BLOCK_SIZE; ++e){
            val += As[row][e] * Bs[e][col];
        }        
        
        // synchronize threads 
        __syncthreads();

               
    }
    // set sub-matrix c_sub element at row, column to val
    // each thread does the following
    set_elt( c_sub, row, col, val);
    
}  


/*
* I_vals: matrix of independent variable values [ [x0,y0,z0...], [x1,y1,z1...], ...]
* b: vector of values corresponsing to elements in I_vals matrix. 
* *r: pointer to result of regression - solution
*/
__global__ void linear_regression( Matrix I_vals, Matrix b, float *r) {
    
}




//TODO linear regression using dot product and inverse/LU/Choelsky




