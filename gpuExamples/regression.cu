#include "common/book.h"

#define imin(a,b) (a<b?a:b)
#define BLOCK_SIZE 3

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N + threadsPerBlock-1)/ threadsPerBlock);


// struct for storing matrices
typedef struct {
    int m;
    int n;
    int stride;
    float *elts; 
} Matrix;


__host__ __device__ void print_matrix( Matrix A){
    printf("--------------------------\n");
    for(int i = 0; i < (A.n * A.m); ++i){
        if( i % 3 == 0) printf("\n");
        printf("%f ", A.elts[i]);
    }
    printf("--------------------------\n");
}



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
__host__ __device__ Matrix get_sub_mtx( const Matrix S, int row, int col){
    Matrix X;
    X.n = BLOCK_SIZE;
    X.m = BLOCK_SIZE;
    X.stride = S.stride;
    X.elts = &S.elts[S.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return X;
}


// Get a matrix element
__device__ float get_elt(const Matrix A, int row, int col){   
    //printf("returning %f\n ", A.elts[row* A.stride + col]); 
    return A.elts[row * A.stride + col];
}

// Set a matrix element
__device__ void set_elt(Matrix A, int row, int col, float value){         
     A.elts[row * A.stride + col] = value;
}


/*
* A: matrix to transpose
* At: tranposed Matrix
*/
__global__ void matrix_transpose(Matrix A, Matrix At){
    
   __shared__ float tile[TILE_DIM][TILE_DIM+1];
   
   //diagonal reorder for transpose implementation

   int block_x, block_y;
   
   if(A.n == A.m){
       block_y = blockIdx.x;
       block_x = (blockIdx.y + blockIdx.x) % gridDim.x;
   } else {
       block_y = ((blockIdx.y * gridDim.x) + blockIdx.x) % gridDim.y;
       block_x = (((blockIdx.y * gridDim.x) + blockIdx.x)/gridDim.y) + block_y) % gridDim.x;
   }
   
   int x = block_x * TILE_DIM + threadIdx.x;
   int y = block_y * TILE_DIM + threadIdx.y;

   int in_index = x + y * A.n; 
   
   int x = block_y * TILE_DIM + threadIdx.x;
   int y = block_x * TILE_DIM + threadIdx.y;

   int out_index = x + y * A.m;


   for (int i = 0; i < TILE_DIM; i+= BLOCK_SIZE){
       tile[threadIdx.y + i][threadIdx.x] = A.elts[in_index + i*A.n];
   }
   __syncthreads();
  
   for(int i = 0; i < TILE_DIM; i+= BLOCK_SIZE){
       At.elts[out_index + i * height] = tile[threadIdx.x][threadIdx.y + i];
   }
   
   
   
}











/*
* A,B: matrices to multiply
* C: resulting matrix of A*B
*/
__global__ void matrix_multiply_matrix( Matrix A, Matrix B, Matrix C){

    //Block row and column
    int row_block = blockIdx.y;
    int col_block = blockIdx.x;

    //each thread block computes submatrix of dimensions BLOCK_SIZE*BLOCK_SIZE
    Matrix sub_c = get_sub_mtx(C, row_block, col_block);
     
    /* 
    printf("submatrix of row %d and col %d  \n", row_block, col_block);
    print_matrix(sub_c);
    */

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
         
        
        if(row == 0 && col == 0){
            print_matrix(sub_a);
            print_matrix(sub_b);        
        }

 
        //shared memory to fill sub matrices 
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];       
        
        As[row][col] = get_elt(sub_a, row, col);
        Bs[row][col] = get_elt(sub_b, row, col);
        
        printf("sub_a elt at row %d and col %d = %f \n ", row, col, get_elt(sub_a, row, col));
        printf("sub_b elt at row %d and col %d = %f \n ", row, col, get_elt(sub_b, row, col));
        

 
	//synchronize the threads
  	__syncthreads();

                
        // multiply sub matrices
        for(int e = 0; e < BLOCK_SIZE; ++e){
            val += As[row][e] * Bs[e][col];
            
	}        
        
        /*
        if(row == 0 && col == 0){
            printf("(0,0) = %f \n", val);
        } else if (row == 2 && col == 2){
            printf("(2,2) = %f \n", val);
        }
        */

        // synchronize threads 
        __syncthreads();

               
    }
    
    
    // set sub-matrix c_sub element at row, column to val
    // each thread does the following
    set_elt( sub_c, row, col, val);

     
}  


/*
* I_vals: matrix of independent variable values [ [x0,y0,z0...], [x1,y1,z1...], ...]
* b: vector of values corresponsing to elements in I_vals matrix. 
* *r: pointer to result of regression - solution
*/
__global__ void linear_regression( Matrix I_vals, Matrix b, float *r) {
    
}



//TODO linear regression using dot product and inverse/LU/Choelsky












int main ( void ) {
    // test matrix multiplication (A_T * A)
    
    Matrix A;
    A.n = 3;
    A.m = 3;
    float a[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    A.elts = a;
           

    Matrix At;
    At.n = 3;
    At.m = 3;
    float at[9] = {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f};
    At.elts = at;
   
    

    Matrix d_A;
    d_A.n = d_A.stride = A.n;
    d_A.m = A.m;
    size_t size = A.n * A.m * sizeof(float);
    
    cudaMalloc(&d_A.elts, size);
    cudaMemcpy(d_A.elts, A.elts, size, cudaMemcpyHostToDevice);
        
    Matrix d_At;
    d_At.n = d_At.stride = At.n;
    d_At.m = At.m;
    
    cudaMalloc(&d_At.elts, size);
    cudaMemcpy(d_At.elts, At.elts, size, cudaMemcpyHostToDevice);
    
    Matrix d_C;
    d_C.n = d_C.stride =  A.n;
    d_C.m = A.n; // square matrix
    size = d_C.m * d_C.n * sizeof(float);
    float c[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f};
    d_C.elts = c;
    cudaMalloc(&d_C.elts, size);
    
       
    // Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(At.n / dimBlock.x, A.m / dimBlock.y);

    matrix_multiply_matrix<<<dimGrid, dimBlock>>>(d_At, d_A, d_C);
    
    //
    Matrix C;
    C.m = At.m;
    C.n = A.n;
    C.elts = (float *)malloc(sizeof(float) * C.m * C.n);
    cudaMemcpy(C.elts, d_C.elts, size, cudaMemcpyDeviceToHost);
   
        
    print_matrix(A);
    print_matrix(At);
    print_matrix(C);    

    
    
    //print_matrix( get_sub_mtx(C, 0, 0));
     
    free(C.elts);    
    cudaFree(d_A.elts);
    cudaFree(d_At.elts);
    cudaFree(d_C.elts); 
     

  
   return 0;
}





