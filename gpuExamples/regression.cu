#include "common/book.h"
#include <math.h>


#define imin(a,b) (a<b?a:b)
#define imax(a,b) (a>b?a:b)

#define TILE_DIM 3
#define EPSILON 0.000000005

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N + threadsPerBlock-1)/ threadsPerBlock);


//const int BLOCK_SIZE = 3;


// struct for storing matrices
typedef struct {
    int m;
    int n;
    int stride;
    float *elts; 
} Matrix;

// struct for vector
typedef struct {
    int length;
    float *elts;
} Vector;




// Some Utility functions

/**
* generates matrix struct with width, height and float elements
*/
Matrix create_matrix( int width, int height, float * elements){
     Matrix A;
     A.n = A.stride = width;
     A.m = height;
     A.elts = elements;
     return A;
}

/**
* generates vector struct with length and elts)
*/
Vector create_vector(int length, float * elements){
    Vector v;
    v.length = length;
    v.elts = elements;
    return v;
}



/**
* Returns true if two matrices are equal
*/
bool mtx_equal(Matrix A, Matrix B){
     if(A.n != B.n || A.m != B.m){
         return false;
     }else{
         int w = A.n;
         int h = A.m;
         for(int z = 0; z < w * h; z++){
             float v1 = A.elts[z];
             float v2 = B.elts[z];
                              
             if((v1 - v2) > EPSILON || (v1-v2) < -EPSILON ){
                  return false;
             }
         }
         return true;
     }
}



/**
* Returns true if two vectors are equal
*/
bool vec_equal(Vector a, Vector b){
     if(b.length != a.length){
          return false;
     }else{
         int len = b.length;
         for(int z = 0; z < len; ++z){
             float v1 = a.elts[z];
             float v2 = b.elts[z];
             
             if((v1-v2) > EPSILON || (v1-v2) < -EPSILON){
                 return false;
             }
         }
         return true;
     }
}




__host__ __device__ void print_matrix( Matrix A){
    printf("--------------------------\n");
    for(int i = 0; i < (A.n * A.m); ++i){
        if( i % A.n == 0) printf("\n");
        printf("%f ", A.elts[i]);
    }
    printf("--------------------------\n");
}


__host__ __device__ void print_vector( Vector v ){
   printf("-------------------------------\n");
   for(int i = 0; i < v.length; ++i){
       printf(" %f \n", v.elts[i]);
   }
   printf("--------------------------------\n");
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
* S: Matrix to get sub matrix from
* row, column: row and column to start at
* return matrix X which is sub matrix of S
*/
template <int BLOCK_SIZE>
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
     //printf("setting row*A.stride + col (%d) to value (%f)\n", row*A.stride + col, value);         
     A.elts[row * A.stride + col] = value;
}



/*
* M: matrix of rows of data points (for out use-case this is a column matrix)
* weights: Vector of weights to apply for each row
* R: matrix result of applying weights
*/
__global__ void apply_weights( Matrix M, Vector weights, Matrix R){

    // only use one dimensional block size
    int b_id = blockIdx.x;
    int t_id = threadIdx.x; // each thread takes care of one data point (i.e one row).

    int index = b_id * gridDim.x + t_id;
      
    if(index < (M.m * M.n)) {        
        int row = (index / M.m);
        int col = (index % M.m);
        int val = get_elt( M, row, col);
        int newVal = val * weights.elts[row]; //multiply by the appropriate weight
        set_elt( R, row, col, newVal);  
    }
}







/*
* L: cholesky lower triangle matrix
* U: should be the transpose of L: upper triangular result from cholesky
* b: vector b 
* x: vector x to solve for
*/
__global__ void fwd_bkwd_elimination( Matrix L, Matrix U, Vector b, Vector r){

  
   int n = b.length;
   //__shared__ float y[n]; // representing vector y
   //__shared__ float x[n]; // representing vector x
 
   float *y = (float *)malloc(n * sizeof(float));
   memset(y, 0.0f, n*sizeof(float));
   

    
   //forward solve Ly = b
   for(int i = 0; i < n; ++i){
       y[i] = b.elts[i];
       for(int j = 0; j < i; ++j){
           y[i] -= get_elt( L, i, j) * y[j];
       }
       y[i] /= get_elt(L, i, i);
   }
   
   //backward solve Ux = y
   for(int i = n-1; i > -1; --i){
       r.elts[i] = y[i];
       for(int j = i+1; j < n; j++){
           r.elts[i] -= get_elt(U, i, j) * r.elts[j];
       }
       r.elts[i] /= get_elt( U,i, i);
   }
   

   //printf("printing x\n");
   //for(int i = 0; i< n; i++) { printf("x = %f\n", r.elts[i]);}

   free(y);
}



/*
* A: matrix (symmetric and positive definite) 
* L: L matrix (lower triangular)
*
* This function does not need to be parallelized
* Maybe place this on host
*
* This should pass Matrix L with all 0.0 values
*
*/
__global__ void cholesky( Matrix A, Matrix L){
               
    int n = L.n;
    
    // init matrix
    for(int z = 0; z < n*n ; ++z){
        L.elts[z] = 0.0f;
    } 

    for(int i=0; i < n; ++i){
        for(int k =0; k < i+1; ++k){
            float tmp_sum = 0;
            for(int j = 0; j < k; ++j){
                tmp_sum += ( get_elt( L, i, j) * get_elt(L, k, j));
            }
            if(i == k){
                float v = sqrt( get_elt(A, i, i) - tmp_sum );
                //printf("v = %f\n", v);                
                set_elt(L, i, k, v);
            }else{
		float v = 1.0 / get_elt(L, k , k) * (get_elt(A, i, k) - tmp_sum);
	        //printf("v = %f \n", v);
                set_elt(L, i , k, v);
            }  
        }
    }      
}




/*
* Compute matrix transpose
* This relies on the small side of matrix being less than maximum grid dimension
*/
template <int BLOCK_SIZE>
__global__ void matrix_transpose(Matrix A, Matrix At){
   
   int row_block = blockIdx.x;
   int col_block = blockIdx.y;   
  

   int row, _row, col, index;
   
   if ( A.m >= A.n ){
       row = (row_block * BLOCK_SIZE + threadIdx.x);
       _row = row * A.n;
       col = col_block;
       index = _row + col;
       if (row < A.m){
           float elt = A.elts[index];
           At.elts[ col * A.m + row] = elt;
       }            

   } else {
       row = row_block;
       _row = row * A.n;
       col = col_block * BLOCK_SIZE + threadIdx.x;
       index = _row + col;//row_block * A.n + col_block * BLOCK_SIZE + threadIdx.x;
       if ( col < A.n) {
            float elt = A.elts[index];
            At.elts[ col * A.m + row ] = elt;
       }
   }

}



/*
* A: matrix to transpose
* At: tranposed Matrix
*/
template <int BLOCK_SIZE>
__global__ void matrix_transpose_x(Matrix A, Matrix At){
  
   //Block row and column
   int row_block = blockIdx.y;
   int col_block = blockIdx.x;
  
   //diagonal reorder for transpose implementation
   int block_x, block_y;
   
   if(A.n == A.m){
       block_y = blockIdx.x;
       block_x = (blockIdx.y + blockIdx.x) % gridDim.x;
   } else {
       block_y = ((blockIdx.y * gridDim.x) + blockIdx.x) % gridDim.y;
       block_x = ((((blockIdx.y * gridDim.x) + blockIdx.x)/gridDim.y) + block_y) % gridDim.x;
   }
   
   int x = block_x * BLOCK_SIZE + threadIdx.x;
   int y = block_y * BLOCK_SIZE + threadIdx.y;

   int in_index = x + y * A.n; 
   
   x = block_y * BLOCK_SIZE + threadIdx.x;
   y = block_x * BLOCK_SIZE + threadIdx.y;

   int out_index = x + y * A.m;
   
   // sub matrix of result transpose
   //Matrix sub_at = get_sub_mtx(At, blockIdx.y, blockIdx.x);
   
   // thread row and column
   int row = threadIdx.y;
   int col = threadIdx.x;


   
   
   int nVal = (A.n > A.m)? A.n : A.m;
   

   for(int j = 0; j < ceil(nVal / (float)BLOCK_SIZE); ++j){
   
       __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];
       
       Matrix sub_a, sub_at;
       //sub_a = get_sub_mtx(A, block_y, block_x);
       //sub_at = get_sub_mtx(At, block_y, block_x);
           
              
       if( A.n >= A.m){
           sub_a = get_sub_mtx<BLOCK_SIZE>(A, block_y, j);               
           sub_at = get_sub_mtx<BLOCK_SIZE>(At, j , block_x);
       } else {
           sub_a = get_sub_mtx<BLOCK_SIZE>(A, j, block_x);
           sub_at = get_sub_mtx<BLOCK_SIZE>(At, block_y , j);
       } 
      
       /*      
       printf("this is iteration %f\n", j);
       printf("sub matrix a\n");
       print_matrix(sub_a);
       printf("sub matrix at\n");
       print_matrix(sub_at);*/
         
         
       for (int i = 0; i < BLOCK_SIZE; i+= BLOCK_SIZE){
           //tile[threadIdx.y + i][threadIdx.x] = A.elts[in_index + i*A.n];
           tile[row + i][col] = sub_a.elts[in_index + i*A.n];//get_elt(sub_a, row, in_index);
           
       }   
   
       __syncthreads();
  
       for(int i = 0; i < BLOCK_SIZE; i+=BLOCK_SIZE){
           sub_at.elts[out_index + i * A.m] = tile[col][row + i];
           //__syncthreads();
       }
       
       __syncthreads();      
         
   }
   //__syncthreads();

}


/*
* A: Matrix A
* x: vector x to multiply matrix A by
* b: result of multiplying A*x.  b = A*x
*/
//__global__ void matrix_multiply_vector(Matrix A, Vector x, Vector b){
   


//}



/*
* A,B: matrices to multiply
* C: resulting matrix of A*B
*/
template <int BLOCK_SIZE>
__global__ void matrix_multiply_matrix( Matrix A, Matrix B, Matrix C){

    //Block row and column
    int row_block = blockIdx.y;
    int col_block = blockIdx.x;

    //each thread block computes submatrix of dimensions BLOCK_SIZE*BLOCK_SIZE
    Matrix sub_c = get_sub_mtx<BLOCK_SIZE>(C, row_block, col_block);
     
    
    // each thread computes one element of sub matrix sub_c
    // we accumulate the results in val
    float val = 0;

    // thread row and column
    int row = threadIdx.y;
    int col = threadIdx.x;

    
    
    // loop the sub matrices of A and B required to compute the sub_c matrix
    // note that this assumes A.n is a multiple of BLOCK_SIZE
    for (int i = 0 ; i < ceil(A.n / (float)BLOCK_SIZE); ++i){
        
        // get sub-mtx sub_a of A and sub_b of B
        Matrix sub_a = get_sub_mtx<BLOCK_SIZE>(A, row_block, i);
        Matrix sub_b = get_sub_mtx<BLOCK_SIZE>(B, i, col_block); 
                
        
        //shared memory to fill sub matrices 
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];       
       
        
        As[row][col] = get_elt(sub_a, row, col);
        Bs[row][col] = get_elt(sub_b, row, col);
        
        //printf("sub_a elt at row %d and col %d = %f \n ", row, col, get_elt(sub_a, row, col));
        //printf("sub_b elt at row %d and col %d = %f \n ", row, col, get_elt(sub_b, row, col));
        
 
	//synchronize the threads
  	__syncthreads();

        if( (i ==  ceil(A.n/ (float)BLOCK_SIZE) - 1) && A.n % BLOCK_SIZE != 0){
            for(int e = 0; e < A.n % BLOCK_SIZE; ++e){
                val += As[row][e] * Bs[e][col];
            }
        } else{        
            // multiply sub matrices
            for(int e = 0; e < BLOCK_SIZE; ++e){
                val += As[row][e] * Bs[e][col];           
	    }
        }    
        
        
        // synchronize threads 
        __syncthreads();

               
    }
    
    
    // set sub-matrix c_sub element at row, column to val
    // each thread does the following
    set_elt( sub_c, row, col, val);

     
}  


/*
* A: matrix of independent variable values [ [1,x0,y0,z0...], [1, x1,y1,z1...], ...] now it assumes you input the pre- ones (for computing constants)
* b: vector of values corresponsing to elements in A matrix
* return a vector representing result equation    C + Dx + Ey ...
*/
template <int BLOCK_SIZE>
__host__ Vector linear_regression( Matrix A, Matrix b) {
   
    // The following code is to transpose the matrix A
    // Invokes the matrix_transpose kernel
    
    int width_A = A.n;
    int height_A = A.m;
    size_t size_A = width_A * height_A * sizeof(float);
    float * elements_A = A.elts;
         
        
    bool w_gt = width_A > height_A;
    int grid_x = (w_gt)? height_A : ceil(height_A / (float)BLOCK_SIZE);
    int grid_y = (w_gt)? ceil(width_A / (float)BLOCK_SIZE) : width_A;
 
    dim3 dimGrid(grid_x, grid_y);//dim3 dimGrid( height_A, ceil(width_A / (float)BLOCK_SIZE));
    dim3 dimBlock( BLOCK_SIZE);


    Matrix d_A = create_matrix(width_A, height_A, elements_A);
    cudaMalloc(&d_A.elts, size_A);
    cudaMemcpy(d_A.elts, A.elts, size_A, cudaMemcpyHostToDevice); 
    
    float * elts_r = (float *)malloc(size_A);
    memset( elts_r, 0.0f, size_A);
    Matrix d_R = create_matrix(height_A, width_A, elts_r);
    cudaMalloc(&d_R.elts, size_A);

    //invoke tranpose kernel
    matrix_transpose<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_A, d_R);
    
    
    float * elts_at = (float *)malloc(size_A);
    Matrix At = create_matrix(height_A, width_A, elts_at);
    cudaMemcpy(At.elts, d_R.elts, size_A, cudaMemcpyDeviceToHost);

    printf("matrix A is \n");
    print_matrix(A);
    printf("matrix At is \n");
    print_matrix(At);
          
   
    //free(elts_c);
    free(elts_r);
    cudaFree(d_A.elts);
    cudaFree(d_R.elts);
    
    // Now we need to multiple the matrices At * A (non-weigheted regression)
    // invoke the matrix_multiply_matrix kernel
    int width_At = At.n;
    int height_At = At.m;
    size_t size_At = size_A;
    float * elements_At = At.elts;

    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2( ceil(height_At / (float)dimBlock2.x), ceil(width_A/ (float)dimBlock2.y));

    Matrix d_At = create_matrix(width_At, height_At, elements_At);
    cudaMalloc(&d_At.elts, size_At);
    cudaMemcpy(d_At.elts, At.elts, size_At, cudaMemcpyHostToDevice);

    cudaMalloc(&d_A.elts, size_A);
    cudaMemcpy(d_A.elts, A.elts, size_A, cudaMemcpyHostToDevice);

    size_t size_C = width_A * height_At * sizeof(float);
    float * elements_C = (float *)malloc(size_C);
    memset(elements_C, 0.0f, size_C);
    Matrix d_R2 = create_matrix(width_A, height_At, elements_C);
    cudaMalloc(&d_R2.elts, size_C);
    
    // invoke matrix_multiply_matrix kernel
    matrix_multiply_matrix<BLOCK_SIZE><<<dimGrid2, dimBlock2>>>(d_At, d_A, d_R2);
    
    float * elts_c = (float *)malloc(size_C);
    Matrix C = create_matrix(width_A, height_At, elts_c);
    cudaMemcpy(C.elts, d_R2.elts, size_C, cudaMemcpyDeviceToHost);
    
    printf("matrix At is  \n");
    print_matrix(At);
    printf("matrix A is   \n");
    print_matrix(A);
    printf("matrix C = At * A is \n");
    print_matrix(C);
    
    
    free(elements_C);
    cudaFree(d_R2.elts);
    cudaFree(d_A.elts);
    cudaFree(d_At.elts);

    

    // Next, multiply the matrix At (tranpose of A) by the vector b. (non-weighted)
    // This also uses the matrix multiply kernel
    
    int width_b = b.n; // should be 1 (b is a vector)
    int height_b = b.m;
    size_t size_b = width_b * height_b * sizeof(float);
    float * elements_b = b.elts;
 
    dim3 dimBlock3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid3(ceil(height_At / (float)dimBlock3.x), ceil(width_b / (float)dimBlock3.y));


    cudaMalloc(&d_At.elts, size_At);
    cudaMemcpy(d_At.elts, At.elts, size_At, cudaMemcpyHostToDevice);

    Matrix d_b = create_matrix(width_b, height_b, elements_b);
    cudaMalloc(&d_b.elts, size_b);
    cudaMemcpy(d_b.elts, b.elts, size_b, cudaMemcpyHostToDevice);

    size_t size_At_b = width_b * height_At * sizeof(float);
    float * elements_At_b = (float *)malloc(size_At_b);
    memset(elements_At_b, 0.0f, size_At_b);
    Matrix d_At_b = create_matrix(width_b, height_At, elements_At_b);
    cudaMalloc(&d_At_b.elts, size_At_b);      

    //invoke matrix_multiply_matrix kernel (multiply by vector instead)
    matrix_multiply_matrix<BLOCK_SIZE><<<dimGrid3, dimBlock3>>>(d_At, d_b, d_At_b);

    float * elts_at_b = (float *)malloc(size_At_b);
    Vector vb = create_vector( width_b * height_At, elts_at_b);
    cudaMemcpy( vb.elts, d_At_b.elts, size_At_b, cudaMemcpyDeviceToHost);

    printf("printing matrix At\n");
    print_matrix(At);
    printf("printing matrix/vector b\n");
    print_matrix(b);
    printf("vector result vb\n");
    print_vector(vb);


    
    free(elements_At_b);
    cudaFree(d_At.elts);
    cudaFree(d_b.elts);
    cudaFree(d_At_b.elts);
    

    // Compute Cholesky for At*A result
    // invoke the cholesky kernel
    int width_C = C.n;
    int height_C = C.m;
    
    Matrix d_C = create_matrix(width_C, height_C, C.elts);
    cudaMalloc(&d_C.elts, size_C);
    cudaMemcpy(d_C.elts, C.elts, size_C, cudaMemcpyHostToDevice);   

    elts_r = (float *)malloc(size_C);
    memset(elts_r, 0.0f, size_C);
    Matrix d_L = create_matrix(width_C, height_C, elts_r);
    cudaMalloc(&d_L.elts, size_C);

    //invoke the kernel here
    cholesky<<<1, 1>>>(d_C, d_L);
    
    float * elts_l =  (float *)malloc(size_C);
    Matrix L = create_matrix(width_C, height_C, elts_l);
    cudaMemcpy(L.elts, d_L.elts, size_C, cudaMemcpyDeviceToHost);

    printf("matrix C is \n");
    print_matrix(C);
    printf("cholesky result L \n");
    print_matrix(L);

    free(elts_r);
    cudaFree(d_C.elts);
    cudaFree(d_L.elts);

    // compute Lt which is transpose of L matrix (cholesky result)
    // invoke the matrix_transpose kernel 
    int width_L = L.n;
    int height_L = L.m;
    size_t size_L = width_L * height_L * sizeof(float);
    float * elements_L = L.elts;

    grid_x = ceil(height_L / (float)BLOCK_SIZE);
    grid_y = width_L;
    dim3 dimGrid4(grid_x, grid_y);
    dim3 dimBlock4(BLOCK_SIZE); 
   
    cudaMalloc(&d_L.elts, size_L);
    cudaMemcpy(d_L.elts, L.elts, size_L, cudaMemcpyHostToDevice);

    elts_r = (float *)malloc(size_L);
    memset(elts_r, 0.0f, size_L);
    Matrix d_U = create_matrix(height_L, width_L, elts_r);
    cudaMalloc(&d_U.elts, size_L);

    //invoke kernel
    matrix_transpose<BLOCK_SIZE><<<dimGrid4, dimBlock4>>>(d_L, d_U);
   
    float * elts_u = (float *)malloc(size_L);
    Matrix U = create_matrix(height_L, width_L, elts_u);
    cudaMemcpy(U.elts, d_U.elts, size_L, cudaMemcpyDeviceToHost);

    printf("matrix L is \n");
    print_matrix(L);
    printf("matrix U is \n");
    print_matrix(U);

    free(elts_r);
    cudaFree(d_L.elts);
    cudaFree(d_U.elts);

    // compute result Vector x by forward-backward elimination
    // invoke forward-backward elimination kernel
    
    int width_U = U.n;
    int height_U = U.m;
    size_t size_U = width_U * height_U * sizeof(float);
    float * elements_U = U.elts;
     
    int v_len = vb.length;
    float * v_elts = vb.elts;

    size_t size_v = v_len * sizeof(float);
    
    cudaMalloc(&d_L.elts, size_L);
    cudaMemcpy(d_L.elts, L.elts, size_L, cudaMemcpyHostToDevice);

    cudaMalloc(&d_U.elts, size_U);
    cudaMemcpy(d_U.elts, U.elts, size_U, cudaMemcpyHostToDevice);

    Vector d_vb = create_vector(v_len, v_elts);
    cudaMalloc(&d_vb.elts, size_v);
    cudaMemcpy(d_vb.elts, vb.elts, size_v, cudaMemcpyHostToDevice);

    elts_r = (float *)malloc(size_v);
    memset(elts_r, 0.0f, size_v);
    Vector d_r = create_vector(v_len, elts_r);
    cudaMalloc(&d_r.elts, size_v);


    //invoke kernel
    fwd_bkwd_elimination<<<1, 1>>>(d_L, d_U, d_vb, d_r);

    float * elts_x = (float *)malloc(size_v);
    memset(elts_x, 0.0f, size_v);
    Vector x = create_vector(v_len, elts_x);
    cudaMemcpy(x.elts, d_r.elts, size_v, cudaMemcpyDeviceToHost);


    printf("matrix L\n");
    print_matrix(L);
    printf("matrix U\n");
    print_matrix(U);
    printf("vector vb\n");
    print_vector(vb);
    printf("vector x result \n");
    print_vector(x);

    //Vector result = create_vector(x.length, x.elts);

    free(elts_c);    
    free(elts_r);
    free(elts_l);
    free(elts_u);
    free(elts_at_b);    
    cudaFree(d_vb.elts);
    cudaFree(d_L.elts);
    cudaFree(d_U.elts);
    cudaFree(d_r.elts);

    return x;
}






//int main ( void ) {
    
    /*   
    // test matrix multiplication (A_T * A)
            
    printf("matrix multiply test\n");
    //Square matrix test    
    int width = 3;
    int height = 3;
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(height / dimBlock.x, height / dimBlock.y);
            
    Matrix A;
    A.n = width;
    A.m = height;
    float a[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    A.elts = a;
           

    Matrix At;
    At.n = height;
    At.m = width;
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

    //invoke kernel
    matrix_multiply_matrix<<<dimGrid, dimBlock>>>(d_At, d_A, d_C);

    
    Matrix C;
    C.m = At.m;
    C.n = A.n;
    C.elts = (float *)malloc(sizeof(float) * C.m * C.n);
    cudaMemcpy(C.elts, d_C.elts, size, cudaMemcpyDeviceToHost);
   
        
    print_matrix(A);
    print_matrix(At);
    print_matrix(C);    
  
        
     
    free(C.elts);    
    cudaFree(d_A.elts);
    cudaFree(d_At.elts);
    cudaFree(d_C.elts); 
    

    //column matrix A test:  A_T * A is row matrix * column matrix 
  
    width = 3;
    height = 9;
    
    dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid2(width / dimBlock.x, width / dimBlock.y);
            
    
    A.n = width;
    A.m = height;
    float a2[27] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,1.0f, 2.0f, 3.0f, 4.0f, 5.0f,6.0f, 7.0f, 8.0f, 9.0f };
    A.elts = a2;
           

    
    At.n = height;
    At.m = width;
    float at2[27] = {1.0f, 4.0f, 7.0f, 1.0f, 4.0f, 7.0f, 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 2.0f, 5.0f, 8.0f, 2.0f, 5.0f, 8.0f,3.0f, 6.0f, 9.0f, 3.0f, 6.0f,9.0f, 3.0f, 6.0f, 9.0f };
    At.elts = at2;
       

    
    d_A.n = d_A.stride = A.n;
    d_A.m = A.m;
    size = A.n * A.m * sizeof(float);
    
    cudaMalloc(&d_A.elts, size);
    cudaMemcpy(d_A.elts, A.elts, size, cudaMemcpyHostToDevice);
        
    
    d_At.n = d_At.stride = At.n;
    d_At.m = At.m;
    
    cudaMalloc(&d_At.elts, size);
    cudaMemcpy(d_At.elts, At.elts, size, cudaMemcpyHostToDevice);
    
    
    d_C.n = d_C.stride =  A.n;
    d_C.m = A.n; // square matrix
    size = d_C.m * d_C.n * sizeof(float);
    float c2[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f };
    d_C.elts = c2;
    cudaMalloc(&d_C.elts, size);

    //invoke kernel
    matrix_multiply_matrix<<<dimGrid2, dimBlock2>>>(d_At, d_A, d_C);// A_t * A

        
    C.m = At.m;
    C.n = A.n;
    C.elts = (float *)malloc(sizeof(float) * C.m * C.n);
    cudaMemcpy(C.elts, d_C.elts, size, cudaMemcpyDeviceToHost);
   
        
    print_matrix(A);
    print_matrix(At);
    print_matrix(C);    
      
     
    free(C.elts);    
    cudaFree(d_A.elts);
    cudaFree(d_At.elts);
    cudaFree(d_C.elts); 


    // test A matrix row matrix -- A_t is column mtx 
   
  
    width = 6;
    height = 3;
    
    dim3 dimBlock3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid3(width / dimBlock.x, width / dimBlock.y);
            
    
    A.n = width;
    A.m = height;
    float a3[18] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 4.0f, 1.0f, 1.0f, 1.0f };
    A.elts = a3;
           

    
    At.n = height;
    At.m = width;
    float at3[18] = {1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f };
    At.elts = at3;
       

    
    d_A.n = d_A.stride = A.n;
    d_A.m = A.m;
    size = A.n * A.m * sizeof(float);
    
    cudaMalloc(&d_A.elts, size);
    cudaMemcpy(d_A.elts, A.elts, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A.elts, A.elts, size, cudaMemcpyHostToDevice);
        
    
    d_At.n = d_At.stride = At.n;
    d_At.m = At.m;
    
    cudaMalloc(&d_At.elts, size);
    cudaMemcpy(d_At.elts, At.elts, size, cudaMemcpyHostToDevice);
    
    
    d_C.n = d_C.stride =  A.n;
    d_C.m = A.n; // square matrix
    size = d_C.m * d_C.n * sizeof(float);
    float c3[36] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    d_C.elts = c3;
    cudaMalloc(&d_C.elts, size);

    //invoke kernel
    matrix_multiply_matrix<<<dimGrid3, dimBlock3>>>(d_At, d_A, d_C);// A_t * A

        
    C.m = At.m;
    C.n = A.n;
    C.elts = (float *)malloc(sizeof(float) * C.m * C.n);
    cudaMemcpy(C.elts, d_C.elts, size, cudaMemcpyDeviceToHost);
   
        
    print_matrix(A);
    print_matrix(At);
    print_matrix(C);    
      
     
    free(C.elts);    
    cudaFree(d_A.elts);
    cudaFree(d_At.elts);
    cudaFree(d_C.elts); 
       
    // test matrix multiply vector 

    
    */    
    /*    
             
    // Test matrix transpose
      
    printf("matrix transpose test\n");


    Matrix X;
    X.n = 3;
    X.m = 6;
    float a[18] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
    X.elts = a;
    
    Matrix d_X;
    d_X.n = d_X.stride = X.n;
    d_X.m = X.m;
    size_t size = X.n * X.m * sizeof(float);
    
    cudaMalloc(&d_X.elts, size);
    cudaMemcpy(d_X.elts, X.elts, size, cudaMemcpyHostToDevice);

    
    Matrix d_Xt;
    d_Xt.n = d_Xt.stride =  X.m;
    d_Xt.m = X.n; 
    size = d_Xt.m * d_Xt.n * sizeof(float);
    float c[18] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    d_Xt.elts = c;
    cudaMalloc(&d_Xt.elts, size);

    
    // Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(X.n / BLOCK_SIZE, X.m / BLOCK_SIZE);

    matrix_transpose<<<dimGrid, dimBlock>>>(d_X, d_Xt);

    Matrix Xt;
    Xt.m = d_Xt.m;
    Xt.n = Xt.stride = d_Xt.n;
    Xt.elts = (float *)malloc(sizeof(float) * Xt.m * Xt.n);
    cudaMemcpy(Xt.elts, d_Xt.elts, size, cudaMemcpyDeviceToHost);
    
    //print_matrix(X);
    //print_matrix(X);
    print_matrix(X);
    printf("transpose result\n");
    print_matrix(Xt);    

    free(Xt.elts);
    cudaFree(d_X.elts);
    cudaFree(d_Xt.elts);
    
    */
    /*   
    // test matrix multiplication (A_T * A)
    
    printf("cholesky decomp tets matrix multiply test\n");
    //Square matrix test    
    int width = 3;
    int height = 3;
    
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(height / dimBlock.x, height / dimBlock.y);
            
    Matrix C;
    C.n = C.stride = width;
    C.m = height;
    float c[9] = {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f};
    //float c[9] = {25.0f, 15.0f, -5.0f, 15.0f, 18.0f, 0.0f, -5.0f, 0.0f, 11.0f};
    C.elts = c;
   
    Matrix d_C;
    d_C.n = d_C.stride = C.n;
    d_C.m = C.m;
    size_t size_C = C.n * C.m * sizeof(float);
    cudaMalloc(&d_C.elts, size_C);
    cudaMemcpy(d_C.elts, C.elts, size_C, cudaMemcpyHostToDevice);

    Matrix d_L;
    d_L.n = d_L.stride = C.n;
    d_L.m = C.m;
    size_t size_L = d_C.m * d_C.n * sizeof(float);
    float dL[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f, 0.0f};
    d_L.elts = dL;
    cudaMalloc(&d_L.elts, size_L);

    

    cholesky<<<1,1>>>(d_C, d_L);


    Matrix L;
    L.n = L.stride = C.n;
    L.m = C.m;
    float _l[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    L.elts = _l;
    //L.elts = (float *)malloc( sizeof(float) * C.m * C.n);
    cudaMemcpy(L.elts, d_L.elts, size_L, cudaMemcpyDeviceToHost);

    printf("start cholesky for C matrix: \n");
    print_matrix(C);
    printf("cholesky result\n");    
    print_matrix(L);


    //free(L.elts);
    cudaFree(d_L.elts);
    cudaFree(d_C.elts);
   */
    
 
//    return 0;
//}


