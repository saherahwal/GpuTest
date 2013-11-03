#include "../regression.cu"

#define EPSILON 0.000000005

// Some Utility functions

/**
* generates matrix struct with width, height and float elements
*/
Matrix create_matrix( int width, int height, float * elements){
     Matrix A;
     A.n = A.stride = width;
     A.m = height;
     //size_t size_A = sizeof(float) * height * width;
     //A.elts = (float *)malloc( size_A );
     //memcpy( A.elts, elements, sizeof(float) * height * width);
     A.elts = elements;
     return A;
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
* A, B: matrices to multiply. order is A * B
* E is expected Result. 
*/
bool mtx_mult_test(Matrix A, Matrix B, Matrix E){
    
    int width_A = A.n;
    int height_A = A.m;
    size_t size_A = width_A * height_A * sizeof(float);
    float * elements_A = A.elts;
    
    int width_B = B.n;
    int height_B = B.m;
    size_t size_B = width_B * height_B * sizeof(float);
    float * elements_B = B.elts;
    

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(height_A / dimBlock.x, height_A / dimBlock.y);
    
    Matrix d_A = create_matrix(width_A, height_A, elements_A);
    cudaMalloc(&d_A.elts, size_A);
    cudaMemcpy(d_A.elts, A.elts, size_A, cudaMemcpyHostToDevice); 

    Matrix d_B = create_matrix(width_B, height_B, elements_B);
    cudaMalloc(&d_B.elts, size_B);
    cudaMemcpy(d_B.elts, B.elts, size_B, cudaMemcpyHostToDevice);

    
    size_t size_R = width_B * height_A * sizeof(float);
    float * elements_R = (float *)malloc( size_R);
    memset(elements_R, 0.0f, size_R);
    Matrix d_R = create_matrix( width_B, height_A, elements_R);
    cudaMalloc(&d_R.elts, size_R); 
        
    //invoke kernel
    matrix_multiply_matrix<<<dimGrid, dimBlock>>>(d_A, d_B, d_R);
    
    float * elts_c = (float *)malloc(size_R);
    Matrix C = create_matrix( width_B, height_A, elts_c);
    cudaMemcpy(C.elts, d_R.elts, size_R, cudaMemcpyDeviceToHost);


    print_matrix(A);
    print_matrix(B);
    print_matrix(C);

   
    bool test = false;
    if(mtx_equal(C, E)){ 
        test = true;
    }
    
    free(elts_c);
    free(elements_R);
    //free(d_A.elts);
    //free(d_B.elts);
    //free(d_R.elts);
    //free(C.elts);
    cudaFree(d_A.elts);
    cudaFree(d_B.elts);
    cudaFree(d_R.elts);
    
    
    return test;    
 
}

/**
* A: Matrix to transpose
* E: expected transpose result
*/
bool mtx_transpose_test(Matrix A, Matrix E){

    int width_A = A.n;
    int height_A = A.m;
    size_t size_A = width_A * height_A * sizeof(float);
    float * elements_A = A.elts;
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid( width_A / BLOCK_SIZE, height_A / BLOCK_SIZE);

    Matrix d_A = create_matrix(width_A, height_A, elements_A);
    cudaMalloc(&d_A.elts, size_A);
    cudaMemcpy(d_A.elts, A.elts, size_A, cudaMemcpyHostToDevice); 
    
    float * elts_r = (float *)malloc(size_A);
    memset( elts_r, 0.0f, size_A);
    Matrix d_R = create_matrix(height_A, width_A, elts_r);
    cudaMalloc(&d_R.elts, size_A);

    //invoke kernel 
    matrix_transpose<<<dimGrid, dimBlock>>>(d_A, d_R);

    float * elts_c = (float *)malloc(size_A);
    Matrix C = create_matrix(height_A, width_A, elts_c);
    cudaMemcpy(C.elts, d_R.elts, size_A, cudaMemcpyDeviceToHost);

    print_matrix(A);
    print_matrix(C);
    
    bool test = false;
    if(mtx_equal(E, C))
        test = true;
   
    

    free(elts_c);
    free(elts_r);
    cudaFree(d_A.elts);
    cudaFree(d_R.elts);
   
    return test;


}


bool mtx_cholesky_test(Matrix A,  Matrix E){
   
    int width_A = A.n;
    int height_A = A.m;
    size_t size_A = width_A * height_A * sizeof(float);
    float * elements_A = A.elts;

    

    Matrix d_A = create_matrix(width_A, height_A, elements_A);
    cudaMalloc(&d_A.elts, size_A);
    cudaMemcpy(d_A.elts, A.elts, size_A, cudaMemcpyHostToDevice);

    float * elts_r = (float *)malloc(size_A);
    memset( elts_r, 0.0f, size_A);
    Matrix d_R = create_matrix(width_A, height_A, elts_r);
    cudaMalloc(&d_R.elts, size_A);

    //invoke kernel 
    cholesky<<<1, 1>>>(d_A, d_R);

    float * elts_c = (float *)malloc(size_A);
    Matrix C = create_matrix(width_A, height_A, elts_c);
    cudaMemcpy(C.elts, d_R.elts, size_A, cudaMemcpyDeviceToHost);

    print_matrix(A);
    print_matrix(C);

    bool test = false;
    if(mtx_equal(E, C))
        test = true;

    free(elts_c);
    free(elts_r);
    cudaFree(d_A.elts);
    cudaFree(d_R.elts);

    return test;



}




int main( void) {

    printf("run matrix multiply tests\n");
    
    // test 1
    float elts_a[18] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 4.0f, 1.0f, 1.0f, 1.0f};
    Matrix A = create_matrix( 6, 3, elts_a);    
    float elts_b[18] = {1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f};
    Matrix B = create_matrix( 3, 6, elts_b);
    float elts_e[9] = {15.0f, 15.0f, 18.0f, 15.0f, 30.0f, 39.0f, 18.0f, 39.0f, 51.0f};
    Matrix E = create_matrix( 3, 3, elts_e);    
    bool t1 = mtx_mult_test( A, B, E);
    if(t1) printf("PASS \n");
    else printf("FAIL\n");
    //free(A.elts);
    //free(B.elts);
    //free(E.elts);

    
    //test 2
    float elts_a2[9] = {1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 3.0f, 1.0f};
    Matrix A2 = create_matrix( 3, 3, elts_a2);
    float elts_b2[9] = {1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 1.0f};
    Matrix B2 = create_matrix( 3, 3, elts_b2);
    //float elts_r2[9] = {0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f};
    //Matrix R2 = create_matrix( 3, 3, elts_r2);
    float elts_e2[9] = {1.0f, 3.0f, 2.0f, 4.0f, 6.0f, 6.0f, 9.0f, 5.0f, 10.f};
    Matrix E2 = create_matrix(3,3 , elts_e2);
    bool t2 = mtx_mult_test( A2, B2, E2);
    if(t2) printf("PASS \n");
    else printf("FAIL \n");
    //free(A2.elts);
    //free(B2.elts);
    //free(E2.elts);
     
    
    //test 3
    float elts_a3[27] = {1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 2.0f, -1.0f, -2.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
    Matrix A3 = create_matrix( 9, 3, elts_a3);
    float elts_b3[27] = {1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 2.0f, 1.0f, -2.0f, 0.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f,3.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 1.0f, -1.0f};
    Matrix B3 = create_matrix( 3, 9, elts_b3);
    float elts_e3[9] = {16.0f, 4.0f, -10.0f, 4.0f, 10.0f, 0.0f, -10.0f, 0.0f, 15.f};
    Matrix E3 = create_matrix(3,3 , elts_e3);
    bool t3 = mtx_mult_test( A3, B3, E3);
    if(t3) printf("PASS \n");
    else printf("FAIL \n");
    //free(A3.elts);
    //free(B3.elts);
    //free(E3.elts);
    

    // test transpose 
    printf("test matrix transpose \n");    
    
    float t_elts_a1[9] = {1.0f ,2.0f, 3.0f, 0.0f, -5.0f, 1.0f, -1.0f, 9.0f, -1.0f};
    Matrix tA = create_matrix( 3, 3, t_elts_a1);
    float t_elts_e1[9] = {1.0f, 0.0f, -1.0f, 2.0f, -5.0f, 9.0f, 3.0f, 1.0f, -1.0f};
    Matrix tE = create_matrix( 3, 3, t_elts_e1);
    bool tt1 = mtx_transpose_test( tA, tE);
    if(tt1) printf("PASS\n");
    else printf("FAIL \n");

    float t_elts_a2[18] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 4.0f, 1.0f, 1.0f, 1.0f};
    Matrix tA1 = create_matrix( 6, 3, t_elts_a2);
    float t_elts_e2[18] = {1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f};
    Matrix tE1 = create_matrix( 3, 6, t_elts_e2);
    bool tt2 = mtx_transpose_test( tA1, tE1);
    if(tt2) printf("PASS\n");
    else printf("FAIL \n");


    float t_elts_a3[18] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 1.0f, 1.0f, 1.0f, 4.0f, 4.0f, 4.0f, 1.0f, 1.0f, 1.0f};
    Matrix tA3 = create_matrix( 3, 6, t_elts_a3);
    float t_elts_e3[18] = {1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 1.0f, 2.0f, 3.0f, 1.0f, 4.0f, 1.0f};
    Matrix tE3 = create_matrix( 6, 3, t_elts_e3);
    bool tt3 = mtx_transpose_test( tA3, tE3);
    if(tt3) printf("PASS\n");
    else printf("FAIL \n");

    printf("cholesky matrix test \n");
    float c_elts_a1[9] = {25.0f ,15.0f, -5.0f, 15.0f, 18.0f, 0.0f, -5.0f, 0.0f, 11.0f};
    Matrix cA1 = create_matrix( 3, 3, c_elts_a1);
    float c_elts_e1[9] = {5.0f, 0.0f, 0.0f, 3.0f, 3.0f, 0.0f, -1.0f, 1.0f, 3.0f};
    Matrix cE1 = create_matrix( 3, 3, c_elts_e1);
    bool ct1 = mtx_cholesky_test( cA1, cE1);
    if(ct1) printf("PASS\n");
    else printf("FAIL \n");

    float c_elts_a2[9] = {4.0f ,12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f};
    Matrix cA2 = create_matrix( 3, 3, c_elts_a2);
    float c_elts_e2[9] = {2.0f, 0.0f, 0.0f, 6.0f, 1.0f, 0.0f, -8.0f, 5.0f, 3.0f};
    Matrix cE2 = create_matrix( 3, 3, c_elts_e2);
    bool ct2 = mtx_cholesky_test( cA2, cE2);
    if(ct2) printf("PASS\n");
    else printf("FAIL \n");
    



    return 0;
}
