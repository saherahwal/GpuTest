#include "../regression.cu"



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
    
    //printf("ceil test test %f\n", ceil(width_A / (float)BLOCK_SIZE));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(height_A / (float)dimBlock.x), ceil(width_B / (float)dimBlock.y));
    
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
    
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid( ceil(width_A / (float)BLOCK_SIZE), ceil(height_A / (float)BLOCK_SIZE));

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

    //invoke kernel 
    //matrix_transpose<<<dimGrid, dimBlock>>>(d_A, d_R);
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

/*
* L, U: cholesky decomposition result
* b: dependent variable vector values
* e: expected result of fwd_bkwd_elimination
*/
bool mtx_fwd_bkwd_elimination_test( Matrix L, Matrix U, Vector b, Vector e){
    
    int width_L = L.n;
    int height_L = L.m;
    size_t size_L = width_L * height_L * sizeof(float);
    float * elements_L = L.elts;

    int width_U = U.n;
    int height_U = U.m;
    size_t size_U = width_U * height_U * sizeof(float);
    float * elements_U = U.elts;
   
    int v_len = b.length;
    float * v_elts = b.elts;


    size_t size_v = v_len * sizeof(float);

    Matrix d_L = create_matrix(width_L, height_L, elements_L);
    cudaMalloc(&d_L.elts, size_L);
    cudaMemcpy(d_L.elts, L.elts, size_L, cudaMemcpyHostToDevice);

    Matrix d_U = create_matrix(width_U, height_U, elements_U);
    cudaMalloc(&d_U.elts, size_U);
    cudaMemcpy(d_U.elts, U.elts, size_U, cudaMemcpyHostToDevice);

    
    Vector d_b = create_vector(v_len, v_elts); 
    cudaMalloc(&d_b.elts, size_v);
    cudaMemcpy(d_b.elts, b.elts, size_v, cudaMemcpyHostToDevice);


    float * elts_r = (float *)malloc(size_v);
    memset( elts_r, 0.0f, size_v);
    Vector d_r = create_vector( v_len, elts_r);
    cudaMalloc(&d_r.elts, size_v);

    //invoke kernel 
    fwd_bkwd_elimination<<<1, 1>>>(d_L, d_U, d_b, d_r);
    
    
    
    float * c_elts = (float *)malloc(size_v);
    memset(c_elts, 0.0f, size_v);
    Vector c = create_vector(v_len, c_elts);
    cudaMemcpy(c.elts, d_r.elts, size_v , cudaMemcpyDeviceToHost);
 
    //print L, U, b, 
    print_matrix(L);
    print_matrix(U);
    print_vector(b);
    print_vector(c);    

    bool test = false;
    if(vec_equal(e, c))
        test = true;

    free(c_elts);
    free(elts_r);
    cudaFree(d_b.elts);
    cudaFree(d_L.elts);
    cudaFree(d_U.elts);
    cudaFree(d_r.elts);    

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
   



    // test different dimensions
    float elts_a4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix A4 = create_matrix( 2, 2, elts_a4);
    float elts_b4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix B4 = create_matrix( 2, 2, elts_b4);
    float elts_e4[4] = {7.0f, 10.0f, 15.0f, 22.0f};
    Matrix E4 = create_matrix( 2, 2, elts_e4);
    bool t4 = mtx_mult_test( A4, B4, E4);
    if(t4) printf("PASS \n");
    else printf("FAIL\n");


    float elts_a6[8] = {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f};
    Matrix A6 = create_matrix( 4, 2, elts_a6);
    float elts_b6[4] = {1.0f, 0.0f, 1.0f, 1.0f};
    Matrix B6 = create_matrix( 1, 4, elts_b6);
    float elts_e6[4] = {3.0f, 2.0f};
    Matrix E6 = create_matrix( 1, 2, elts_e6);
    bool t6 = mtx_mult_test( A6, B6, E6);
    if(t6) printf("PASS \n");
    else printf("FAIL\n");

    
    float elts_a7[30] = {1.0f, 2.0f, 2.0f, 5.0f, 2.0f, 3.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 5.0f, 4.0f, 6.0f, 5.0f, 5.0f, 5.0f, 6.0f, 5.0f, 7.0f, 6.0f, 8.0f, 7.0f, 6.0f, 8.0f, 4.0f, 8.0f, 9.0f, 9.0f, 8.0f};
    Matrix A7 = create_matrix( 2, 15, elts_a7);
    float elts_b7[30] = {1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 9.0f, 2.0f, 5.0f, 3.0f, 2.0f, 4.0f, 5.0f, 6.0f, 5.0f, 6.0f, 7.0f, 8.0f, 6.0f, 4.0f, 9.0f, 8.0f};
    Matrix B7 = create_matrix( 15, 2, elts_b7);
    float elts_e7[4] = {416.0f, 429.0f, 429.0f, 490.0f};
    Matrix E7 = create_matrix( 2, 2, elts_e7);
    bool t7 = mtx_mult_test( B7, A7, E7);
    if(t7) printf("PASS \n");
    else printf("FAIL\n");

 

    //test matrix multiply vector 
    float elts_a5[9] = {1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f, -3.0f, 1.0f, 2.0f};
    Matrix A5 = create_matrix( 3, 3, elts_a5);
    float elts_b5[3] = {1.0f, -2.0f, 3.0f};
    Matrix B5 = create_matrix( 1, 3, elts_b5); 
    float elts_e5[3] = {0.0f, 1.0f, 1.0f};
    Matrix E5 = create_matrix( 1, 3, elts_e5);
    bool t5 = mtx_mult_test( A5, B5, E5);
    if(t5) printf("PASS \n");
    else printf("FAIL\n");




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


    float t_elts_a4[4] = {1.0f ,2.0f, 7.0f, 9.0f};
    Matrix tA4 = create_matrix( 2, 2, t_elts_a4);
    float t_elts_e4[9] = {1.0f, 7.0f, 2.0f, 9.0f};
    Matrix tE4 = create_matrix( 2, 2, t_elts_e4);
    bool tt4 = mtx_transpose_test( tA4, tE4);
    if(tt4) printf("PASS\n");
    else printf("FAIL \n");

    float t_elts_a5[25] = {1.0f ,2.0f, 3.0f, 4.0f, 5.0f, 1.0f ,2.0f, 3.0f, 4.0f, 5.0f, 1.0f ,2.0f, 3.0f, 4.0f, 5.0f, 1.0f ,2.0f, 3.0f, 4.0f, 5.0f, 1.0f ,2.0f, 3.0f, 4.0f, 5.0f};
    Matrix tA5 = create_matrix( 5, 5, t_elts_a5);
    float t_elts_e5[25] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    Matrix tE5 = create_matrix( 5, 5, t_elts_e5);
    bool tt5 = mtx_transpose_test( tA5, tE5);
    if(tt5) printf("PASS\n");
    else printf("FAIL \n");
 


    
    // start cholesky tests
    
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
    
    // start fwd_bkwd_elimination test
    
    printf("fwd_bkwd eliminiation test \n");
    float fb_elts_l1[9] = {2.0f, 0.0f, 0.0f, 6.0f, 1.0f, 0.0f, -8.0f, 5.0f, 3.0f};
    Matrix fL1 = create_matrix(3,3, fb_elts_l1);
    float fb_elts_u1[9] = {2.0f, 6.0f, -8.0f, 0.0f, 1.0f, 5.0f, 0.0f, 0.0f, 3.0f};
    Matrix fU1 = create_matrix(3,3, fb_elts_u1);
    float fb_elts_b1[3] = {0.0f, 6.0f, 39.0f};
    Vector b1 = create_vector(3, fb_elts_b1);
    float fb_elts_e1[3] = {1.0f, 1.0f, 1.0f};
    Vector e1 = create_vector( 3, fb_elts_e1); 
    bool fbt1 = mtx_fwd_bkwd_elimination_test(fL1, fU1, b1, e1);
    if(fbt1) printf("PASS\n");
    else printf("FAIL\n");

    
    float fb_elts_l2[9] = {5.0f, 0.0f, 0.0f, 3.0f, 3.0f, 0.0f, -1.0f, 1.0f, 3.0f};
    Matrix fL2 = create_matrix(3,3, fb_elts_l2);
    float fb_elts_u2[9] = {5.0f, 3.0f, -1.0f, 0.0f, 3.0f, 1.0f, 0.0f, 0.0f, 3.0f};
    Matrix fU2 = create_matrix(3,3, fb_elts_u2);
    float fb_elts_b2[3] = {-65.0f, -30.0f, 43.0f};
    Vector b2 = create_vector(3, fb_elts_b2);
    float fb_elts_e2[3] = {-2.0f, 0.0f, 3.0f};
    Vector e2 = create_vector( 3, fb_elts_e2);
    bool fbt2 = mtx_fwd_bkwd_elimination_test(fL2, fU2, b2, e2);
    if(fbt2) printf("PASS\n");
    else printf("FAIL\n");
   
    
    //test regression 
    float reg_elts_a1[45] = {1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 5.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 2.0f, 1.0f,3.0f, 4.0f, 1.0f,3.0f, 5.0f, 1.0f,4.0f, 6.0f, 1.0f, 5.0f, 5.0f,1.0f,  5.0f, 6.0f, 1.0f, 5.0f, 7.0f, 1.0f, 6.0f, 8.0f, 1.0f, 7.0f, 6.0f, 1.0f, 8.0f, 4.0f, 1.0f, 8.0f, 9.0f, 1.0f, 9.0f, 8.0f};
    Matrix reg_A1 = create_matrix(3, 15, reg_elts_a1);
    float reg_elts_b1[15] = {2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 3.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 2.0f, 4.0f, 3.0f, 4.0f};
    Matrix reg_b1 = create_matrix(1, 15, reg_elts_b1);
    Vector reg_x1 = linear_regression(reg_A1, reg_b1);
    print_vector(reg_x1);



    return 0;
}
