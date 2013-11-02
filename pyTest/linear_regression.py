#################################################################
# simple test library to implement linear regression
#################################################################
from math import sqrt

def dot_product( v1, v2):
    """
    v1,v2: vectors to dot product
    return dot product of two vectors v1 and v2
    """
    assert len(v1) == len(v2)
    result = 0
    for i in xrange(len(v1)):
        result += (v1[i] * v2[i])

    return result


def transpose_matrix(matrix_x):
    """ given matrix_x (a matrix) return the transposed matrix """
    n = len(matrix_x[0])
    m = len(matrix_x)
    
    # construct the transpose of matrix
    matrix_x_t = []
    for i in xrange(n):
        row = []
        for j in xrange(m):
            row.append(matrix_x[j][i])
	matrix_x_t.append(row)

    return matrix_x_t



def cholesky(matrix_x):
    """Performs a Cholesky decomposition of matrix_x, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    n = len(matrix_x)

    # initialize  _L matrix (lower triangular matrix)
    _L = [[0.0] * n for i in xrange(n)]

    # Perform the Cholesky decomposition
    for i in xrange(n):
        for k in xrange(i+1):
            tmp_sum = sum(_L[i][j] * _L[k][j] for j in xrange(k))
              
            if (i == k): # Diagonal elements
                _L[i][k] = sqrt(matrix_x[i][i] - tmp_sum)
            else:
                _L[i][k] = (1.0 / _L[k][k] * (matrix_x[i][k] - tmp_sum))
    return _L



def solve_fwd_bkwd(matrix_a, b):
    """ solved Ax = b linear by using Cholesky decomposition
    this assumes that matrix_a is symmetric and positive definite """
    _L = cholesky(matrix_a) 
    _U = transpose_matrix(_L) 
    
    n = len(b)
    x = [0 for i in xrange(n)] 
    y = [0 for i in xrange(n)]    

    #forward solve _Ly = b
    for i in xrange(n):
        y[i] = b[i]
        for j in xrange(i):
	    y[i] -= _L[i][j] * y[j]
	y[i] /= _L[i][i]

    #backward solve _Ux = y
    for i in xrange(n-1, -1, -1):
	x[i] = y[i]
        for j in xrange(i+1, n):
            x[i] -= _U[i][j] * x[j]
        x[i] /= _U[i][i]

    return x

    



def get_matrices( ind_v):
    """
    ind_v: list of lists, where each element is a list of independent variable values [ [x0,y0,z0], [x1,y1,z1] ... ]
    
    return matrix_a, matrix_a_t : where matrix_a_t is the transpose of the matrix_a
    """
    matrix_a = [ [1] + e for e in ind_v] ## add 1 elt to list to account for constant value in equations
    matrix_a_t = transpose_matrix(matrix_a)
    return (matrix_a, matrix_a_t)





def matrix_mult_vec(matrix_a, x):
    """ multiply matrix matrix_a by vector x
    """
    m = len(matrix_a)
    b = [0 for i in xrange(m)]
    for i in xrange(m):
        b[i] = dot_product(matrix_a[i], x)
    return b




def matrix_mult_matrix(matrix_a, matrix_b):
    """ return matrix result of multiplying matrix_a and matrix_b
    """
    m = len(matrix_a)
    n = len(matrix_b)
    result = []
    matrix_b_t = transpose_matrix(matrix_b)
    for i in xrange(m):
        row = []
	for j in xrange(m):
            row.append(dot_product(matrix_a[i], matrix_b_t[j]))
	result.append(row)
    return result




def get_matrices( ind_v):
    """
    ind_v: list of lists, where each element is a list of independent variable values [ [x0,y0,z0], [x1,y1,z1] ... ]
    
    return matrix_a, matrix_a_t : where matrix_a_t is the transpose of the matrix_a
    """
    matrix_a = [ [1] + e for e in ind_v] ## add 1 elt to list to account for constant value in equations
    matrix_a_t = transpose_matrix(matrix_a)
    return (matrix_a, matrix_a_t)






def linear_regression( ind_v , b ):
    """ 
    ind_v: list of lists, where each element is a list of independent variable values [ [x0,y0,z0], [x1,y1,z1] ...]
    b : list of values corresponding to each ind_v value
    """
    ## get matrices A and A_transpose
    a, a_t = get_matrices( ind_v )
    a_t_mult_a = matrix_mult_matrix(a_t, a) ## A_transpose * A
    a_t_mult_b = matrix_mult_vec(a_t, b)  
    result =  solve_fwd_bkwd( a_t_mult_a, a_t_mult_b) ## Solve A_transpose*A x = A_transpose * b
    return result



##### simple test #######
if __name__== "__main__":
    ## test1
    A = [[0], [1], [2]]
    b = [6, 0, 0]

    r= linear_regression(A, b)
    print "test1"
    print "A", A
    print "b", b
    print "result", r , "should be close to [5, -3]"

    print "test2"
    A = [ [-2], [1], [3]]
    b = [-1, 1, 2]
    r = linear_regression(A,b)
    print "A", A
    print "b", b
    print "result", r, "should be close to [", str(float(5)/19), ",", str(float(23)/38),"]"
    
    print "test3"
    A = [ [1,2],
          [2,5],
          [2,3],
	  [2,2],
          [3,4],
          [3,5],
          [4,6],
          [5,5],
          [5,6],
          [5,7],
          [6,8],
          [7,6],
          [8,4],
          [8,9],
          [9,8]]
    b = [2,1,2,2,1,3,2,3,4,3,4,2,4,3,4]
    r = linear_regression(A,b)
    print "A", A
    print "b", b
    print "result", r, "should be close to [1.353480, 0.286191, -0.004195]"

      

    print linear_regression(A, b)

    ### Cholesky Test
    C = [ [25.0, 15, -5], [15, 18, 0], [-5, 0,11]]
    L = cholesky(C);
    print L;

