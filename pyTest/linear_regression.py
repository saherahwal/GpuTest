################################################################
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
    pass 




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
    pass
    








##### simple test #######

if __name__== "__main__":
    matrix_x =  [ [4,12,-16], [12, 37, -43], [-16, -43, 98]]
    print transpose_matrix(matrix_x)
    print cholesky(matrix_x)

    









