#include <iostream>
#include "../eigen3/Eigen/Dense"

using namespace Eigen;
using namespace std;

void split_matrix(const MatrixXd& A, MatrixXd& D, MatrixXd& L, MatrixXd& U);
VectorXd solve_jacobi(const MatrixXd& D, const MatrixXd& L, const MatrixXd& U, const VectorXd& b);
VectorXd solve_jacobi_sum(const MatrixXd& A, const VectorXd& b);
VectorXd solve_gauss_seidel(const MatrixXd& D, const MatrixXd& L, const MatrixXd& U, const VectorXd& b);
VectorXd solve_gauss_seidel_sum(const MatrixXd& A, const VectorXd& b);

const int ITERACIONES = 50;

int main()
{
    size_t n = 3;
    MatrixXd A(n, n), D, L, U;
    VectorXd b(n);

    A << 1, -0.5, 0.5,
         1, 1, 1,
         -0.5, -0.5, 1;
    
    b << 4,
         -1,
         1;
    
    split_matrix(A, D, L, U);
    
    cout << "Jacobi: " << endl << solve_jacobi(D, L, U, b) << endl;
    cout << "Jacobi Sumatoria: " << endl << solve_jacobi_sum(A, b) << endl;
    cout << "Gauss-Seidel: " << endl << solve_gauss_seidel(D, L, U, b) << endl;
    cout << "Gauss-Seidel Sumatoria: " << endl << solve_gauss_seidel_sum(A, b) << endl;

    return 0;
}

void split_matrix(const MatrixXd& A, MatrixXd& D, MatrixXd& L, MatrixXd& U)
{
    size_t n = A.rows();
    
    D = MatrixXd::Zero(n, n);
    L = MatrixXd::Zero(n, n);
    U = MatrixXd::Zero(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < j)
                U(i, j) = -A(i, j);
            else if (i > j)
                L(i, j) = -A(i, j);
            else
                D(i, j) = A(i, j);
        }
    }
}

VectorXd solve_jacobi(const MatrixXd& D, const MatrixXd& L, const MatrixXd& U, const VectorXd& b)
{
    MatrixXd inv = D.inverse();
    MatrixXd sum = L + U;
    VectorXd x0 = VectorXd::Random(D.rows());

    for (int k = 0; k < ITERACIONES; k++) {
        VectorXd x = inv * (b + sum * x0);
        x0 = x;
    }

    return x0;
}

VectorXd solve_jacobi_sum(const MatrixXd& A, const VectorXd& b)
{
    size_t n = A.rows();
    VectorXd x0 = VectorXd::Random(n);

    for (int k = 0; k < ITERACIONES; k++) {
        VectorXd x(n);
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                if (j != i)
                    sum += A(i, j) * x0(j);
            }
            x(i) = 1 / A(i, i) * (b(i) - sum);
        }
        x0 = x;
    }

    return x0;
}

VectorXd solve_gauss_seidel(const MatrixXd& D, const MatrixXd& L, const MatrixXd& U, const VectorXd& b)
{
    MatrixXd inv = (D - L).inverse();
    VectorXd x0 = VectorXd::Random(D.rows());

    for (int i = 0; i < ITERACIONES; i++) {
        VectorXd x = inv * (b + U * x0);
        x0 = x;
    }

    return x0;
}

VectorXd solve_gauss_seidel_sum(const MatrixXd& A, const VectorXd& b)
{
    size_t n = A.rows();
    VectorXd x0 = VectorXd::Random(n);
    VectorXd xk_1(n);

    for (int k = 0; k < ITERACIONES; k++) {
        for (int i = 0; i < n; i++) {
            float sum, sum2; sum = sum2 = 0;
            for (int j = 0; j < i; j++)
                sum += A(i, j) * xk_1(j);

            for (int j = i + 1; j < n; j++)
                sum2 += A(i, j) * x0(j);

            xk_1(i) = 1 / A(i, i) * (b(i) - sum - sum2);
        }
        x0 = xk_1;
    }

    return x0;
}