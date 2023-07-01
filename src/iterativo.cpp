#include <iostream>
#include <fstream>
#include "Eigen/Dense"

enum {
  Jacobi,
  JacobiSum,
  GaussSeidel,
  GaussSeidelSum,
  FactLU
};

using namespace Eigen;
using namespace std;

void splitMatrix(MatrixXd &, MatrixXd &, MatrixXd &);
VectorXd jacobi(const MatrixXd &, const MatrixXd &, const MatrixXd &);
VectorXd jacobiSum();
VectorXd gaussSeidel(const MatrixXd &, const MatrixXd &, const MatrixXd &);
VectorXd gaussSeidelSum();

MatrixXd A;
VectorXd b;

size_t iteraciones;
double corte;

int main(int argc, const char **argv) {
  if (argc < 5) {
    cerr << "Uso: " << argv[0] << " <matriz_extendida> <metodo> <iteraciones> <corte>" << endl;
    return -1;
  }

  ifstream archivo(argv[1]);
  if (!archivo.is_open()) {
    cerr << "Error al abrir el sistema de entrada" << endl;
    return -1;
  }

  string arg = argv[2];
  int metodo;

  if (arg == "J") 
    metodo = Jacobi;
  else if (arg == "JS")
    metodo = JacobiSum;
  else if (arg == "GS")
    metodo = GaussSeidel;
  else if (arg == "GSS")
    metodo = GaussSeidelSum;
  else if (arg == "LU")
    metodo = FactLU;
  else {
    cerr << "Metodo invalido" << endl;
    return -1;
  }

  size_t n; archivo >> n;
  A.resize(n, n), b.resize(n);

  // Cargo el sistema
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= n; j++) {
      if (j == n)
        archivo >> b(i);
      else
        archivo >> A(i, j);
    } 
  }

  archivo.close();

  iteraciones = atoi(argv[3]);
  corte = atof(argv[4]);

  MatrixXd D, L, U;
  VectorXd solucion;

  switch (metodo) {
  case Jacobi:
    splitMatrix(D, L, U);
    solucion = jacobi(D, L, U);
    break;
  case JacobiSum:
    solucion = jacobiSum();
    break;
  case GaussSeidel:
    splitMatrix(D, L, U);
    solucion = gaussSeidel(D, L, U);
    break;
  case GaussSeidelSum:  
    solucion = gaussSeidelSum();
    break;
  case FactLU:
    solucion = A.lu().solve(b);
    break;
  }
  IOFormat fmt(StreamPrecision, DontAlignCols);
  cout << solucion.transpose().format(fmt);
  return 0;
}

void splitMatrix(MatrixXd &D, MatrixXd &L, MatrixXd &U) {
  size_t n = A.rows();

  D = MatrixXd::Zero(n, n);
  L = MatrixXd::Zero(n, n);
  U = MatrixXd::Zero(n, n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i < j)
        U(i, j) = -A(i, j);
      else if (i > j)
        L(i, j) = -A(i, j);
      else
        D(i, j) = A(i, j);
    }
  }
}

VectorXd jacobi(const MatrixXd &D, const MatrixXd &L, const MatrixXd &U) {
  MatrixXd inv = D.inverse();
  MatrixXd sum = L + U;
  VectorXd x0 = VectorXd::Random(D.rows());

  for (size_t k = 0; k < iteraciones; k++) {
    VectorXd x_k = inv * (b + sum * x0);

    if ((x_k - x0).norm() < corte)
      return x_k;

    x0 = x_k;

  }
  cerr << "Número de iteraciones máximo alcanzado, el método posiblemente diverge." << endl;
  return x0;
}

VectorXd jacobiSum() {
  size_t n = A.rows();
  VectorXd x0 = VectorXd::Random(n);

  for (size_t k = 0; k < iteraciones; k++) {
    VectorXd x_k(n);
    for (size_t i = 0; i < n; i++) {
      double sum = 0;

      for (size_t j = 0; j < n; j++) {
        if (j != i)
          sum += A(i, j) * x0(j);
      }

      x_k(i) = (b(i) - sum) / A(i, i);
    }

    if ((x_k - x0).norm() < corte)
      return x_k;


    x0 = x_k;

  }
  cerr << "Número de iteraciones máximo alcanzado, el método posiblemente diverge." << endl;
  return x0;
}

VectorXd gaussSeidel(const MatrixXd &D, const MatrixXd &L, const MatrixXd &U) {
  MatrixXd inv = (D - L).inverse();
  VectorXd x0 = VectorXd::Random(D.rows());

  for (size_t i = 0; i < iteraciones; i++) {
    VectorXd x_k = inv * (b + U * x0);

    if ((x_k - x0).norm() < corte)
      return x_k;

    x0 = x_k;

  }
  cerr << "Número de iteraciones máximo alcanzado, el método posiblemente diverge." << endl;
  return x0;
}

VectorXd gaussSeidelSum() {
  size_t n = A.rows();

  VectorXd x0 = VectorXd::Random(n);
  VectorXd xk_1(n);

  for (size_t k = 0; k < iteraciones; k++) {
    for (size_t i = 0; i < n; i++) {
      double sum, sum2;
      sum = sum2 = 0;

      for (size_t j = 0; j < i; j++)
        sum += A(i, j) * xk_1(j);

      for (size_t j = i + 1; j < n; j++)
        sum2 += A(i, j) * x0(j);

      xk_1(i) = (b(i) - sum - sum2) / A(i, i);
    }

    if ((xk_1 - x0).norm() < corte)
      return xk_1;
      
    x0 = xk_1;

  }
  cerr << "Número de iteraciones máximo alcanzado, el método posiblemente diverge." << endl;
  return x0;
}