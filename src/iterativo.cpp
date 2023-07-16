#include <iostream>
#include <fstream>
#include <Eigen/Dense>

enum {
  Jacobi,
  JacobiSum,
  GaussSeidel,
  GaussSeidelSum,
  FactLU
};

using namespace Eigen;
using namespace std;

void splitMatrix(const MatrixXd &, MatrixXd &, MatrixXd &, MatrixXd &);
VectorXd jacobi(const MatrixXd &, const VectorXd &, size_t, double);
VectorXd jacobiSum(const MatrixXd &, const VectorXd &, size_t, double);
VectorXd gaussSeidel(const MatrixXd &, const VectorXd &, size_t, double);
VectorXd gaussSeidelSum(const MatrixXd &, const VectorXd &, size_t, double);

int main(int argc, const char **argv) {
  if (argc < 5) {
    cerr << "Uso: " << argv[0] << " <matriz_extendida> <metodo> <iteraciones> <tolerancia>" << endl;
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
  MatrixXd A(n, n);
  VectorXd b(n);

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

  size_t iteraciones = atoi(argv[3]);
  double tolerancia = atof(argv[4]);

  VectorXd solucion;

  switch (metodo) {
  case Jacobi:
    solucion = jacobi(A, b, iteraciones, tolerancia);
    break;
  case JacobiSum:
    solucion = jacobiSum(A, b, iteraciones, tolerancia);
    break;
  case GaussSeidel:
    solucion = gaussSeidel(A, b, iteraciones, tolerancia);
    break;
  case GaussSeidelSum:  
    solucion = gaussSeidelSum(A, b, iteraciones, tolerancia);
    break;
  case FactLU:
    solucion = A.lu().solve(b);
    break;
  }

  IOFormat fmt(StreamPrecision, DontAlignCols);
  cout << solucion.transpose().format(fmt);
  return 0;
}

void splitMatrix(const MatrixXd &A, MatrixXd &D, MatrixXd &L, MatrixXd &U) {
  D = A.diagonal().asDiagonal();
  L = -(MatrixXd)(A.triangularView<StrictlyLower>());
  U = -(MatrixXd)(A.triangularView<StrictlyUpper>());
}

VectorXd jacobi(const MatrixXd &A, const VectorXd &b, size_t iteraciones, double tolerancia) {
  MatrixXd D, L, U;
  splitMatrix(A, D, L, U);

  MatrixXd D_inv = D.inverse();
  VectorXd x0 = VectorXd::Random(D.rows());
  MatrixXd T = D_inv * (L + U);
  VectorXd c = D_inv * b;

  for (size_t k = 0; k < iteraciones; k++) {
    VectorXd x = T * x0 + c;

    if ((x - x0).norm() < tolerancia)
      return x;

    x0 = x;
  }

  cerr << "El método no converge para " << iteraciones << " iteracion(es)." << endl;
  return x0;
}

VectorXd jacobiSum(const MatrixXd &A, const VectorXd &b, size_t iteraciones, double tolerancia) {
  size_t n = A.rows();
  VectorXd x0 = VectorXd::Random(n);
  VectorXd x = VectorXd::Zero(n);

  for (size_t k = 0; k < iteraciones; k++) {
    for (size_t i = 0; i < n; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        if (j != i)
          sum += A(i, j) * x0(j);
      }

      x(i) = (b(i) - sum) / A(i, i);
    }

    if ((x - x0).norm() < tolerancia)
      return x;

    x0 = x;
  }

  cerr << "El método no converge para " << iteraciones << " iteracion(es)." << endl;
  return x0;
}

VectorXd gaussSeidel(const MatrixXd &A, const VectorXd &b, size_t iteraciones, double tolerancia) {
  MatrixXd D, L, U;
  splitMatrix(A, D, L, U);
  
  MatrixXd D_L = (D - L).inverse();
  VectorXd x0 = VectorXd::Random(D.rows());
  MatrixXd T = D_L * U;
  VectorXd c = D_L * b;

  for (size_t k = 0; k < iteraciones; k++) {
    VectorXd x = T * x0 + c;
    
    if ((x - x0).norm() < tolerancia)
      return x;

    x0 = x;
  }

  cerr << "El método no converge para " << iteraciones << " iteracion(es)." << endl;
  return x0;
}

VectorXd gaussSeidelSum(const MatrixXd &A, const VectorXd &b, size_t iteraciones, double tolerancia) {
  size_t n = A.rows();
  VectorXd x0 = VectorXd::Random(n);
  VectorXd x = VectorXd::Zero(n);

  for (size_t k = 0; k < iteraciones; k++) {
    for (size_t i = 0; i < n; i++) {
      double sum1 = 0.0, sum2 = 0.0;
      for (size_t j = 0; j < i; j++)
        sum1 += A(i, j) * x(j);

      for (size_t j = i + 1; j < n; j++)
        sum2 += A(i, j) * x0(j);

      x(i) = (b(i) - sum1 - sum2) / A(i, i);
    }

    if ((x - x0).norm() < tolerancia)
      return x;
      
    x0 = x;
  }

  cerr << "El método no converge para " << iteraciones << " iteracion(es)." << endl;
  return x0;
}