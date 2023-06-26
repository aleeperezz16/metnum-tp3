#include <iostream>
#include <fstream>
#include <chrono>
#include <eigen3/Eigen/Dense>

enum {
  Jacobi,
  JacobiSum,
  GaussSeidel,
  GaussSeidelSum
};

using namespace Eigen;
using namespace std;

void splitMatrix(MatrixXd &, MatrixXd &, MatrixXd &);
VectorXd jacobi(const MatrixXd &, const MatrixXd &, const MatrixXd &);
VectorXd jacobiSum();
VectorXd gaussSeidel(const MatrixXd &, const MatrixXd &, const MatrixXd &);
VectorXd gaussSeidelSum();

MatrixXd A;
VectorXd x, b;

size_t iteraciones;
double error;

int main(int argc, const char **argv) {
  if (argc < 5) {
    cerr << "Uso: " << argv[0] << " <sistema> <metodo> <iteraciones> <maximo_error>" << endl;
    return -1;
  }

  ifstream sistema(argv[1]);
  if (!sistema.is_open()) {
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
  else {
    cerr << "Metodo invalido" << endl;
    return -1;
  }

  int n; sistema >> n;
  A.resize(n, n), x.resize(n), b.resize(n);

  // Cargo la matriz A
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      sistema >> A(i, j);
  }

  // Cargo la solución del sistema
  for (int i = 0; i < n; i++)
    sistema >> x(i);

  // Cargo el resultado del sistema
  for (int i = 0; i < n; i++)
    sistema >> b(i);

  sistema.close();

  iteraciones = atoi(argv[3]);
  error = atof(argv[4]);

  MatrixXd D, L, U;
  VectorXd solucion;
  splitMatrix(D, L, U);

  auto inicio = chrono::high_resolution_clock::now();

  switch (metodo) {
  case Jacobi:
    solucion = jacobi(D, L, U);
    break;
  case JacobiSum:
    solucion = jacobiSum();
    break;
  case GaussSeidel:
    solucion = gaussSeidel(D, L, U);
    break;
  case GaussSeidelSum:  
    solucion = gaussSeidelSum();
    break;
  }
  
  auto fin = chrono::high_resolution_clock::now();
  cout << "Tiempo: " << chrono::duration<double, milli>(fin - inicio).count() << endl;
  cout << "Solucion: " << solucion.transpose() << endl;

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
    x0 = x_k;

    if ((x0 - x).norm() > error) {
      cerr << "El metodo diverge a la solucion" << endl;
      break;
    }
  }

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
    x0 = x_k;

    if ((x0 - x).norm() > error) {
      cerr << "El metodo diverge a la solucion" << endl;
      break;
    }
  }

  return x0;
}

VectorXd gaussSeidel(const MatrixXd &D, const MatrixXd &L, const MatrixXd &U) {
  MatrixXd inv = (D - L).inverse();
  VectorXd x0 = VectorXd::Random(D.rows());

  for (size_t i = 0; i < iteraciones; i++) {
    VectorXd x_k = inv * (b + U * x0);
    x0 = x_k;

    if ((x0 - x).norm() > error) {
      cerr << "El metodo diverge a la solucion" << endl;
      break;
    }
  }

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

    x0 = xk_1;

    if ((x0 - x).norm() > error) {
      cerr << "El metodo diverge a la solucion" << endl;
      break;
    }
  }

  return x0;
}