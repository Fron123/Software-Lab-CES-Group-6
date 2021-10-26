#include "dco.hpp"
#include "ColPackHeaders.h"
#include <iostream>
#include <array>
#include <cmath>
#include <typeinfo>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

using namespace ColPack;

//objective function
//auslagern, dann folgendes:
template<typename T, typename TP, size_t N, size_t NP>
void f(
    const Eigen::Matrix<T,N,1>& x,
    const Eigen::Matrix<TP,NP,1>& p,
    T& y
){
    using namespace std;
    y=p[0]*x[0]*x[0] + exp(x[1]);  //funktionsinput
}

//first derivative (gradient)
template<typename T, typename TP, size_t N, size_t NP>
void df(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<T,N,1>& dydx
){
    typedef typename dco::ga1s<T> DCO_M;
    typedef typename DCO_M::type DCO_T;
    typedef typename DCO_M::tape_t DCO_TT;
    DCO_M::global_tape=DCO_TT::create();
    Eigen::Matrix<DCO_T,N,1> x;
    DCO_T y;
    for (size_t i=0;i<N;i++) x(i)=xv(i);
    for (auto& i:x) DCO_M::global_tape->register_variable(i);
    f<DCO_T,TP,N,NP>(x,p,y);
    yv=dco::value(y);
    DCO_M::global_tape->register_output_variable(y);
    dco::derivative(y)=1;
    DCO_M::global_tape->interpret_adjoint();
    for (size_t i=0;i<N;i++) dydx(i)=dco::derivative(x(i));
    DCO_TT::remove(DCO_M::global_tape);
}

//dydx an "Hauptfunktion übergeben mit y_s = dydx
//Ende f

//System F
template<typename T, typename TP, size_t N, size_t NP>
void F(
    const Eigen::Matrix<T,N,1>& x,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& y
){
    y << x(0)*x(0),x(1)*x(1),x(2)*x(2),x(3);
}

//first derivative (Jacobian)
template<typename T, typename TP, size_t N, size_t NP>
void dF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<T,N,N>& dydx
){
    typedef typename dco::gt1s<T>::type DCO_T;
    Eigen::Matrix<DCO_T,N,1> x,y;
    for (size_t i=0;i<N;i++){
        y[i]=yv[i];
        x[i]=xv[i];
    }
    for(size_t i=0;i<N;i++){
        dco::derivative(x(i))=1;
        F<DCO_T,TP,N,NP>(x,p,y);
        for (size_t j=0;j<N;j++) dydx(j,i)=dco::derivative(y(j));
        dco::derivative(x(i)) = 0;
    }
}

//third derivative (tensor)
template<typename T, typename TP, size_t N, size_t NP>
void ddF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<T,N,N>& dydx_v,
    Eigen::Matrix<T,N,N>& ddydxx
){
    typedef typename dco::gt1s<T>::type DCO_T;
    Eigen::Matrix<DCO_T, N, 1> x; Eigen::Matrix<DCO_T, N, N> dydx;
    Eigen::Matrix<DCO_T, N, 1> y;

for(int i = 0;i<N;i++){
    for(int j = 0;j<N;j++) ddydxx(i,j) = 0;
  }

  for (size_t i=0;i<N;i++){
     x(i)=xv(i);
  }
  for(size_t k=0;k<N; k++){
    for (size_t i=0;i<N;i++){
      dco::derivative(x(i)) = 1;
      dF<DCO_T, TP, N, NP>(x, p, y, dydx);
    for (size_t j=0; j<N; j++) ddydxx(j,k) += dco::derivative(dydx(j,k));
      dco::derivative(x(i))=0;
    }
  }
  for(auto i=0;i<N;i++){
    yv(i)=dco::value(y(i));
  for(auto j=0;j<N;j++) dydx_v(i,j)=dco::value(dydx(i,j));
  }
}

//sparsity F'
template<typename T, typename TP, size_t N, size_t NP>
void S_dF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<bool,N,N> &S_dF
) {
    using DCO_T=dco::p1f::type;
    Eigen::Matrix<DCO_T,N,1> x,y;
    Eigen::Matrix<DCO_T,N,N> dydx;

    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,i);
    }
    F<DCO_T,TP,N,NP>(x,p,y);
    for (size_t i=0; i<N;i++){
        dco::p1f::get(y(i),yv(i));
    for (size_t j=0;j<N;j++) dco::p1f::get(y(i),S_dF(i,j),j);
    }
}

template<typename T, typename TP, size_t N, size_t NP>
void S_ddF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<bool,N,N> &S_ddF
) {
    using DCO_T=dco::p1f::type;
    Eigen::Matrix<DCO_T,N,1> x,y;
    Eigen::Matrix<DCO_T,N,N> dydx;

    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,i);
    }
    dF<DCO_T,TP,N,NP>(x,p,y,dydx);
    for (size_t i=0; i<N;i++){
        dco::p1f::get(y(i),yv(i));
    for (size_t j=0;j<N;j++) dco::p1f::get(dydx(i,j),S_ddF(i,j),j);
    }
}


//Namen der ganzen übergaben ordentlich machen. Was ist was ?
//Namensgebung wie folgt:
//Ableitungen: dF, ddF etc.
//Sparsity: S_
//Variabler teil Jacobi dFv
//konstanter Teil Jacobi dFc
//Seedmatrix V_
//compressed comp_
//Konstante elemente C_

//Beispiel comp_S_dFv ist das komprimierte Sparsitypattern der variablen Teilmatrix der Jacobimatrix

//Decompressed
template<typename T, typename TP, size_t N, size_t NP>
void dFv(
  const Eigen::Matrix<T, N, 1>& xv,
  const Eigen::Matrix<TP, NP, 1>& p,
  Eigen::Matrix<T, N, 1>& y_s,
  Eigen::Matrix<bool, N, N> sparsity_pattern,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& seed,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& sparsity_pattern_dFv,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& CompressedJacobian,
  Eigen::Matrix<double, N, N>& CdF_,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& compressed_dFv_v,
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& full_dFv_v
) {

    Eigen::Matrix<int, N,1> x_;
    x_.setZero();

    for(int i = 0; i<N; i++){
        for(int j = 0; j<N;j++){
          if(sparsity_pattern(i,j) == 1){
            x_(i) += 1;
          }
        }
      }


int max = x_.maxCoeff() + 1;

sparsity_pattern_dFv = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N, max);

for(int i = 0; i<N; i++){
    sparsity_pattern_dFv(i,0) = x_(i);
    int counter = 1;
    for(int j = 0; j<N ;j++){

      if(sparsity_pattern(i,j) == 1){
        sparsity_pattern_dFv(i,counter) = j;
        counter++;
      }
    }
  }

unsigned int **uip2_JacobianSparsityPattern = new unsigned int *[N];
for(int i=0;i<N;i++) uip2_JacobianSparsityPattern[i] = new unsigned int[N];

  for(int i = 0; i < N; i++){
    for(int j = 0; j < max; j++){
        uip2_JacobianSparsityPattern[i][j] = sparsity_pattern_dFv(i,j);
    }
  }

  double*** dp3_Seed = new double**;
  int *ip1_SeedRowCount = new int;
  int *ip1_SeedColumnCount = new int;

  BipartiteGraphPartialColoringInterface* g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC,uip2_JacobianSparsityPattern, N ,N);

  g->PartialDistanceTwoColoring( "SMALLEST_LAST", "COLUMN_PARTIAL_DISTANCE_TWO");

  (*dp3_Seed) = g->GetSeedMatrix(ip1_SeedRowCount, ip1_SeedColumnCount);


  int rows = g->GetColumnVertexCount();
  int cols = g->GetRightVertexColorCount();
  double **Seed = *dp3_Seed;

  seed = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);

  for(int i=0; i<rows; i++){
    for(int j=0; j<cols; j++){
      seed(i,j) = Seed[i][j];
    }
  }

using DCO_T=typename dco::gt1s<T>::type;

int rows_seed = seed.rows();
int cols_seed = seed.cols();

  Eigen::Matrix<DCO_T, N, 1> x;
  Eigen::Matrix<DCO_T, N, 1> yv;
  Eigen::Matrix<DCO_T, N, 1> y;

  for(size_t i = 0;i<N;i++){
      x(i) = xv(i);
      yv(i) = y_s(i);
  }

  double** compressedJacobian = new double*[N];
    for (size_t i = 0; i < N; i++)
        compressedJacobian[i] = new double[cols_seed];

    for(size_t i = 0; i < cols_seed; i++) {
        for(size_t j = 0; j < rows_seed; j++){
            dco::derivative(x(j)) = seed(j,i); //der Clou dahinter
        }
        F<DCO_T,TP,N,NP>(x,p,y,yv);
        for(size_t j = 0; j < rows_seed; j++) {
            compressedJacobian[j][i] = dco::derivative(y(j));

            std::cout<<y(j) << " ";
        }
        std::cout <<"" <<std::endl;
    }


  CompressedJacobian = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows_seed, cols_seed);

    for(int i=0; i<rows_seed; i++){
    for(int j=0; j<cols_seed; j++){
      CompressedJacobian(i,j) = compressedJacobian[i][j];
    }
  }

  compressed_dFv_v = CompressedJacobian - CdF_ * seed;

//Recovery Teil - fixed by Matteo

  // std::vector<std::vector<double>> fulldFv(N,std::vector<double>(N,0.0));

    // int rows_cj = compressed_dFv_v.rows();
    // int cols_cj = compressed_dFv_v.cols();

    // double **compressed_dFv = new  double *[rows_cj];
    // for(int i=0;i<rows_cj;i++) compressed_dFv[i] = new double[cols_cj];

    // for(int i = 0; i < rows_cj; i++){
    //   for(int j = 0; j < cols_cj; j++){
    //       compressed_dFv[i][j] = compressed_dFv_v(i,j);
    //   }
    // }

    // int rows_spdF_v = sparsity_pattern_dFv.rows();
    // int cols_spdF_v = sparsity_pattern_dFv.cols();

    // unsigned int **sparsity_pattern_dFv_array = new unsigned int *[rows_spdF_v];
    // for(int i=0;i<rows_spdF_v;i++) sparsity_pattern_dFv_array[i] = new unsigned int[rows_spdF_v];

    // for(int i = 0; i < rows_spdF_v; i++){
    //   for(int j = 0; j < rows_spdF_v; j++){
    //       sparsity_pattern_dFv_array[i][j] = sparsity_pattern_dFv(i,j);
    //   }
    // }

    //   JacobianRecovery1D jr1d;
    //   unsigned int* rowIndex;
    //   unsigned int* colIndex;
    //   double* jacValue;
    //   int nnz = jr1d.RecoverD2Cln_CoordinateFormat(g, compressed_dFv, sparsity_pattern_dFv_array, &rowIndex, &colIndex, &jacValue);
    //   for(int i = 0; i < nnz; i++){
    //       fulldFv[rowIndex[i]][colIndex[i]] = jacValue[i];
    //   }

      // Hier noch fulldFV in eigen übertragen!!

      // for(int i=0; i<N; i++){
      //     for(int j=0; j<N; j++){
      //         std::cout << fulldFv[i][j] << " ";
      //     }
      //     std::cout << std::endl;
      //}
}

template<typename T, typename TP, size_t N, size_t NP>
void Newton_Solver(
  const Eigen::Matrix<T, N, 1>& xv,
  const Eigen::Matrix<TP, NP, 1>& p,
  Eigen::Matrix<T, N, 1>& y_s,
  Eigen::Matrix<double, N, N>& dFc,
  //Eigen::Matrix<double, N, N>& dFv,
  Eigen::Matrix<T, N, 1>& dx
) {

    //Jacobi berechnen
    Eigen::Matrix<T,N,N> J;
    J = dFc; //+ dFv;

    //Sparse Matrix A generieren
    Eigen::SparseMatrix<T> A(N,N);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            if(J(i,j)!=0){
                A.insert(i,j) = J(i,j);
            }
        }
    }

    //Rechte Seite y_s wird hier überschrieben
    F<T,TP,N,NP>(xv,p,y_s);

    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    dx = solver.solve(y_s);

}

int main() {
    using T=double; using TP=float;
    const size_t N=4, NP=0;
    Eigen::Matrix<T,N,1> x,y;
    Eigen::Matrix<TP,NP,1> p;
    float tol = 0.3;

    x << 1,1,3,4;
//Der teil ist irrelevant fürs system und kann mit f und df ausgelagert werden
/*
    std::cout << "f:" << std::endl;
    f<T,TP,N,NP>(x,p,y);
    std::cout << y << std::endl;

    std::cout << "df:" << std::endl;
    Eigen::Matrix<T,N,1> dydx;
    df<T,TP,N,NP>(x,p,y,dydx);
    for (const auto& i:dydx) std::cout << i << std::endl;
*/
    Eigen::Matrix<T,N,1> y_s;
    F<T,TP,N,NP>(x,p,y_s);

    std::cout << "dF:" << std::endl;
    Eigen::Matrix<T,N,N> ddydxx;
    dF<T,TP,N,NP>(x,p,y_s,ddydxx);
    std::cout << ddydxx << std::endl;

    std::cout <<"ddF:" << std::endl;
    Eigen::Matrix<T,N,N> dddydxxx;
    ddF<T,TP,N,NP>(x,p,y_s,ddydxx,dddydxxx);
    std::cout << dddydxxx << std::endl;

    std::cout << "S_dF:" << std::endl;
    Eigen::Matrix<bool,N,N> dsddf;
    S_dF<T,TP,N,NP>(x,p,y_s,dsddf);
    std::cout << dsddf << std::endl;


    std::cout << "S_ddF:" << std::endl;
    Eigen::Matrix<bool,N,N> dsdddf;
    S_ddF<T,TP,N,NP>(x,p,y_s,dsdddf);
    std::cout << dsdddf << std::endl;

    std::cout << "Cddf:" << std::endl;
    Eigen::Matrix<bool,N,N> Cddf;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++)
            Cddf(i,j) = (dsddf(i,j)!= dsdddf(i,j));
        }
    std::cout << Cddf << std::endl;

    std::cout << "dFc:" << std::endl;
    Eigen::Matrix<double,N,N> dFc;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++){
                if(Cddf(i,j)!=0)
                    dFc(i,j) = ddydxx(i,j);
                else
                    dFc(i,j) = 0;
            }
    }
    std::cout << dFc << std::endl;

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompressedJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compressed_dFv_v;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> full_dFv_v;

    Eigen::Matrix<double,N,1> dx,x_curr,x_prev;

    x_curr = x;


    // Anpassen der aufrufe
    while(x_curr.norm() > tol){
       x_prev = x_curr;
       dFv<T,TP,N,NP>(x_prev,p,y_s,sVdF,seed,sparsity_pattern_dFv,CompressedJacobian,dFc,compressed_dFv_v,full_dFv_v);
       Newton_Solver<T,TP,N,NP>(x,p,y_s,ddydxx,dx);
       x_curr = x_prev + dx;

    }


    std::cout << dx << std::endl;


    return 0;
}
