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

//System F
template<typename T, typename TP, size_t N, size_t NP>
void F(
    const Eigen::Matrix<T,N,1>& x,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& y
){
    T sum = 0;
     for(size_t i = 0; i < N; i++){
         for(size_t j = 0; j < N; j++){
             sum += p(0)*x(j);
         }
     y(i) = x(i)*x(i)+sum-p(0)*x(i);
}
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
  Eigen::Matrix<double, N,N>& full_dFv_v
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

if(max > N){
    using DCO_T=typename dco::gt1s<T>::type;

      Eigen::Matrix<DCO_T, N, 1> x;
      Eigen::Matrix<DCO_T, N, 1> yv;
      Eigen::Matrix<DCO_T, N, 1> y;

      Eigen::MatrixXd seed2 = Eigen::MatrixXd::Identity(N,N);

      for(size_t i = 0;i<N;i++){
          x(i) = xv(i);
          yv(i) = y_s(i);
      }

      double** compressedJacobian = new double*[N];
        for (size_t i = 0; i < N; i++)
            compressedJacobian[i] = new double[N];

        for(size_t i = 0; i < N; i++) {
            for(size_t j = 0; j < N; j++){
                dco::derivative(x(j)) = seed2(j,i); //der Clou dahinter
            }
            F<DCO_T,TP,N,NP>(x,p,yv);
            for(size_t j = 0; j < N; j++) {
                compressedJacobian[j][i] = dco::derivative(yv(j));
            }
        }

      CompressedJacobian = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N, N);

        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                CompressedJacobian(i,j) = compressedJacobian[i][j];
            }
        }

      full_dFv_v = CompressedJacobian - CdF_ * seed2;

}
else{
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
for(int i=0;i<N;i++) uip2_JacobianSparsityPattern[i] = new unsigned int[max];
//std::cout<< max <<std::endl;

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
        F<DCO_T,TP,N,NP>(x,p,yv);
        for(size_t j = 0; j < rows_seed; j++) {
            compressedJacobian[j][i] = dco::derivative(yv(j));
        }
    }


  CompressedJacobian = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows_seed, cols_seed);

    for(int i=0; i<rows_seed; i++){
        for(int j=0; j<cols_seed; j++){
            CompressedJacobian(i,j) = compressedJacobian[i][j];
        }
    }

  compressed_dFv_v = CompressedJacobian - CdF_ * seed;

   std::vector<std::vector<double>> fulldFv(N,std::vector<double>(N,0.0));

    int rows_cj = compressed_dFv_v.rows();
    int cols_cj = compressed_dFv_v.cols();

    double **compressed_dFv = new  double *[rows_cj];
    for(int i=0;i<rows_cj;i++) compressed_dFv[i] = new double[cols_cj];

    for(int i = 0; i < rows_cj; i++){
      for(int j = 0; j < cols_cj; j++){
          compressed_dFv[i][j] = compressed_dFv_v(i,j);
      }
    }

    int rows_spdF_v = sparsity_pattern_dFv.rows();
    int cols_spdF_v = sparsity_pattern_dFv.cols();

    unsigned int **sparsity_pattern_dFv_array = new unsigned int *[rows_spdF_v];
    for(int i=0;i<rows_spdF_v;i++) sparsity_pattern_dFv_array[i] = new unsigned int[rows_spdF_v];

    for(int i = 0; i < rows_spdF_v; i++){
        for(int j = 0; j < cols_spdF_v; j++){
          sparsity_pattern_dFv_array[i][j] = sparsity_pattern_dFv(i,j);
        }
    }


      JacobianRecovery1D jr1d;
      unsigned int* rowIndex;
      unsigned int* colIndex;
      double* jacValue;
      int nnz = jr1d.RecoverD2Cln_CoordinateFormat(g, compressed_dFv, sparsity_pattern_dFv_array, &rowIndex, &colIndex, &jacValue);
      for(int i = 0; i < nnz; i++){
          fulldFv[rowIndex[i]][colIndex[i]] = jacValue[i];
      }
  
  //fulldFV is conveyed into eigen
       for(int i=0; i<N; i++){
           for(int j=0; j<N; j++){
               full_dFv_v(i,j) = fulldFv[i][j];
           }

       }
}
}

template<typename T, typename TP, size_t N, size_t NP>
void Newton_Solver(
  const Eigen::Matrix<T, N, 1>& xv,
  const Eigen::Matrix<TP, NP, 1>& p,
  Eigen::Matrix<T, N, 1>& y_s,
  Eigen::Matrix<double,N,N>& full_dFc,
  Eigen::Matrix<double, N,N>& full_dFv,
  Eigen::Matrix<double, N, 1>& dx
) {

    //calculate Jacobian
    Eigen::Matrix<T,N,N> J;
    J = full_dFc + full_dFv;
    //Sparse Matrix A generieren
    Eigen::SparseMatrix<T> A(N,N);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            if(J(i,j)!=0){
                A.insert(i,j) = J(i,j);
            }
        }
    }

    //right-hand side y_s is overwritten here
    F<T,TP,N,NP>(xv,p,y_s);

    A.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    dx = solver.solve(-y_s);

}
