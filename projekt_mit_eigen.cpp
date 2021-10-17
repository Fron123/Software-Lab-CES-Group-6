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

using namespace ColPack;

//objective function
template<typename T, typename TP, size_t N, size_t NP>
void f(
    const Eigen::Matrix<T,N,1>& x,
    const Eigen::Matrix<TP,NP,1>& p,
    T& y
){
    using namespace std;
    y=p[0]*x[0]*x[0] + exp(x[1]);
}

template<typename T, typename TP, size_t N, size_t NP>
void F(
    const Eigen::Matrix<T,N,1>& x,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& y
){
    y << x(0)*x(0),x(1)*x(1),x(2)*x(2),x(3);
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

//second derivative(Jacobian/Hessian)
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

//sparsity f'
template<typename T, typename TP, size_t N, size_t NP>
void Sdf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<bool,N,1> &sdf
){
    using DCO_T=dco::p1f::type;
    Eigen::Matrix<DCO_T,N,1> x;
    DCO_T y;
    for (size_t i=0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,i);
    }
    f<DCO_T,TP,N,NP>(x,p,y);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++) dco::p1f::get(y,sdf(i),i);
}

//sparsity f''
template<typename T, typename TP, size_t N, size_t NP>
void dSdF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<bool,N,N> &dSddf
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
    for (size_t j=0;j<N;j++) dco::p1f::get(y(i),dSddf(i,j),j);
    }
}

//sparsity f'''
template<typename T, typename TP, size_t N, size_t NP>
void dSdddf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<bool,N,N> &dSdddf,
    std::array<std::array<std::array<T,N>,N>,N>& dddydxxx
){

    T sum ;
        for (size_t i=0;i<N;i++){
            for(size_t j=0;j<N;j++){
                sum = 0;
                for(size_t k=0;k<N;k++){
                    sum += dddydxxx[i][j][k];

                }
                if (sum == 0) {
                    dSdddf(i,j)=0;
                }
                else {
                    dSdddf(i,j)=1;
                    }
            }
        }
    }

template<typename T, typename TP, size_t N, size_t NP>
void dSddF(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    Eigen::Matrix<T,N,1>& yv,
    Eigen::Matrix<bool,N,N> &dSddf
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
    for (size_t j=0;j<N;j++) dco::p1f::get(dydx(i,j),dSddf(i,j),j);
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


int main() {
    using T=double; using TP=float;
    const size_t N=4, NP=0;
    Eigen::Matrix<T,N,1> x,y;
    Eigen::Matrix<TP,NP,1> p;

    x << 1,1,3,4;

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

    std::cout << "ddf:" << std::endl;
    Eigen::Matrix<T,N,N> ddydxx;
    dF<T,TP,N,NP>(x,p,y_s,ddydxx);
    std::cout << ddydxx << std::endl;

    // TENSOR ODER SO, kp bin kein Iformatiker
    std::cout <<"dddf:" << std::endl;
    Eigen::Matrix<T,N,N> dddydxxx;
    ddF<T,TP,N,NP>(x,p,y_s,ddydxx,dddydxxx);
    std::cout << dddydxxx << std::endl;
/*
    for (const auto& i:dddydxxx)
    for (const auto& j:i)
    for (const auto& k:j)
    std::cout << k << std::endl;


    std::cout << "Sdf:" << std::endl;
    Eigen::Matrix<bool,N,1> sdf;
    Sdf<T,TP,N,NP>(x,p,y,sdf);
    std::cout << sdf << std::endl;
*/
    std::cout << "dSddf:" << std::endl;
    Eigen::Matrix<bool,N,N> dsddf;
    dSdF<T,TP,N,NP>(x,p,y_s,dsddf);
    std::cout << dsddf << std::endl;


    std::cout << "dSdddf:" << std::endl;
    Eigen::Matrix<bool,N,N> dsdddf;
    dSddF<T,TP,N,NP>(x,p,y_s,dsdddf);
    std::cout << dsdddf << std::endl;
/*
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

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompressedJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compressed_dFv_v;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> full_dFv_v;




   dFv<T,TP,N,NP>(x,p,y_s,sVdF,seed,sparsity_pattern_dFv,CompressedJacobian,dFc,compressed_dFv_v,full_dFv_v);

   cout << "SeedMatrix: " << endl;
   cout << seed << endl;

   std::cout << "Compressed Jacobian" << std::endl;
   std::cout << CompressedJacobian << std::endl;

   std::cout << "compressed_dFv_v" << std::endl;
   std::cout << compressed_dFv_v << std::endl;



/*

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    cout << "sVdF: " << endl << sVdF << endl;


    Eigen::Matrix<int, N,1> x_;
    x_.setZero();

    int counter = 0;
    int max = 0;

    for(size_t i = 0; i<N; i++){
      for(size_t j = 0; j<N;j++){
        if(sVdF(i,j) == 1){
          x_(i) += 1;
          counter++;
        }
      }
      if(counter > max){
        max = counter + 1;
      }
      counter = 0;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> test;

    test = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(N, max);

    for(size_t i = 0; i<N; i++){
      test(i,0) = x_(i);
      int counter = 1;
      for(size_t j = 0; j<N ;j++){

        if(sVdF(i,j) == 1){
          test(i,counter) = j;
          counter++;
        }
      }
    }

    cout << "Adol-C: " << endl << test << endl;


    unsigned int **uip2_JacobianSparsityPattern = new unsigned int *[N];
      for(size_t i=0;i<N;i++) uip2_JacobianSparsityPattern[i] = new unsigned int[N];

    for(size_t i = 0; i < N; i++){
      for(int j = 0; j < max; j++){
          uip2_JacobianSparsityPattern[i][j] = test(i,j);
      }
    }

    double*** dp3_Seed = new double**;
    int *ip1_SeedRowCount = new int;
    int *ip1_SeedColumnCount = new int;

    BipartiteGraphPartialColoringInterface * g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC,uip2_JacobianSparsityPattern, N ,N);

          g->PartialDistanceTwoColoring( "SMALLEST_LAST", "COLUMN_PARTIAL_DISTANCE_TWO");

    (*dp3_Seed) = g->GetSeedMatrix(ip1_SeedRowCount, ip1_SeedColumnCount);


    int rows = g->GetColumnVertexCount();
          int cols = g->GetRightVertexColorCount();
    double **Seed = *dp3_Seed;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;

    seed = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);

    for(int i=0; i<rows; i++){
      for(int j=0; j<cols; j++){
        seed(i,j) = Seed[i][j];
      }
    }

    cout << "SeedMatrix: " << endl;
    cout << seed << endl;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Compress;
    Compress = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);

    Compress = ddydxx*seed;
    std::cout << "Compressed Jacobian" << std::endl;
    std::cout << Compress << std::endl;
/*
    Eigen::Matrix<T,N,N> DeCompress;

    DeCompress = Compress*dsddf;
    std::cout << "DeCompressed Jacobian" << std::endl;
    std::cout << DeCompress << std::endl;
*/

/*
    typedef typename dco::gt1s<T>::type DCO_T;

    // compute compressed Jacobian \in R^{n x 2}
    double** compressedJacobian = new double*[N];
    for (int i = 0; i < N; i++)
        compressedJacobian[i] = new double[cols];

    for(int i = 0; i < cols; i++) {
        for(int j = 0; j < rows; j++){
            seed(j,i) = dco::derivative(x(j));
        }
        f<DCO_T,TP,N,NP>(x,p,y);
        for(size_t j = 0; j < rows; j++) {
            compressedJacobian[j][i] = dco::derivative(dydx(j));
            std::cout << "Compressed Jacobian" << std::endl;
            std::cout << compressedJacobian[j][i] << std::endl;
        }
    }

*/

    return 0;
}
