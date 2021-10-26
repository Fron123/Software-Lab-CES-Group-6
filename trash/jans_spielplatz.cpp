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

//residual
template<typename T, typename TP, size_t N, size_t NP>
void F(
    const std::array<T,N>& x,
    const std::array<TP,NP>& p,
    std::array<T,N>& y
) {
    //residual
    for(size_t i = 0; i < N; i++)
        y[i] = -(x[i]*x[i])+i+1;
}

//jacobian
template<typename T, typename TP, size_t N, size_t NP>
void dF(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& yv,
    std::array<std::array<T,N>,N>& dydx
){
    typedef typename dco::gt1v<T,N>::type DCO_T;
    std::array<DCO_T,N> x,y;
    for(size_t i=0; i<N; i++){
        y[i] = yv[i];
        x[i] = xv[i];
    }

    for(size_t i=0; i<N; i++){
        dco::derivative(x[i]) = 1;
    }
    F(x,p,y);
    for(size_t i=0; i<N; i++){
        for(size_t j=0; j<N; j++) dydx[j][i]=dco::derivative(y[j]);
        dco::derivative(x[i]) = 0;
    }
}

//second derivative F''
template<typename T, typename TP, size_t N, size_t NP>
void ddF(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& yv,
    std::array<std::array<T,N>,N>& dydx_v,
    std::array<std::array<T,N>,N>& d2ydx2
){
    typedef typename dco::gt1s<T>::type DCO_T;
    std::array<DCO_T,N> x,y;
    std::array<std::array<DCO_T,N>,N> dydx;

    for(size_t i=0; i<N; i++){
        x[i] = xv[i];
        dco::derivative(x[i]) = 1;
        for(size_t j=0; j<N; j++){
            d2ydx2[i][j] = 0;
        } 
    }
    dF(x,p,y,dydx);

    for(size_t i=0; i<N; i++) {
        for(size_t j=0; j<N; j++) {
            //dydx_v[i][j] = dco::derivative(dydx[i][j]);
            d2ydx2[j][i] += dco::derivative(dydx[j][i]);
        }
        dco::derivative(x[i])=0;
        //yv[i]=dco::value(y[i]);
    }
}

//sparsity jacobian
template<typename T, typename TP, size_t N, size_t NP>
void SdF(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& yv,
    std::array<std::array<bool,N>,N>& S_dF
){
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,y;
    //std::array<std::array<DCO_T,N>,N> dydx;

    for(size_t i=0; i<N; i++){
        x[i] = xv[i];
        dco::p1f::set(x[i],true,0);
    }
    F(x,p,y);
    for(size_t i=0; i<N; i++) {
        dco::p1f::get(y[i],yv[i]);
        for(size_t j=0; j<N; j++) dco::p1f::get(y[i],S_dF[i][j],0);
    }
}

//sparsity F''
template<typename T, typename TP, size_t N, size_t NP>
void SddF(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& yv,
    std::array<std::array<bool,N>,N>& S_ddF
){
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,y;
    std::array<std::array<DCO_T,N>,N> dydx;

    for(size_t i=0; i<N; i++){
        x[i] = xv[i];
        dco::p1f::set(x[i],true,0);
    }
    dF(x,p,y,dydx);
    for(size_t i=0; i<N; i++) {
        dco::p1f::get(y[i],yv[i]);
        for(size_t j=0; j<N; j++) dco::p1f::get(y[i][j],S_ddF[i][j],0);
    }
}
template<typename T, typename TP, size_t N, size_t NP>
void dFv(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& y_s,
    std::array<std::array<bool,N>,N>& sparsity_pattern,
    std::array<std::array<T,N>,N>& CdF_,
    Eigen::Matrix<T,N,N>& full_dFv_v
){
    std::array<int,N> x_;
    std::fill_n(x_.begin(),N,0);

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(sparsity_pattern[i][j] == 1) x_[i] += 1;
        }
    }

    for(int i = 0;i < N; ++i) {
       if(x_[0] < x_[i])
           x_[0] = x_[i];
    }
    int max = x_[0] + 1;

    std::array<std::array<T,N>,max> sparsity_pattern_dFv;

    for(int i=0; i<N; i++){
        sparsity_pattern_dFv[i][0] = x_[i];
        int counter = 1;
        for(int j=0; j<N; j++) {
            if(sparsity_pattern[i][j] == 1){
                sparsity_pattern_dFv[i][counter] = j;
                counter++;
            }
        }
    }

    unsigned int **uip2_JacobianSparsityPattern = new unsigned int *[N];
    for(int i=0;i<N;i++) uip2_JacobianSparsityPattern[i] = new unsigned int[N];

    for(int i = 0; i < N; i++){
        for(int j = 0; j < max; j++){
            uip2_JacobianSparsityPattern[i][j] = sparsity_pattern_dFv[i][j];
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

    std::array<std::array<T,rows>,cols> seed;

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            seed[i][j] = Seed[i][j];
        }
    }

    using DCO_T=typename dco::gt1s<T>::type;

    std::array<DCO_T,N>x,y,yv;

    for(size_t i = 0;i<N;i++){
        x[i] = xv[i];
        yv[i] = y_s[i];
    }
    F(x,p,yv);
    double** compressedJacobian = new double*[N];
    for (size_t i = 0; i < N; i++)
        compressedJacobian[i] = new double[cols];

    for(size_t i = 0; i < cols; i++) {
        for(size_t j = 0; j < rows; j++){
            dco::derivative(x[j]) = seed[j][i]; //der Clou dahinter
            compressedJacobian[j][i] = dco::derivative(yv[j]);
        }
    }
    //F(x,p,yv);
    //for(size_t i=0; i<cols; i++){    
    //    for(size_t j = 0; j < rows; j++) {
    //        compressedJacobian[j][i] = dco::derivative(yv[j]);
    //    }
    //}

    std::array<std::array<T,rows>,cols> CompressedJacobian;

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            CompressedJacobian[i][j] = compressedJacobian[i][j];
        }
    }

    std::array<std::array<T,N>,cols> CompressedCdF_ ;
    
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < cols; ++j)
        {
            CompressedCdF_[i][j]=0;
        }

    
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < cols; ++j)
            for(int k = 0; k < N; ++k)
            {
                CompressedCdF_[i][j] += CdF_[i][k] * seed[k][j];
            }

    std::array<std::array<T,N>,cols> compressed_dFv_v;

    for(int i=0; i<N; i++) {
        for(int j=0; j<cols; j++) compressed_dFv_v[i][j] = CompressedJacobian[i][j] - CompressedCdF_[i][j];
    }

    std::vector<std::vector<double>> fulldFv(N,std::vector<double>(N,0.0));

    int cols_cj = cols;

    double **compressed_dFv = new  double *[N];
    for(int i=0;i<N;i++) compressed_dFv[i] = new double[cols_cj];

    for(int i = 0; i < N; i++){
      for(int j = 0; j < cols_cj; j++){
          compressed_dFv[i][j] = compressed_dFv_v(i,j);
      }
    }

    int cols_spdF_v = cols;

    unsigned int **sparsity_pattern_dFv_array = new unsigned int *[N];
    for(int i=0;i<N;i++) sparsity_pattern_dFv_array[i] = new unsigned int[N];

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
    // Hier noch fulldFV in eigen übertragen!!

       for(int i=0; i<N; i++){
           for(int j=0; j<N; j++){
               full_dFv_v(i,j) = fulldFv[i][j];
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

    //Jacobi berechnen
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

    //Rechte Seite y_s wird hier überschrieben
    F<T,TP,N,NP>(xv,p,y_s);

    A.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    dx = solver.solve(-y_s);

}
template<typename T, typename TP, size_t N, size_t NP>
void norm(
    std::array<T,N>& y,
    T& norm
){  
    T accum = 0.;
    for (int i = 0; i < N; ++i) {
        accum += y[i] * y[i];
    }
    norm = std::sqrt(accum);
}

int main() {
    using T=double; using TP=float;
    const size_t N=2, NP=0;
    std::array<T,N> x={1,1};
    std::array<TP,NP> p;
    
    std::array<T,N> y;
    F(x,p,y);
    std::cout << "F: " << std::endl;
    for(const auto& i:y) std::cout << i << std::endl;

    std::array<std::array<T,N>,N> dF, ddF;
    dF(x,p,y,dF);
    std::cout << "dF: " << std::endl;
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << dF[i][j];
        if(j == N-1)
            std::cout << endl;
    }

    ddF(x,p,y,dF,ddF);
    std::cout << "ddF: " << std::endl;
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << ddF[i][j];
        if(j == N-1)
            std::cout << endl;
    }

    std::array<std::array<bool,N>,N> S_dF, S_ddF;
    SdF(x,p,y,S_dF);
    std::cout << "S_dF: " << std::endl;
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << S_dF[i][j];
        if(j == N-1)
            std::cout << endl;
    }
    SddF(x,p,y,S_ddF);
    std::cout << "S_ddF: " << std::endl;
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << S_ddF[i][j];
        if(j == N-1)
            std::cout << endl;
    }
    std::cout << "Cddf:" << std::endl;
    std::array<std::array<bool,N>,N> Cddf;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++)
            Cddf[i][j] = (S_dF[i][j]!= S_ddF[i][j]);
        }
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << Cddf[i][j];
        if(j == N-1)
            std::cout << endl;
    }

    std::cout << "dFc:" << std::endl;
    std::array<std::array<T,N>,N> dFc;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++){
                if(Cddf[i][j]!=0)
                    dFc[i][j] = dF[i][j];
                else
                    dFc[i][j] = 0;
            }
    }
    for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < N; ++j)
    {
        std::cout << " " << dFc[i][j];
        if(j == N-1)
            std::cout << endl;
    }

    std::array<std::array<bool,N>,N> sVdF;
    
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++) sVdF[i][j] = S_dF[i][j] - Cddf[i][j];
    }

    Eigen::Matrix<T,N,N>& full_dFv_v;
    Eigen::Matrix<T,N,1> dx_eigen;
    Eigen::Matrix<T,N,N>& dFc_eigen;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++) dFc_eigen(i,j) = dFc[i][j];
    }

    std::array<T,N> dx;
    std::array<T,N> x_curr;
    for (size_t i=0;i<N;i++) x_curr[i] = x[i]; 
    
    int i =0;

    std::array<T,N> y_s;
    // Anpassen der aufrufe
    while(norm(y_s) > tol){
       dFv(x_curr,p,y_s,sVdF,dFc,full_dFv_v);
       Newton_Solver(x_curr,p,y_s,dFc,full_dFv_v,dx);
       x_curr = x_curr + dx;
       for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++) dx= dx_eigen(i,j);
    }
       F(x_curr,p,y_s);
    }


    //std::cout << "x_curr:" << std::endl << x_curr << std::endl;

    return 0;
}
