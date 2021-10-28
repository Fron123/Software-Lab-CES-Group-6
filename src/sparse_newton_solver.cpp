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


namespace Newton_Solver {

template<typename T, typename TP, size_t N, size_t NP>
void solve(
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

    A.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    dx = solver.solve(-y_s);
}

} //end namespace
