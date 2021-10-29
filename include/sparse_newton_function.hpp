#pragma once
#include "dco.hpp"
#include "ColPackHeaders.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

template<typename T,typename TP,size_t N,size_t NP>
class sparse_newton{
public:
    //Variablen
    const Eigen::Matrix<T,N,1>& x;
    const Eigen::Matrix<TP,NP,1>& p;
    T& y;
    const Eigen::Matrix<T,N,1>& xv;
    Eigen::Matrix<T,N,1>& y_s;
    Eigen::Matrix<T,N,N>& dydx;
    Eigen::Matrix<T,N,N>& ddydxx;
    Eigen::Matrix<T,N,N>& dydx_v;
    Eigen::Matrix<bool,N,N> &S_dF_2;
    Eigen::Matrix<bool,N,N> &S_ddF_2;
    Eigen::Matrix<bool, N, N> sparsity_pattern;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& CompressedJacobian;
    Eigen::Matrix<double, N, N>& CdF_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& compressed_dFv_v;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& full_dFv_v;
    Eigen::Matrix<double, N, 1>& dx;
    Eigen::Matrix<double,N,N> full_dFc;


    //Methoden
 /**
 *The given convex nonlinear objective function (y) with their parameters (p) and the initial vector(x) needs to be defined here 
 */   
    void f(
        const Eigen::Matrix<T,N,1>& x,
        const Eigen::Matrix<TP,NP,1>& p,
        T& y
    );
    
 /**
  * Calculation of the first derivative of the function
  */
    void df(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        T& yv,
        Eigen::Matrix<T,N,1>& dydx
    );
    
 /**
  *The given sparse nonlinear system (y) with their parameters (p) and the initial vector(x) needs to be defined here 
  */
    void F(
        const Eigen::Matrix<T,N,1>& x,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& y
    );
    
 /**
  * Calculation of the first derivative of the system
  */
    void dF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<T,N,N>& dydx
    );
    
 /**
  * Calculation of the second derivative of the system
  */
    void ddF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<T,N,N>& dydx_v,
        Eigen::Matrix<T,N,N>& ddydxx
    );
    
 /**
  * Calculation of the sparsity pattern of the first derivative of the system, constant entries are set to true
  */
    void S_dF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<bool,N,N> &S_dF_2
    );
    
 /**
  * Calculation of the sparsity pattern of the second derivative of the system, constant entries are set to true
  */
    void S_ddF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<bool,N,N> &S_ddF_2
    );
    
 /**
  * Calculation of the sparsity pattern of the first derivative of the system, variable entries are set to true
  */
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
    );
    
 /**
  * Calculate the solution of the sparse System using Newton's method and Jacobion compression
  */
    void Newton_Solver(
      const Eigen::Matrix<T, N, 1>& xv,
      const Eigen::Matrix<TP, NP, 1>& p,
      Eigen::Matrix<T, N, 1>& y_s,
      Eigen::Matrix<double,N,N>& full_dFc,
      Eigen::Matrix<double, N,N>& full_dFv,
      Eigen::Matrix<double, N, 1>& dx
    );

};


#include "../src/sparse_newton_function.cpp"
