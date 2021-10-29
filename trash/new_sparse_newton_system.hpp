#pragma once
#include "dco.hpp"
#include "ColPackHeaders.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

namespace System {


template<typename T,typename TP,size_t N,size_t NP>
class newton_system{
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
    void F(
        const Eigen::Matrix<T,N,1>& x,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& y
    );

    void dF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<T,N,N>& dydx
    );

    void ddF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<T,N,N>& dydx_v,
        Eigen::Matrix<T,N,N>& ddydxx
    );

    void S_dF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<bool,N,N> &S_dF
    );

    void S_ddF(
        const Eigen::Matrix<T,N,1>& xv,
        const Eigen::Matrix<TP,NP,1>& p,
        Eigen::Matrix<T,N,1>& yv,
        Eigen::Matrix<bool,N,N> &S_ddF
    );
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
};

}
#include "../src/new_sparse_newton_system.cpp"
