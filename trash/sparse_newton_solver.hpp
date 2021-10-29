#pragma once
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

namespace Newton_Solver{

template<typename T, typename TP, size_t N, size_t NP>
class sparse_newton_solver {
public:
    //Variablen
    const Eigen::Matrix<T, N, 1>& xv;
    const Eigen::Matrix<TP, NP, 1>& p;
    Eigen::Matrix<T, N, 1>& y_s;
    Eigen::Matrix<double,N,N>& full_dFc;
    Eigen::Matrix<double, N,N>& full_dFv;
    Eigen::Matrix<double, N, 1>& dx;

    //Methoden
    void solve(
      const Eigen::Matrix<T, N, 1>& xv,
      const Eigen::Matrix<TP, NP, 1>& p,
      Eigen::Matrix<T, N, 1>& y_s,
      Eigen::Matrix<double,N,N>& full_dFc,
      Eigen::Matrix<double, N,N>& full_dFv,
      Eigen::Matrix<double, N, 1>& dx
    );

};
}

#include "../src/sparse_newton_solver.cpp"
