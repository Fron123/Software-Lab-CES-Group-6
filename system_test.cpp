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
#include <sparse_newton_system.hpp>
#include <chrono>


int main() {

    //Laufzeitanalyse
    auto start = std::chrono::steady_clock::now();
    std::srand(std::time(nullptr));

    using T=double; using TP=float;
    const size_t N=5, NP=4;
    Eigen::Matrix<T,N,1> x,y;
    Eigen::Matrix<TP,NP,1> p;
    float tol = 1e-6;

    //x << 1,1,1,1,1,1,1,1,1,1;
    for(size_t i=0;i<N;i++) x(i) = -3;

    p(0) = 3;
    p(1) = 0.5;
    p(2) = 2;
    p(3) = 1;

    std::cout << "Startvektor:" << std::endl << x << std::endl;
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
    std::cout << "F:" << std::endl;
    std::cout << y_s << std::endl;


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

   // std::cout << "Cddf:" << std::endl;
    Eigen::Matrix<bool,N,N> Cddf;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++)
            Cddf(i,j) = (dsddf(i,j)!= dsdddf(i,j));
        }
 //   std::cout << Cddf << std::endl;

 //   std::cout << "dFc:" << std::endl;
    Eigen::Matrix<double,N,N> dFc;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++){
                if(Cddf(i,j)!=0)
                    dFc(i,j) = ddydxx(i,j);
                else
                    dFc(i,j) = 0;
            }
    }
  //  std::cout << dFc << std::endl;

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompressedJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compressed_dFv_v;
    Eigen::Matrix<double, N,N> full_dFv_v;

    Eigen::Matrix<double,N,1> dx,x_curr,x_prev;

    x_curr = x;

    int i =0;
    F<T,TP,N,NP>(x,p,y_s);
//    std::cout << y_s << std::endl;

  //  std::cout << y_s.norm() << std::endl;

    // Anpassen der aufrufe
    while(y_s.norm() > tol){
       dFv<T,TP,N,NP>(x_curr,p,y_s,sVdF,seed,sparsity_pattern_dFv,CompressedJacobian,dFc,compressed_dFv_v,full_dFv_v);
       Newton_Solver<T,TP,N,NP>(x_curr,p,y_s,dFc,full_dFv_v,dx);
       x_curr = x_curr + dx;
       F<T,TP,N,NP>(x_curr,p,y_s);
       i++;
    }

    std::cout << "Iterationen:" << std::endl << i << std::endl;

    std::cout << "x_stationary:" << std::endl << x_curr << std::endl;

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Elapsed time in milliseconds: " << elapsed.count()*1e-6;



    return 0;
}
