//#include "dco.hpp"
//#include "ColPackHeaders.h"
#include <iostream>
#include <array>
#include <cmath>
//#include <typeinfo>
//#include <string>
//#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
//#include <Eigen/Sparse>
//#include <Eigen/SparseLU>
//#include <Eigen/OrderingMethods>
#include <new_sparse_newton_system.hpp>
#include <new_sparse_newton_function.hpp>
#include <sparse_newton_solver.hpp>
#include <chrono>

//system
template<typename T, typename TP, size_t N, size_t NP>
void solve_system(
    Eigen::Matrix<T,N,1>& x,
    Eigen::Matrix<TP,NP,1> p,
    Eigen::Matrix<T,N,1>& x_stationary,
    float& tol
){

    using namespace System;
    using namespace Newton_Solver;

    //////////////////////////////////////////////////////////////
    // if you want to print out the different derivatives       //
    // sparsity patterns or constant/variable submatrizes       //
    // delete the // infront of the related std::cout           //  
    //////////////////////////////////////////////////////////////

    Eigen::Matrix<T,N,1> y_s;
    F<T,TP,N,NP>(x,p,y_s);
    //std::cout << "F:" << std::endl << y_s << std::endl;

    Eigen::Matrix<T,N,N> ddydxx;
    dF<T,TP,N,NP>(x,p,y_s,ddydxx);
    //std::cout << "dF:" << std::endl << ddydxx << std::endl;

    Eigen::Matrix<T,N,N> dddydxxx;
    ddF<T,TP,N,NP>(x,p,y_s,ddydxx,dddydxxx);
    //std::cout <<"ddF:" << std::endl << dddydxxx << std::endl;

    Eigen::Matrix<bool,N,N> dsddf;
    S_dF<T,TP,N,NP>(x,p,y_s,dsddf);
    //std::cout << "S_dF:" << std::endl << dsddf << std::endl;

    Eigen::Matrix<bool,N,N> dsdddf;
    S_ddF<T,TP,N,NP>(dsdddf, dddydxxx);
    //std::cout << "S_ddF:" << std::endl << dsdddf << std::endl;

    Eigen::Matrix<bool,N,N> Cddf;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++)
            Cddf(i,j) = (dsddf(i,j)!= dsdddf(i,j));
        }
    //std::cout << "Cddf:" << std::endl << Cddf << std::endl;

    Eigen::Matrix<double,N,N> dFc;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++){
                if(Cddf(i,j)!=0)
                    dFc(i,j) = ddydxx(i,j);
                else
                    dFc(i,j) = 0;
            }
    }
   //std::cout << "dFc:" << std::endl << dFc << std::endl;

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompressedJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compressed_dFv_v;
    Eigen::Matrix<double, N,N> full_dFv_v;

    Eigen::Matrix<double,N,1> dx,x_curr;

    x_curr = x;

    int i =0;
    F<T,TP,N,NP>(x,p,y_s);

    // Anpassen der aufrufe

    while(y_s.norm() > tol){
       dFv<T,TP,N,NP>(x_curr,p,y_s,sVdF,seed,sparsity_pattern_dFv,CompressedJacobian,dFc,compressed_dFv_v,full_dFv_v);
       solve<T,TP,N,NP>(x_curr,p,y_s,dFc,full_dFv_v,dx);
       x_curr = x_curr + dx;

       F<T,TP,N,NP>(x_curr,p,y_s);

       i++;
    }
/*
    std::cout << "Iterations:" << std::endl << i << std::endl;
    std::cout << "x_statonary:" << std::endl << x_curr << std::endl;
*/
    x_stationary = x_curr;
}

//objective function
template<typename T, typename TP, size_t N, size_t NP>
void solve_objective(
    Eigen::Matrix<T,N,1>& x,
    Eigen::Matrix<TP,NP,1> p,
    Eigen::Matrix<T,N,1>& x_stationary,
    float& tol
){
    using namespace Function;
    using namespace Newton_Solver;


    //////////////////////////////////////////////////////////////
    // if you want to print out the different derivatives       //
    // sparsity patterns or constant/variable submatrizes       //
    // delete the // infront of the related std::cout           //  
    //////////////////////////////////////////////////////////////

    T y;
    f<T,TP,N,NP>(x,p,y);
    //std::cout << "f:" << std::endl << y << std::endl;

    Eigen::Matrix<T,N,1> dydx;
    df<T,TP,N,NP>(x,p,y,dydx);
    //std::cout << "df:" << std::endl << dydx << std::endl;

    Eigen::Matrix<T,N,1> y_f;
    F<T,TP,N,NP>(x,p,y_f);
    //std::cout << "F:" << std::endl << y_s << std::endl;

    Eigen::Matrix<T,N,N> ddydxx;
    dF<T,TP,N,NP>(x,p,y_f,ddydxx);
    //std::cout << "dF:" << std::endl << ddydxx << std::endl;

    Eigen::Matrix<T,N,N> dddydxxx;
    ddF<T,TP,N,NP>(x,p,y_f,ddydxx,dddydxxx);
    //std::cout <<"ddF:" << std::endl << dddydxxx << std::endl;

    Eigen::Matrix<bool,N,N> dsddf;
    S_dF<T,TP,N,NP>(x,p,y_f,dsddf);
    //std::cout << "S_dF:" << std::endl << dsddf << std::endl;

    Eigen::Matrix<bool,N,N> dsdddf;
    S_ddF<T,TP,N,NP>(dsdddf, dddydxxx);
    //std::cout << "S_ddF:" << std::endl << dsdddf << std::endl;

    Eigen::Matrix<bool,N,N> Cddf;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++)
            Cddf(i,j) = (dsddf(i,j)!= dsdddf(i,j));
        }
    //std::cout << "Cddf:" << std::endl << Cddf << std::endl;

    Eigen::Matrix<double,N,N> dFc;
    for (size_t i=0;i<N;i++) {
            for(size_t j=0;j<N;j++){
                if(Cddf(i,j)!=0)
                    dFc(i,j) = ddydxx(i,j);
                else
                    dFc(i,j) = 0;
            }
    }
   //std::cout << "dFc:" << std::endl << dFc << std::endl;

    Eigen::Matrix<bool, N,N> sVdF;
    sVdF = dsddf - Cddf;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> seed;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sparsity_pattern_dFv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CompressedJacobian;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> compressed_dFv_v;
    Eigen::Matrix<double, N,N> full_dFv_v;

    Eigen::Matrix<double,N,1> dx,x_curr;

    x_curr = x;

    int i =0;
    F<T,TP,N,NP>(x,p,y_f);

    // Anpassen der aufrufe

    while(y_f.norm() > tol){
       dFv<T,TP,N,NP>(x_curr,p,y_f,sVdF,seed,sparsity_pattern_dFv,CompressedJacobian,dFc,compressed_dFv_v,full_dFv_v);
       solve<T,TP,N,NP>(x_curr,p,y_f,dFc,full_dFv_v,dx);
       x_curr = x_curr + dx;

       F<T,TP,N,NP>(x_curr,p,y_f);

       i++;
    }

   /* std::cout << "Iterations:" << std::endl << i << std::endl;
    std::cout << "x_statonary:" << std::endl << x_curr << std::endl;*/
    x_stationary = x_curr;
}

int main() {

    //Laufzeitanalyse
    auto start = std::chrono::steady_clock::now();
    std::srand(std::time(nullptr));

    using T=double; using TP=float;

    /************************************************************//**
    *  define the dimensions of your Problem here.
    *  N being the dimensions of x
    *  NP being the dimensions of p
    *
    *  select a tolerance tol for newtons method
    *
    *  fill initial values of x and values of p
    ***************************************************************/

    const size_t N=5, NP=4;

    Eigen::Matrix<T,N,1> x;
    Eigen::Matrix<TP,NP,1> p;

    float tol = 1e-6;

    for(size_t i=0;i<N;i++) x(i) = -3;

    p(0) = 3;
    p(1) = 0.5;
    p(2) = 2;
    p(3) = 1;

    char decider;


    std::cout << "Please select the Problem you want to solve:" << std::endl;
    std::cout << "To solve a convex unconstraigned minimization problem, type 'm' " << std::endl;
    std::cout << "To solve a system of nonlinear equations, type 's' " <<std::endl;
    std::cout << "To solve both, please type 'b' " <<std::endl;
    std::cin >> decider;

    if(decider == 'b') {

        Eigen::Matrix<T,N,1> x_stationary_obj;
        Eigen::Matrix<T,N,1> x_stationary_sys;

        solve_objective<T,TP,N,NP>(x,p,x_stationary_obj,tol);
        solve_system<T,TP,N,NP>(x,p,x_stationary_sys,tol);

        std::cout << "The solution x of the system of nonlinear equations is: " << std::endl << x_stationary_sys << std::endl << " " << std::endl;

        Eigen::Matrix<T,N,1> y_stationary;
        Function::F<T,TP,N,NP>(x_stationary_obj,p,y_stationary);

        Eigen::Matrix<T,N,N> ddydxx;
        Function::dF<T,TP,N,NP>(x_stationary_obj,p,y_stationary,ddydxx);

        std::cout << "The stationary x for the minimization is: " << std::endl << x_stationary_obj << std::endl;

        T output = x_stationary_obj.transpose() * ddydxx * x_stationary_obj;
        if(output > 0 )
            std::cout << "X erfüllt Minimierung" << std::endl;
        else std::cout << "X erfüllt Minimierung nicht" << std::endl;
    }
    else if(decider == 'm') {
        Eigen::Matrix<T,N,1> x_stationary_obj;
        solve_objective<T,TP,N,NP>(x,p,x_stationary_obj,tol);
        Eigen::Matrix<T,N,1> y_stationary;
        Function::F<T,TP,N,NP>(x_stationary_obj,p,y_stationary);

        Eigen::Matrix<T,N,N> ddydxx;
        Function::dF<T,TP,N,NP>(x_stationary_obj,p,y_stationary,ddydxx);

        std::cout << "The stationary x for the minimization is: " << std::endl << x_stationary_obj << std::endl;

        T output = x_stationary_obj.transpose() * ddydxx * x_stationary_obj;
        if(output > 0 )
            std::cout << "X erfüllt Minimierung" << std::endl;
        else std::cout << "X erfüllt Minimierung nicht" << std::endl;
    }
    else if(decider == 's') {
        Eigen::Matrix<T,N,1> x_stationary_sys;
        solve_system<T,TP,N,NP>(x,p,x_stationary_sys,tol);
        std::cout << "The solution x of the system of nonlinear equations is: " << std::endl << x_stationary_sys << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Elapsed time in milliseconds: " << elapsed.count()*1e-6;



    return 0;
}
