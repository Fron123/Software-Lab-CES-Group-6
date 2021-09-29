#include "dco.hpp"
#include <iostream>
#include <array> 
#include <cmath>
#include <typeinfo>
#include <string>

template<typename T, typename TP, size_t N, size_t NP> 
void F(
    const std::array<T,N>& x, 
    const std::array<TP,NP>& p, 
    std::array<T,N>& y
);

template<typename T, typename TP, size_t N, size_t NP> 
void f(
    const std::array<T,N>& x, 
    const std::array<TP,NP>& p, 
    T& y
){
    using namespace std; 
    y=p[0]*x[0]+sin(x[1]);
}

template<typename T, typename TP, size_t N, size_t NP> 
void grad_f(
    const std::array<T,N>& xv, 
    const std::array<TP,NP>& p, 
    T& yv,
    std::array<T,N>& dydx
){
    typedef typename dco::ga1s<T> DCO_M;
    typedef typename DCO_M::type DCO_T;
    typedef typename DCO_M::tape_t DCO_TT; 
    DCO_M::global_tape=DCO_TT::create();
    std::array<DCO_T,N> x; 
    DCO_T y;
    for (size_t i=0;i<N;i++) x[i]=xv[i];
    for (auto& i:x) DCO_M::global_tape->register_variable(i);
    f(x,p,y);
    yv=dco::value(y); 
    DCO_M::global_tape->register_output_variable(y); 
    dco::derivative(y)=1; 
    DCO_M::global_tape->interpret_adjoint();
    for (size_t i=0;i<N;i++) dydx[i]=dco::derivative(x[i]); 
    DCO_TT::remove(DCO_M::global_tape);
}

template<typename T, typename TP, size_t N, size_t NP> 
void jacobi_F(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& dydx_v, 
    std::array<std::array<T,N>,N>& ddydxx
){
    typedef typename dco::gt1v<T,N>::type DCO_T; 
    std::array<DCO_T,N> x,dydx; 
    DCO_T y;
    for (size_t i=0;i<N;i++) {
        x[i]=xv[i];
        dco::derivative(x[i])[i]=1;
    }
    grad_f(x,p,y,dydx); 
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        dydx_v[i]=dco::value(dydx[i]); 
        for (size_t j=0;j<N;j++) ddydxx[i][j]=dco::derivative(dydx[i])[j];
    }
}

template<typename T, typename TP, size_t N, size_t NP> 
void S_grad_f(
    const std::array<T,N>& xv, 
    const std::array<TP,NP>& p, 
    T& yv,
    std::array<bool,N> &sdf
){
    using DCO_T=dco::p1f::type; 
    std::array<DCO_T,N> x; 
    DCO_T y; 
    for (size_t i=0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,i); 
    }
    f(x,p,y);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++) dco::p1f::get(y,sdf[i],i);
}

template<typename T, typename TP, size_t N, size_t NP>
void S_jacobi_F(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<bool,N> &dSddf
) {
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,dydx;
    DCO_T y;
    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,0);
    }
    grad_f(x,p,y,dydx);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++) dco::p1f::get(dydx[i],dSddf[i],0);
}

template<typename T, typename TP, size_t N, size_t NP>
void C_grad_f(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& cdf
) {
    std::array<T,N> dydx;
    grad_f(xv,p,yv,dydx);

    std::array<std::array<T,N>,N> ddydxx;
    jacobi_F(xv,p,yv,dydx,ddydxx);

    std::array<bool,N> sdf;
    S_grad_f(xv,p,yv,sdf);

    std::array<bool,N> dsddf;
    S_jacobi_F(xv,p,yv,dsddf);
    
    for(int i=0;i<N,i++)
    cdf[i] = (sdf[i]!=dsddf[i]);
}

//input F:
template<typename T, typename TP, size_t N, size_t NP> 
void input_F(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    std::array<T,N>& yv,
    double*** dp3_seed
){
    std::array<T,N> cdf;
    unsigned int** uip2_SparsityPattern = new unsigned int**;
    std::array<std::array<T,N>,N> cdf_jacobian;
    
        for (int i = 0; i<yv.size(); i++) {
            Cdf(xv,p,yv[i],cdf);
            for(int j=0;j<cdf.size();j++) {
                cdf_jacobian[i][j] = cdf[j];
                uip2_SparsityPattern[i][j] = 1 - cdf_jacobian[i][j];
            }
        }
    
    //Matrix Komprimieren Comp: ColPack
    std::string s_OrderingVariant = "LARGEST_FIRST";
    std::string s_ColoringVariant = "DISTANCE ONE";
    int i_rowCount = N;
    int* ip1_SeedRowCount = new int*;
    int* ip1_SeedColumnCount = new int*;


    ColPack::GraphColoringInterface * GCI = new ColPack::GraphColoringInterface(SRC_MEM_ADOLC, uip2_SparsityPattern, i_rowCount);
    GCI->Coloring(s_OrderingVariant,s_ColoringVariant);
    (*dp3_seed) = GCI->GetSeedMatrix(ip1_SeedRowCount,ip1_SeedColumnCount);

    //newtonverfahren aufrufen
    //eventuell einfach extern in main/function-file    
    //Newton (dp3_seed);
    
}

template<typename T, typename TP, size_t N, size_t NP>
void input_f(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
){
    std::array<T,N> dydx;
    grad_f(xv,p,yv,dydx);
    double*** dp3_seed = new double***;
    input_F(xv,p,dydx);
    //eingabe genau anschauen, was brauchen wir hier?
}
int main() {
    using T=double; using TP=float;
    //initialisierung
    const size_t N=2, NP=1;
    std::array<T,N> x={1,1};
    std::array<TP,NP> p={1.1};
    Ty; 
}
