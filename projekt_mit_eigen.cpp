#include "dco.hpp"
#include "ColPackHeaders.h"
#include <iostream>
#include <array>
#include <cmath>
#include <typeinfo>
#include <string>
#include <fstream>
#include <Eigen/Dense>

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
void ddf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<T,N,1>& dydx_v,
    Eigen::Matrix<T,N,N>& ddydxx
){
    typedef typename dco::gt1v<T,N>::type DCO_T;
    Eigen::Matrix<DCO_T,N,1> x,dydx;
    DCO_T y;
    for (size_t i=0;i<N;i++) {
        x[i]=xv[i];
        dco::derivative(x[i])[i]=1;
    }
    df<DCO_T,TP,N,NP>(x,p,y,dydx);
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        dydx_v[i]=dco::value(dydx[i]);
        for (size_t j=0;j<N;j++) ddydxx(i,j)=dco::derivative(dydx[i])[j];
    }
}

//third derivative (tensor)
template<typename T, typename TP, size_t N, size_t NP>
void dddf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<T,N,1>& dydx_v,
    Eigen::Matrix<T,N,N>& ddydxx_v,
    Eigen::Matrix<T,N,N>& dddydxxx
){
    typedef typename dco::gt1v<T,N>::type DCO_T;
    Eigen::Matrix<DCO_T,N,1> x,dydx;
    Eigen::Matrix<DCO_T,N,N> ddydxx;
    DCO_T y;
    for(size_t i=0;i<N;i++) {
        x[i]=xv[i];
        dco::derivative(x[i])[i]=1;
    }
    ddf(x,p,y,dydx,ddydxx);
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        dydx_v[i]=dco::value(dydx[i]);
        for (size_t j=0;j<N;j++){
            ddydxx_v[i][j] = dco::value(ddydxx[i][j]);
            for (size_t k=0;k<N;k++) dddydxxx[i][j][k]=dco::derivative(ddydxx[i][j])[k];
        }
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
void dSddf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    Eigen::Matrix<bool,N,N> &dSddf
) {
    using DCO_T=dco::p1f::type;
    Eigen::Matrix<DCO_T,N,1> x;
    Eigen::Matrix<DCO_T,N,1> dydx;
    //Eigen::Matrix<bool,Eigen::Matrix<bool,DCO_T,N>,N> ddydxx;
    DCO_T y;
    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,i);
    }
    df<DCO_T,TP,N,NP>(x,p,y,dydx);
    dco::p1f::get(y,yv);
    for (size_t i=0; i<N;i++){
    for (size_t j=0;j<N;j++) dco::p1f::get(dydx(i),dSddf(i,j),j);
    }
}
/*
//sparsity f'''
template<typename T, typename TP, size_t N, size_t NP>
void dSdddf(
    const Eigen::Matrix<T,N,1>& xv,
    const Eigen::Matrix<TP,NP,1>& p,
    T& yv,
    //Eigen::Matrix<bool,Eigen::Matrix<bool,bool,N>,N> &dSdddf,
    //Eigen::Matrix<bool,Eigen::Matrix<bool,Eigen::Matrix<bool,T,N>,N>,N>& dddydxxx
){
    /*
    using DCO_T=dco::p1f::type;
    Eigen::Matrix<bool,DCO_T,N> x,dydx;
    Eigen::Matrix<bool,Eigen::Matrix<bool,DCO_T,N>,N> ddydxx; //,dddydxxx;
    DCO_T y;
    for (size_t i=0;i<N;i++) {
        x[i] = xv[i];
        dco::p1f::set(x[i],true,i);
    }
    ddf(x,p,y,dydx,ddydxx);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++){
    for (size_t j=0;j<N;j++) {
    for (size_t k=0;k<N;k++)
        dco::p1f::get(ddydxx[i][j],dSdddf[i][j][k],k);
    }
    }

    T sum ;
    for (size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            sum = 0;
            for(size_t k=0;k<N;k++){
                sum = dddydxxx[i][j][k];
            }
            dSdddf[i][j] = sum;
        }

   }
}
*/


//Constant Awareness
template<typename T, typename TP, size_t N, size_t NP>
void constant_awareness(
    const Eigen::Matrix<T,N,N> ddf_c,
    const Eigen::Matrix<T,N,N> ddf_v,
    Eigen::Matrix<T,N,N>& ddydxx,
    Eigen::Matrix<bool,N,1> &dsdddf,
    Eigen::Matrix<bool,N,1> &dsddf
){
    Eigen::Matrix<T,N,N> cdf;

    //std::cout << "Cdf:" << std::endl;
    for (size_t i=0;i<dsddf.size();i++) cdf[i] = (dsddf[i]!= dsdddf[i]);
    //    std::cout << (dsddf[i]!=dsdddf[i]) << std::endl;

    //einteilung in die teil-matrizen
    //Theoretisch kann man sich zeile 216 sparen, wenn man die dsddf != dsdddf als bedingung in if packt.
    //zum "debugging" ist die ausgabe von cdf aber eventuell sinnvoll

    // Flo :Hier in die Schleife muss ein j statt dem i hin oder irre ich mich ?

    for (size_t j=0;j<dsddf.size();j++) {
        if (cdf[j] = 0) {
            ddf_c[j] = ddydxx[j];
        }
        else {
            ddf_v[j] = ddydxx[j];
        }
    }


}

int main() {
    using T=double; using TP=float;
    const size_t N=2, NP=2;
    Eigen::Matrix<T,N,1> x;
    Eigen::Matrix<TP,NP,1> p;
    T y;

    x << 1,2;
    p << 2,2;


    std::cout << "f:" << std::endl;
    f<T,TP,N,NP>(x,p,y);
    std::cout << y << std::endl;

    std::cout << "df:" << std::endl;
    Eigen::Matrix<T,N,1> dydx;
    df<T,TP,N,NP>(x,p,y,dydx);
    for (const auto& i:dydx) std::cout << i << std::endl;

    std::cout << "ddf:" << std::endl;
    Eigen::Matrix<T,N,N> ddydxx;
    ddf<T,TP,N,NP>(x,p,y,dydx,ddydxx);
    std::cout << ddydxx << std::endl;

    /* TENSOR ODER SO, kp bin kein Iformatiker
    std::cout <<"dddf:" << std::endl;
    Eigen::Matrix<T,Eigen::Matrix<T,N,N>,1> dddydxxx;
    dddf<T,TP,N,NP>(x,p,y,dydx,ddydxx,dddydxxx);
    std::cout << dddydxxx << std::endl;
*/

    std::cout << "Sdf:" << std::endl;
    Eigen::Matrix<bool,N,1> sdf;
    Sdf<T,TP,N,NP>(x,p,y,sdf);
    std::cout << sdf << std::endl;

    std::cout << "dSddf:" << std::endl;
    Eigen::Matrix<bool,N,N> dsddf;
    dSddf<T,TP,N,NP>(x,p,y,dsddf);
    std::cout << dsddf << std::endl;

    /*
    std::cout << "dSdddf:" << std::endl;
    Eigen::Matrix<bool,N,N> dsdddf;
    dSdddf<T,TP,N,NP>(x,p,y,dsdddf,dddydxxx);
    std::cout << dsdddf << std::endl;
/*
    std::cout << "Cdf:"<< std::endl;
    for (size_t i=0; i<N; i++)
        std::cout << (sdf[i] != dsddf[i]) << std::endl;
    std::cout << "dCddf:" << std::endl;
    for (size_t i=0;i<N;i++)
        std::cout << (dsddf[i]!= dsdddf[i]) << std::endl;
*/

    // Mal in die main Graph Coloring
    int rowCount, columnCount;
    std::string Matrixmaket, s1, s2, s3, s4, s5, s6, s7;
    std::stringstream ss;
    int treffer = 0;
    double*** dp3_Value = new double**;
    unsigned int *** uip3_SparsityPattern = new unsigned int **;

/*

//for-schleife anpassen
//Matrix-Name ändern
    std::cout << "Dim i " << ddydxx[0].size() << std::endl;
    std::cout << "Dim j " << ddydxx.size() << std::endl;

for (int i = 0; i<ddydxx[0].size(); i++) {
    for(int j=0;j<ddydxx.size();j++) {
            if(ddydxx[i][j] != 0){
                s1 = s1 + std::to_string(i+1) + " " + std::to_string(j+1) + " "+ std::to_string(ddydxx[i][j]) + "\n";
                treffer = treffer +1;
            }
    }
}



Matrixmaket = "%%MatrixMarket matrix coordinate real general\n";

Matrixmaket = Matrixmaket + std::to_string(ddydxx[0].size()) + " " + std::to_string(ddydxx.size()) + " "+ std::to_string(treffer) + "\n" +s1;


fstream datei;
datei.open("MatrixMarket.mtx", ios::out);         //Hier möchte ich den Inhalt eines Strings
datei << Matrixmaket << endl;
datei.close();






string s_InputFile; //path of the input file. PICK A SYMMETRIC MATRIX!!!
s_InputFile = "/home/dc350267/Dokumente/SP_CES/Code/NONLINEAR_SYSTEM/libnls_apps/NamenloserOrdner";
s_InputFile += "/MatrixMarket.mtx";


std::ifstream input(s_InputFile);
/*
if (!input)
{
  std::cerr << "Datei beim Oeffnen der Datei " << s_InputFile << "\n";
  return 1;
}
std::string line;
while (std::getline(input, line))
{
   std::cout << s_InputFile << '\n';
}


std::cout <<  s_InputFile << std::endl;
GraphColoringInterface * g = new GraphColoringInterface(SRC_FILE, s_InputFile.c_str(),"AUTO_DETECTED");

//Color the bipartite graph with the specified ordering
g->Coloring("LARGEST_FIRST", "DISTANCE_TWO");

/*Done with coloring. Below are possible things that you may
want to do after coloring:
//*/

/* 1. Check DISTANCE_TWO coloring result
cout<<"Check DISTANCE_TWO coloring result"<<endl;
g->CheckDistanceTwoColoring();
//

//* 2. Print coloring results
g->PrintVertexColoringMetrics();
//
//* 3. Get the list of colorID of vertices
vector<int> vi_VertexColors;
g->GetVertexColors(vi_VertexColors);

//Display vector of VertexColors
printf("vector of VertexColors (size %d) \n", (int)vi_VertexColors.size());
displayVector(&vi_VertexColors[0], vi_VertexColors.size(), 1);
//

// 4. Get seed matrix
int i_SeedRowCount = 0;
int i_SeedColumnCount = 0;
double** Seed = g->GetSeedMatrix(&i_SeedRowCount, &i_SeedColumnCount);

//Display Seed
printf("Seed matrix %d x %d \n", i_SeedRowCount, i_SeedColumnCount);
displayMatrix(Seed, i_SeedRowCount, i_SeedColumnCount, 1);

*/
    return 0;

}
