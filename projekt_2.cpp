#include "dco.hpp"
#include "ColPackHeaders.h"
#include <iostream>
#include <array>
#include <cmath>
#include <typeinfo>
#include <string>

//Für Flo:
//wir wollen zuerst die konstanten einträge in einer Matrix finden (dafür CDF,siehe briefing).
//Dann teilen wir die Matrix auf in Konstante und Variable Teilmatrix
//Die beiden sind dann sparse, wir wollen die variable Matrix komprimieren.
//das CDF-Pattern sollte demnach identisch zum sparsity-pattern sein (hier können wir also den string schon schreiben)


//Für My:
//CDF ist im briefing nochmal genauer erklärt, da siehst du, was die Idee war. Julia ist gerade im Urlaub
//Du kannst ja mal die Tiefenaddition ausprobieren (einfach jeden n-ten eintrag addieren, das ganze in zwei forschleifen wie folgt)
//for i=0;i<N,i++
//  for j=i;j<N*N;j+N
//oder so denke ich

//ich konnte nicht finden, wie man das sonst in dco machen soll. vielleicht fragt da sonst einfach mal die andere Gruppe.
using namespace ColPack;


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
void df(
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
void ddf(
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
    df(x,p,y,dydx);
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        dydx_v[i]=dco::value(dydx[i]);
        for (size_t j=0;j<N;j++) ddydxx[i][j]=dco::derivative(dydx[i])[j];
    }
}


//neue Mehtode für f'''
template<typename T, typename TP, size_t N, size_t NP>
void dddf(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& dydx_v,
    std::array<std::array<T,N>,N>& ddydxx_v,
    std::array<std::array<std::array<T,N>,N>,N>& dddydxxx
){
    typedef typename dco::gt1v<T,N>::type DCO_T;    //typedef typename dco::gt1s<T,N>::type DCO_T;(Stand vorher da) -> Fehler?
                                                    //Kommentar Jan: gt1s ist eigentlich richtig und sollte auch funktionieren. Bei My hat das (dachte ich) fehlerlos kompiliert
                                                    //Kommentar My: ich weiß nicht warum hier gt1s stand, weil bei mir steht gt1v
    std::array<DCO_T,N> x,dydx;
    std::array<std::array<DCO_T,N>,N> ddydxx;
    DCO_T y;
    for(size_t i=0;i<N;i++) {
        x[i]=xv[i];
	dco::derivative(x[i])[i]=1; 
        //was ist mit Zeile 62 ?
    }
    ddf(x,p,y,dydx,ddydxx);
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        for (size_t j=0;j<N;j++){
                        ddydxx_v[i][j] = dco::value(ddydxx[i][j]);
            for (size_t k=0;k<N;k++) dddydxxx[i][j][k]=dco::derivative(ddydxx[i][j])[k];  // Hier wurden die Klammern flasch gesetzt ich gehe mal davon aus das die 3 Schleife mit der zuweosung gleichzeitig laufen soll oder ?
	    //Kommentar My: also ich habe es so getestet wie es ist mit einigen Testfunktionen und so kam das richtige raus
        }
    }
}

//dddf mit Tiefenaddition
template<typename T, typename TP, size_t N, size_t NP>
void dddf_a(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& dydx_v,
    std::array<std::array<T,N>,N>& ddydxx_v,
    std::array<std::array<T,N>,N>& dddydxxx_a
){
    std::array<std::array<std::array<T,N>,N>,N> dddydxxx;
    typedef typename dco::gt1v<T,N>::type DCO_T;
    std::array<DCO_T,N> x,dydx;
    std::array<std::array<DCO_T,N>,N> ddydxx;
    DCO_T y;
    for(size_t i=0;i<N;i++) {
        x[i]=xv[i];
	dco::derivative(x[i])[i]=1;
    }
    ddf(x,p,y,dydx,ddydxx);
    yv=dco::value(y);
    for (size_t i=0;i<N;i++) {
        for (size_t j=0;j<N;j++){ ddydxx_v[i][j] = dco::value(ddydxx[i][j]);
        for (size_t k=0;k<N;k++) dddydxxx[i][j][k]=dco::derivative(ddydxx[i][j])[k];}
    }
    
    for (int k=0; k<N; k++){
	for(int j=0; j<N; j++){
			T v=0;
		for(int i=0; i<N; i++){
			v += dddydxxx[i][j][k];
			dddydxxx_a[j][k] = v;
			
			
	}}
}


template<typename T, typename TP, size_t N, size_t NP>
void Sdf(
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
void dSddf(
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
    df(x,p,y,dydx);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++) dco::p1f::get(dydx[i],dSddf[i],0);
}

//wie kann ich hier die Matrix dddf mit Tiefenaddition reinbringen?
template<typename T, typename TP, size_t N, size_t NP>
void dSdddf(
    const std::array< T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<bool,N> &sdddf
){
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,dydx;
    std::array<std::array<DCO_T,N>,N> ddydxx;
    DCO_T y;
    for (size_t i=0;i<N;i++) {
        x[i] = xv[i];
        dco::p1f::set(x[i],true,0);
    }
    ddf(x,p,y,dydx,ddydxx);
    dco::p1f::get(y,yv);
    //for (size_t i=0;i<N;i++) //Hier ist auch noch etwas falsch an der Funktion ich weiß aber leider nicht was genau
    //for(size_t j=0;j<N;j++)
    //  dco::p1f::get(ddydxx[i][j],sdddf[i][j],0);
    //Kommentar Jan: Auch hier dachte ich, dass das bei My schonmal kompiliert. Zumindest sah ihr output gut aus.
    //Kommentar My: habe das mit dem Verglichen, was ich hatte und es sind Zeilen irgendwie verloren gegangen ich habe sie mal hinzugefügt, das Problem ist hier kommen manchmal die richtigen und manchmal nicht richtige Werte raus und die Frage ist, wie hier dddf mit der Tiefensuche eingebuden werden kann 
}
/*
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
*/

//Constant Awareness
template<typename T, typename TP, size_t N, size_t NP>
void constant_awareness(
    const std::array<std::array<T,N>,N> ddf_c, 
    const std::array<std::array<T,N>,N> ddf_v,
    std::array<std::array<T,N>,N>& ddydxx,
    std::array<bool,N> &dsdddf,
    std::array<bool,N> &dsddf
){
    std::array<std::array<T,N>,N> cdf;

    //std::cout << "Cdf:" << std::endl;
    for (size_t i=0;i<dsddf.size();i++) cdf[i] = (dsddf[i]!= dsdddf[i]);
    //    std::cout << (dsddf[i]!=dsdddf[i]) << std::endl;

    //einteilung in die teil-matrizen
    //Theoretisch kann man sich zeile 216 sparen, wenn man die dsddf != dsdddf als bedingung in if packt. 
    //zum "debugging" ist die ausgabe von cdf aber eventuell sinnvoll

    for (size_t j=0;j<dsddf.size();j++) {
        if cdf[i] = 0 {
            ddf_c[i] = ddydxx[i];
        }
        else {
            ddf_v[i] = ddydxx[i];
        }
    }


}

//ddf_v soll dann an compression übergeben werden
//ddf_c eventuell auch einmal, aber das bleibt ja eh konstant. 


//Graphcoloring Flo
template<typename T, typename TP, size_t N, size_t NP>
void compression(
    const std::array<std::array<T,N>,N>& ddydxx,
    double*** &dp3_Seed
        ){
                /*
        zuerst muss die Jacobi Matrix in ein Matrix Marketformat überführt werden
                Ne ich Lüge ich benutze direkt das Row Compressed format
        */
                int rowCount, columnCount;
        std::string Matrixmaket, s1;
        int treffer = 0;
                double*** dp3_Value = new double**;
                unsigned int *** uip3_SparsityPattern = new unsigned int **;

        //for-schleife anpassen
        //Matrix-Name ändern
    for (int i = 0; i<ddydxx[0].size(); i++) {
                for(int j=0;j<ddydxx.size();j++) {
                        if(ddydxx[i][j] != 0){
                                s1.insert(i,' ',j,' ',ddydxx[i][j],"\n");
                                treffer = treffer +1;
                        }
                }
        }

    Matrixmaket.insert(ddydxx[0].size(),' ',ddydxx.size(),' ',treffer,'\n',s1);

    ConvertMatrixMarketFormat2RowCompressedFormat(Matrixmaket, uip3_SparsityPattern, dp3_Value,rowCount, columnCount);

        std::cout<<"(*uip3_SparsityPattern)"<<endl;
    displayCompressedRowMatrix((*uip3_SparsityPattern),rowCount);
    std::cout<<"(*dp3_Value)"<<endl;
    displayCompressedRowMatrix((*dp3_Value),rowCount);
    std::cout<<"Finish ConvertMatrixMarketFormatToRowCompressedFormat()"<<endl;
    Pause();

    int *ip1_SeedRowCount = new int;
    int *ip1_SeedColumnCount = new int;
    int *ip1_ColorCount = new int;
        BipartiteGraphPartialColoringInterface *g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, *uip3_SparsityPattern, rowCount, columnCount);

    g->PartialDistanceTwoColoring("SMALLEST_LAST", "COLUMN_PARTIAL_DISTANCE_TWO");
    (*dp3_Seed) = g->GetSeedMatrix(ip1_SeedRowCount, ip1_SeedColumnCount);
}


int main() {
    using T=double; using TP=float;
    const size_t N=2, NP=1;
    std::array<T,N> x={1,1};
    std::array<TP,NP> p={1.1};
    T y;

    std::cout << "f:" << std::endl;
    f(x,p,y);
    std::cout << y << std::endl;

    std::cout << "df:" << std::endl;
    std::array<T,N> dydx;
    df(x,p,y,dydx);
    for (const auto& i:dydx) std::cout << i << std::endl;

    std::cout << "ddf:" << std::endl;
    std::array<std::array<T,N>,N> ddydxx;
    ddf(x,p,y,dydx,ddydxx);
    for (const auto& i:ddydxx)
    for (const auto& j:i)
    std::cout << j << std::endl;

    std::cout <<"dddf:" << std::endl;
    std::array<std::array<std::array<T,N>,N>,N> dddydxxx;
    dddf(x,p,y,dydx,ddydxx,dddydxxx);
    for (const auto& i:dddydxxx)
    for (const auto& j:i)
    for (const auto& k:j)
    std::cout << k << std::endl;

    std::cout << "Sdf:" << std::endl;
    std::array<bool,N> sdf;
    Sdf(x,p,y,sdf);
    for (const auto& i:sdf) std::cout << i << std::endl;

    std::cout << "dSddf:" << std::endl;
    std::array<bool,N> dsddf;
    dSddf(x,p,y,dsddf);
    for (const auto& i:dsddf) std::cout << i << std::endl;

        //Schaut mal drüber ob das richtig ist aber dsdddf muss ja noch berechnet werden
        std::array<bool,N> dsdddf;
        dSdddf(x,p,y,dsdddf);



    std::cout << "Cdf:" << std::endl;
    for (size_t i=0;i<N;i++)
        std::cout << (dsddf[i]!=dsdddf[i]) << std::endl;
    return 0;


        double*** dp3_Seed;
        compression(ddydxx,dp3_Seed);


}
