#include "dco.hpp"
#include <iostream>
#include <array> 
#include <cmath>
#include <typeinfo>
#include <string>
#include <vector>

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
    y=p[0]*x[0]*x[0]*x[1]+exp(x[1]);
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


//neue Mehtode fÃ¼r f'''
template<typename T, typename TP, size_t N, size_t NP>
void dddf(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& dydx_v,
    std::array<std::array<T,N>,N>& ddydxx_v,
    std::array<std::array<std::array<T,N>,N>,N>& dddydxxx
){
    typedef typename dco::gt1v<T,N>::type DCO_T;
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
        for (size_t j=0;j<N;j++){ ddydxx_v[i][j] = dco::value(ddydxx[i][j]);
        for (size_t k=0;k<N;k++) dddydxxx[i][j][k]=dco::derivative(ddydxx[i][j])[k];}
    }
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

//Neu
template<typename T, typename TP, size_t N, size_t NP>
void Sdddf(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<std::array<bool,N>,N>& sdddf
) {
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,dydx;
    std::array<std::array<DCO_T,N>,N> ddydxx;
    DCO_T y;
    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,0);
    }
    
    ddf(x,p,y,dydx,ddydxx);
    dco::p1f::get(y,yv);
    for (size_t i=0;i<N;i++){
	for(size_t j=0;j<N;j++){
		dco::p1f::get(ddydxx[i][j],sdddf[i][j],0);
		}
	}
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


/*

//input F:
template<typename T, typename TP, size_t N, size_t NP> 
void input_F(
    
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

*/



//Graphcoloring Flo
template<typename T, typename TP, size_t N, size_t NP> 
void input_F(
    const std::array<std::array<T,N>,N>& ddydxx,
    double*** dp3_seed
	){
		zuerst muss die Jacobi Matrix in ein Matrix Marketformat überführt werden
		Ne ich Lüge ich benutze direkt das Row Compressed format
	/*	
	std::array<T,N> cdf;
    std::array<std::array<T,N>,N> cdf_jacobian;
    
        for (int i = 0; i<yv.size(); i++) {
            Cdf(xv,p,yv[i],cdf);
            for(int j=0;j<cdf.size();j++) {
                cdf_jacobian[i][j] = cdf[j];
            }
        }
	*/	
	string Matrixmaket, s1;
	int treffer = 0;
	//for-schleife anpassen
	//Matrix-Name ändern
    for (int i = 0; i<yv.size(); i++) {
		for(int j=0;j<cdf.size();j++) {
			if(cdf_jacobian[i][j] != 0){
				s1.insert(i,' ',j,' ',cdf_jacobian[i][j],"\n")
				treffer = treffer +1;
			}
		}
	}
		
	Matrixmaket.insert(yv.size(),' ',cdf.size(),' ',treffer,'\n',s1)  Ob das alles so richtig ist weiß ich nicht es klingt auf jedenfall logisch und würde die richtige Ausgabe erzeugen 
	
	ConvertMatrixMarketFormatToRowCompressedFormat(Matrixmaket, uip3_SparsityPattern, dp3_Value,rowCount, columnCount);
		
	cout<<"(*uip3_SparsityPattern)"<<endl;
    displayCompressedRowMatrix((*uip3_SparsityPattern),rowCount);
    cout<<"(*dp3_Value)"<<endl;
    displayCompressedRowMatrix((*dp3_Value),rowCount);
    cout<<"Finish ConvertMatrixMarketFormatToRowCompressedFormat()"<<endl;
    Pause();				

	double*** dp3_Seed = new double**;
	int *ip1_SeedRowCount = new int;
	int *ip1_SeedColumnCount = new int;
	int *ip1_ColorCount = new int;

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

    //Kommentar fÃ¼r My: 
    //Methode dddf prÃ¼fen, schauen ob der output richtig ist, wenn nicht fehler finden und anpassen
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

    //Neu 
    std::cout << "Sdddf:" << std::endl;
    std::array<std::array<bool,N>,N> sdddf;
    Sdddf(x,p,y,sdddf);
    for (const auto& i:sdddf) 
	for(const auto& j:i)
		std::cout << j << std::endl;

	
    std::cout << "Cdf:" << std::endl; 
    for (size_t i=0;i<N;i++)
        std::cout << (sdf[i]!=dsddf[i]) << std::endl; 
    return 0;
  }
