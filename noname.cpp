//Tensor mit Tiefenaddition
template<typename T, typename TP, size_t N, size_t NP>
void dddf(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<T,N>& dydx_v,
    std::array<std::array<T,N>,N>& ddydxx_v,
    std::array<std::array<T,N>,N>& dddydxxx_v
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
                        dddydxxx_v[j][k] = v;
                }
        }
    }
}



//Sparsity pattern mit Tiefenaddition
template<typename T, typename TP, size_t N, size_t NP>
void dSdddf(
    const std::array<T,N>& xv,
    const std::array<TP,NP>& p,
    T& yv,
    std::array<std::array<bool,N>,N> &dSdddf
) {
    using DCO_T=dco::p1f::type;
    std::array<DCO_T,N> x,dydx;
    std::array<std::array<DCO_T,N>,N> ddydxx;
    std::array<std::array<DCO_T,N>,N> dddydxxx;

    DCO_T y;
    for (size_t i =0;i<N;i++) {
        x[i]=xv[i];
        dco::p1f::set(x[i],true,0);
    }
    dddf_a(x,p,y,dydx,ddydxx,dddydxxx);
    dco::p1f::get(y,yv);
    for(size_t i=0; i<N;i++){
        for(size_t j=0; j<N; j++){
        
        if (dddydxxx[i][j] == 0)
                dSdddf[i][j] = 0;
        else if (dddydxxx[i][j] != 0)
                dSdddf[i][j] = 1;
	}
    }
}
  

   std::cout<<"dddf:" << std::endl;
    std::array<std::array<T,N>,N> dddydxxx;
    dddf(x,p,y,dydx,ddydxx,dddydxxx);
    for(const auto& i:dddydxxx)
        for(const auto j:i)
                std::cout<< j<< std::endl;

std::cout << "dSdddf:" << std::endl;
    std::array<std::array<bool,N>,N> dsdddf;
    dSdddf(x,p,y,dsdddf);
    for (const auto& i:dsdddf)
	for(const auto& j:i)
	 std::cout << j << std::endl;

