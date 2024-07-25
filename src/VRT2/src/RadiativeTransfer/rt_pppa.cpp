// Include Statements
#include "rt_pppa.h"

#define N_INTS 4 // ft,fr,fth,fph
#define TINY 1.0e-10

namespace VRT2 {
RT_PP_PA::RT_PP_PA(Metric& g)
  : RadiativeTransfer(g)
{
}

RT_PP_PA::RT_PP_PA(const double y[], Metric& g)
  : RadiativeTransfer(g)
{
}

RT_PP_PA::RT_PP_PA(FourVector<double>& x, FourVector<double>& k, Metric& g)
  : RadiativeTransfer(g)
{
}

// Do Geometric rotation
void RT_PP_PA::IQUV_rotate(double iquv[], double lambdai, const double yi[], const double dydxi[], double lambdaf, const double yf[], const double dydxf[])
{
  double qtmp = iquv[1];
  double utmp = iquv[2];

  double dpa = delta_tPA(lambdai,lambdaf);
  double cdp = std::cos(dpa);
  double sdp = std::sin(dpa);

  iquv[1] = cdp*qtmp + sdp*utmp;
  iquv[2] = -sdp*qtmp + cdp*utmp;
}

double RT_PP_PA::PP_polarization_angle(double lambda)
{
  double val;
  if (lambda<=_lt[0])
    val = _pat[0];
  else if (lambda>=_lt[_lt.size()-1])
    val = _pat[_lt.size()-1];
  else
  {
    double* iter = std::lower_bound(&_lt[0],&_lt[_lt.size()],lambda);
    int i = iter - &_lt[0];
    double dx = (lambda - _lt[i-1])/(_lt[i]-_lt[i-1]);
    // Attempt to fix branch-jumps: get difference into 0-2pi and then
    // choose +- 2pi branch based upon which is closer.  Underlying assumption
    // is that we constructed the interpolation tables with sufficient resolution
    // to resolve the pi jumps (which we had better have done!).
    double dpat = std::fmod(_pat[i]-_pat[i-1]+2*M_PI,2*M_PI);
    if (dpat>M_PI) 
      dpat -= 2*M_PI;
    val = _pat[i-1] + dx*dpat;
    //val = _pat[i-1] + dx*(_pat[i]-_pat[i-1]);
  }

  /*
  std::cout << std::setw(15) << lambda
	    << std::setw(15) << val
	    << std::endl;
  */

  return val;


}

double RT_PP_PA::delta_tPA(double lambdai, double lambdaf)
{
  return 2.0*std::fmod( PP_polarization_angle(lambdaf) - PP_polarization_angle(lambdai),M_PI ); // factor of 2 comes from Stoke's vs. polarization angle.
}


// Integration of the parallel propagation equation to get the
// orientation of the polarization basis.  Begins with the projected
// vertical location at the image plane and goes backwards, identifying
// the angle relative to teh locally projected vertical direction.
void RT_PP_PA::IQUV_integrate_initialize(std::vector<double> ya[], std::vector<double> dydxa[], std::valarray<double>&)
{
  // Reset tables
  _lt.resize(0);
  _pat.resize(0);


  // If only one element in ray, don't bother integrating
  size_t N = ya[0].size();
  if (N<=1)
  {
    _lt.push_back(ya[0][0]);
    _pat.push_back(0);
    return;
  }
  // Set pointers to these arrays  ASSUMES THAT THE LAMBDA IS INCREASING
  _ya = ya;
  _dydxa = dydxa;
  static double y[8], dydx[8];


  /* // DEBUG TEST
  std::cout << "\n+++++++++++++++++\n\n";
  for (size_t i=0; i<2*N; ++i)
  {
    double lam;
    if (i%2==0)
      lam = _ya[0][i/2];
    else
      lam = 0.5*(_ya[0][i/2]+_ya[0][i/2+1]);
    PPPAinterp(lam,y,dydx);
    _g.reset(y);
    reinitialize(y);
    std::cout << std::setw(15) << lam
	      << " | "
	      << std::setw(15) << _x.con(0)
	      << std::setw(15) << _x.con(1)
	      << std::setw(15) << _x.con(2)
	      << std::setw(15) << _x.con(3)
	      << " | "
	      << std::setw(15) << _k.cov(0)
	      << std::setw(15) << _k.cov(1)
	      << std::setw(15) << _k.cov(2)
	      << std::setw(15) << _k.cov(3)
	      << " | "
	      << std::setw(15) << (_k*_k)
	      << std::endl;
  }
  std::cout << "\n------------------\n\n";
  */


  // Start the polarization vector in the projected vertical direction
  // Specifically, this is orthogonal to uZAMO and k, but as much along z
  // as otherwise possible.  This direction is chosen by standard.
  PPPAinterp(_ya[0][0],y,dydx);
  _g.reset(y);
  reinitialize(y);
  FourVector<double> z(_g), uZAMO(_g);
  z.mkcon(0.0,std::cos(_x.con(2)),-std::sin(_x.con(2))/_x.con(1),0.0);
  uZAMO.mkcov(1.0,0.0,0.0,0.0);
  uZAMO *= 1.0/std::sqrt(-(uZAMO*uZAMO) );
  z = std::pow( (uZAMO*_k), 2.0)*z - ((uZAMO*_k)*(z*_k))*uZAMO + (z*_k)*(uZAMO*uZAMO)*_k;
  z *= 1.0/std::sqrt((z*z));
  static double f[4], df[4], fscal[4];

  
  for (int i=0; i<4; ++i)
    f[i] = z.con(i);

  // First table entry
  _lt.push_back(_ya[0][0]);
  _pat.push_back(0.0);




  // Integration step details (hmax is 4 ray steps)
  double lambda, h, hdid, hnext, hmax = (_ya[0][N-1]/N);

  // Orientation details
  FourVector<double> epara(_g), eperp(_g), fv(_g);
  double norm;

  // Accuracy level
  const double eps = 1.0e-6;



  // Initialize integration (will now go backwards in time)
  lambda = _ya[0][0]; // Start at image plane
  PPPAinterp(lambda,y,dydx);
  _g.reset(y);
  reinitialize(y);
  h = 1e-2*hmax;

  for (lambda=_ya[0][0]; lambda<_ya[0][N-1];)
  {
    // Make sure that we stay lambda<=_ya[0][N-1]
    if ( (lambda+h) > _ya[0][N-1] )
      h = _ya[0][N-1]-lambda;

    // Call derivs first (Note that reinitialization is done in derivs!)
    PPPAderivs(lambda,f,df);

    // Get error scaling
    for (int i=0;i<N_INTS;++i)
      fscal[i] = std::fabs(1.0) + std::fabs(h*df[i]);
    
    // Take Runge-Kutta step
    PPPArkqs(f,df,N_INTS,lambda,h,eps,fscal,hdid,hnext);

    // Get next step size
    h = std::min(hnext,hmax); // Remember h is positive now (inverted lambda array to make sorted)


    // Now save table entry
    _lt.push_back(lambda);    
    PPPAinterp(lambda,y,dydx);
    _g.reset(y);
    reinitialize(y);
    z.mkcon(0.0,std::cos(_x.con(2)),-std::sin(_x.con(2))/_x.con(1),0.0);
    uZAMO.mkcov(1.0,0.0,0.0,0.0);
    uZAMO *= 1.0/std::sqrt(-(uZAMO*uZAMO) );
    fv.mkcon(f);
    epara = std::pow( (uZAMO*_k), 2.0)*z - ((uZAMO*_k)*(z*_k))*uZAMO + (z*_k)*(uZAMO*uZAMO)*_k;
    norm = epara*epara;
    epara *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
    eperp = cross_product(uZAMO,_k,epara);
    norm = eperp*eperp;
    eperp *= 1.0/(norm>0 ? std::sqrt(norm) : 1.0);
    _pat.push_back(std::atan2((fv*eperp),(fv*epara)));

    /*
    std::cout << std::setw(15) << lambda               // 1
	      << std::setw(15) << _pat[_pat.size()-1]  // 2
	      << " |x: "
	      << std::setw(15) << _x.con(0)            // 3
	      << std::setw(15) << _x.con(1)            // 4
	      << std::setw(15) << _x.con(2)            // 5
	      << std::setw(15) << _x.con(3)            // 6
	      << " |k: "
	      << std::setw(15) << _k.con(0)            // 7
	      << std::setw(15) << _k.con(1)            // 8
	      << std::setw(15) << _k.con(2)            // 9
	      << std::setw(15) << _k.con(3)            // 10
	      << " |0: "
	      << std::setw(15) << (epara*eperp)      // 11
	      << std::setw(15) << (epara*_k)          // 12
	      << std::setw(15) << (eperp*_k)          // 13
	      << std::setw(15) << (epara*uZAMO)       // 14
	      << std::setw(15) << (eperp*uZAMO)       // 15
	      << std::setw(15) << (fv*_k)              // 16
	      << std::setw(15) << (_k*_k)              // 17
	      << " |1: "
	      << std::setw(15) << (epara*epara)      // 18
	      << std::setw(15) << (eperp*eperp)      // 19
	      << std::setw(15) << (fv*fv)              // 20
	      << std::setw(15) << (-(uZAMO*uZAMO))     // 21
	      << std::endl;
    */
  
    /*
    PPPAderivs(lambda,f,df);
    std::cout << std::setw(15) << lambda
	      << std::setw(15) << _pat[_pat.size()-1]
	      << std::setw(15) << _x.con(1)*sin(_x.con(2))*cos(_x.con(3))
	      << std::setw(15) << _x.con(1)*sin(_x.con(2))*sin(_x.con(3))
	      << std::setw(15) << _x.con(1)*cos(_x.con(2))
	      << std::setw(15) << (fv*_k)
	      << std::setw(15) << (fv*eperp)
	      << std::setw(15) << (fv*epara)
	      << std::setw(15) << (epara*uZAMO)
	      << std::setw(15) << (epara*_k)
	      << std::setw(15) << (z*_k)
	      << std::setw(15) << (z*uZAMO)
	      << std::setw(15) << (z*z)
	      << std::setw(15) << (_k*_k)
	      << std::setw(15) << (fv*fv)
	      << " |epara"
	      << std::setw(15) << epara.con(0)
	      << std::setw(15) << epara.con(1)
	      << std::setw(15) << epara.con(2)
	      << std::setw(15) << epara.con(3)
	      << " |f "
	      << std::setw(15) << f[0]
	      << std::setw(15) << f[1]
	      << std::setw(15) << f[2]
	      << std::setw(15) << f[3]
	      << " |ffoo "
	      << std::setw(15) << ffoo[0]
	      << std::setw(15) << ffoo[1]
	      << std::setw(15) << ffoo[2]
	      << std::setw(15) << ffoo[3]
	      << " |z "
	      << std::setw(15) << z.con(0)
	      << std::setw(15) << z.con(1)
	      << std::setw(15) << z.con(2)
	      << std::setw(15) << z.con(3)
	      << " |k "
	      << std::setw(15) << _k.con(0)
	      << std::setw(15) << _k.con(1)
	      << std::setw(15) << _k.con(2)
	      << std::setw(15) << _k.con(3)
	      << " |df "
	      << std::setw(15) << df[0]
	      << std::setw(15) << df[1]
	      << std::setw(15) << df[2]
	      << std::setw(15) << df[3]
	      << std::endl;
    */    

    /*
    std::cout << z  << '\n'
	      << epara << '\n'
	      << uZAMO << std::endl;
    */
      
      
  }
  //std::cout << "\n\n==============\n\n" << std::endl;

  /*
  std::cout << '\n' << std::endl;
  for (size_t i=0; i<_lt.size(); ++i)
    std::cout << std::setw(15) << _lt[i]
	      << std::setw(15) << _pat[i]
	      << std::endl;
  std::cout << '\n' << std::endl;
  */
}


// Inerpolate to a given lambda
void RT_PP_PA::PPPAinterp(double lambda, double y[], double dydx[])
{

  if (lambda<=_ya[0][0]){
    for (int j=0; j<8; ++j){
      y[j] = _ya[j+1][0];
      dydx[j] = _dydxa[j+1][0];
    }
  }
  else if (lambda>=_ya[0][_ya[0].size()-1]){
    for (int j=0; j<8; ++j){
      y[j] = _ya[j+1][_ya[0].size()-1];
      dydx[j] = _dydxa[j+1][_ya[0].size()-1];
    }
  }
  else{
    double* iter = std::lower_bound(&_ya[0][0],&_ya[0][_ya[0].size()],lambda);
    int i = iter - &_ya[0][0];
    double dx = (lambda - _ya[0][i-1])/(_ya[0][i]-_ya[0][i-1]);
    double mdx = 1.0-dx;
    if (std::fabs(std::fmod(_ya[3][i],M_PI))>0.1
	&& std::fabs(std::fmod(M_PI-_ya[3][i],M_PI))>0.1
	&& std::fabs(std::fmod(_ya[3][i-1],M_PI))>0.1
	&& std::fabs(std::fmod(M_PI-_ya[3][i-1],M_PI))>0.1)	
    {
      for (int j=0; j<8; ++j){
	y[j] = dx*_ya[j+1][i] + mdx*_ya[j+1][i-1];
	dydx[j] = dx*_dydxa[j+1][i] + mdx*_dydxa[j+1][i-1];
      }
    }
    else // Do pseudo-Cartesian interpolation
    {
      double vtmp[4][3];

      // 1st do x^a
      for (int j=-1; j<=0; ++j)
      {
	vtmp[0][1+j] = _ya[1][i+j];
	vtmp[1][1+j] = _ya[2][i+j]*std::sin(_ya[3][i+j])*std::cos(_ya[4][i+j]);
	vtmp[2][1+j] = _ya[2][i+j]*std::sin(_ya[3][i+j])*std::sin(_ya[4][i+j]);
	vtmp[3][1+j] = _ya[2][i+j]*std::cos(_ya[3][i+j]);
      }
      for (size_t j=0; j<4; ++j)
	vtmp[j][2] = dx*vtmp[j][1]+mdx*vtmp[j][0];
      y[0] = vtmp[0][2];
      y[1] = std::sqrt( vtmp[1][2]*vtmp[1][2] + vtmp[2][2]*vtmp[2][2] + vtmp[3][2]*vtmp[3][2] );
      y[2] = std::atan2( std::sqrt( vtmp[1][2]*vtmp[1][2] + vtmp[2][2]*vtmp[2][2] ), vtmp[3][2] );
      y[3] = std::atan2( vtmp[2][2], vtmp[1][2] );
      // Fix branch cuts
      int npi = int((_ya[3][i]-y[2])/M_PI+0.5);
      if (npi%2)
	y[3] = y[3]+M_PI;
      y[2] += npi*M_PI;
      y[3] += int((_ya[4][i]-y[3])/(2.0*M_PI)+0.5)*2.0*M_PI;


      // 2nd do k_a
      for (int j=-1; j<=0; ++j)
      {
	// dt = dt
	vtmp[0][1+j] = _ya[5][i+j];
	// dx = sT cP dr + r cT cP dT - r sT sP dP
	vtmp[1][1+j] = ( std::sin(_ya[3][i+j])*std::cos(_ya[4][i+j])*_ya[6][i+j]
		       + _ya[2][i+j]*std::cos(_ya[3][i+j])*std::cos(_ya[4][i+j])*_ya[7][i+j]
		       - _ya[2][i+j]*std::sin(_ya[3][i+j])*std::sin(_ya[4][i+j])*_ya[8][i+j] );
	// dy = sT sP dr + r cT sP dT + r sT CP dP
	vtmp[2][1+j] = ( std::sin(_ya[3][i+j])*std::sin(_ya[4][i+j])*_ya[6][i+j]
		       + _ya[2][i+j]*std::cos(_ya[3][i+j])*std::sin(_ya[4][i+j])*_ya[7][i+j]
		       + _ya[2][i+j]*std::sin(_ya[3][i+j])*std::cos(_ya[4][i+j])*_ya[8][i+j] );
	// dz = cT dr - r sT dT
	vtmp[3][1+j] = ( std::cos(_ya[3][i+j])*_ya[6][i+j]
		       - _ya[2][i+j]*std::sin(_ya[3][i+j])*_ya[7][i+j] );
      }
      for (size_t j=0; j<4; ++j)
	vtmp[j][2] = dx*vtmp[j][1]+mdx*vtmp[j][0];

      // dt = dt
      y[4] = vtmp[0][2];
      // dr = sT cP dx + sT sP dy + cT dz 
      y[5] = std::sin(y[2])*std::cos(y[3])*vtmp[1][2] + std::sin(y[2])*std::sin(y[3])*vtmp[2][2] + std::cos(y[2])*vtmp[3][2];
      // dT = (cT cP/r) dx + (cT sP/r) dy - (sT/r) dz
      y[6] = std::cos(y[2])*std::cos(y[3])/y[1] * vtmp[1][2] + std::cos(y[2])*std::sin(y[3])/y[1] * vtmp[2][2] - std::sin(y[2])/y[1] * vtmp[3][2];
      // dP = (-sP/r sT) dx + (cP/r sT) dy
      y[7] = -std::sin(y[3])/(y[1]*std::sin(y[2])) * vtmp[1][2] + std::cos(y[3])/(y[1]*std::sin(y[2])) * vtmp[2][2];

      
      // Fix signs!  Under T->-T and P->P+pi,  dT->-dT, dP->dP
      if ( (dx*_ya[7][i]+mdx*_ya[7][i-1])*y[6] < 0 )
	y[6] *= -1;

      /*
      std::cout << "PPPAinterp:\n"
		<< std::setw(15) << _ya[5][i-1]
		<< std::setw(15) << _ya[6][i-1]
		<< std::setw(15) << _ya[7][i-1]
		<< std::setw(15) << _ya[8][i-1]
		<< std::endl
		<< std::setw(15) << vtmp[0][0]
		<< std::setw(15) << vtmp[1][0]
		<< std::setw(15) << vtmp[2][0]
		<< std::setw(15) << vtmp[3][0]
		<< std::endl
		<< std::setw(15) << vtmp[0][1]
		<< std::setw(15) << vtmp[1][1]
		<< std::setw(15) << vtmp[2][1]
		<< std::setw(15) << vtmp[3][1]
		<< std::endl
		<< std::setw(15) << vtmp[0][2]
		<< std::setw(15) << vtmp[1][2]
		<< std::setw(15) << vtmp[2][2]
		<< std::setw(15) << vtmp[3][2]
		<< std::endl
		<< std::setw(15) << y[4]
		<< std::setw(15) << y[5]
		<< std::setw(15) << y[6]
		<< std::setw(15) << y[7]
		<< " | "
		<< std::setw(15) << -std::sin(y[2])*std::sin(y[3])/y[1] * vtmp[1][2]
		<< std::setw(15) << std::sin(y[2])*std::cos(y[3])/y[1] * vtmp[2][2]
		<< std::endl;
      */




      //  Finally get the rest
      for (int j=0; j<8; ++j){
	dydx[j] = dx*_dydxa[j+1][i] + mdx*_dydxa[j+1][i-1];
      }
    }
  }
}

// Derivs
void RT_PP_PA::PPPAderivs(double lambda, const double f[], double df[])
{
  static double y[8], dydx[8];
  // Linearly interpolate
  PPPAinterp(lambda,y,dydx);
    
  // Initialize to current point
  _g.reset(y);
  reinitialize(y);

  // Now we just go with the parallel propagation equation:
  //  df^a/dl + G^a_bc k^b f^c = 0  // SIGN CHECK
  for (int i=0; i<4; ++i)
    df[i] = 0.0;
  
  for (int i=0; i<_g.NG; ++i)
    df[_g.Gi[i]] += _g.Gamma(i)*dydx[_g.Gj[i]]*f[_g.Gk[i]];

  /*
  for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j)
      for (int k=0; k<4; ++k)
	df[i] += _g.Gamma(i,j,k)*dydx[j]*f[k];
  */
}

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
int RT_PP_PA::PPPArkqs(double y[], double dydx[], int n, double& x, double htry, double eps, double yscal[], double& hdid, double& hnext)
{
  //int imax=-1;
  static int i;
  static double errmax,h,htemp,yerr[N_INTS],ytemp[N_INTS]; 
  //static double xnew;
  h=htry;
  for (;;) {
    PPPArkck(y,dydx,n,x,h,ytemp,yerr);
    errmax=0.0;
    for (i=0;i<n;i++){
      //std::cout << std::setw(15) << yerr[i]
      //	<< std::setw(15) << yscal[i];
      if (yscal[i]>0.0 && std::fabs(yerr[i]/yscal[i])>errmax){
	      //imax = i;
      	errmax = std::fabs(yerr[i]/yscal[i]);
      }
    }
    errmax /= eps;
    if (errmax <= 1.0)
      break;
    htemp=SAFETY*h*std::pow(errmax,PSHRNK);
    htemp=(h >= 0.0 ? std::max(htemp,0.1*h) : std::min(htemp,0.1*h));
    //xnew=x+htemp;
    h = htemp;
  }
  if (errmax > ERRCON) 
    hnext=SAFETY*h*std::pow(errmax,PGROW);
  else
    hnext=5.0*h;
  x += (hdid=h);
  for (i=0;i<n;i++)
    y[i]=ytemp[i];
  return 0;
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

void RT_PP_PA::PPPArkck(double y[], double dydx[], int n, double x, double h, double yout[], double yerr[])
{
  int i;
  static double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  static double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  static double ak2[N_INTS],ak3[N_INTS],ak4[N_INTS],ak5[N_INTS],ak6[N_INTS],ytemp[N_INTS];

  for (i=0;i<n;i++)
    ytemp[i]=y[i]+b21*h*dydx[i];
  PPPAderivs(x+a2*h,ytemp,ak2);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  PPPAderivs(x+a3*h,ytemp,ak3);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  PPPAderivs(x+a4*h,ytemp,ak4);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  PPPAderivs(x+a5*h,ytemp,ak5);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  PPPAderivs(x+a6*h,ytemp,ak6);
  for (i=0;i<n;i++)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
  for (i=0;i<n;i++)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
}
#undef TINY
#undef N_INTS



};
