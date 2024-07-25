#include "johannsen.h"

namespace VRT2{

Johannsen::Johannsen(double M, double spin, double alpha13, double alpha22, double alpha52, double epsilon, double beta)
  : Metric(6,12,32), _M(M), _a(spin*M), _alpha13(alpha13), _alpha22(alpha22), _alpha52(alpha52), _epsilon(epsilon), _beta(beta)
{
  initialize();
  mk_hash_table();
  check_compiler_optimization();

  _M2 = _M*_M;
  _a2 = _a*_a;
  _M3 = _M2*_M;
  _2M = 2.0*M;
  _rh = _M + std::sqrt(std::max(0.0,_M2 - _a2 - _M2*_beta*_beta));

  //Solve for the ISCO
  _risco = 0;
  
  if ( _rh == _M){
    std::cerr << "!!!! warning beta is too large have a naked singularity\n"
              << "beta*M^2  < M^2 - a^2\n";
  }

  if ( _epsilon <= -_rh*_rh*_rh/_M3 ){
    std::cerr << "epsilon parameter outside of allowed parameter range.\n"
              << "epsilon > rHoriz^3/M^3\n";
    std::exit(1);
  }

  if ( _alpha52 <= -_rh*_rh/_M2 ){
    std::cerr << "alpha52 parameter outside of allowed parameter range.\n"
              << "alpha52 > rHoriz*rHoriz/M^2\n";
    std::exit(1);
  }

  if ( _alpha13 <= -_rh*_rh*_rh/_M3 ){
    std::cerr << "alpha13 parameter outside of recommended parameter range.\n"
              << "alpha13 > rHoriz*rHoriz/M^2\n"
              << "There will be CTC's\n";
  }

  if ( _alpha22 <= -_rh*_rh/_M2 ){
    std::cerr << "alpha22 parameter outside of recommended parameter range.\n"
              << "alpha22 > rHoriz*rHoriz/M^2\n"
              << "There will be CTC's\n";
  }
  
}

// One time initialization stuff that must be virtual
void Johannsen::initialize()
{
  // Memory management for vector
  _x.resize(4);

  // non-zero metric entries
  gi.resize(Ng);
  gj.resize(Ng);
  _g.resize(Ng);
  _ginv.resize(Ng);

  gi[0] = 0;
  gj[0] = 0;

  gi[1] = 1;
  gj[1] = 1;

  gi[2] = 2;
  gj[2] = 2;

  gi[3] = 3;
  gj[3] = 3;

  gi[4] = 0;
  gj[4] = 3;

  gi[5] = 3;
  gj[5] = 0;

  // non-zero g_ij,k entries
  Dgi.resize(NDg);
  Dgj.resize(NDg);
  Dgk.resize(NDg);
  _Dg.resize(NDg);
  _Dginv.resize(NDg); 

  Dgi[0] = 0;
  Dgj[0] = 0;
  Dgk[0] = 1;

  Dgi[1] = 1;
  Dgj[1] = 1;
  Dgk[1] = 1;

  Dgi[2] = 2;
  Dgj[2] = 2;
  Dgk[2] = 1;

  Dgi[3] = 3;
  Dgj[3] = 3;
  Dgk[3] = 1;

  Dgi[4] = 0;
  Dgj[4] = 3;
  Dgk[4] = 1;

  Dgi[5] = 3;
  Dgj[5] = 0;
  Dgk[5] = 1;

  Dgi[6] = 0;
  Dgj[6] = 0;
  Dgk[6] = 2;

  Dgi[7] = 1;
  Dgj[7] = 1;
  Dgk[7] = 2;

  Dgi[8] = 2;
  Dgj[8] = 2;
  Dgk[8] = 2;

  Dgi[9] = 3;
  Dgj[9] = 3;
  Dgk[9] = 2;

  Dgi[10] = 3;
  Dgj[10] = 0;
  Dgk[10] = 2;

  Dgi[11] = 0;
  Dgj[11] = 3;
  Dgk[11] = 2;

  // non-zero Gamma^i_jk entries
  Gi.resize(NG);
  Gj.resize(NG);
  Gk.resize(NG);
  _Gamma.resize(NG);

  Gi[0] = 0;
  Gj[0] = 0;
  Gk[0] = 1;

  Gi[1] = 0;
  Gj[1] = 1;
  Gk[1] = 0;

  Gi[2] = 0;
  Gj[2] = 0;
  Gk[2] = 2;

  Gi[3] = 0;
  Gj[3] = 2;
  Gk[3] = 0;

  Gi[4] = 0;
  Gj[4] = 1;
  Gk[4] = 3;

  Gi[5] = 0;
  Gj[5] = 3;
  Gk[5] = 1;

  Gi[6] = 0;
  Gj[6] = 2;
  Gk[6] = 3;

  Gi[7] = 0;
  Gj[7] = 3;
  Gk[7] = 2;

  Gi[8] = 1;
  Gj[8] = 0;
  Gk[8] = 0;

  Gi[9] = 1;
  Gj[9] = 0;
  Gk[9] = 3;

  Gi[10] = 1;
  Gj[10] = 3;
  Gk[10] = 0;

  Gi[11] = 1;
  Gj[11] = 1;
  Gk[11] = 1;

  Gi[12] = 1;
  Gj[12] = 1;
  Gk[12] = 2;

  Gi[13] = 1;
  Gj[13] = 2;
  Gk[13] = 1;

  Gi[14] = 1;
  Gj[14] = 2;
  Gk[14] = 2;

  Gi[15] = 1;
  Gj[15] = 3;
  Gk[15] = 3;

  Gi[16] = 2;
  Gj[16] = 0;
  Gk[16] = 0;

  Gi[17] = 2;
  Gj[17] = 0;
  Gk[17] = 3;

  Gi[18] = 2;
  Gj[18] = 3;
  Gk[18] = 0;

  Gi[19] = 2;
  Gj[19] = 1;
  Gk[19] = 1;

  Gi[20] = 2;
  Gj[20] = 1;
  Gk[20] = 2;

  Gi[21] = 2;
  Gj[21] = 2;
  Gk[21] = 1;

  Gi[22] = 2;
  Gj[22] = 2;
  Gk[22] = 2;

  Gi[23] = 2;
  Gj[23] = 3;
  Gk[23] = 3;

  Gi[24] = 3;
  Gj[24] = 0;
  Gk[24] = 1;

  Gi[25] = 3;
  Gj[25] = 1;
  Gk[25] = 0;

  Gi[26] = 3;
  Gj[26] = 0;
  Gk[26] = 2;

  Gi[27] = 3;
  Gj[27] = 2;
  Gk[27] = 0;

  Gi[28] = 3;
  Gj[28] = 1;
  Gk[28] = 3;

  Gi[29] = 3;
  Gj[29] = 3;
  Gk[29] = 1;

  Gi[30] = 3;
  Gj[30] = 2;
  Gk[30] = 3;

  Gi[31] = 3;
  Gj[31] = 3;
  Gk[31] = 2;
}

void Johannsen::get_fcns()
{
  //Position stuff
  _r = _x[1];
  _sn = std::sin(_x[2]);
  _cs = std::cos(_x[2]);
  _r2 = _r*_r;
  _sn2 = _sn*_sn;
  _cs2 = _cs*_cs;
  _sncs = _sn*_cs;
  _ra2 = _r2 + _a2;
  _a2sn2 = _a2*_sn2;

  //Function stuff
  _f = _epsilon*_M3/_r;
  _A1 = 1 + _alpha13*_M3/(_r2*_r);
  _A2 = 1 + _alpha22*_M2/_r2;
  _A5 = 1 + _alpha52*_M2/_r2;
  _A22 = _A2*_A2;
  _A1A2 = _A1*_A2;
  _tDelta = _ra2 - 2.0*_M*_r + _beta*_M2;
  _tSigma = _r2 + _a2*_cs2 + _f;
  _d = _ra2*_A1 - _a2*_A2*_sn2;
  _d2 = _d*_d;
  _d3 = _d2*_d;
  _X = _tSigma/_d2;
  _pg_tt = _tDelta - _A22*_a2sn2;
  _pg_pp = _ra2*_ra2*_A1*_A1 - _tDelta*_a2sn2;
  _pg_tp = _ra2*_A1A2 - _tDelta;

}

double Johannsen::g(int i)
{
  if (!defined_list[0]){
    _g[0] = -_X*_pg_tt;
    _g[1] = _tSigma/(_tDelta*_A5);
    _g[2] = _tSigma;
    _g[3] = _X*_sn2*_pg_pp;
    _g[4] = _g[5] = -_a*_X*_sn2*_pg_tp;
    defined_list.set(0);
  }
  return (_g[i]);
}

// g^ij (For expedience assume i and j are  in (gi,gj))
double Johannsen::ginv(int i)
{
  // If not defined, define ginv
  if (!defined_list[1]) { 
    if (_sn2>0)
    {
      double odg = 1.0/(g(0)*g(3)-g(4)*g(5));
      _ginv[0] = g(3)*odg;
      _ginv[3] = g(0)*odg;
      _ginv[4] = _ginv[5] = -g(4)*odg;
    }
    else
    {
      _ginv[0] = 1.0/g(0);
      _ginv[3] = 1.0/g(3);
      _ginv[4] = _ginv[5] = 0;
    }

    _ginv[1] = 1.0/g(1);
    _ginv[2] = 1.0/g(2);

    defined_list.set(1);
  }
  return (_ginv[i]);
} 

// sqrt(-det(g))
double Johannsen::detg()
{
  // If not defined, define detg
  if (!defined_list[2]) { 
    _detg = std::sqrt( - (g(0)*g(3)-g(4)*g(5))*g(1)*g(2) );
    defined_list.set(2);
  }
  return (_detg);
}

//Dg_ij,k
double Johannsen::Dg(int i)
{
  if (!defined_list[3]){
    //Derivative stuff
    //
    //Radial stuff
    double _dA1dr = (1.0-_A1)*3.0/_r;
    double _dA2dr = (1.0-_A2)*2.0/_r;
    double _dA5dr = (1.0-_A5)*2.0/_r;
    double _dfdr = -_f/_r;
    double _dtDeltadr = 2.0*_r - _2M;
    double _dtSigmadr = 2.0*_r + _dfdr;
    double _Ddr = (2.0*_r*_A1 + _ra2*_dA1dr - _a2sn2*_dA2dr); //d/dr d(r,th)
    double _dXdr = (2.0*(-_tSigma*_Ddr/_d +_r) + _dfdr)/_d2;
    double _tDA5 = _tDelta*_A5;

    //Theta stuff
    double _da2sn2dth = 2.0*_a2*_sncs;
    double _dXdth = _da2sn2dth/_d3*(2.0*_A2*_tSigma - _d);
    //g_tt_,c
    _Dg[0] = -_dXdr*_pg_tt - _X*(_dtDeltadr-2.0*_A2*_dA2dr*_a2sn2);// ,r
    _Dg[6] = -_dXdth*_pg_tt + _X*_A22*_da2sn2dth;// ,theta

    // g_rr_,c
    _Dg[1] = _dtSigmadr/(_tDA5) - 
             _tSigma*(_dtDeltadr*_A5 + _tDelta*_dA5dr)/(_tDA5*_tDA5);// ,r
    _Dg[7] = -_da2sn2dth/_tDA5;// ,theta

    //g_thth_,c
    _Dg[2] = _dtSigmadr; // ,r
    _Dg[8] = -_da2sn2dth;// ,theta

    //g_phph_,c
    _Dg[3] = _sn2*(_dXdr*_pg_pp + _X*(2.0*_A1*_ra2*(2.0*_r*_A1 + _ra2*_dA1dr) 
                      - _dtDeltadr*_a2sn2) );// ,r
    _Dg[9] = 2.0*_sncs*_X*_pg_pp + _sn2*(_dXdth*_pg_pp - _X*_tDelta*_da2sn2dth);// ,th

    //g_tphi_,c
    _Dg[4] = _Dg[5] = -_a*_sn2*(_dXdr*_pg_tp +
                        _X*((2.0*_r*_A1A2 + _ra2*(_A1*_dA2dr + _dA1dr*_A2))
                          - _dtDeltadr));// ,r
    _Dg[10] = _Dg[11] = -_a*_pg_tp*(_sn2*_dXdth + 2.0*_sncs*_X);// ,theta
    defined_list.set(3);
  }
  return (_Dg[i]);
}

// Dg^ij,k
double Johannsen::Dginv(int i)
{
  // If not defined, define Dg
  if (!defined_list[4]) { 
    // g^tt_,c
    _Dginv[0] = -ginv(0)*ginv(0)*Dg(0) - 2.0*ginv(0)*ginv(4)*Dg(4) - ginv(4)*ginv(4)*Dg(3); // ,r
    _Dginv[6] = -ginv(0)*ginv(0)*Dg(6) - 2.0*ginv(0)*ginv(4)*Dg(10) - ginv(4)*ginv(4)*Dg(9); // ,theta

    // g^rr_,c
    _Dginv[1] = -ginv(1)*ginv(1)*Dg(1); // ,r
    _Dginv[7] = -ginv(1)*ginv(1)*Dg(7); // ,theta

    // g^thth_,c
    _Dginv[2] = -ginv(2)*ginv(2)*Dg(2); // ,r
    _Dginv[8] = -ginv(2)*ginv(2)*Dg(8); // ,theta

    // g^phph_,c
    _Dginv[3] = -ginv(3)*ginv(3)*Dg(3) - 2.0*ginv(3)*ginv(4)*Dg(4) - ginv(4)*ginv(4)*Dg(0); // ,r
    _Dginv[9] = -ginv(3)*ginv(3)*Dg(9) - 2.0*ginv(3)*ginv(4)*Dg(10) - ginv(4)*ginv(4)*Dg(6); // ,theta

    // g^tph_,c
    _Dginv[4] = _Dginv[5] = -ginv(0)*ginv(3)*Dg(4) - ginv(4)*ginv(3)*Dg(3) - ginv(0)*ginv(4)*Dg(0) - ginv(4)*ginv(4)*Dg(4); // ,r
    _Dginv[10] = _Dginv[11] = -ginv(0)*ginv(3)*Dg(10) - ginv(4)*ginv(3)*Dg(9) - ginv(0)*ginv(4)*Dg(6) - ginv(4)*ginv(4)*Dg(10); // ,theta
        
    defined_list.set(4);
  }
  return (_Dginv[i]);
} 

// Gamma^i_jk
double Johannsen::Gamma(int i)
{
  // If not defined, define Gamma
  if (!defined_list[5]) { 
    
    for (int i=0; i<NG; ++i)
    {
      _Gamma[i] = 0.0;
      for (int j=0; j<4; ++j)
	_Gamma[i] += 0.5*Metric::ginv(Gi[i],j)*( Metric::Dg(Gj[i],j,Gk[i])+Metric::Dg(Gk[i],j,Gj[i])-Metric::Dg(Gj[i],Gk[i],j) );
    }

    defined_list.set(5);
  }
  return (_Gamma[i]);
} 

// Horizon
double Johannsen::horizon() const
{
  return ( _rh );
}
double Johannsen::rISCO() 
{
  if ( _risco == 0 )
    _risco = get_ISCO();
  return ( _risco );
}

// Parameters
double Johannsen::mass() const
{
  return ( _M );
}

double Johannsen::ang_mom() const
{
  return ( _a*_M );
}

double Johannsen::alpha13() const
{
  return _alpha13;
}

double Johannsen::alpha22() const
{
  return _alpha22;
}

double Johannsen::alpha52() const
{
  return _alpha52;
}

double Johannsen::epsilon() const
{
  return _epsilon;
}

double Johannsen::beta() const
{
  return _beta;
}
double Johannsen::get_ISCO()
{
  // Finds ISCOs as long as separate from the horizon
  // Note for certain parameter values bad shit can happen
  double x2 = 8.0;
  double bracket = ISCOFunc(x2);
  while (bracket < 0.0 || std::isnan(bracket) || !std::isfinite(bracket))
  {
    x2+=1e-1;
    if (_rh+1e-1 < x2){
      std::cerr << "Bracket failed can't find ISCO\n";
      std::exit(1);
    }
    bracket = ISCOFunc(x2);
  }
 
  //return rtflsp(x1,x2,1e-6);
  //return zriddr(x1,x2,1e-6);
  
  return grad_descent(x2, 1e-6);
}


double Johannsen::ISCOFunc(double r)
{

  // ISCO defined by the zero of the energy function 
  double dr=1e-9;
  double Ep = energy(r+dr); 
  double Em = energy(r-dr);
  return ( (Ep-Em)/(2.0*dr) );

}


double Johannsen::energy(double r)
{
  double gttr, gppr, gtpr;
  double gtt,gtp,gpp;

  reset(0,r,0.5*M_PI,0);
  gtt = Metric::g(0,0);
  gtp = Metric::g(0,3);
  gpp = Metric::g(3,3);
  gttr = Metric::Dg(0,0,1);
  gppr = Metric::Dg(3,3,1);
  gtpr = Metric::Dg(0,3,1);
  double OmegaP = (-gtpr + std::sqrt(gtpr*gtpr - gttr*gppr))/gppr;//std::pow(r+dr,-1.5);
  double Ep = -(gtt + gtp*OmegaP)/std::sqrt(-gtt-2.0*gtp*OmegaP - gpp*OmegaP*OmegaP);

  return std::pow(Ep,4.0);
}




#define MAXIT 10000
double Johannsen::grad_descent(double x0, double xacc)
{
  //Gradient descent with backtracking to find the minimum
  //where it is adapted to deal with the ISCOFunc going imaginary
  double x1 = x0;
  double eps;
  double df = ISCOFunc(x1);

  double i = 0;
  do{
    double t = 1;

    double xNew = x1 - df;
    double tmin = 1e-2;
    while ( ( (energy(xNew) > energy(x1) - t/2.0*df*df) || std::isnan(energy(x1-t*df)))  && t > tmin ){
      t *= 0.8; 
      if ( std::isnan(energy(x1 - tmin*df))){
        tmin *= 0.5;
      }
      if (tmin < 1e-6)
        break;
    }
    x1 -= t*df;
    if ( std::isnan(energy(x1)))
      return x1+=t*df;

    if ( x1 > x0 || x1 <= _rh){
      std::cerr << "Finding ISCO failed  setting to horizon\n"
                << "Spacetime params: \n"
                << "a       = " << _a << std::endl
                << "alpha13 = " << _alpha13 << std::endl
                << "alpha22 = " << _alpha22 << std::endl
                << "alpha52 = " << _alpha52 << std::endl
                << "epsilon = " << _epsilon << std::endl
                << "beta    = " << _beta << std::endl;

      return _rh +1e-2;
    } 
    df = ISCOFunc(x1);
    eps = std::fabs(df);
    i++;
    /*
    std::cerr << std::setw(15) << i
              << std::setw(15) << x1 
              << std::setw(15) << df 
              << std::setw(15) << t 
              << std::setw(15) << eps 
              << std::setw(15) << Hess << std::endl;
    */
    if (i == MAXIT){
      std::cerr << "MAXIT ISCO: "
                << std::setw(5) << x1 
                << std::setw(15) << df 
                << std::setw(15) << t 
                << std::setw(15) << eps << std::endl;

      break;
    }
  }while (eps > xacc );
  return x1;
}
/*
double Johannsen::rtflsp(double x1, double x2, double xacc)
{
  int j;
  double fl,fh,xl,xh,swap,dx,del,f,rtf;
 
  fl=ISCOFunc(x1);
  fh=ISCOFunc(x2);
 
  if (fl*fh > 0.0){
    std::cerr << "Root must be bracketed in rtflsp" << std::endl;
    return -999;
  }
  if (fl < 0.0) {
    xl=x1;
    xh=x2;
  }
  else {
    xl=x2;
    xh=x1;
    swap=fl;
    fl=fh;
    fh=swap;
  }
  dx=xh-xl;
  for (j=1;j<=MAXIT;j++) {
    rtf=xl+dx*fl/(fl-fh);
    f=ISCOFunc(rtf);
 
    if (f < 0.0) {
      del=xl-rtf;
      xl=rtf;
      fl=f;
    }
    else {
      del=xh-rtf;
      xh=rtf;
      fh=f;
    }
    dx=xh-xl;
    if (std::fabs(del) < xacc || f == 0.0)
      return rtf;
  }
  std::cerr << "Maximum number of iterations exceeded in rtflsp" << std::endl;
  return -999;
}
#undef MAXIT

#define NRANSI
#define MAXIT 1000000
#define UNUSED (-1.11e30)
#define SIGN(a,b) ((b) >= 0.0 ? std::fabs(a) : -std::fabs(a)) 
 
double Johannsen::zriddr(double x1, double x2, double xacc)
{
  int j;
  double ans,fh,fl,fm,fnew,s,xh,xl,xm,xnew;

  fl=ISCOFunc(x1);
  fh=ISCOFunc(x2);

  if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0)) {
    xl=x1;
    xh=x2;
    ans=UNUSED;
    for (j=1;j<=MAXIT;j++) {
      xm=0.5*(xl+xh);
      fm=ISCOFunc(xm);
      s=std::sqrt(fm*fm-fl*fh);
      if (s == 0.0)
	return ans;
      xnew=xm+(xm-xl)*((fl >= fh ? 1.0 : -1.0)*fm/s);
      if (std::fabs(xnew-ans) <= xacc)
	return ans;
      ans=xnew;
      fnew=ISCOFunc(ans);
      if (fnew == 0.0)
	return ans;
      if (SIGN(fm,fnew) != fm) {
	xl=xm;
	fl=fm;
	xh=ans;
	fh=fnew;
      }
      else if (SIGN(fl,fnew) != fl) {
	xh=ans;
	fh=fnew;
      }
      else if (SIGN(fh,fnew) != fh) {
	xl=ans;
	fl=fnew;
      }
      else
	std::cerr << "never get here in zridder." << std::endl;
      if (std::fabs(xh-xl) <= xacc)
	return ans;
    }
    std::cerr << "Maximum number of iterations exceeded in zriddr" << std::endl;
    return -999;
  }
  else {
    if (fl == 0.0)
      return x1;
    if (fh == 0.0)
      return x2;
    std::cerr << "Root not bracketed in zridder\n";
    return -999;
  }
  std::cerr << "never get here 2\n";
  return 0.0;
}

#undef SIGN
#undef MAXIT
#undef UNUSED
#undef NRANSI
*/
#undef MAXIT
};
