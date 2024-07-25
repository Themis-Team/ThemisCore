// Include Header
#include "hartle_thorne.h"


namespace VRT2 {

// Metric Parameters
HartleThorne::HartleThorne(double Mass, double Spin, double epsilon)
  : Metric(6,12,32), _M(Mass), _a(Spin*Mass), _ep(epsilon)
{
  initialize();
  mk_hash_table();
  check_compiler_optimization();

  _rh = 1.0;
  _risco = 6;

  _risco = get_ISCO();
}

// One time initialization stuff that must be virtual
void HartleThorne::initialize()
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

// Position dependent funcs used by g and ginv
void HartleThorne::get_fcns()
{
  _r = _x[1];
  _sn = std::sin(_x[2]);
  _cs = std::cos(_x[2]);
  _r2 = _r*_r;
  _sn2 = _sn*_sn;
  _cs2 = _cs*_cs;
  _sncs = _sn*_cs;
  _a2 = _a*_a;
  _Delta = _r2 + _a2 - 2.0*_M*_r;
  _Sigma = _r2 + _a2 * _cs2;
  _ra2 = _r2 + _a2;
  _M2 = _M*_M;
  _M3 = _M2*_M;
  _r3 = _r2*_r;
  _sw = _r - 2.0*_M;
  _sw2 = _sw*_sw;
  //_lg = std::log( _r/_sw );
  _lg = std::log( std::min(_r/std::fabs(_sw),2e100) );
  //_lg = (std::fabs(_sw)>1e-15 ? std::log( _r/std::fabs(_sw) ) : 1e50 );
  _2M = 2.0*_M;
  _c2s = 1.0 + 3.0*std::cos(2.0*_x[2]);
} 

/*** Elements ***/
// g_ij (For expedience assume i and j are  in (gi,gj))
double HartleThorne::g(int i)
{
  // THINGS TO DO:
  //    (1) Get asymptotic forms for the corrections at infinity so that we can smoothly obtain the cancellations
  //    (2) Get asymptotic forms for the metric coefficients at r=2M so that we can smoothly pass over this region (here I think
  //          some of the metric coefficients are divergent, so this is problematic!)
  //    (3) Check to see if g_tphi is really identical to the Boyer-Lindquist values!

  // If not defined, define g
  if (!defined_list[0]) { 
    double _c3s = 1.0 - 3.0*_cs2;
    _g[0] = -(_Delta - _a2*_sn2)/_Sigma + 5.0*_ep*_c2s * (_2M * (2.0*_M3 + 4.0*_M2*_r - 9.0*_M*_r2 + 3.0*_r3) - 3.0*_r2*_sw2*_lg ) / (32.0*_M2*_r2);

    double _schw = _sw / _r;
    double _F1 = -5.0*(_r - _M) * (2.0*_M2 + 6.0*_M*_r - 3.0*_r2) / (8.0*_M*_r*_sw) - 15.0*_r*_sw*_lg / (16.0*_M2);
    _g[1] = 1.0/( _Delta / _Sigma + _ep*_schw * _c3s * _F1 );

    double _poly = - 5.0*_r*_ep*_c2s * (_2M * (2.0*_M2 - 3.0*_M*_r - 3.0*_r2) + (3.0*_r3 - 6.0*_M2*_r)*_lg ) / (32.0*_M2);

    _g[2] = _Sigma + _poly;
    _g[3] = ( _ra2*_ra2 - _Delta * _a2 * _sn2) * _sn2/ _Sigma + _poly*_sn2;

    _g[4] = _g[5] = -_2M*_a*_r*_sn2 / _Sigma;

    defined_list.set(0);
  }
  return (_g[i]);
}
 
// g^ij (For expedience assume i and j are  in (gi,gj))
double HartleThorne::ginv(int i)
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
double HartleThorne::detg()
{
  // If not defined, define detg
  if (!defined_list[2]) { 
    _detg = std::sqrt( - (g(0)*g(3)-g(4)*g(5))*g(1)*g(2) );
    defined_list.set(2);
  }
  return (_detg);
}

// Dg_ij,k
double HartleThorne::Dg(int i)
{
  // If not defined, define Dg
  if (!defined_list[3]) { 
    double _Sigma2 = _Sigma*_Sigma;
    // ORIGINAL
    double _poly = - 5.0*_r*_ep * (_2M * (2.0*_M2 - 3.0*_M*_r - 3.0*_r2) + (3.0*_r3 - 6.0*_M2*_r)*_lg ) / (32.0*_M2);
    double _poly2 =  5.0*_ep*(_r + _M)*_c2s * (_2M * (_M2 - 6.0*_M*_r + 3.0*_r2) - 3.0*_r * (2.0*_M2 - 3.0*_M*_r + _r2)*_lg ) / (8.0*_M2*_sw);

    _Dg[0] = -_2M*(_r2 - _a2*_cs2)/_Sigma2 - 5.0*_ep*_c2s * (_2M * (2.0*_M3 + 2.0*_M2*_r + 3.0*_M*_r2 - 3.0*_r3) + 3.0*_r3*_sw*_lg ) / (16.0*_M2*_r3);

    _Dg[1] = - ( ( 2.0*(_r-_M)*_Sigma - 2.0*_r*_Delta )/_Sigma2 + 5.0*_ep*_c2s * (_2M * (2.0*_M3 + 2.0*_M2*_r + 3.0*_M*_r2 - 3.0*_r3) + 3.0*_r3*_sw*_lg) / (16.0*_M2*_r3) ) * (g(1)*g(1));

    _Dg[2] = 2.0*_r + _poly2;
    _Dg[3] = 2.0*_sn2 * ( (_r-_M)*_Sigma2 + _M*_r2*_Sigma + _M*_a2*_ra2*_cs2 - _M*_a2*_r2*_sn2 ) / _Sigma2 + _poly2*_sn2;
    _Dg[4] = _Dg[5] = _2M*_a*_sn2*(_r2 - _a2*_cs2) / _Sigma2;
    _Dg[6] = 4.0*_r*_a2*_M*_sncs/_Sigma2 + 15.0*_ep * (_2M*(_r-_M) * (2.0*_M2 + 6.0*_M*_r - 3.0*_r2) + 3.0*_r2*_sw2*_lg)*_sncs / (8.0*_M2*_r2);

    _Dg[7] = - ( 2.0*_a2*_Delta*_sncs/_Sigma2 + 15.0*_ep*_sncs * (_2M * (2.0*_M3 + 4.0*_M2*_r - 9.0*_M*_r2 + 3.0*_r3) - 3.0*_r2*_sw2*_lg) / (8.0*_M2*_r2)) * (g(1)*g(1));

    _Dg[8] = -2.0*_a2*_sncs - 12.0*_sncs*_poly;

    _Dg[9] = 2.0*_sncs * ( _Delta*_Sigma2 + _2M*_r*_ra2*_ra2 ) /_Sigma2 - 8.0*_poly* _sncs*(2.0 - 3.0*_cs2);

    _Dg[10] = _Dg[11] = -4.0*_a*_r*_M*_ra2*_sncs/_Sigma2;

    defined_list.set(3);
  }
  return (_Dg[i]);
}
 
// Dg^ij,k
double HartleThorne::Dginv(int i)
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
double HartleThorne::Gamma(int i)
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
double HartleThorne::horizon() const
{
  return ( _rh );
}
double HartleThorne::rISCO() const
{
  return ( _risco );
}

// Parameters
double HartleThorne::mass() const
{
  return ( _M );
}

double HartleThorne::ang_mom() const
{
  return ( _a*_M );
}

double HartleThorne::quad_mom() const
{
  return ( -_M * (_a*_a + _ep*_M*_M ) );
}



double HartleThorne::get_ISCO()
{
  // Finds ISCOs as long as they are above x1=4.0.  Inside of this point
  //  bad shit can happen for the HT metric.
  double x1 = 4;
  double x2 = 15.0;

  if (ISCOFunc(x1)*ISCOFunc(x2)>0.0)
  {
    std::cerr << "Couldn't Find ISCO, using 4.0\n";
    return 4.0;
  }
  else
  {
    //return rtflsp(x1,x2,1e-6);
    return zriddr(x1,x2,1e-6);
  }
}


double HartleThorne::ISCOFunc(double r)
{
  // ISCO defined by the zero of the 
  double dr=1e-8;
  double gttr, gppr, gtpr;

  reset(0,r+dr,0.5*M_PI,0);
  gttr = Metric::Dginv(0,0,1);
  gppr = Metric::Dginv(3,3,1);
  gtpr = Metric::Dginv(0,3,1);
  double lp = -gtpr/gppr - sqrt( std::pow(gtpr/gppr,2.0) - gttr/gppr );

  reset(0,r-dr,0.5*M_PI,0);
  gttr = Metric::Dginv(0,0,1);
  gppr = Metric::Dginv(3,3,1);
  gtpr = Metric::Dginv(0,3,1);
  double lm = -gtpr/gppr - sqrt( std::pow(gtpr/gppr,2.0) - gttr/gppr );
  
  return ( (lp-lm)/(2.0*dr) );

}







#define MAXIT 10000
 
double HartleThorne::rtflsp(double x1, double x2, double xacc)
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
 
double HartleThorne::zriddr(double x1, double x2, double xacc)
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


};
