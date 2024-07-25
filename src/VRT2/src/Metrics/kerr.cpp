// Include Header
#include "kerr.h"

namespace VRT2 {

// Metric Parameters
Kerr::Kerr(double Mass, double Spin)
  : Metric(6,12,32), _M(Mass), _a(Spin*Mass)
{
  initialize();
  mk_hash_table();
  check_compiler_optimization();
}

// One time initialization stuff that must be virtual
void Kerr::initialize()
{
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
void Kerr::get_fcns()
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
} 

/*** Elements ***/
// g_ij (For expedience assume i and j are  in (gi,gj))
double Kerr::g(int i)
{
  // If not defined, define g
  if (!defined_list[0]) { 
    _g[0] = -(_Delta - _a2*_sn2)/_Sigma;
    _g[1] = _Sigma / _Delta;
    _g[2] = _Sigma;
    _g[3] = ( _ra2*_ra2 - _Delta * _a2 * _sn2) * _sn2/ _Sigma;
    _g[4] = _g[5] = - 2.0*_M*_a*_r*_sn2 / _Sigma;
    defined_list.set(0);
  }
  return (_g[i]);
}
 
// g^ij (For expedience assume i and j are  in (gi,gj))
double Kerr::ginv(int i)
{
  // If not defined, define ginv
  if (!defined_list[1]){
    _ginv[0] = -(_ra2*_ra2 - _a2*_Delta*_sn2) / (_Sigma*_Delta);
    _ginv[1] = _Delta / _Sigma;
    _ginv[2] = 1.0 / _Sigma;
    _ginv[3] = (_Delta - _a2*_sn2) / (_Sigma*_Delta*_sn2);
    _ginv[4] = _ginv[5] = -2.0*_M*_a*_r / (_Sigma*_Delta);
    defined_list.set(1);
  }
  return (_ginv[i]);
} 

// sqrt(-det(g))
double Kerr::detg()
{
  // If not defined, define detg
  if (!defined_list[2]){
    _detg = _Sigma*_sn;
    defined_list.set(2);
  }
  return (_detg);
}

// Dg_ij,k
double Kerr::Dg(int i)
{
  // If not defined, define Dg
  if (!defined_list[3]){
    double _Sigma2 = _Sigma*_Sigma;
    _Dg[0] = -2.0*_M*(_r2 - _a2*_cs2)/_Sigma2;
    _Dg[1] = 2.0*( _r*_Delta - (_r-_M)*_Sigma )/(_Delta*_Delta);
    _Dg[2] = 2.0*_r;
    _Dg[3] = 2.0*_sn2 
      * ( (_r-_M)*_Sigma2 + _M*_r2*_Sigma 
	  + _M*_a2*_ra2*_cs2 
	  - _M*_a2*_r2*_sn2 ) / _Sigma2;
    _Dg[4] = _Dg[5] = 2.0*_a*_M*_sn2*(_r2 - _a2*_cs2)
      /_Sigma2;

    _Dg[6] = 4.0*_r*_a2*_M*_sncs/_Sigma2;
    _Dg[7] = -2.0*_a2*_sncs/_Delta;
    _Dg[8] = -2.0*_a2*_sncs;
    _Dg[9] = 2.0*_sncs * ( _Delta*_Sigma2 + 2.0*_M*_r*_ra2*_ra2 )
      /_Sigma2;
    _Dg[10] = _Dg[11] = -4.0*_a*_r*_M*_ra2*_sncs/_Sigma2;
    defined_list.set(3);
  }
  return (_Dg[i]);
}
 
// Dg^ij,k
double Kerr::Dginv(int i)
{
  // If not defined, define Dg
  if (!defined_list[4]){
    double _Sigma2 = _Sigma*_Sigma;
    double _Delta2 = _Delta*_Delta;
    _Dginv[0] = 2.0*_M 
      * ( _ra2*_ra2*(_r2-_a2*_cs2) - 4.0*_r2*_r*_a2*_M*_sn2 )
      / ( _Sigma2*_Delta2 );
    _Dginv[1] = ( 2.0*(_r-_M)*_Sigma - 2.0*_r*_Delta )/_Sigma2;
    _Dginv[2] = - 2.0*_r/_Sigma2;
    _Dginv[3] = 2.0 * 
      ( (_M-_r)*_Sigma2 + (_r2-_a2)*_M*_Sigma 
	+ 2.0*_M*_r2*_Delta ) /( _Sigma2*_Delta2*_sn2 );
    _Dginv[4] = _Dginv[5] = 2.0*_a*_M *
      ( _Sigma*(_r2-_a2) + 2.0*_r2*_Delta )/( _Sigma2*_Delta2 );


    _Dginv[6] = -4.0*_a2*_r*_M*_sncs*_ra2/(_Sigma2*_Delta);
    _Dginv[7] = 2.0*_a2*_Delta*_sncs/_Sigma2;
    _Dginv[8] = 2.0*_a2*_sncs/_Sigma2;
    _Dginv[9] = 2.0*_cs * ( 2.0*_M*_r*(_Sigma-_a2*_sn2) - _Sigma2 )
      /(_Sigma2*_Delta*_sn2*_sn);
    _Dginv[10] = _Dginv[11] = - 4.0*_a2*_a*_r*_M*_sncs
      /(_Sigma2*_Delta);
    defined_list.set(4);
  }
  return (_Dginv[i]);
} 

// Gamma^i_jk
double Kerr::Gamma(int i)
{
  // If not defined, define Gamma
  if (!defined_list[5]){
    double _Sigma2 = _Sigma*_Sigma;
    double _mSigma = _r2-_a2*_cs2;
    _Gamma[0] = _Gamma[1] = _M * _ra2*_mSigma/ ( _Delta*_Sigma2 );
    _Gamma[2] = _Gamma[3] = -2.0*_M*_a2*_r*_sncs/_Sigma2;
    _Gamma[4] = _Gamma[5] = _M*_a*_sn2
      *( (_r2-_a2)*_mSigma - 4.0*_r2*_r2 )/( _Delta*_Sigma2 );
    _Gamma[6] = _Gamma[7] = 2.0*_M*_a*_a2*_r*_sn2*_sncs
      /_Sigma2;
    _Gamma[8] = _M*_Delta*_mSigma/(_Sigma*_Sigma2);
    _Gamma[9] = _Gamma[10] = -_M*_a*_sn2*_mSigma*_Delta
      /(_Sigma*_Sigma2);
    _Gamma[11] = -( _r*(_r2-_a2) - (_r-_M)*_mSigma )
      /( _Delta*_Sigma );
    _Gamma[12] = _Gamma[13] = -_a2*_sncs/_Sigma;
    _Gamma[14] = -_r*_Delta/_Sigma;
    _Gamma[15] = _Delta*_sn2*( 2.0*_M*_r2*_ra2
			       - (_r-_M)*_Sigma2
			       - _M*(2.0*_r2+_ra2)*_Sigma )
      / (_Sigma*_Sigma2);
    _Gamma[16] = -2.0*_M*_a2*_r*_sncs/(_Sigma*_Sigma2);
    _Gamma[17] = _Gamma[18] = 2.0*_M*_a*_r*_ra2*_sncs
      /(_Sigma*_Sigma2);
    _Gamma[19] = _a2*_sncs/( _Delta*_Sigma );
    _Gamma[20] = _Gamma[21] = _r/_Sigma;
    _Gamma[22] = -_a2*_sncs/_Sigma;
    _Gamma[23] = - _sncs * ( _Delta*_Sigma2 + 2.0*_M*_r*_ra2*_ra2 )
      /(_Sigma*_Sigma2);
    _Gamma[24] = _Gamma[25] = _M*_a*_mSigma/(_Delta*_Sigma2);
    _Gamma[26] = _Gamma[27] = -2.0*_M*_a*_r*_cs/(_sn*_Sigma2);
    _Gamma[28] = _Gamma[29] = ( _r*(_Sigma-2.0*_M*_r)*_Sigma
				-_M*_mSigma*_a2*_sn2 )
      /(_Delta*_Sigma2);
    _Gamma[30] = _Gamma[31] = (_Sigma2+2.0*_M*_a2*_r*_sn2)
      * _cs/(_sn*_Sigma2);
    defined_list.set(5);
  }
  return (_Gamma[i]);
} 

// Horizon
double Kerr::horizon() const
{
  return ( _M + _M*std::sqrt(1.0-_a*_a) );
}

// Parameters
double Kerr::mass() const
{
  return ( _M );
}

double Kerr::ang_mom() const
{
  return ( _a*_M );
}


double Kerr::rISCO() const
{
  // From Bardeen 1972
  
  //set _Z1 = ( 1 + (1-_a**2)**(1.0/3.0) * (  (1+_a)**(1.0/3.0) + (1.0-_a)**(1.0/3.0) ) )
  //set _Z2 = ( sqrt(3*_a**2+_Z1**2) )
  //set $5 = ( 3 + _Z2 - $_sgn*sqrt( (3-_Z1)*(3+_Z1+2*_Z2) ) )

  double a2 = _a*_a;						    
  double Z1 = 1.0 + std::pow(1.0-a2,1.0/3.0)*( std::pow(1+_a,1.0/3.0) + std::pow(1-_a,1.0/3.0) );
  double Z2 = std::sqrt(3.0*a2+Z1*Z1);
  
  return ( _M*( 3.0 + Z2 - (_a<0 ? -1 : 1)*std::sqrt( (3.0-Z1)*(3.0+Z1+2.0*Z2) ) ) );
}

};
