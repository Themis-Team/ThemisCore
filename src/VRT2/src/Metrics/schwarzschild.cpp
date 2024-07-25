// Include Header
#include "schwarzschild.h"

namespace VRT2 {

Schwarzschild::Schwarzschild(double Mass)
  : Metric(4,5,13), _M(Mass)
{
  initialize();
  mk_hash_table();
  check_compiler_optimization();
}

// One time initialization stuff that must be virtual
void Schwarzschild::initialize()
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

  Dgi[4] = 3;
  Dgj[4] = 3;
  Dgk[4] = 2;

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

  Gi[2] = 1;
  Gj[2] = 0;
  Gk[2] = 0;

  Gi[3] = 1;
  Gj[3] = 1;
  Gk[3] = 1;

  Gi[4] = 1;
  Gj[4] = 2;
  Gk[4] = 2;

  Gi[5] = 1;
  Gj[5] = 3;
  Gk[5] = 3;

  Gi[6] = 2;
  Gj[6] = 1;
  Gk[6] = 2;

  Gi[7] = 2;
  Gj[7] = 2;
  Gk[7] = 1;

  Gi[8] = 2;
  Gj[8] = 3;
  Gk[8] = 3;

  Gi[9] = 3;
  Gj[9] = 1;
  Gk[9] = 3;

  Gi[10] = 3;
  Gj[10] = 3;
  Gk[10] = 1;

  Gi[11] = 3;
  Gj[11] = 2;
  Gk[11] = 3;

  Gi[12] = 3;
  Gj[12] = 3;
  Gk[12] = 2;
}

// Position dependent funcs used by g and ginv
void Schwarzschild::get_fcns()
{
  _r = _x[1];
  _sn = std::sin(_x[2]);
  _sigma = (1.0 - 2.0*_M/_r);
} 

/*** Elements ***/
// g_ij (For expedience assume i and j are  in (gi,gj))
double Schwarzschild::g(int i)
{
  // If not defined, define g
  if (!defined_list[0]) { 
    _g[0] = -_sigma;
    _g[1] = 1.0/_sigma;
    _g[2] = _r*_r;
    _g[3] = _r*_r*_sn*_sn;
    defined_list.set(0);
  }
  return (_g[i]);
}
 
// g^ij (For expedience assume i and j are  in (gi,gj))
double Schwarzschild::ginv(int i)
{
  // If not defined, define ginv
  if (!defined_list[1]) { 
    _ginv[0] = -1.0/_sigma;
    _ginv[1] = _sigma;
    _ginv[2] = 1.0/(_r*_r);
    _ginv[3] = 1.0/(_r*_r*_sn*_sn);
    defined_list.set(1);
  }
  return (_ginv[i]);
} 

// sqrt(-det(g))
double Schwarzschild::detg()
{
  // If not defined, define detg
  if (!defined_list[2]) { 
    _detg = _r*_r*_sn;
    defined_list.set(2);
  }
  return (_detg);
}

// Dg_ij,k
double Schwarzschild::Dg(int i)
{
  // If not defined, define Dg
  if (!defined_list[3]) { 
    _Dg[0] = - 2.0*_M/(_r*_r);
    _Dg[1] = - 2.0*_M / (_r*_r*_sigma*_sigma);
    _Dg[2] = 2.0*_r;
    _Dg[3] = 2.0*_r*_sn*_sn;
    _Dg[4] = 2.0*_r*_r*_sn*std::cos(_x[2]);
    defined_list.set(3);
  }
  return (_Dg[i]);
}
 
// Dg^ij,k
double Schwarzschild::Dginv(int i)
{
  // If not defined, define Dg
  if (!defined_list[4]) { 
    _Dginv[0] = 2.0*_M/(_r*_r*_sigma*_sigma);
    _Dginv[1] = 2.0*_M/(_r*_r);
    _Dginv[2] = -2.0/(_r*_r*_r);
    _Dginv[3] = -2.0/(_r*_r*_r*_sn*_sn);
    _Dginv[4] = -2.0*std::cos(_x[2])/(_r*_r*_sn*_sn*_sn);
    defined_list.set(4);
  }
  return (_Dginv[i]);
} 

// Gamma^i_jk
double Schwarzschild::Gamma(int i)
{
  // If not defined, define Gamma
  if (!defined_list[5]) { 
    _Gamma[0] = _Gamma[1] = _M/(_r*_r*_sigma);
    _Gamma[2] = _M*_sigma/(_r*_r);
    _Gamma[3] = -_M/(_r*_r*_sigma);
    _Gamma[4] = -_r*_sigma;
    _Gamma[5] = -_r*_sigma*_sn*_sn;
    _Gamma[6] = _Gamma[7] = 1.0/_r;
    _Gamma[8] = -_sn*std::cos(_x[2]);
    _Gamma[9] = _Gamma[10] = 1.0/_r;
    _Gamma[11] = _Gamma[12] = std::cos(_x[2])/_sn;
    defined_list.set(5);
  }
  return (_Gamma[i]);
}

// Horizon
double Schwarzschild::horizon() const
{
  return ( 2.0*_M );
}

// Parameters
double Schwarzschild::mass() const
{
  return ( _M );
}

};
