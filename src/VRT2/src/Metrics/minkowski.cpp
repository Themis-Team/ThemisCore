// Include Statements
#include "minkowski.h"

namespace VRT2 {

// Constructor
Minkowski::Minkowski()
  : Metric(4,3,9)
{ 
  initialize();
  mk_hash_table();
  check_compiler_optimization();
}

// One time initialization stuff that must be virtual
void Minkowski::initialize()
{
  // Memory management for vector
  _x.resize(4);

  // non-zero minkowski entries
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

  Dgi[0] = 2;
  Dgj[0] = 2;
  Dgk[0] = 1;

  Dgi[1] = 3;
  Dgj[1] = 3;
  Dgk[1] = 1;

  Dgi[2] = 3;
  Dgj[2] = 3;
  Dgk[2] = 2;

  // non-zero Gamma^i_j,k entries
  Gi.resize(NG);
  Gj.resize(NG);
  Gk.resize(NG);
  _Gamma.resize(NG);

  Gi[0] = 1;
  Gj[0] = 2;
  Gk[0] = 2;

  Gi[1] = 1;
  Gj[1] = 3;
  Gk[1] = 3;

  Gi[2] = 2;
  Gj[2] = 1;
  Gk[2] = 2;

  Gi[3] = 2;
  Gj[3] = 2;
  Gk[3] = 1;  

  Gi[4] = 2;
  Gj[4] = 3;
  Gk[4] = 3;

  Gi[5] = 3;
  Gj[5] = 1;
  Gk[5] = 3;

  Gi[6] = 3;
  Gj[6] = 3;
  Gk[6] = 1;

  Gi[7] = 3;
  Gj[7] = 2;
  Gk[7] = 3;

  Gi[8] = 3;
  Gj[8] = 3;
  Gk[8] = 2;
}

// Position dependent funcs used by g and ginv
void Minkowski::get_fcns()
{
  _r = _x[1];
  _sn = std::sin(_x[2]);
}

/*** Elements ***/
// g_ij (For expedience assume i and j are  in (gi,gj))
double Minkowski::g(int i)
{
  // If not defined, define g
  if (!defined_list[0]) {
    _g[0] = -1.0;
    _g[1] = 1.0;
    _g[2] = _r*_r;
    _g[3] = _r*_r*_sn*_sn;
    defined_list.set(0);
  }
  return (_g[i]);
}

// g^ij (For expedience assume i and j are  in (gi,gj))
double Minkowski::ginv(int i)
{
  // If not defined, define ginv
  if (!defined_list[1]) {
    _ginv[0] = -1.0;
    _ginv[1] = 1.0;
    _ginv[2] = 1.0/(_r*_r);
    _ginv[3] = 1.0/(_r*_r*_sn*_sn);
    defined_list.set(1);
  }
  return (_ginv[i]);
}

// sqrt(-det(g))
double Minkowski::detg()
{
  // If not defined, define detg
  if (!defined_list[2]) {
    _detg = _r*_r*_sn;
    defined_list.set(2);
  }
  return (_detg);
}

// Dg_ij,k
double Minkowski::Dg(int i)
{
  // If not defined, define Dg
  if (!defined_list[3]) {
    _Dg[0] = 2*_r;
    _Dg[1] = 2*_r*_sn*_sn;
    _Dg[2] = 2*_r*_r*_sn*std::cos(_x[2]);
    defined_list.set(3);
  }
  return (_Dg[i]);
}

// Dg^ij,k
double Minkowski::Dginv(int i)
{
  // If not defined, define Dginv
  if (!defined_list[4]) {
    _Dginv[0] = -2/(_r*_r*_r);
    _Dginv[1] = -2/(_r*_r*_r*_sn*_sn);
    _Dginv[2] = -2*std::cos(_x[2])/(_r*_r*_sn*_sn*_sn);
    defined_list.set(4);
  }
  return (_Dginv[i]);
}

// Gamma^i_jk
double Minkowski::Gamma(int i)
{
  // If not defined, define Gamma
  if (!defined_list[5]) {
    _Gamma[0] = -_r;
    _Gamma[1] = -_r*_sn*_sn;
    _Gamma[2] = _Gamma[3] = 1/_r;
    _Gamma[4] = -_sn*std::cos(_x[2]);
    _Gamma[5] = _Gamma[6] = 1/_r;
    _Gamma[7] = _Gamma[8] = std::cos(_x[2])/_sn;
    defined_list.set(5);
  }
  return (_Gamma[i]);
}

// Horizon
double Minkowski::horizon() const
{
  return ( 0.0 );				
}

};
