/*!
  \file random_number_generator.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements various random number generators using the algorithms in "Numerical Recipes in C" by Flannery, Teukolsky, Press, and Vettering.
  \details To be added
*/

#include "random_number_generator.h"


namespace Themis {

//----------------------------- Implemetations -------------------------------
// System supplied generator
SystemSuppliedRNG::SystemSuppliedRNG(int seed)
{
  std::srand(seed);
}
double SystemSuppliedRNG::rand()
{
  return ( double(std::rand())/(double(RAND_MAX)+1.0) );
}
// Minimal Standard
MinimalStandardRNG::MinimalStandardRNG(int seed)
  : _seed(seed), _IA(16807), _IM(2147483647), _IQ(127773), _IR(2836),
     _MASK(123459876), _AM(1.0/double(_IM))
{
}
double MinimalStandardRNG::rand()
{
  int k;
  double ans;

  _seed ^= _MASK;
  k = _seed/_IQ;
  _seed = _IA*( _seed - k*_IQ ) - _IR*k;
  if ( _seed < 0 )
    _seed += _IM;
  ans = _AM*_seed;
  _seed ^= _MASK;

  return ans;
}
// NR Ran2
Ran2RNG::Ran2RNG(int seed)
  : _idum(seed),
     _IM1(2147483563),
     _IM2(2147483399),
     _IMM1(_IM1-1),
     _IA1(40014),
     _IA2(40692),
     _IQ1(53668),
     _IQ2(52774),
     _IR1(12211),
     _IR2(3791),
     _NTAB(32),
     _NDIV(1+_IMM1/_NTAB),
     _AM(1.0/_IM1),
     _EPS(1.0e-14),
     _RNMX(1.0-_EPS)
{
  _iv = new int[_NTAB];
  if (_idum==0)
    _idum=1;
  _idum2 = _idum;
  int j, k;
  for (j=_NTAB+7; j>=0; j--) {
    k=_idum/_IQ1;
    _idum=_IA1*( _idum - k*_IQ1 ) - k*_IR1;
    if (_idum<0)
      _idum += _IM1;
    if (j<_NTAB)
      _iv[j] = _idum;
  }
  _iy = _iv[0];
}
Ran2RNG::~Ran2RNG()
{
  delete[] _iv;
}
void Ran2RNG::reset_seed(int seed)
{
  _idum = seed;
  _IM1 = 2147483563;
  _IM2 = 2147483399;
  _IMM1 = _IM1-1;
  _IA1 = 40014;
  _IA2 = 40692;
  _IQ1 = 53668;
  _IQ2 = 52774;
  _IR1 = 12211;
  _IR2 = 3791;
  _NTAB = 32;
  _NDIV = 1+_IMM1/_NTAB;
  _AM = 1.0/_IM1;
  _EPS = 1.0e-14;
  _RNMX = 1.0-_EPS;

  if (_idum==0)
    _idum=1;
  _idum2 = _idum;
  int j, k;
  for (j=_NTAB+7; j>=0; j--) {
    k=_idum/_IQ1;
    _idum=_IA1*( _idum - k*_IQ1 ) - k*_IR1;
    if (_idum<0)
      _idum += _IM1;
    if (j<_NTAB)
      _iv[j] = _idum;
  }
  _iy = _iv[0];
}
double Ran2RNG::rand()
{
  int j, k;
  double temp;

  k = _idum/_IQ1;
  _idum = _IA1*( _idum - k*_IQ1 ) - k*_IR1;
  if (_idum<0)
    _idum += _IM1;
  k = _idum2/_IQ2;
  _idum2 = _IA2*( _idum2 - k*_IQ2 ) - k*_IR2;
  if (_idum2<0)
    _idum2 += _IM2;
  j = _iy/_NDIV;
  _iy = _iv[j] - _idum2;
  _iv[j] = _idum;
  if (_iy<1)
    _iy += _IMM1;
  if ((temp=_AM*_iy)>_RNMX)
    return _RNMX;
  else
    return temp;
}

};
