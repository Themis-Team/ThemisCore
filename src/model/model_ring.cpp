/*!
  \file model_ring.cpp
  \author Jorge A. Preciado
  \date  February 2018
  \brief Implements a ring model class.
  \details To be added
*/

#include "model_ring.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace Themis {

model_ring::model_ring()
  : _V0(0.0), _Rext(0.0), _Rint(0.0)
{
}


void model_ring::generate_model(std::vector<double> parameters)
{
  _V0 = parameters[0];
  _Rext = parameters[1];
  _Rint = (1 - parameters[2]) * parameters[1];
}


double model_ring::BesselJ1(double x)
{
  double ax,z;
  double xx,y,ans,ans1,ans2;

  if ((ax=std::fabs(x)) < 8.0) {
    y=x*x;
    ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
      +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
    ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
      +y*(99447.43394+y*(376.9991397+y*1.0))));
    ans=ans1/ans2;
  } else {
    z=8.0/ax;
    y=z*z;
    xx=ax-2.356194491;
    ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
      +y*(0.2457520174e-5+y*(-0.240337019e-6))));
    ans2=0.04687499995+y*(-0.2002690873e-3
      +y*(0.8449199096e-5+y*(-0.88228987e-6
      +y*0.105787412e-6)));
    ans=std::sqrt(0.636619772/ax)*(std::cos(xx)*ans1-z*std::sin(xx)*ans2);
    if (x < 0.0) ans = -ans;
  }
  return ans;

}


std::complex<double> model_ring::complex_visibility(double u, double v)
{    
  double k = 2.* M_PI * std::sqrt(u*u + v*v); //+ 1.e-15*_Rext;
  std::complex<double> V;
  
  V = 2.*_V0/(k*(_Rext*_Rext - _Rint*_Rint)) * (_Rext*BesselJ1(k * _Rext) - _Rint*BesselJ1(k*_Rint));
  
  return ( V );
}


double model_ring::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{    
  double VM = std::abs(complex_visibility(d.u, d.v));    
  return ( VM );
}


double model_ring::closure_phase(datum_closure_phase& d, double acc)
{
  std::complex<double> V123 = complex_visibility(d.u1,d.v1)*complex_visibility(d.u2,d.v2)*complex_visibility(d.u3,d.v3);
  return ( std::imag(std::log(V123))*180.0/M_PI );
}


double model_ring::closure_amplitude(datum_closure_amplitude& d, double acc)
{
  double V1234 = std::abs(complex_visibility(d.u1,d.v1)*complex_visibility(d.u3,d.v3)/complex_visibility(d.u2,d.v2)*complex_visibility(d.u4,d.v4));
  return ( V1234 );  
}

};
