/*!
  \file model_image_gaussian.cpp
  \author Avery Broderick
  \date  June, 2017
  \brief Implements symmetric Gaussian image class.
  \details To be added
*/

#include "model_symmetric_gaussian.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace Themis {

model_symmetric_gaussian::model_symmetric_gaussian()
  : _Itotal(0.0), _sigma(0.0)
{
}

void model_symmetric_gaussian::generate_model(std::vector<double> parameters)
{
  _Itotal = parameters[0];
  _sigma = parameters[1];
}
  
double model_symmetric_gaussian::closure_phase(datum_closure_phase&, double)
{
  return 0.;
}

std::complex<double> model_symmetric_gaussian::visibility(datum_visibility& d, double acc)
{
  double u2 = 4.0*M_PI*M_PI*(d.u*d.u+d.v*d.v) * _sigma*_sigma;
    
  double VM = _Itotal * std::exp( - 0.5 * u2 );
    
  return ( std::complex<double>(VM,0.0) );
}

double model_symmetric_gaussian::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{
  double u2 = 4.0*M_PI*M_PI*(d.u*d.u+d.v*d.v) * _sigma*_sigma;
    
  double VM = _Itotal * std::exp( - 0.5 * u2 );
    
  return ( VM );
}

double model_symmetric_gaussian::closure_amplitude(datum_closure_amplitude& d, double acc)
{
  double u2 = 4.0*M_PI*M_PI*( (d.u1*d.u1+d.v1*d.v1) + (d.u3*d.u3+d.v3*d.v3) - (d.u2*d.u2+d.v2*d.v2) - (d.u4*d.u4+d.v4*d.v4) ) * _sigma*_sigma;
  
  double VM = std::exp( - 0.5 * u2 );
    
  return ( VM );
}

};
