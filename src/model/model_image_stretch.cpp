/*!
    \file model_image_stretch.cpp  
    \author Paul Tiede
    \date April 2021
    \brief Header file for ensemble averaged, parameterized scattering interface.
    \details 
*/

#include "model_image_stretch.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace Themis {

model_image_stretch::model_image_stretch(model_image& model)
    : _model(model), _smoothing_parameters(2,0.0)
{
}

std::string model_image_stretch::model_tag() const
{
  std::stringstream tag;  
  tag << "model_image_stretch\n";
  tag << "SUBTAG START\n";
  tag << _model.model_tag() << '\n';
  tag << "SUBTAG FINISH";
  
  return tag.str();
}

void model_image_stretch::generate_model(std::vector<double> parameters)
{
  // Check to see if these differ from last set used.
  if (_generated_model && parameters==_current_parameters)
    return;
  else
  {
    _current_parameters = parameters;

    
    // Grab the last 3 parameters for the smoothing kernel
    for (int j=0; j<2; ++j)
    {
      _smoothing_parameters[1-j] = parameters[parameters.size()-1];
      parameters.pop_back();
    }
    // Generate the intrinsic model
    _model.generate_model(parameters);
    _generated_model=true;
    _generated_visibilities=false;

  }
}

std::complex<double> model_image_stretch::visibility(datum_visibility& d, double acc)
  {  
    
    //_smoothing_parameters[0] = std::min(std::max(_smoothing_parameters[0],0.0),0.999);

    
    double semi_major = 1.0/std::sqrt(1-_smoothing_parameters[0]); // Major axis smoothing_parameters[0];
    double semi_minor = std::sqrt(1-_smoothing_parameters[0]);

    // Position angle relative to East of the major axis (Major axis is 78 degrees East of North)
    double PA_minor_rad = _smoothing_parameters[1];
    // The properly normalized and major and minor scattering sigmas (in rad)
    // The associated sigmas in the u-v plane (in m)
    double us = d.u;///semi_minor;
    double vs = d.v;///semi_major;
    double uv_minor = us*std::cos(PA_minor_rad) + vs * std::sin(PA_minor_rad);
    double uv_major = -us * std::sin(PA_minor_rad) + vs*std::cos(PA_minor_rad);
    datum_visibility dmod(uv_minor/semi_minor, uv_major/semi_major, d.V, d.err, d.frequency, d.tJ2000, d.Station1, d.Station2, d.Source);

    return (_model.visibility(dmod, acc) );
  
  }

  double model_image_stretch::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    datum_visibility tmp(d.u,d.v,std::complex<double>(d.V,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);

    return ( std::abs(visibility(tmp,acc)) );
  }


  double model_image_stretch::closure_phase(datum_closure_phase& d, double acc)
  {
    return (_model.closure_phase( d, acc));
  }




double model_image_stretch::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
    datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);

    double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
    return ( V1234 );
  }

void model_image_stretch::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "model_image_stretch::generate_image : This function is not implemented because because the stretching is applied in visibility space.\n";
    std::exit(1);
  } 


void model_image_stretch::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}


};
