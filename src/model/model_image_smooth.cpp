/*!
    \file model_image_smooth.cpp  
    \author Avery Broderick & Hung-Yi Pu
    \date Jun 2018
    \brief Header file for ensemble averaged, parameterized scattering interface.
    \details 
*/

#include "model_image_smooth.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace Themis {

model_image_smooth::model_image_smooth(model_image& model)
    : _model(model), _smoothing_parameters(3,0.0)
{
}

std::string model_image_smooth::model_tag() const
{
  std::stringstream tag;  
  tag << "model_image_smooth\n";
  tag << "SUBTAG START\n";
  tag << _model.model_tag() << '\n';
  tag << "SUBTAG FINISH";
  
  return tag.str();
}

void model_image_smooth::generate_model(std::vector<double> parameters)
{
  // Check to see if these differ from last set used.
  if (_generated_model && parameters==_current_parameters)
    return;
  else
  {
    _current_parameters = parameters;

    
    // Grab the last 3 parameters for the smoothing kernel
    for (int j=0; j<3; ++j)
    {
      _smoothing_parameters[2-j] = parameters[parameters.size()-1];
      parameters.pop_back();
    }
    // Generate the intrinsic model
    _model.generate_model(parameters);
    _generated_model=true;
    _generated_visibilities=false;

  }
}

std::complex<double> model_image_smooth::visibility(datum_visibility& d, double acc)
  {  
    
    _smoothing_parameters[0] = std::fabs(_smoothing_parameters[0]);
    _smoothing_parameters[1] = std::min(std::max(_smoothing_parameters[1],0.0),0.999);

    
    double sig_major = _smoothing_parameters[0] * std::sqrt( 1.0 / (1.0-_smoothing_parameters[1]) ); // Major axis smoothing_parameters[0];
    double sig_minor = _smoothing_parameters[0] * std::sqrt( 1.0 / (1.0+_smoothing_parameters[1]) ); // Minor axis smoothing_parameters[1];

    // Position angle relative to East of the minor axis (Major axis is 78 degrees East of North)
    double PA_minor_rad = _smoothing_parameters[2];
  
    // The properly normalized and major and minor scattering sigmas (in rad)
    // The associated sigmas in the u-v plane (in m)
    double uv_sig_major = 1.0/(2.0*M_PI*sig_major); //* lambda;		//we want it in units of lambda
    double uv_sig_minor = 1.0/(2.0*M_PI*sig_minor); //* lambda;		//we want it in units of lambda


    double uv_minor = d.u*std::cos(PA_minor_rad) + d.v * std::sin(PA_minor_rad);
    double uv_major = -d.u * std::sin(PA_minor_rad) + d.v*std::cos(PA_minor_rad) ;
    double scattering_kernel = std::exp(-0.5*std::pow(uv_major/uv_sig_major,2) - 0.5*std::pow(uv_minor/uv_sig_minor,2));

    return ( scattering_kernel*_model.visibility(d, acc) );
  
  }

  double model_image_smooth::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    datum_visibility tmp(d.u,d.v,std::complex<double>(d.V,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);

    return ( std::abs(visibility(tmp,acc)) );
  }


  double model_image_smooth::closure_phase(datum_closure_phase& d, double acc)
  {
    return (_model.closure_phase( d, acc));
  }




double model_image_smooth::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
    datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);

    double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
    return ( V1234 );
  }

void model_image_smooth::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "model_image_smooth::generate_image : This function is not implemented because because the smoothing is applied in visibility space.\n";
    std::exit(1);
  } 


void model_image_smooth::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}


};
