/*!
    \file model_ensemble_averaged_parameterized_scattered_image.cpp  
    \author Avery Broderick & Hung-Yi Pu
    \date Jun 2018
    \brief Header file for ensemble averaged, parameterized scattering interface.
    \details 
*/

#include "model_ensemble_averaged_parameterized_scattered_image.h"
#include <cmath>
#include <iostream>
#include <iomanip>

namespace Themis {

model_ensemble_averaged_parameterized_scattered_image::model_ensemble_averaged_parameterized_scattered_image(model_visibility_amplitude& model, double pivot_frequency)
    : _model(model), _scattering_parameters(7,0.0), _pivot_frequency(pivot_frequency)
{
}

void model_ensemble_averaged_parameterized_scattered_image::generate_model(std::vector<double> parameters)
{
  // Grab the last 7 parameters for the scattering screen
  for (int j=0; j<7; ++j)
  {
    _scattering_parameters[6-j] = parameters[parameters.size()-1];
    parameters.pop_back();
  }
  // Generate the intrinsic model
  _model.generate_model(parameters);
}

void model_ensemble_averaged_parameterized_scattered_image::set_mpi_communicator(MPI_Comm comm)
{
  _model.set_mpi_communicator(comm);
}



double model_ensemble_averaged_parameterized_scattered_image::visibility_amplitude(datum_visibility_amplitude& d, double acc)
{

  // Major and minor scattering axes in mas cm^-2
  double uas2rad = 1e-6/3600.*M_PI/180.;
  double sigscat_major = _scattering_parameters[0] * std::pow(d.frequency/_pivot_frequency, -_scattering_parameters[1]) * uas2rad;
  double sigscat_minor = _scattering_parameters[2] * std::pow(d.frequency/_pivot_frequency, -_scattering_parameters[3]) * uas2rad;

  // Position angle relative to East of the minor axis (Major axis is 78 degrees East of North)
  double PAscat_minor_rad = ( _scattering_parameters[4] + _scattering_parameters[5]*( std::pow(d.frequency/_pivot_frequency,-_scattering_parameters[6]) - 1.0 ) ) * M_PI/180.0; // in radians
  
  // The properly normalized and major and minor scattering sigmas (in rad)
  // The associated sigmas in the u-v plane (in m)
  double uv_sigscat_major = 1.0/(2.0*M_PI*sigscat_major); //* lambda;		//we want it in units of lambda
  double uv_sigscat_minor = 1.0/(2.0*M_PI*sigscat_minor); //* lambda;		//we want it in units of lambda

  // Vij = V_in[i][j]*exp(-pow(uv_major/uv_sigscat_major,2.0)/2.0
  // 		       -pow(uv_minor/uv_sigscat_minor,2.0)/2.0)

  double uv_minor = d.u*std::cos(PAscat_minor_rad) + d.v * std::sin(PAscat_minor_rad);
  double uv_major = -d.u * std::sin(PAscat_minor_rad) + d.v*std::cos(PAscat_minor_rad) ;
  double scattering_kernel = std::exp(-0.5*std::pow(uv_major/uv_sigscat_major,2) - 0.5*std::pow(uv_minor/uv_sigscat_minor,2));

  // TODO:
  // Using interpolation class, bicubic, maybe replace with type 1
  // Final two parameters are extraneous and should be stripped out, relating to the
  // Galactic center scattering screen, which now is dealt with elsewhere
  return ( scattering_kernel*_model.visibility_amplitude(d, acc) );
}

};
