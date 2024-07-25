/*!
    \file model_gaussian_scattered_image.cpp  
    \author Avery Broderick
    \date Dec 2017
    \brief Header file for gaussian scattering interface (like model_ensemble_averaged_scattered_image, but with parameters that define the Gaussian and its wavelength dependence).
    \details 
*/

#include "model_gaussian_scattered_image.h"
#include <cmath>

#include <iostream>
#include <iomanip>

namespace Themis {

  model_gaussian_scattered_image::model_gaussian_scattered_image(model_visibility_amplitude& model, double pivot_frequency)
    : _model(model), _generated_model(false), _screen_params(7), _pivot_frequency(pivot_frequency) 
  {
  }

  void model_gaussian_scattered_image::generate_model(std::vector<double> parameters)
  {
    std::vector<double> model_params(_model.size());
    for (size_t j=0; j<_model.size(); ++j)
      model_params[j] = parameters[j];
    for (size_t j=0; j<size()-_model.size(); ++j)
      _screen_params[j] = parameters[j+_model.size()];

    _model.generate_model(parameters);

    _generated_model = true;
  }

  void model_gaussian_scattered_image::set_mpi_communicator(MPI_Comm comm)
  {
    _model.set_mpi_communicator(comm);
  }

  double model_gaussian_scattered_image::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    // The properly normalized and major and minor scattering sigmas (in rad)
    double sigscat_major = _screen_params[0] * std::pow(d.frequency/_pivot_frequency,-_screen_params[1]);
    double sigscat_minor = _screen_params[2] * std::pow(d.frequency/_pivot_frequency,-_screen_params[3]);

    // Position angle relative to East of the minor axis
    double PAscat_minor_rad = _screen_params[4] + _screen_params[5] * std::pow(d.frequency/_pivot_frequency,-_screen_params[6]);

    // The associated sigmas in the u-v plane (in m)
    double uv_sigscat_major = 1.0/(2.0*M_PI*sigscat_major); //* lambda;		//we want it in units of lambda
    double uv_sigscat_minor = 1.0/(2.0*M_PI*sigscat_minor); //* lambda;		//we want it in units of lambda
    
    // Vij = V_in[i][j]*exp(-pow(uv_major/uv_sigscat_major,2.0)/2.0
    // 		       -pow(uv_minor/uv_sigscat_minor,2.0)/2.0)    
    double uv_minor = d.u*std::cos(PAscat_minor_rad) + d.v * std::sin(PAscat_minor_rad);
    double uv_major = -d.u * std::sin(PAscat_minor_rad) + d.v*std::cos(PAscat_minor_rad) ;
    double scattering_kernel = std::exp(-0.5*std::pow(uv_major/uv_sigscat_major,2) - 0.5*std::pow(uv_minor/uv_sigscat_minor,2));
    
    return ( scattering_kernel*_model.visibility_amplitude(d, acc) );
  }
};
