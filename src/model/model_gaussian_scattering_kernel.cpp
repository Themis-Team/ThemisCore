 /*!
  \file model_image_asymmetric_gaussian.h
  \author Avery Broderick & Hung-Yi Pu
  \date  December, 2017
  \brief Implements wavelength-dependent, asymmetric Gaussian image class.
  \details To be added
*/

#include "model_gaussian_scattering_kernel.h"
#include <cmath>


namespace Themis {

  model_gaussian_scattering_kernel::model_gaussian_scattering_kernel(std::vector<double> frequencies, double pivot_frequency)
    : _frequencies(frequencies), _images(frequencies.size()), _generated_model(false), _pivot_frequency(pivot_frequency)
  {
    for (size_t j=0; j<_frequencies.size(); ++j)
      _images[j].use_analytical_visibilities();
  }


  void model_gaussian_scattering_kernel::generate_model(std::vector<double> parameters)
  {
    _parameters = parameters;
    _generated_model = true;
  }

  std::vector<double> model_gaussian_scattering_kernel::generate_asymmetric_gaussian_parameter_list(double frequency) const
  {
    double sigA = _parameters[0] * std::pow(frequency/_pivot_frequency,-_parameters[1]);
    double sigB = _parameters[2] * std::pow(frequency/_pivot_frequency,-_parameters[3]);
    double xi = _parameters[4] + _parameters[5] * ( std::pow(frequency/_pivot_frequency,-_parameters[6]) - 1.0 );

    std::vector<double> params(4);
    params[0] = 1.0; // No intensity
    params[1] = std::sqrt( 2.0 /( 1.0/(sigA*sigA) + 1.0/(sigB*sigB) ) );
    params[2] = std::fabs( (sigA*sigA) - (sigB*sigB) ) / ( (sigA*sigA) + (sigB*sigB) );
    params[3] = xi + ( sigA>sigB ? 0 : 0.5*M_PI );
    
    return params;
  }

  
  double model_gaussian_scattering_kernel::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
  {
    size_t ii=find_frequency_index(d.frequency);
    if (_generated_model)
    {
      std::vector<double> params = generate_asymmetric_gaussian_parameter_list(_frequencies[ii]);
      _images[ii].generate_model(params);
      return ( _images[ii].visibility_amplitude(d,accuracy) );
    } 
    else 
    {
      std::cerr << "model_riaf::visibility_amplitude : Must generate model before visibility_amplitude\n";
      std::exit(1);
    }
  }

  double model_gaussian_scattering_kernel::closure_phase(datum_closure_phase& d, double accuracy)
  {
    size_t ii=find_frequency_index(d.frequency);
    if (_generated_model)
    {
      std::vector<double> params = generate_asymmetric_gaussian_parameter_list(_frequencies[ii]);
      _images[ii].generate_model(params);
      return ( _images[ii].closure_phase(d,accuracy) );
    } 
    else 
    {
      std::cerr << "model_riaf::closure_amplitude : Must generate model before closure_amplitude\n";
      std::exit(1);
    }
  }

    double model_gaussian_scattering_kernel::closure_amplitude(datum_closure_amplitude& d, double accuracy)
  {
    size_t ii=find_frequency_index(d.frequency);
    if (_generated_model)
    {
      std::vector<double> params = generate_asymmetric_gaussian_parameter_list(_frequencies[ii]);
      _images[ii].generate_model(params);
      return ( _images[ii].closure_amplitude(d,accuracy) );
    } 
    else 
    {
      std::cerr << "model_riaf::closure_amplitude : Must generate model before closure_amplitude\n";
      std::exit(1);
    }
  }


  size_t model_gaussian_scattering_kernel::find_frequency_index(double frequency) const
  {
    size_t imin=0;
    double dfmin=0;
    for (size_t i=0; i<_frequencies.size(); ++i)
    {
      double df = std::fabs(std::log(frequency/_frequencies[i]));
      if (i==0 || df<dfmin)
      {
	imin=i;
	dfmin=df;
      }
    }
    return imin;
  }


};
