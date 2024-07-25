/*! 
  \file uncertainty_visibility_broken_power_change.cpp
  \author Avery E Broderick
  \date  October, 2020
  \brief Implementation file for the uncertainty_visibility_broken_power_change class
*/


#include "uncertainty_visibility_broken_power_change.h"

#include <iostream>
#include <iomanip>

namespace Themis{

  uncertainty_visibility_broken_power_change::uncertainty_visibility_broken_power_change()
    : _var_at_fixed_baseline(false), _amplitude_baseline(1.0), _logarithmic_ranges(false)
  {
  }

  uncertainty_visibility_broken_power_change::~uncertainty_visibility_broken_power_change()
  {
  }

  void uncertainty_visibility_broken_power_change::generate_uncertainty(std::vector<double> parameters)
  {
    if (_logarithmic_ranges)
    {
      _error_threshold = std::exp(2.0*parameters[0]); // Variance
      _error_fraction = std::exp(2.0*parameters[1]); // Variance
      _error_zero_baseline = std::exp(2.0*parameters[2]); // Variance
      _error_baseline_break = std::exp(parameters[3])*1.0e9; // Positive definite
      _error_baseline_index = parameters[4];
      _error_short_baseline_index = parameters[5];

      /*
      std::cerr << "FOO:"
		<< std::setw(15) << _error_threshold
		<< std::setw(15) << _error_fraction
		<< std::setw(15) << _error_zero_baseline
		<< std::setw(15) << _error_baseline_break
		<< std::setw(15) << _error_baseline_index
		<< std::setw(15) << _error_short_baseline_index
		<< std::endl;
      */
      
    }
    else
    {
      _error_threshold = parameters[0]*parameters[0]; // Variance
      _error_fraction = parameters[1]*parameters[1]; // Variance
      _error_zero_baseline = parameters[2]*parameters[2]; // Variance
      _error_baseline_break = std::fabs(parameters[3])*1.0e9; // Positive definite
      _error_baseline_index = parameters[4];
      _error_short_baseline_index = parameters[5];
    }

    // If specifying the variance at 4 Glambda, then
    if (_var_at_fixed_baseline)
    {
      double umag = _amplitude_baseline/_error_baseline_break;
      _error_zero_baseline *= ( 1.0 + std::pow(umag,_error_short_baseline_index+_error_baseline_index) ) / std::pow(umag,_error_short_baseline_index);
    }
  }
  
  size_t uncertainty_visibility_broken_power_change::size() const
  {
    return 6;
  }

  
  void uncertainty_visibility_broken_power_change::constrain_noise_at_fixed_baseline(double baseline)
  {
    _var_at_fixed_baseline = true;
    _amplitude_baseline = baseline;
    std::cerr << "uncertainty_visibility_broken_power_change using normalization at fixed baselines " << baseline << std::endl;
  }

  void uncertainty_visibility_broken_power_change::constrain_noise_at_4Glambda()
  {
    constrain_noise_at_fixed_baseline(4.0e9);
  }

  void uncertainty_visibility_broken_power_change::logarithmic_ranges()
  {
    _logarithmic_ranges=true;
  }
  
  std::complex<double> uncertainty_visibility_broken_power_change::error(datum_visibility& d)
  {
    double Vmag2 = std::pow(std::abs(d.V),2);
    double umag = std::sqrt(d.u*d.u+d.v*d.v) / _error_baseline_break;
    double broken_power_var = _error_zero_baseline*std::pow(umag,_error_short_baseline_index)/( 1.0 + std::pow(umag,_error_short_baseline_index+_error_baseline_index) );

    double errr = std::sqrt( std::pow(d.err.real(),2) + _error_threshold + _error_fraction*Vmag2 + broken_power_var );
    double erri = std::sqrt( std::pow(d.err.imag(),2) + _error_threshold + _error_fraction*Vmag2 + broken_power_var );
    
    return ( std::complex<double>(errr,erri) );
  }

  double uncertainty_visibility_broken_power_change::log_normalization(datum_visibility& d)
  {
    std::complex<double> err = error(d);
    return -std::log( err.real()*err.imag() );
  }
  
  void uncertainty_visibility_broken_power_change::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
