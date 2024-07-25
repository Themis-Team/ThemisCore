/*! 
  \file uncertainty_crosshand_visibilities_broken_power_change.cpp
  \author Avery E Broderick
  \date  March, 2022
  \brief Implementation file for the uncertainty_crosshand_visibilities_broken_power_change class
*/


#include "uncertainty_crosshand_visibilities_broken_power_change.h"
#include <iostream>
#include <iomanip>

namespace Themis{

  uncertainty_crosshand_visibilities_broken_power_change::uncertainty_crosshand_visibilities_broken_power_change()
    : _var_at_fixed_baseline(false), _amplitude_baseline(1.0), _logarithmic_ranges(false), _crosshand_ratio(1.0)
  {
  }

  uncertainty_crosshand_visibilities_broken_power_change::~uncertainty_crosshand_visibilities_broken_power_change()
  {
  }

  void uncertainty_crosshand_visibilities_broken_power_change::generate_uncertainty(std::vector<double> parameters)
  {
    if (_logarithmic_ranges)
    {
      _error_threshold = std::exp(2.0*parameters[0]); // Variance
      _error_fraction = std::exp(2.0*parameters[1]); // Variance
      _error_zero_baseline = std::exp(2.0*parameters[2]); // Variance
      _error_baseline_break = std::exp(parameters[3])*1.0e9; // Positive definite
      _error_baseline_index = parameters[4];
      _error_short_baseline_index = parameters[5];      
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
  
  size_t uncertainty_crosshand_visibilities_broken_power_change::size() const
  {
    return 6;
  }

  
  void uncertainty_crosshand_visibilities_broken_power_change::constrain_noise_at_fixed_baseline(double baseline)
  {
    _var_at_fixed_baseline = true;
    _amplitude_baseline = baseline;
    std::cerr << "uncertainty_crosshand_visibilities_broken_power_change using normalization at fixed baselines " << baseline << std::endl;
  }

  void uncertainty_crosshand_visibilities_broken_power_change::constrain_noise_at_4Glambda()
  {
    constrain_noise_at_fixed_baseline(4.0e9);
  }

  void uncertainty_crosshand_visibilities_broken_power_change::logarithmic_ranges()
  {
    _logarithmic_ranges = true;
  }

  void uncertainty_crosshand_visibilities_broken_power_change::set_crosshand_ratio(double ratio)
  {
    _crosshand_ratio = ratio*ratio;
  }
  
  std::vector< std::complex<double> >& uncertainty_crosshand_visibilities_broken_power_change::error(datum_crosshand_visibilities& d)
  {
    double Vmag2 = std::pow(0.5*std::abs(d.RR+d.LL),2);
    double umag = std::sqrt(d.u*d.u+d.v*d.v) / _error_baseline_break;
    double broken_power_var = _error_zero_baseline*std::pow(umag,_error_short_baseline_index)/( 1.0 + std::pow(umag,_error_short_baseline_index+_error_baseline_index) );

    double ea2 = _error_threshold + _error_fraction*Vmag2 + broken_power_var;
    
    _err[0] = std::complex<double>( std::sqrt( std::pow(d.RRerr.real(),2) + ea2 ), std::sqrt( std::pow(d.RRerr.imag(),2) + ea2 ) );
    _err[1] = std::complex<double>( std::sqrt( std::pow(d.LLerr.real(),2) + ea2 ), std::sqrt( std::pow(d.LLerr.imag(),2) + ea2 ) );

    ea2 *= _crosshand_ratio;
    _err[2] = std::complex<double>( std::sqrt( std::pow(d.RLerr.real(),2) + ea2 ), std::sqrt( std::pow(d.RLerr.imag(),2) + ea2 ) );
    _err[3] = std::complex<double>( std::sqrt( std::pow(d.LRerr.real(),2) + ea2 ), std::sqrt( std::pow(d.LRerr.imag(),2) + ea2 ) );
    
    return _err;
  }

  double uncertainty_crosshand_visibilities_broken_power_change::log_normalization(datum_crosshand_visibilities& d)
  {
    std::vector< std::complex<double> > err = error(d);
    return -std::log( err[0].real()*err[0].imag()
		    * err[1].real()*err[1].imag()
		    * err[2].real()*err[2].imag()
		    * err[3].real()*err[3].imag() );
  }
  
  void uncertainty_crosshand_visibilities_broken_power_change::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
