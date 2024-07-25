/*! 
  \file uncertainty_visibility_power_change.cpp
  \author Avery E Broderick
  \date  October, 2020
  \brief Implementation file for the uncertainty_visibility_power_change class
*/


#include "uncertainty_visibility_power_change.h"

#include <iostream>
#include <iomanip>

namespace Themis{

  uncertainty_visibility_power_change::uncertainty_visibility_power_change()
  {
  }

  uncertainty_visibility_power_change::~uncertainty_visibility_power_change()
  {
  }

  void uncertainty_visibility_power_change::generate_uncertainty(std::vector<double> parameters)
  {
    _error_threshold = parameters[0]*parameters[0]; // Variance
    _error_fraction = parameters[1]*parameters[1]; // Variance
    _error_zero_baseline = parameters[2]*parameters[2]; // Variance
    _error_baseline_break = std::fabs(parameters[3])*1.0e9; // Positive definite
    _error_baseline_index = parameters[4];
  }
  
  size_t uncertainty_visibility_power_change::size() const
  {
    return 5;
  }
  
  std::complex<double> uncertainty_visibility_power_change::error(datum_visibility& d)
  {
    double Vmag2 = std::pow(std::abs(d.V),2);
    double umag = std::sqrt(d.u*d.u+d.v*d.v) / _error_baseline_break;
    double power_var = _error_zero_baseline/( 1.0 + std::pow(umag, _error_baseline_index) );
    double errr = std::sqrt( std::pow(d.err.real(),2) + _error_threshold + _error_fraction*Vmag2 + power_var );
    double erri = std::sqrt( std::pow(d.err.imag(),2) + _error_threshold + _error_fraction*Vmag2 + power_var );
    
    return ( std::complex<double>(errr,erri) );
  }

  double uncertainty_visibility_power_change::log_normalization(datum_visibility& d)
  {
    std::complex<double> err = error(d);
    return -std::log( err.real()*err.imag() );
  }
  
  void uncertainty_visibility_power_change::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
