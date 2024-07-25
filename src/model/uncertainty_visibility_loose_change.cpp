/*! 
  \file uncertainty_visibility_loose_change.cpp
  \author Avery E Broderick
  \date  September, 2020
  \brief Implementation file for the uncertainty_visibility_loose_change class
*/


#include "uncertainty_visibility_loose_change.h"

#include <iostream>
#include <iomanip>

namespace Themis{

  uncertainty_visibility_loose_change::uncertainty_visibility_loose_change()
  {
  }

  uncertainty_visibility_loose_change::~uncertainty_visibility_loose_change()
  {
  }

  void uncertainty_visibility_loose_change::generate_uncertainty(std::vector<double> parameters)
  {
    _error_threshold = parameters[0]*parameters[0]; // Variance
    _error_fraction = parameters[1]*parameters[1]; // Variance
  }
  
  size_t uncertainty_visibility_loose_change::size() const
  {
    return 2;
  }
  
  std::complex<double> uncertainty_visibility_loose_change::error(datum_visibility& d)
  {
    double Vmag2 = std::pow(std::abs(d.V),2);
    double errr = std::sqrt( std::pow(d.err.real(),2) + _error_threshold + _error_fraction*Vmag2 );
    double erri = std::sqrt( std::pow(d.err.imag(),2) + _error_threshold + _error_fraction*Vmag2 );
    
    return ( std::complex<double>(errr,erri) );
  }

  double uncertainty_visibility_loose_change::log_normalization(datum_visibility& d)
  {
    std::complex<double> err = error(d);
    return -std::log( err.real()*err.imag() );
  }
  
  void uncertainty_visibility_loose_change::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
