/*! 
  \file uncertainty_visibility.cpp
  \author Avery E Broderick
  \date  September, 2020
  \brief Implementation file for the uncertainty_visibility class
*/


#include "uncertainty_visibility.h"

namespace Themis{

  uncertainty_visibility::uncertainty_visibility()
  {
  }

  uncertainty_visibility::~uncertainty_visibility()
  {
  }

  void uncertainty_visibility::generate_uncertainty(std::vector<double>)
  {
  }
  
  size_t uncertainty_visibility::size() const
  {
    return 0;
  }
  
  std::complex<double> uncertainty_visibility::error(datum_visibility& d)
  {
    return d.err;
  }

  double uncertainty_visibility::log_normalization(datum_visibility& d)
  {
    return 0.0;
  }
  
  void uncertainty_visibility::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
