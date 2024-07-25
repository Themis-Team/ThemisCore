/*! 
  \file uncertainty_crosshand_visibilities.cpp
  \author Avery E Broderick
  \date  March, 2022
  \brief Implementation file for the uncertainty_crosshand_visibilities class
*/


#include "uncertainty_crosshand_visibilities.h"

#include <iostream>
#include <iomanip>

namespace Themis{

  uncertainty_crosshand_visibilities::uncertainty_crosshand_visibilities()
    : _err(4)
  {
  }

  uncertainty_crosshand_visibilities::~uncertainty_crosshand_visibilities()
  {
  }

  void uncertainty_crosshand_visibilities::generate_uncertainty(std::vector<double>)
  {
  }
  
  size_t uncertainty_crosshand_visibilities::size() const
  {
    return 0;
  }
  
  std::vector< std::complex<double> >& uncertainty_crosshand_visibilities::error(datum_crosshand_visibilities& d)
  {
    _err[0] = d.RRerr;
    _err[1] = d.LLerr;
    _err[2] = d.RLerr;
    _err[3] = d.LRerr;
    
    //std::cerr << "BAR:" << std::setw(15) << _err[0] << std::setw(15) << _err[1] << '\n';

    
    return _err;
  }

  double uncertainty_crosshand_visibilities::log_normalization(datum_crosshand_visibilities& d)
  {
    return 0.0;
  }
  
  void uncertainty_crosshand_visibilities::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
  }
  
};
