/*! 
  \file likelihood_visibility_amplitude.cpp
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Implementation file for the Visibility Amplitude Likelihood class
*/


#include "likelihood_visibility_amplitude.h"
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_visibility_amplitude::likelihood_visibility_amplitude(data_visibility_amplitude& data,
								   model_visibility_amplitude& model)
    : _data(data), _model(model)
  {
  }

  void likelihood_visibility_amplitude::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  double likelihood_visibility_amplitude::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
    double sum = 0.0;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      sum += - 0.5*std::pow((_data.datum(i).V 
			     - _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err))
			    /_data.datum(i).err,2);
    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_visibility_amplitude::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }

  void likelihood_visibility_amplitude::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_visibility_amplitude output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "|V| (Jy)"
	  << std::setw(15) << "err (Jy)"
	  << std::setw(15) << "model |V| (Jy)"
	  << std::setw(15) << "residual (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      double V = _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err);
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).V
	    << std::setw(15) << _data.datum(i).err
	    << std::setw(15) << V
	    << std::setw(15) << (_data.datum(i).V-V)
	    << '\n';
    }
  }
  
};
