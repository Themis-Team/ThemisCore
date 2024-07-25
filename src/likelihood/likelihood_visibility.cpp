/*! 
  \file likelihood_visibility.cpp
  \author Avery E Broderick
  \date  February, 2020
  \brief Implementation file for the Visibility Likelihood class
*/


#include "likelihood_visibility.h"
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty)
  {
  }

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model,
					       uncertainty_visibility& uncertainty)
    : _data(data), _model(model), _uncertainty(uncertainty)
  {
  }

  likelihood_visibility::~likelihood_visibility()
  {
  }
  
  void likelihood_visibility::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
    _uncertainty.set_mpi_communicator(comm);
  }

  double likelihood_visibility::operator()(std::vector<double>& x)
  {
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    for(i = 0; i < _data.size(); ++i)
    {
      std::complex<double> V = _data.datum(i).V;
      std::complex<double> err = _uncertainty.error(_data.datum(i));
      std::complex<double> Vm = _model.visibility(_data.datum(i),0.25*std::abs(err));

      sum += - 0.5*( std::pow( (V.real()-Vm.real())/err.real(), 2)
		     +
		     std::pow( (V.imag()-Vm.imag())/err.imag(), 2) );

      sum += _uncertainty.log_normalization(_data.datum(i));
    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_visibility::chi_squared(std::vector<double>& x)
  {
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    for(i = 0; i < _data.size(); ++i)
    {
      std::complex<double> V = _data.datum(i).V;
      std::complex<double> err = _uncertainty.error(_data.datum(i));
      std::complex<double> Vm = _model.visibility(_data.datum(i),0.25*std::abs(err));

      sum += 0.5*( std::pow( (V.real()-Vm.real())/err.real(), 2)
		   +
		   std::pow( (V.imag()-Vm.imag())/err.imag(), 2) );

    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }

  void likelihood_visibility::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_visibility output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "V.r (Jy)"
	  << std::setw(15) << "err.r (Jy)"
	  << std::setw(15) << "model V.r (Jy)"
	  << std::setw(15) << "residual.r (Jy)"
	  << std::setw(15) << "V.i (Jy)"
	  << std::setw(15) << "err.i (Jy)"
	  << std::setw(15) << "model V.i (Jy)"
	  << std::setw(15) << "residual.i (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      std::complex<double> V = _model.visibility(_data.datum(i),0.25*std::abs(_data.datum(i).err));
      std::complex<double> err = _uncertainty.error(_data.datum(i));
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).V.real()
	    << std::setw(15) << err.real()
	    << std::setw(15) << V.real()
	    << std::setw(15) << (_data.datum(i).V-V).real()
	    << std::setw(15) << _data.datum(i).V.imag()
	    << std::setw(15) << err.imag()
	    << std::setw(15) << V.imag()
	    << std::setw(15) << (_data.datum(i).V-V).imag()
	    << '\n';
    }
  }
  
};
