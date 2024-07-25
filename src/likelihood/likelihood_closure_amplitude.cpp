/*! 
  \file likelihood_closure_amplitude.cpp
  \author Avery E. Broderick
  \date  June, 2018
  \brief Implementation file for the Closure Amplitude Likelihood class
*/

#include <iostream>
#include <iomanip>

#include "likelihood_closure_amplitude.h"

namespace Themis{

  likelihood_closure_amplitude::likelihood_closure_amplitude(data_closure_amplitude& data,
							     model_closure_amplitude& model)
    : _data(data), _model(model)
  {
  }

  void likelihood_closure_amplitude::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }
  
  double likelihood_closure_amplitude::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
    double sum = 0.0;
    double Sigma2, Delta, Omega, rho2;
    double CA0, CAsig, CA;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      CA = _data.datum(i).CA;
      CAsig = _data.datum(i).err;
      CA0 = _model.closure_amplitude(_data.datum(i),0.25*CAsig);
      // We can introduce information about the relative errors in the numerator and denominator
      rho2 = 1.0; 
      
      Sigma2 = (CAsig*CAsig) * (rho2+CA*CA)/(rho2+CA0*CA0);
      Delta = (rho2+CA*CA0)/(rho2+CA*CA);
      Omega = CAsig * std::sqrt( rho2/ ((rho2+CA*CA)*(rho2+CA0*CA0)) );
      double expfac = -0.5*(Delta*Delta)/(Omega*Omega);
      expfac = std::max(expfac,-200.0);
      
      sum += -0.5*(CA-CA0)*(CA-CA0)/Sigma2
	- 0.5 * std::log(Sigma2/(CAsig*CAsig))
	+ std::log( Delta*std::erf(Delta/std::sqrt(2.)/Omega)
		    -
		    2.*Omega/std::sqrt(2.*M_PI)*std::exp(expfac) );    
    }
    
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_closure_amplitude::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }
  
  void likelihood_closure_amplitude::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_closure_amplitude output file\n#"
	  << std::setw(14) << "u1 (Gl)"
	  << std::setw(15) << "v1 (Gl)"
	  << std::setw(15) << "u2 (Gl)"
	  << std::setw(15) << "v2 (Gl)"
	  << std::setw(15) << "u3 (Gl)"
	  << std::setw(15) << "v3 (Gl)"
	  << std::setw(15) << "u4 (Gl)"
	  << std::setw(15) << "v4 (Gl)"
	  << std::setw(15) << "CA"
	  << std::setw(15) << "err"
	  << std::setw(15) << "model CA"
	  << std::setw(15) << "residual"
	  << '\n';


    double CA, CAsig, CA0;
    for (size_t i=0; i<_data.size(); ++i)
    {
      CA = _data.datum(i).CA;
      CAsig = _data.datum(i).err;
      CA0 = _model.closure_amplitude(_data.datum(i),0.25*CAsig);
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u1/1e9
	    << std::setw(15) << _data.datum(i).v1/1e9
	    << std::setw(15) << _data.datum(i).u2/1e9
	    << std::setw(15) << _data.datum(i).v2/1e9
	    << std::setw(15) << _data.datum(i).u3/1e9
	    << std::setw(15) << _data.datum(i).v3/1e9
	    << std::setw(15) << _data.datum(i).u4/1e9
	    << std::setw(15) << _data.datum(i).v4/1e9
	    << std::setw(15) << CA
	    << std::setw(15) << CAsig
	    << std::setw(15) << CA0
	    << std::setw(15) << (CA-CA0)
	    << '\n';
    }
  }

};
