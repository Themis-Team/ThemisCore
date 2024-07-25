/*! 
  \file likelihood_polarization_fraction.cpp
  \author Roman Gold
  \date  June, 2018
  \brief Implementation file for the fractional polarization Likelihood class
*/


#include "likelihood_polarization_fraction.h"
#include <cmath>
#include <iomanip>

namespace Themis{

  likelihood_polarization_fraction::likelihood_polarization_fraction(data_polarization_fraction& data,
									 model_polarization_fraction& model)
    : _data(data), _model(model)
  {
  }
  
  void likelihood_polarization_fraction::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }
  
  double likelihood_polarization_fraction::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
    double sum = 0.0;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      // sum += - 0.5*std::pow((_data.datum(i).V 
      // - _model.polarization_fraction(_data.datum(i),0.25*_data.datum(i).err))
      // /_data.datum(i).err,2);
      
      // See [THEMIS code paper Appendix eq B.17]
      double mbreve_0 = _model.polarization_fraction(_data.datum(i),0.25*_data.datum(i).err);
      double mbreve_0_sq = mbreve_0*mbreve_0;
      double mbreve = _data.datum(i).mbreve_amp;
      double mbreve_sq = std::pow(_data.datum(i).mbreve_amp,2);
      
      double Sigma_sq = std::pow(_data.datum(i).err,2) * (1. + mbreve_sq)/(1.+ mbreve_0_sq);
      
      double Delta = (1. + mbreve_0 * mbreve)/(1.+ mbreve_sq);
      double Omega = _data.datum(i).err * 1./std::sqrt( ( 1. + mbreve_sq) * ( 1. + mbreve_0_sq ) );
      
      sum += - 0.5*std::pow( mbreve - mbreve_0 ,2) / Sigma_sq 
	- 0.5*std::log( (1.+ mbreve_sq)/(1.+ mbreve_0_sq ) )
	// - 0.5*std::log(Sigma_sq/std::pow(_data.datum(i).err,2))
	// - std::log(std::sqrt(2.*M_PI) * std::sqrt(Sigma_sq))
	+ std::log(Delta * std::erf(Delta/std::sqrt(2.)/Omega) - 2.*Omega/std::sqrt(2.*M_PI)*std::exp(-std::pow(Delta/Omega,2)/2.) )
	;
      
    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_polarization_fraction::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }
   
  void likelihood_polarization_fraction::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_visibility_amplitude output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "mbreve"
	  << std::setw(15) << "err"
	  << std::setw(15) << "model mbreve"
	  << std::setw(15) << "residual"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      double m = _model.polarization_fraction(_data.datum(i),0.25*_data.datum(i).err);
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).mbreve_amp
	    << std::setw(15) << _data.datum(i).err
	    << std::setw(15) << m
	    << std::setw(15) << (_data.datum(i).mbreve_amp-m)
	    << '\n';
    }
  }

};
