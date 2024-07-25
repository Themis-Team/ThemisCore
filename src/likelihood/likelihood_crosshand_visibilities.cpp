/*! 
  \file likelihood_crosshand_visibilities.cpp
  \author Avery E Broderick
  \date  March, 2020
  \brief Implementation file for the crosshand visibilities likelihood class
*/


#include "likelihood_crosshand_visibilities.h"
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_crosshand_visibilities::likelihood_crosshand_visibilities(data_crosshand_visibilities& data,
								       model_crosshand_visibilities& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty)
  {
  }

  likelihood_crosshand_visibilities::likelihood_crosshand_visibilities(data_crosshand_visibilities& data,
								       model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty)
    : _data(data), _model(model), _uncertainty(uncertainty)
  {
  }

  void likelihood_crosshand_visibilities::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  double likelihood_crosshand_visibilities::operator()(std::vector<double>& x)
  {
    //_model.generate_model(x);
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      std::complex<double> RR = _data.datum(i).RR;
      //std::complex<double> RRerr = _data.datum(i).RRerr;
      std::complex<double> LL = _data.datum(i).LL;
      //std::complex<double> LLerr = _data.datum(i).LLerr;
      std::complex<double> RL = _data.datum(i).RL;
      //std::complex<double> RLerr = _data.datum(i).RLerr;
      std::complex<double> LR = _data.datum(i).LR;
      //std::complex<double> LRerr = _data.datum(i).LRerr;

      std::vector<std::complex<double> > err = _uncertainty.error(_data.datum(i));
      std::vector<std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(i),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));

      // RR
      sum += - 0.5*( std::pow( (RR.real()-cvo[0].real())/err[0].real(), 2)
		     +
		     std::pow( (RR.imag()-cvo[0].imag())/err[0].imag(), 2) );

      // LL
      sum += - 0.5*( std::pow( (LL.real()-cvo[1].real())/err[1].real(), 2)
		     +
		     std::pow( (LL.imag()-cvo[1].imag())/err[1].imag(), 2) );

      // RL
      sum += - 0.5*( std::pow( (RL.real()-cvo[2].real())/err[2].real(), 2)
		     +
		     std::pow( (RL.imag()-cvo[2].imag())/err[2].imag(), 2) );

      // LR
      sum += - 0.5*( std::pow( (LR.real()-cvo[3].real())/err[3].real(), 2)
		     +
		     std::pow( (LR.imag()-cvo[3].imag())/err[3].imag(), 2) );

    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_crosshand_visibilities::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }

  void likelihood_crosshand_visibilities::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_crosshand_visibilities output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "phi1 (rad)"
	  << std::setw(15) << "phi2 (rad)"
	  << std::setw(15) << "RR.r (Jy)"
	  << std::setw(15) << "RRerr.r (Jy)"
	  << std::setw(15) << "mod RR.r (Jy)"
	  << std::setw(15) << "RRres.r (Jy)"
	  << std::setw(15) << "RR.i (Jy)"
	  << std::setw(15) << "RRerr.i (Jy)"
	  << std::setw(15) << "mod RR.i (Jy)"
	  << std::setw(15) << "RRres.i (Jy)"
	  << std::setw(15) << "LL.r (Jy)"
	  << std::setw(15) << "LLerr.r (Jy)"
	  << std::setw(15) << "mod LL.r (Jy)"
	  << std::setw(15) << "LLres.r (Jy)"
	  << std::setw(15) << "LL.i (Jy)"
	  << std::setw(15) << "LLerr.i (Jy)"
	  << std::setw(15) << "mod LL.i (Jy)"
	  << std::setw(15) << "LLres.i (Jy)"
	  << std::setw(15) << "RL.r (Jy)"
	  << std::setw(15) << "RLerr.r (Jy)"
	  << std::setw(15) << "mod RL.r (Jy)"
	  << std::setw(15) << "RLres.r (Jy)"
	  << std::setw(15) << "RL.i (Jy)"
	  << std::setw(15) << "RLerr.i (Jy)"
	  << std::setw(15) << "mod RL.i (Jy)"
	  << std::setw(15) << "RLres.i (Jy)"
	  << std::setw(15) << "LR.r (Jy)"
	  << std::setw(15) << "LRerr.r (Jy)"
	  << std::setw(15) << "mod LR.r (Jy)"
	  << std::setw(15) << "LRres.r (Jy)"
	  << std::setw(15) << "LR.i (Jy)"
	  << std::setw(15) << "LRerr.i (Jy)"
	  << std::setw(15) << "mod LR.i (Jy)"
	  << std::setw(15) << "LRres.i (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      // std::complex<double> RRerr = _data.datum(i).RRerr;
      // std::complex<double> LLerr = _data.datum(i).LLerr;
      std::vector<std::complex<double> > err = _uncertainty.error(_data.datum(i));
      std::vector<std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(i),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).phi1
	    << std::setw(15) << _data.datum(i).phi2
	  // RR
	    << std::setw(15) << _data.datum(i).RR.real()
	    << std::setw(15) << err[0].real()
	    << std::setw(15) << cvo[0].real()
	    << std::setw(15) << (_data.datum(i).RR-cvo[0]).real()
	    << std::setw(15) << _data.datum(i).RR.imag()
	    << std::setw(15) << err[0].imag()
	    << std::setw(15) << cvo[0].imag()
	    << std::setw(15) << (_data.datum(i).RR-cvo[0]).imag()
	  // LL
	    << std::setw(15) << _data.datum(i).LL.real()
	    << std::setw(15) << err[1].real()
	    << std::setw(15) << cvo[1].real()
	    << std::setw(15) << (_data.datum(i).LL-cvo[1]).real()
	    << std::setw(15) << _data.datum(i).LL.imag()
	    << std::setw(15) << err[1].imag()
	    << std::setw(15) << cvo[1].imag()
	    << std::setw(15) << (_data.datum(i).LL-cvo[1]).imag()
	  // RL
	    << std::setw(15) << _data.datum(i).RL.real()
	    << std::setw(15) << err[2].real()
	    << std::setw(15) << cvo[2].real()
	    << std::setw(15) << (_data.datum(i).RL-cvo[2]).real()
	    << std::setw(15) << _data.datum(i).RL.imag()
	    << std::setw(15) << err[2].imag()
	    << std::setw(15) << cvo[2].imag()
	    << std::setw(15) << (_data.datum(i).RL-cvo[2]).imag()
	  // LR
	    << std::setw(15) << _data.datum(i).LR.real()
	    << std::setw(15) << err[3].real()
	    << std::setw(15) << cvo[3].real()
	    << std::setw(15) << (_data.datum(i).LR-cvo[3]).real()
	    << std::setw(15) << _data.datum(i).LR.imag()
	    << std::setw(15) << err[3].imag()
	    << std::setw(15) << cvo[3].imag()
	    << std::setw(15) << (_data.datum(i).LR-cvo[3]).imag()
	    << '\n';
    }
  }
  
};
