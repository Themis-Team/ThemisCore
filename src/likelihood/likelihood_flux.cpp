/*!
  \file likelihood_flux.cpp
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Implementation file for the flux likelihood class
*/

#include "likelihood_flux.h"
#include <iomanip>

namespace Themis
{

  likelihood_flux::likelihood_flux(data_flux& data, model_flux& model)
    : _data(data), _model(model)
  {
  }
  
  void likelihood_flux::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  double likelihood_flux::operator()(std::vector<double>& x)
  {
    _model.generate_model(x);
    double sum = 0.0;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      //std::cout << "  " <<_model.flux(_data.datum(i),0.25*_data.datum(i).err) << std::endl;
      double modelF = _model.flux(_data.datum(i), 0.25*_data.datum(i).err);
      double dataF = _data.datum(i).Fnu;
      double err = _data.datum(i).err;
      //Need to check if flux measurement is an upper bound
      if ( err <= 0){
        if ( modelF >= dataF){
          sum += -std::numeric_limits<double>::infinity();
        }
        else{
          sum += 0;
        }
        
      }
      else{
        sum += - 0.5*std::pow((dataF - modelF)/_data.datum(i).err, 2);
      }
      //std::cout << "Flux data:  " << _data.datum(i).frequency << "  " << _data.datum(i).Fnu << "  " << _data.datum(i).err  << " Flux model: " << _model.flux(_data.datum(i),0.25*_data.datum(i).err)<<std::endl; 
      //std::cout << " Flux Chi2 sum: " << -2.0*sum << std::endl;
    }
    
    // the factor 0.25 accounts for finite accuracy in model prediction;
    // it currently gives 3 percent error in the reconstructed uncertainties
      
    return sum;
  }

  double likelihood_flux::chi_squared(std::vector<double>& x)
  {
    //std::cout << "Flux Chi2: " << ( -2.0*operator()(x) ) << std::endl;
    return ( -2.0*operator()(x) );
  }

  void likelihood_flux::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_flux output file\n#"
	  << std::setw(14) << "freq (GHz)"
	  << std::setw(15) << "flux (Jy)"
	  << std::setw(15) << "err (Jy)"
	  << std::setw(15) << "model flux (Jy)"
	  << std::setw(15) << "residual (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      double Fnu = _model.flux(_data.datum(i),0.25*_data.datum(i).err);
      if (rank==0)
	out << std::setw(15) << _data.datum(i).frequency/1e9
	    << std::setw(15) << _data.datum(i).Fnu
	    << std::setw(15) << _data.datum(i).err
	    << std::setw(15) << Fnu
	    << std::setw(15) << (_data.datum(i).Fnu-Fnu)
	    << '\n';
    }
  }

};
