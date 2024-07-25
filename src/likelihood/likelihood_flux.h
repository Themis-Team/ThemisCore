/*!
  \file likelihood_flux.h
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Header file for the flux likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_FLUX_H_
#define THEMIS_LIKELIHOOD_FLUX_H_
#include <vector>
#include <iostream>
#include "likelihood_base.h"
#include "data_flux.h"
#include "model_flux.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a flux likelihood object
    
    \details This class takes a flux data object and a flux model object, 
    and then returns the log likelihood by direct comparison of the model
    predictions to the observational data assuming that the measured 
    visibility amplitudes have Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality
  */
  class likelihood_flux:public likelihood_base
  {
    public:
      likelihood_flux(data_flux& data, model_flux& model);
                                      
      ~likelihood_flux() {};
    
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_flux& _data;
      model_flux& _model;  
  };
};

#endif 
