/*! 
  \file likelihood_visibility_amplitude.h
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Header file for the Visibility Amplitude Likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_VISIBILITY_AMPLITUDE_H_
#define THEMIS_LIKELIHOOD_VISIBILITY_AMPLITUDE_H_
#include <vector>
#include "likelihood_base.h"
#include "data_visibility_amplitude.h"
#include "model_visibility_amplitude.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a visibility amplitude likelihood object
    
    \details This class takes a visibility amplitude data object and
    a visibility amplitude model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    visibility amplitudes has Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality
  */
  class likelihood_visibility_amplitude : public likelihood_base
  {
    public:
      likelihood_visibility_amplitude(data_visibility_amplitude& data,
                                      model_visibility_amplitude& model);

      ~likelihood_visibility_amplitude() {};
      
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_visibility_amplitude& _data;
      model_visibility_amplitude& _model;  
  };
};

#endif 
