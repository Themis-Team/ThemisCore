/*! 
  \file likelihood_marginalized_visibility_amplitude.h
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Header file for the Marginalized Visibility Amplitude Likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor
*/

#ifndef THEMIS_LIKELIHOOD_MARGINALIZED_VISIBILITY_AMPLITUDE_H_
#define THEMIS_LIKELIHOOD_MARGINALIZED_VISIBILITY_AMPLITUDE_H_

#include <vector>
#include "likelihood_base.h"
#include "data_visibility_amplitude.h"
#include "model_visibility_amplitude.h"
#include <mpi.h>

namespace Themis
{

  /*! 
    \brief Defines a class that constructs a marginalized visibility amplitude 
    likelihood object
    
    \details This class takes a visibility amplitude data object and a 
    visibility amplitude model object, and then returns the marginalized 
    log likelihood to account for an unknown flux normalization present at
    each observational epoch and for which a flat prior is used. It's assumed
    that the observational data has Gaussian errors. 
  */
  class likelihood_marginalized_visibility_amplitude:public likelihood_base
  {
    public:
      likelihood_marginalized_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model);
                            
      ~likelihood_marginalized_visibility_amplitude() {};
      
      //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
      virtual double operator()(std::vector<double>& x);
      
      //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
      virtual double chi_squared(std::vector<double>& x);
      
      //! This functions gets the marginalized visbility amplitude \f$ V_{00}^{M} \f$
      virtual double get_disk_intensity_normalization();

      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
    	
    private:
      data_visibility_amplitude& _data;
      model_visibility_amplitude& _model;  
      double _V00M;
  };
  
};

#endif 
