/*! 
  \file likelihood_visibility.h
  \author Avery E Broderick
  \date  February, 2017
  \brief Header file for the Visibility Likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_VISIBILITY_H_
#define THEMIS_LIKELIHOOD_VISIBILITY_H_
#include <vector>
#include <complex>
#include "likelihood_base.h"
#include "data_visibility.h"
#include "model_visibility.h"
#include "uncertainty_visibility.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a visibility amplitude likelihood object
    
    \details This class takes a visibility data object and
    a visibility model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    visibilities have Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality

    \todo Must address phase centering.  Seems like it should live in models as offsets.
  */
  class likelihood_visibility : public likelihood_base
  {
    public:
      likelihood_visibility(data_visibility& data, model_visibility& model);
      likelihood_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& uncertainty);

      ~likelihood_visibility();
      
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_visibility& _data;
      model_visibility& _model;
      uncertainty_visibility _local_uncertainty; // Default if none is passed
      uncertainty_visibility& _uncertainty;
  };
};

#endif 
