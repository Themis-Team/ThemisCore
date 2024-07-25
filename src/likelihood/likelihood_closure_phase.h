/*!
  \file likelihood_closure_phase.h
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Header file for the Closure Phase Likelihood class
  
  \todo Enhance the documentation of this file
*/

#ifndef THEMIS_LIKELIHOOD_CLOSURE_PHASE_H_
#define THEMIS_LIKELIHOOD_CLOSURE_PHASE_H_
#include <vector>
#include "likelihood_base.h"
#include "../data/data_closure_phase.h"
#include "../model/model_closure_phase.h"

#include <mpi.h>

namespace Themis
{

  /*!
    \brief Defines a class that constructs a closure phase likelihood object
    
    \details This class takes a closure phase data object and
    a closure phase model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    closure phase data has Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality
  */
  class likelihood_closure_phase:public likelihood_base
  {
    public:
      likelihood_closure_phase(data_closure_phase& data,
      model_closure_phase& model);
                                      
      ~likelihood_closure_phase() {};
    
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);

    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_closure_phase& _data;
      model_closure_phase& _model;  

      double angle_difference(double a, double b) const;
  };
};

#endif 
