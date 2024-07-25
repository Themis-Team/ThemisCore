/*! 
  \file likelihood_closure_amplitude.h
  \author Avery E. BRoderick
  \date  June, 2018
  \brief Header file for the clsoure Amplitude Likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_CLOSURE_AMPLITUDE_H_
#define THEMIS_LIKELIHOOD_CLOSURE_AMPLITUDE_H_
#include <vector>
#include <cmath>
#include "likelihood_base.h"
#include "data_closure_amplitude.h"
#include "model_closure_amplitude.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a closure amplitude likelihood object
    
    \details This class takes a closure amplitude data object and
    a closure amplitude model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    visibility amplitudes has a Gaussian quotient distribution.  This is a good
    approximation at large SNRs (\f$\ge4\f$) and for closure amplitudes constructed such
    that they are \f$\le1\f$.  Note that doing so is always possible; however, this is
    not performed during likelihood evaluation.

    Currently assumes that the ratio of the combined thermal errors of the terms 
    in numerator and denominator is unity.  This can be relaxed in the future, 
    exploiting the known SEFDs of the stations involved in the closure phase.
    
    This class also includes an utility function for estimating the \f$ \chi^2 = -2 \mathcal{L} \f$ to
    assess fit quality.  This is approximately the normal \f$ \chi^2 \f$, i.e., similar
    to order unity.
  */
  class likelihood_closure_amplitude : public likelihood_base
  {
    public:
      likelihood_closure_amplitude(data_closure_amplitude& data,
                                      model_closure_amplitude& model);

      ~likelihood_closure_amplitude() {};
      
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:
      
      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_closure_amplitude& _data;
      model_closure_amplitude& _model;  
  };
};

#endif 
