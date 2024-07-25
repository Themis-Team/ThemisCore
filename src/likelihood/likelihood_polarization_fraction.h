/*! 
  \file likelihood_polarization_fraction.h
  \author Roman Gold
  \date  June, 2018
  \brief Header file for the Fractional Polarization Likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_POLARIZATION_FRACTION_H_
#define THEMIS_LIKELIHOOD_POLARIZATION_FRACTION_H_
#include <vector>
#include <iostream>
#include "likelihood_base.h"
#include "data_polarization_fraction.h"
#include "model_polarization_fraction.h"

#include <mpi.h>

namespace Themis{

  /*!
    \brief Defines a class that constructs a fractional polarization likelihood object
    
    \details This class takes a fractional polarization data object and
    a fractional polarization model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    fractional polarizations has Gaussian Quotient errors with equal errors [see THEMIS code paper]. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fitquality
  */
  class likelihood_polarization_fraction:public likelihood_base
  {
    public:
      likelihood_polarization_fraction(data_polarization_fraction& data,
      model_polarization_fraction& model);

      ~likelihood_polarization_fraction() {};
      
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      data_polarization_fraction& _data;
      model_polarization_fraction& _model;  
  };
};

#endif 
