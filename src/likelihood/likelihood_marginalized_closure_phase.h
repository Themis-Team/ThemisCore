/*! 
  \file likelihood_marginalized_closure_phase.h
  \author Jorge A. Preciado
  \date  April, 2017
  \brief Header file for the Marginalized Closure Phase Likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor
*/

#ifndef THEMIS_LIKELIHOOD_MARGINALIZED_CLOSURE_PHASE_H_
#define THEMIS_LIKELIHOOD_MARGINALIZED_CLOSURE_PHASE_H_

#include <vector>
#include "likelihood_base.h"
#include "data_closure_phase.h"
#include "model_closure_phase.h"
#include <mpi.h>

namespace Themis
{
  
  /*!
    \brief Defines a class that constructs a marginalized closure phase likelihood object
    
    \details This class takes a closure phase data object and
    a closure phase model object, and then returns the log likelihood. 
    by direct comparison to the observational data assuming that the measured 
    closure phase data has Gaussian errors. 
    
    This class also includes an utility function for computing the \f$ \chi^2 \f$ to
    assess fit quality
  */
  class likelihood_marginalized_closure_phase:public likelihood_base
  {
  public:
    likelihood_marginalized_closure_phase(data_closure_phase& data, 
    model_closure_phase& model, double sigma_phi);
                          
    ~likelihood_marginalized_closure_phase() {};
    
    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator()(std::vector<double>& x);
    
    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x);
    
    //! This functions gets the marginalized closure phase$
    double get_marginalized_phi() const;

    //! This functions gets the marginalized closure phase$
    double get_maximizing_phi() const;

    //! Defines a set of processors provided to the model for parallel 
    //! computation via an MPI communicator.  Only facilates code 
    //! parallelization if the model computation is parallelized via MPI.
    virtual void set_mpi_communicator(MPI_Comm comm);
      
  protected:

    //! Outputs the data and model, as modified by the likelihood appropriately,
    //! to the specified output stream.  Useful for comparison later.
    virtual void output(std::ostream& out);
  	
  private:
    data_closure_phase& _data;
    model_closure_phase& _model;  
    const double _sigma_phi;
    double _phiMarg, _phiMax;


    double angle_difference(double a, double b) const;
  };
};

#endif 
