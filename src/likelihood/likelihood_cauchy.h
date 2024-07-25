/*! 
  \file likelihood_cauchy.h
  \author Roman Gold
  \date  February, 2020
  \brief Header file for the Egg box likelihood class. Ref: https://mathworld.wolfram.com/LorentzianFunction.html
  \details Derived from the base likelihood class. Returns the natural log of   the likelihood  
*/

#ifndef THEMIS_LIKELIHOOD_CAUCHY_H_
#define THEMIS_LIKELIHOOD_CAUCHY_H_

#include <vector>
#include "likelihood_base.h"
#include <cmath>

namespace Themis {

  /*! \brief Defines the Lorentzian likelihood 
  
  \details Defines a multi-dimensional Lorentzian likelihood as a 
  test example for the sampler routines. This likelihood consists of 
  multiple well separated sharp modes and thus makes for a challenging 
  sampling problem
  */  
  class likelihood_cauchy:public likelihood_base
  {

  public:

    likelihood_cauchy(double mean, double alpha):_m(mean),_alpha(alpha){};
    virtual ~likelihood_cauchy() {};

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator() (std::vector<double>& x)
    {
      double norm2 = 0.0;
      size_t d = x.size();
      for(int i = 0; i < int(x.size()); i++){
	norm2 += (x[i]-_m)*(x[i]-_m);
      }

      return -(d+1.0)/2.0*std::log(norm2 + _alpha*_alpha);
    };

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x)
    {
      
      return -2*operator()(x);
    };

    //! Defines a set of processors provided to the likelihood for parallel computation via an MPI communicator
    virtual void set_mpi_communicator(MPI_Comm comm)
    {
      _comm = comm;
    };

  private:

    double _m,_alpha;
    MPI_Comm _comm;

  };
  
};

#endif /* LIKELIHOOD_LORENTZIAN_H_ */
