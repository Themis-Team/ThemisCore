/*! 
  \file likelihood_griewank.h
  \author Roman Gold
  \date  February, 2020
  \brief Header file for the Egg box likelihood class. Ref: https://mathworld.wolfram.com/GriewankFunction.html
  \details Derived from the base likelihood class. Returns the natural log of   the likelihood  
*/

#ifndef THEMIS_LIKELIHOOD_GRIEWANK_H_
#define THEMIS_LIKELIHOOD_GRIEWANK_H_

#include <vector>
#include "likelihood_base.h"

namespace Themis {

  /*! \brief Defines the egg box likelihood 
  
  \details Defines a multi-dimensional egg box likelihood as a 
  test example for the sampler routines. This likelihood consists of 
  multiple well separated sharp modes and thus makes for a challenging 
  sampling problem
  */  
  class likelihood_griewank:public likelihood_base
  {

  public:

    likelihood_griewank(double alpha):_alpha(alpha){};
    virtual ~likelihood_griewank() {};

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator() (std::vector<double>& x)
    {
      double res_prod=1.0;
      double res_sum=0.0;
      
      for(int i = 0; i < int(x.size()); i++)
      {
	res_sum  += 0.00025* x[i]*x[i];
	res_prod *= cos(x[i]/std::sqrt(i+1));
      }
      
      return -1.0*pow(1.0 + res_sum - res_prod, _alpha);    
    };

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x)
    {
      double res_prod=1.0;
      double res_sum=0.0;
      
      for(int i = 0; i < int(x.size()); i++)
      {
	res_sum  += 0.00025* x[i]*x[i];
	res_prod *= cos(x[i]/std::sqrt(i+1));
      }
      
      return 1.0*pow(1.0 + res_sum - res_prod, _alpha);    
    };

    //! Defines a set of processors provided to the likelihood for parallel computation via an MPI communicator
    virtual void set_mpi_communicator(MPI_Comm comm)
    {
      _comm = comm;
    };

  private:

    double _alpha;
    MPI_Comm _comm;

  };
  
};

#endif /* LIKELIHOOD_GRIEWANK_H_ */
