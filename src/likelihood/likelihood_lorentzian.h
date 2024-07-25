/*! 
  \file likelihood_lorentzian.h
  \author Roman Gold
  \date  February, 2020
  \brief Header file for the Egg box likelihood class. Ref: https://mathworld.wolfram.com/LorentzianFunction.html
  \details Derived from the base likelihood class. Returns the natural log of   the likelihood  
*/

#ifndef THEMIS_LIKELIHOOD_LORENTZIAN_H_
#define THEMIS_LIKELIHOOD_LORENTZIAN_H_

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
  class likelihood_lorentzian:public likelihood_base
  {

  public:

    virtual ~likelihood_lorentzian() {};

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator() (std::vector<double>& x)
    {
      double Gamma = 1./M_PI;
      double s=Gamma/2;
      double m=0.;
      double res_sum=0.0;
      
      for(int i = 0; i < int(x.size()); i++)
      {
	res_sum += 1.0/M_PI/pow((std::pow(x[i]-m,2)+s*s),x.size()+1/2);
      }

      // REFs:
      // https://mathworld.wolfram.com/LorentzianFunction.html
      return std::log(res_sum);
    };

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x)
    {
      double Gamma = 1.;
      double s=Gamma/2;
      double m=0.;
      double res_sum=0.0;
      
      for(int i = 0; i < int(x.size()); i++)
      {
	res_sum += 1.0/M_PI*s/(std::pow(x[i]-m,2)+s*s);
      }

      return -std::log(res_sum);
    };

    //! Defines a set of processors provided to the likelihood for parallel computation via an MPI communicator
    virtual void set_mpi_communicator(MPI_Comm comm)
    {
      _comm = comm;
    };

  private:

    MPI_Comm _comm;

  };
  
};

#endif /* LIKELIHOOD_LORENTZIAN_H_ */
