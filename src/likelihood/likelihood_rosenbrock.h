/*! 
  \file likelihood_rosenbrock.h
  \author Roman Gold
  \date  Feb 2020
  \brief Header file for the Rosenbrock likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood specified in the Rosenbrock test. 
  Reference: arXiv [stat.CO]:1903.09556v1 
*/

#ifndef THEMIS_LIKELIHOOD_ROSENBROCK_H_
#define THEMIS_LIKELIHOOD_ROSENBROCK_H_

#include <vector>
#include "likelihood_base.h"

namespace Themis {

  /*! \brief Defines the egg box likelihood 
  
  \details Defines a multi-dimensional egg box likelihood as a 
  test example for the sampler routines. This likelihood consists of 
  multiple well separated sharp modes and thus makes for a challenging 
  sampling problem
  */  
  class likelihood_rosenbrock:public likelihood_base
  {

  public:

    virtual ~likelihood_rosenbrock() {};

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator() (std::vector<double>& x)
    {
      double res=1.0;

      if (x.size()==2) {
	res = (100.0*pow(x[1]-std::pow(x[0],2),2)+std::pow(1-x[0],2))/20.0; // 2D hardcoded # http://www.pyopt.org/examples/examples.rosenbrock.html
	  }
      else
	{
	std::cout<<"ONLY 2D Rosenbrock supported for now! You have asked for "<<x.size()<<"D"<<std::endl;
	std::exit(1);
	}
      return -res;    
    };

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x)
    {
      double res=1.0;

      // Some multi D stuff. REF: arXiv [stat.CO]:1903.09556v1
      // P(x, y) ~ exp( −a(x − μ)**2 − b(y − x**2 )**2) // 2D, integrates to pi/sqrt(ab) , see eq (6)
      // P(x) ~ exp −a(x − mu1 )2 − b(y − x2 )2 − c(y − mu2 )2 − d(z − y2)2 // full Rosenbrock , eq (8)

      if (x.size()==2) {
	res = 100*pow(x[1]-std::pow(x[0],2),2)+std::pow(1-x[0],2); // 2D hardcoded
	  }
      else
	{
	std::cout<<"ONLY 2D Rosenbrock supported for now! You have asked for "<<x.size()<<"D"<<std::endl;
	std::exit(1);
	}
      
      return res;

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

#endif /* LIKELIHOOD_ROSENBROCK_H_ */
