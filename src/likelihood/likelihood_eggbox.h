/*! 
  \file likelihood_eggbox.h
  \author 
  \date  April, 2017
  \brief Header file for the Egg box likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood  
*/

#ifndef THEMIS_LIKELIHOOD_EGGBOX_H_
#define THEMIS_LIKELIHOOD_EGGBOX_H_

#include <vector>
#include "likelihood_base.h"

namespace Themis {

  /*! \brief Defines the egg box likelihood 
  
  \details Defines a multi-dimensional egg box likelihood as a 
  test example for the sampler routines. This likelihood consists of 
  multiple well separated sharp modes and thus makes for a challenging 
  sampling problem
  */  
  class likelihood_eggbox:public likelihood_base
  {

  public:

    likelihood_eggbox(double alpha):_alpha(alpha){};
    virtual ~likelihood_eggbox() {};

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator() (std::vector<double>& x)
    {
      unsigned int i;
      double res=1.0;
      
      for(i = 0; i < x.size(); i++)
      {
  	    res *= cos(x[i]);
      }
      
      return pow(2.0 + res, _alpha);    
    };

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x)
    {
      unsigned int i;
      double res=1.0;
      
      for(i = 0; i < x.size(); i++)
      {
  	    res *= cos(x[i]);
      }
      
      return -2.0*pow(2.0 + res, _alpha);    
    };


  private:

    double _alpha;

  };
  
};

#endif /* LIKELIHOOD_EGGBOX_H_ */
