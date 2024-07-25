/*! 
  \file prior.h
  \author
  \date
  
  \brief Header file for prior class
  \details Calculates the combined log-prior for an input vector
*/
#ifndef THEMIS_PRIOR_H_
#define THEMIS_PRIOR_H_

#include <string>
#include <vector>
#include <limits>
#include "prior_base.h"
#include "prior_linear.h"
#include "prior_gaussian.h"
#include "prior_truncated_gaussian.h"
#include "prior_logarithmic.h"
#include "prior_none.h"

namespace Themis {

  /*! 
    \brief Defines the combined prior class 
    
    \details Returns the natural log of the prior.
    A vector of pointers to prior classes is accepted by the constructor
    This class is a functor returning a combined log-prior for a vector of parameters
  */
  class prior
  {

    public:

      /*!
        \brief Prior class constructor, accepts a vector of pointers to prior classes
        as input
        
        \param type A vector of pointers to prior classes
      */
      prior(std::vector<prior_base*> type):prior_type(type) {};
  
      ~prior() {};

      
      //! Overloaded paranthesis operator to return the log-prior for each input vector    
      double operator()(std::vector<double>& x)
      {
        double logPrior = 0.0;
        for (size_t i=0; i<prior_type.size(); ++i)
	  logPrior += prior_type[i]->operator()(x[i]);
        return logPrior;
      };

      double lognorm()
      {
          double lnorm = 0.0;
          for (size_t i = 0; i<prior_type.size(); ++i)
              lnorm += prior_type[i]->lognorm();
          return lnorm;
      };

      double lower_bound(size_t i)
      {
	return (prior_type[i]->lower_bound());
      };

      double upper_bound(size_t i)
      {
	return (prior_type[i]->upper_bound());
      };
      
      std::vector<double> gradient(std::vector<double>& x)
      {
	std::vector<double> grad(x.size(),0.0);
	for (size_t i=0; i<prior_type.size(); ++i)
	  grad[i] += prior_type[i]->derivative(x[i]);
	return grad;
      };
  
    private:
      //! A vector of pointers to prior objects, these are the priors on the
      //! each parameter in the parameter space
      std::vector<prior_base*> prior_type;

  };

};
#endif 
