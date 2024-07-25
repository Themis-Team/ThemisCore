/*! 
  \file prior_trunctated_gaussian.h
  \author Avery Broderick & Ali Sar Toosi
  \date Aug 2023
  \brief Header file for prior_truncated_gaussian class
    
  \details Defines the truncated gaussian prior class 
*/

#ifndef THEMIS_PRIOR_TRUNCATED_GAUSSIAN_H_
#define THEMIS_PRIOR_TRUNCATED_GAUSSIAN_H_
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include "prior_base.h"



namespace Themis
{

  /*! 
    \class prior_truncated_gaussian
    \brief Defines the gaussian prior class 
    
    \details Derived from the base prior class. Returns the natural log of the prior
    The mean and the standard deviation are passed as arguments to the constructor
  */
  class prior_truncated_gaussian:public prior_base
  {
    
    public:

      /*!
        \brief
        Truncated Gaussian prior class constructor, accepts the mean and the standard deviation  
        as input
        \param mean Mean value
        \param sigma Standard deviation value
        \param xmin Minimum value (if not given, assumes -xmax)
	\param xmax Maximum value
      */
      prior_truncated_gaussian(double mean, double sigma, double xmax):_mean(mean), _sigma(sigma), _xmin(-xmax), _xmax(xmax){};
    
      prior_truncated_gaussian(double mean, double sigma, double xmin, double xmax):_mean(mean), _sigma(sigma), _xmin(xmin), _xmax(xmax){};
    
      ~prior_truncated_gaussian() {};
  
      //! Overloaded paranthesis operator to return the log-prior for each input value    
      virtual double operator() (double x)
      {
	if ( (x<_xmin) || (x>_xmax) )
	  return -std::numeric_limits< double >::infinity();
	else 	    
	  return (-(x-_mean)*(x-_mean) / (2.0*_sigma*_sigma));
      };

      //! Derivative operator
      virtual double derivative(double x)
      {
	if ( (x<_xmin) || (x>_xmax) )
	  return -std::numeric_limits< double >::infinity();
	else 
	  return ( -(x-_mean)/(_sigma*_sigma) );
      };

      //! Overloaded bounds operators.
      virtual double lower_bound()
      {
	return _xmin;
      };
      virtual double upper_bound()
      {
	return _xmax;
      };

      virtual double lognorm()
      {
	return -std::log(M_PI*_sigma*(std::erf(_xmax/(std::sqrt(2.0)*_sigma))
				     -std::erf(_xmin/(std::sqrt(2.0)*_sigma))));
      };
  
    private:
      double _mean, _sigma, _xmin, _xmax;
  };
  
};

#endif 
