/*! 
  \file prior_gaussian.h
  \author
  \date
  \brief Header file for prior_gaussian class
    
  \details Defines the gaussian prior class 
*/

#ifndef THEMIS_PRIOR_GAUSSIAN_H_
#define THEMIS_PRIOR_GAUSSIAN_H_
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include "prior_base.h"



namespace Themis
{

  /*! 
    \class prior_gaussian
    \brief Defines the gaussian prior class 
    
    \details Derived from the base prior class. Returns the natural log of the prior
    The mean and the standard deviation are passed as arguments to the constructor
  */
  class prior_gaussian:public prior_base
  {
    
    public:

      /*!
        \brief
        Gaussian prior class constructor, accepts the mean and the standard deviation  
        as input
        \param mean Mean value
        \param sigma Standard deviation value
        
      */
      prior_gaussian(double mean, double sigma):_mean(mean), _sigma(sigma){};
    
      ~prior_gaussian() {};
  
      //! Overloaded paranthesis operator to return the log-prior for each input value    
      virtual double operator () (double x)
      {
        return (-(x-_mean)*(x-_mean) / (2.0*_sigma*_sigma)); 
      };

      //! Derivative operator
      virtual double derivative(double x)
      {
	return ( -(x-_mean)/(_sigma*_sigma) );
      };

      //! Overloaded bounds operators.
      virtual double lower_bound()
      {
	return _mean-10.0*_sigma; // Avoid numerical underflows
      };
      virtual double upper_bound()
      {
	return _mean+10.0*_sigma; // Avoid numerical underflows
      };

      virtual double lognorm()
      {
          return -0.5*std::log(_sigma*_sigma * 2*M_PI);
      };
  
    private:
      double _mean, _sigma;
  };
  
};

#endif 
