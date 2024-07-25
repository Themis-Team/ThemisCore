/*! 
  \file prior_logarithmic.h
  \author Mansour Karami
  \date
  \brief Header file for the prior_logarithmic class
  \details Defines the logarithmic prior class     
*/

#ifndef THEMIS_PRIOR_LOG_H_
#define THEMIS_PRIOR_LOG_H_
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include "prior_base.h"


namespace Themis
{

  /*! 
    \brief Defines a prior that is uniform in log space
    \details Derived from the base prior class. Returns the natural log of the prior
  */
  class prior_logarithmic:public prior_base
  {
    
    public:

      /*!
      \brief Logarithmic prior class constructor, accepts the min and max values as input
        \param min Minimum value
        \param max Maximum value
      */
      prior_logarithmic(double min, double max):_min(min), _max(max){};
      ~prior_logarithmic() {};
  
      //! Overloaded paranthesis operator to return the log-prior for each input value    
      virtual double operator () (double x)
      {
        if ((x < 0.0) || (x < _min) || (x > _max))
	  return -std::numeric_limits< double >::infinity();    
        else
	  return -std::log(x);
      };

      //! Derivative operator
      virtual double derivative(double x)
      {
	if ( (x>_min) && (x<_max) )
	  return -1.0/x;
	else if (x<=_min)
	  return std::numeric_limits< double >::infinity();
	else
	  return -std::numeric_limits< double >::infinity();
      };
      
      //! Overloaded bounds operators.
      virtual double lower_bound()
      {
	return _min;
      };
      virtual double upper_bound()
      {
	return _max;
      };

      virtual double lognorm()
      {
          return std::log(std::log(_max/_min));
      };
    

    private:
      double _min, _max;
  
  };
  
};

#endif 
