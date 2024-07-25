/*! 
  \file prior_linear.h
  \author
  \date
  \brief Header file for prior_base class
  \details Defines the linear prior class
*/

#ifndef THEMIS_PRIOR_LINEAR_H_
#define THEMIS_PRIOR_LINEAR_H_
#include <string>
#include <vector>
#include <limits>
#include "prior_base.h"


namespace Themis
{

  /*! 
    \class prior_linear
    \brief Defines the linear(flat) prior class 
    
    \details Derived from the base prior class. Returns the natural log of the prior.
    The minimum and maximum range are passed as arguments to the class constructor.
  */
  class prior_linear:public prior_base
  {
    public:

      /*!
        \brief
        Linear prior class constructor, accepts the minimum and maximum values
        as input
        \param min Minimum value
        \param max Maximum value
      */
      prior_linear(double min, double max):_min(min), _max(max){};
      ~prior_linear() {};

      //! Overloaded paranthesis operator to return the log-prior for each input value
      virtual double operator () (double x)
      {
        if ( (x>_min) && (x<_max) )
	  return 0;
        else
	  return -std::numeric_limits< double >::infinity();
      };

      //! Derivative operator
      virtual double derivative(double x)
      {
        if( (x>_min) && (x<_max) )
	  return 0;
        else if (x>=_max)
	  return -std::numeric_limits< double >::infinity();
	else
	  return std::numeric_limits< double >::infinity();
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
          return -std::log(_max-_min);
      };

     
    private:
      double _min, _max;
  };
  
};

#endif 
