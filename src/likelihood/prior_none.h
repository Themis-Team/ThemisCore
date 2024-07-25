/*! 
  \file prior_linear.h
  \author Mansour Karami
  \date
  \brief Header file for prior_none class
  \details Defines the no-prior class
*/

#ifndef THEMIS_PRIOR_NONE_H_
#define THEMIS_PRIOR_NONE_H_
#include <string>
#include <vector>
#include <limits>
#include "prior_base.h"


namespace Themis
{

  /*! 
    \class prior_none
    \brief Defines the no-prior class 
    
    \details Derived from the base prior class. Returns the natural log of the prior.
  */
  class prior_none:public prior_base
  {
    public:

      /*!
        \brief
        No-prior class constructor
   
      */
    prior_none() {};
    ~prior_none() {};

    //! Overloaded paranthesis operator to return the log-prior for each input value
    virtual double operator () (double x)
    {
      return 0.0;
    };

    //! Derivative operator
    virtual double derivative(double x)
    {
      return 0;
    };

    //! Overloaded bounds operators.
    virtual double lower_bound()
    {
      return -std::numeric_limits< double >::infinity();
    };
    virtual double upper_bound()
    {
      return std::numeric_limits< double >::infinity();
    };

    virtual double lognorm()
    {
        return std::numeric_limits<double>::infinity();
    };
  };
};

#endif 
