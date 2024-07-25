/*! 
  \file transform_logarithmic.h
  \author
  \date
  \brief Header file for transform_logarithmic class
  \details Defines the logarithmic transformation class
*/
#ifndef THEMIS_TRANSFORM_LOG_H_
#define THEMIS_TRANSFORM_LOG_H_

#include <vector>
#include <limits>
#include <cmath>
#include "transform_base.h"



namespace Themis {

  /*! 
    \class transform_logarithmic
    \brief Defines the logarithmic transformation class
    \details Defines the logarithmic transformation class
  */
  class transform_logarithmic:public transform_base
  {
    
   public:

    transform_logarithmic(){};
    ~transform_logarithmic() {};

    //! Forward transformation function
    virtual double forward(double X)
    {
      if (X > 0.0)
	return std::log(X);
      else
	return std::numeric_limits< double >::infinity();    
    };
    
    //! Inverse transformation function
    virtual double inverse(double x)
    {
      return std::exp(x);
    };


    //! Jacobian for forward transform \f$ dx/dX \f$
    virtual double forward_jacobian(double X)
    {
      if (X>0.0)
	return 1.0/X;
      else
	return std::numeric_limits< double >::infinity();
    };
    
    //! Jacobian for inverse transform \f$ dX/dx \f$
    virtual double inverse_jacobian(double x)
    {
      return std::exp(x);
    };
    

  };
  
};

#endif 
