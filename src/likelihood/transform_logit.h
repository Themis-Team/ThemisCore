/*! 
  \file transform_logit.h
  \author
  \date
  \brief Header file for transform_logit class
  \details Defines the logit transformation class
*/
#ifndef THEMIS_TRANSFORM_LOGIT_H_
#define THEMIS_TRANSFORM_LOGIT_H_

#include <vector>
#include <limits>
#include <cmath>
#include "transform_base.h"



namespace Themis {

  /*! 
    \class transform_logit
    \brief Defines the logit transformation class
    \details Defines the logit transformation class a is the lower limit
    and b is the upper
  */
  class transform_logit:public transform_base
  {
    
    public:

      transform_logit(double a, double b):_a(a),_b(b){};
      ~transform_logit() {};

      //! Forward transformation function
      virtual double forward(double x)
      {
        if(x > _a && x < _b)
        {
          double z = (x - _a)/(_b - _a);
  	  return log(z/(1.0-z));
        }
        else
  	  return std::numeric_limits< double >::infinity();    
      };

      //! Inverse transformation function
      virtual double inverse(double x)
      { 
        double z = 1.0/(1.0+exp(-x));
        return _a + (_b - _a)*z;
      };

      //! Jacobian for forward transform y(x) = logit(z(x)) where z(x) = (x-a)/(b-a)
      virtual double forward_jacobian(double x)
      {
        double z = (x - _a)/(_b - _a);
        return 1.0/(z*(1-z))*1.0/(_b - _a);
      }

      //! Jacobian for inverse transform x(y) = z^{-1}(logit^{-1}(y)) 
      virtual double inverse_jacobian(double y)
      {
        double expmy = exp(-y);
        return (_b - _a)*expmy/((1.0+expmy)*(1.0+expmy));
      }

    private:
      const double _a,_b;

  };
  
};

#endif 
