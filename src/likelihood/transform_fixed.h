/*! 
  \file transform_fixed.h
  \author
  \date
  \brief Header file for transform_fixed class
    
  \details Defines the fixed transformation class, maps a parameter 
  to a constant value.
*/
#ifndef THEMIS_TRANSFORM_FIXED_H_
#define THEMIS_TRANSFORM_FIXED_H_

#include <vector>
#include <limits>
#include <cmath>
#include "transform_base.h"


namespace Themis {

  /*! 
    \class transform_fixed
    \brief Defines the fixed value transformation class
    \details Defines the fixed transformation class, maps a parameter to a constant value .
  */
  class transform_fixed:public transform_base
  {
  
   public:
    
      /*!
        \brief fixed value transformation class constructor, accepts the fixed value as argument
        \param value The fixed value.
      */
    transform_fixed(double value):_value(value){};
    ~transform_fixed() {};
    
    //! Forward transformation function
    virtual double forward(double X) { return _value; };
    
    //! Inverse transformation function
    virtual double inverse(double x) { return _value; };

    //! Jacobian for forward transform \f$ dx/dX \f$
    virtual double forward_jacobian(double X) { return 0.0; }; 
    
    //! Jacobian for inverse transform \f$ dX/dx \f$
    virtual double inverse_jacobian(double x) { return 0.0; };
    
   private:
    double _value;
    
  };
  
};

#endif 
