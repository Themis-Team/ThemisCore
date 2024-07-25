/*! 
  \file transform_none.h
  \author
  \date
  \brief Header file for transform_none class
  \details Defines the identity transformation class, indicates that a parameter is not transformed.
*/
#ifndef THEMIS_TRANSFORM_NONE_H_
#define THEMIS_TRANSFORM_NONE_H_
#include <string>
#include <vector>
#include <limits>
#include "transform_base.h"

namespace Themis {

  /*! 
    \class transform_none
    \brief Defines the identity transformation class 
    \details Defines the identity transformation class, used for parameters that are not transformed 
  */
  class transform_none:public transform_base
  {

  public:

    transform_none(){};

    ~transform_none() {};

    //! Forward transformation function
    virtual double forward(double X){ return X; };
    
    //! Inverse transformation function
    virtual double inverse(double x){ return x; };

    //! Jacobian for forward transform \f$ dx/dX \f$
    virtual double forward_jacobian(double X) { return 1.0; }; 
    
    //! Jacobian for inverse transform \f$ dX/dx \f$
    virtual double inverse_jacobian(double x) { return 1.0; };    
  };
  
};

#endif 
