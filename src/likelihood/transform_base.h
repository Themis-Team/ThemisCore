/*! 
  \file transform_base.h
  \author
  \date
  \brief Header file for transform_base class
  \details Defines the base transformation class from which all the parameter
  transformations are derived 
*/
#ifndef THEMIS_TRANSFORM_BASE_H_
#define THEMIS_TRANSFORM_BASE_H_
#include <string>
#include <vector>
#include <limits>

namespace Themis {

  /*! 
    \class transform_base
    \brief Defines the base transformation class 
    \details Defines the base parameter transformation class form which all the parameter transformations are derived  
  */
  class transform_base
  {
    public:
      virtual ~transform_base() {};

      //! Forward transformation function \f$ x=T(X) \f$
      virtual double forward(double) = 0;
      
      //! Inverse transformation function \f$ X=T^{-1}(x) \f$
      virtual double inverse(double) = 0;

      //! Jacobian for forward transform \f$ dx/dX \f$
      virtual double forward_jacobian(double) = 0;

      //! Jacobian for inverse transform \f$ dX/dx \f$
      virtual double inverse_jacobian(double) = 0;
  };
};

#endif 
