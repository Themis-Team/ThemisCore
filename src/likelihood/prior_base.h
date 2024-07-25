/*! 
  \file prior_base.h
  \author
  \date
  \brief Header file for prior_base class
    
  \details Defines the base prior class from which all the priors are derived 
*/

#ifndef THEMIS_PRIOR_BASE_H_
#define THEMIS_PRIOR_BASE_H_
#include <string>
#include <vector>
#include <limits>


namespace Themis
{

  /*! 
    \class prior_base
    \brief Defines the base prior class 
    \details Defines the base prior class from which all the priors are derived 
  */
  class prior_base
  {
    public:
      prior_base(){};
      virtual ~prior_base() {};
      
      //! Overloaded paranthesis operator to return the log-prior for each input value
      virtual double operator ()(double ){return 0;};

      //! Derivative operator
      virtual double derivative(double){return 0;};
      
      //! Overloaded bounds operators.
      virtual double lower_bound()
      {
	return -std::numeric_limits< double >::infinity();
      };
      virtual double upper_bound()
      {
	return std::numeric_limits< double >::infinity();
      };

      virtual double lognorm(){return 0.0;};

    protected:

    private:
    
  };
  
};

#endif 
