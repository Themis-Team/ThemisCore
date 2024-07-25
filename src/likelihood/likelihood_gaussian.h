/*! 
  \file likelihood_gaussian.h
  \author 
  \date  April, 2017
  \brief Header file for the Gaussian likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor
*/

#ifndef THEMIS_LIKELIHOOD_GAUSSIAN_H_
#define THEMIS_LIKELIHOOD_GAUSSIAN_H_

#include <vector>
#include "likelihood_base.h"

namespace Themis {

  /*! 
    \brief Defines a class that constructs a likelihood object for a 
    Nd gaussian distribution with diagonal covariance matrix
    
    \details This class returns the log likelihood for a gaussian distribution
    of mean \f$ \mu \f$ and variance \f$ \sigma^{2} \f$
  */  
  class likelihood_gaussian:public likelihood_base
  {

  public:

    likelihood_gaussian(std::vector<double> mean, std::vector<double> cov):_mean(mean), _cov(cov){};

    virtual ~likelihood_gaussian() {};

    virtual double operator() (std::vector<double>& x)
    {
      double sum = 0.0;
      for(size_t i = 0; i < _mean.size(); ++i)
      {
        sum += -((x[i]-_mean[i])*(x[i]-_mean[i]) / (2.0*_cov[i])) 
               - 0.5*std::log( 2*M_PI * _cov[i]);
        //std::cout << "PT: Themis inside x_"<< i 
        //          << " = " << x[i] << std::endl;
      }
      //std::cout << "PT: Themis inside likelihood: " << sum << std::endl;
      return sum;
    };

    virtual double chi_squared(std::vector<double>& x)
    {
      double chi2 = 0.0;
      for(size_t i = 0; i < _mean.size(); ++i)
        chi2 += ((x[i]-_mean[i])*(x[i]-_mean[i]) / (2.0*_cov[i]));  
      return chi2;
    };


  private:
    const std::vector<double> _mean, _cov;
    
  };

};

#endif

