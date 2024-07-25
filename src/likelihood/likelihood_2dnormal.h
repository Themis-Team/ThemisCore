/*! 
  \file likelihood_2dnormal.h
  \author 
  \date  April, 2017
  \brief Header file for the Gaussian likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor
*/

#ifndef THEMIS_LIKELIHOOD_2DNORMAL_H_
#define THEMIS_LIKELIHOOD_2DNORMAL_H_

#include <vector>
#include "likelihood_base.h"

namespace Themis {

  /*! 
    \brief Defines a class that constructs a likelihood object for a 
    Nd gaussian distribution with diagonal covariance matrix
    
    \details This class returns the log likelihood for a gaussian distribution
    of mean \f$ \mu \f$ and variance \f$ \sigma^{2} \f$
  */  
  class likelihood_2dnormal:public likelihood_base
  {

  public:

    likelihood_2dnormal(std::vector<double> mean, std::vector<double> cov, double corr)
      :_mean(mean), _cov(cov), _corr(corr){};

    virtual ~likelihood_2dnormal() {};

    virtual double operator() (std::vector<double>& x)
    {
      double lnorm = -std::log(2*M_PI*std::sqrt(_cov[0]*_cov[1]*(1.0-_corr*_corr)));
      double dx = (x[0] - _mean[0]);
      double dy = (x[1] - _mean[1]);
      double cross = 2*_corr*dx*dy;
      double llog = -1.0/(2.0*(1.0-_corr*_corr))*(dx*dx/_cov[0] + dy*dy/_cov[1] - cross/std::sqrt(_cov[0]*_cov[1]));
      return lnorm + llog;
    };

    virtual double chi_squared(std::vector<double>& x)
    {
      return 1.0;
    };


  private:
    const std::vector<double> _mean, _cov;
    const double _corr;
    
  };

};

#endif

