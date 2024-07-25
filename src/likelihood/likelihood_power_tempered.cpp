#include "likelihood_power_tempered.h"

namespace Themis
{
  likelihood_power_tempered::likelihood_power_tempered(likelihood& L)
    : likelihood(L.priors(), L.transforms(), L.likelihoods(), L.weights()),
    _L(&L), _beta(1.0), _lklhd_no_temp(0.0)
  {

  }
  
  void likelihood_power_tempered::set_beta(double beta)
  {
    if (beta > 1.0 || beta < 0.0)
    {
      std::cerr << "likelihood_power_tempered::set_beta, beta must be in [0,1]\n";
      std::exit(1);
    }
    _beta = beta;
  }

  double likelihood_power_tempered::get_lklhd_no_temp()
  {
    return _lklhd_no_temp;
  }

  double likelihood_power_tempered::operator()(std::vector<double>& x)
  {
    _lklhd_no_temp = _L->operator()(x);
    return _beta*_lklhd_no_temp;
  }
  std::vector<double> likelihood_power_tempered::gradient(std::vector<double>& x)
  {
    std::vector<double> grad = _L->gradient(x);
    for ( size_t i = 0; i < grad.size(); ++i )
      grad[i] *= _beta;
    return grad;
  }

}//end Themis
