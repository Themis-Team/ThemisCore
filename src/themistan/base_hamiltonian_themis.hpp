#ifndef BASE_HAMILTONIAN_THEMIS_HPP
#define BASE_HAMILTONIAN_THEMIS_HPP
#include <stan/mcmc/hmc/hamiltonians/base_hamiltonian.hpp>
#include "themistan_model.h"
namespace stan {
namespace mcmc {


template <class Point, class BaseRNG>
class base_hamiltonian<Themis::themistan_model, Point, BaseRNG>
{
public:
  explicit base_hamiltonian(const Themis::themistan_model& model) : model_(model) {}

  ~base_hamiltonian() {}

  typedef Point PointType;

  virtual double T(Point& z) = 0;

  double V(Point& z) { return z.V; }

  virtual double tau(Point& z) = 0;

  virtual double phi(Point& z) = 0;

  //Potential already has the temperature in it.
  double H(Point& z) { return T(z) + V(z); }

  // The time derivative of the virial, G = \sum_{d = 1}^{D} q^{d} p_{d}.
  virtual double dG_dt(Point& z, callbacks::logger& logger) = 0;

  // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q)
  virtual Eigen::VectorXd dtau_dq(Point& z, callbacks::logger& logger) = 0;

  virtual Eigen::VectorXd dtau_dp(Point& z) = 0;

  // phi = 0.5 * log | Lambda (q) | + V(q)
  virtual Eigen::VectorXd dphi_dq(Point& z, callbacks::logger& logger) = 0;

  virtual void sample_p(Point& z, BaseRNG& rng) = 0;

  void init(Point& z, callbacks::logger& logger) {
    this->update_potential_gradient(z, logger);
  }

  void update_potential(Point& z, callbacks::logger& logger) {
    try {
      z.V = -stan::model::log_prob_propto<true>(model_, z.q);
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger);
      z.V = std::numeric_limits<double>::infinity();
    }
  }

  void update_potential_gradient(Point& z, callbacks::logger& logger) {
    try {
      gradient_themis(z.q, z.V, z.g);
      z.V = -z.V;
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger);
      z.V = std::numeric_limits<double>::infinity();
    }
    z.g = -z.g;
  }

  void update_metric(Point& z, callbacks::logger& logger) {}

  void update_metric_gradient(Point& z, callbacks::logger& logger) {}

  void update_gradients(Point& z, callbacks::logger& logger) {
    update_potential_gradient(z, logger);
  }

 protected:
  const Themis::themistan_model& model_;

  void write_error_msg_(const std::exception& e, callbacks::logger& logger) {
    logger.error(
        "Informational Message: The current Metropolis proposal "
        "is about to be rejected because of the following issue:");
    logger.error(e.what());
    logger.error(
        "If this warning occurs sporadically, such as for highly "
        "constrained variable types like covariance matrices, "
        "then the sampler is fine,");
    logger.error(
        "but if this warning occurs often then your model may be "
        "either severely ill-conditioned or misspecified.");
    logger.error("");
  }


  inline double step_size(double u)
  {
    static const double cbrt_epsilon
        = std::cbrt(std::numeric_limits<double>::epsilon());
    return cbrt_epsilon * std::fmax(1, fabs(u));
  }
  
  void gradient_themis(
                       const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                       double& fx, 
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_fx
                      )
  {
    //Create the model functional that calls the correct log_prob
    
    //For whatever reason I have to use var to get Stan to actually eval the shit
    //this is something weird to do with the autodiff shit. For Themis I can probably 
    //make this simpler
    Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> x_var(x.size());
    std::vector<double> x_vec(x.size());
    for ( int i = 0; i < x.size(); ++i )
    {
      stan::math::var var_i(x(i));
      x_var(i) = var_i;
      x_vec[i] = x(i);
      //std::cout << "PT: x_"<< i << " = " << x_var(i).val() << std::endl;
    }
    fx = model_.log_prob_jacobian(x_var, &std::cout).val();
    std::vector<double> lower = model_.get_lower_bounds();
    std::vector<double> upper = model_.get_upper_bounds();
  
    std::vector<double> dxdy(x.size(),1.0);
    std::vector<double> dlog_jac(x.size(),0.0);
    for ( int i = 0; i < x.size(); ++i )
    {
      double jac = 0.0;
      if ( (std::isfinite(lower[i])&&std::isfinite(upper[i])) ){
        dlog_jac[i] = 1.0-2.0/(std::exp(-x_vec[i])+1);
        x_vec[i] = stan::math::lub_constrain(x_vec[i], lower[i], upper[i], jac);
        dxdy[i] = std::exp(jac);
      } else if (std::isfinite(lower[i])){ 
        x_vec[i] = stan::math::lb_constrain(x_vec[i], lower[i], jac);
        dxdy[i] = std::exp(jac);
        dlog_jac[i] = 1.0;
      }else if (std::isfinite(upper[i])){
        x_vec[i] = stan::math::ub_constrain(x_vec[i], upper[i], jac);
        dxdy[i] = -std::exp(jac);
        dlog_jac[i] = 1.0;
      }else{
        x_vec[i] = x_vec[i];
        dxdy[i] = 1.0;
      }
    //std::cout << "PT: x_"<< i 
    //          << " = " << params_r[i] << std::endl;
    }

    //Now get the derivative:
    std::vector<double> gradient = model_.get_themis_gradient(x_vec);
    //set the derivative 
    for ( int i = 0; i < x.size(); ++i ){
      grad_fx(i) = gradient[i]*dxdy[i] + dlog_jac[i];
    }
    //std::cout << "PT: Themis likelihood: " << _L->operator()(params_t) << std::endl;
    stan::math::recover_memory();
  }
  

};




} //mcmc
} //stan
#endif
