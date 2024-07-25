/*!
  \file themistan_model.cpp
  \author Paul Tiede
  \brief Header file for hook of Themis model into Stan.  
*/

#include "themistan_model.h"
namespace Themis
{
/******************************************************************************************
 * ****************************************************************************************
 * ThemiStan model wrapper implementation
 * ****************************************************************************************
*/ 
themistan_model::themistan_model(likelihood& L, 
                                                 std::vector<std::string> var_names, 
                                                 std::ostream* msg_stream)
 :stan::model::model_base(L.priors().size()), _L(&L), _var_names(var_names)
{ 
  //get the param bounds
  _lower.resize(0);
  _upper.resize(0);
  size_t nparams = L.priors().size();
  _gradient.resize(nparams);
  for ( size_t i = 0; i < nparams; ++i )
  {
    double lo = L.priors()[i]->lower_bound();
    double up = L.priors()[i]->upper_bound();
    _lower.push_back(lo);
    _upper.push_back(up);
  } 
}


std::vector<double> themistan_model::get_themis_gradient(std::vector<double>& x) const
{
  // Distribute x
  //double *xbuff = new double[x.size()];
  //for (size_t i=0; i<x.size(); ++i)
  //  xbuff[i] = x[i];
  //MPI_Bcast(xbuff,x.size(),MPI_DOUBLE,0,_comm);
  //for (size_t i=0; i<x.size(); ++i)
  //  x[i] = xbuff[i];
  //delete[] xbuff;

  return _L->gradient(x);
}
  
std::string themistan_model::model_name() const
{
  return "themistan_model";
}

void themistan_model::get_param_names(std::vector<std::string>& names) const
{
  //Grab the param names from var_names
  names.resize(0);
  names.reserve(num_params_r__);
  for ( size_t i = 0; i < num_params_r__; ++i )
    names.push_back(_var_names[i]);
}

void themistan_model::get_dims(std::vector<std::vector<size_t> >& dimss) const
{
  dimss.resize(0);
  dimss.reserve(num_params_r__);
  std::vector<size_t> dims;
  for ( size_t i = 0; i < num_params_r__; ++i )
  {
    dimss.push_back(dims);
    dims.resize(0);
  }
}

void themistan_model::constrained_param_names( std::vector<std::string>& param_names, 
                                                        bool include_tparams, bool include_gqs) const
{
  for ( size_t i = 0; i < _var_names.size(); ++i )
    param_names.push_back(_var_names[i]);
}

void themistan_model::unconstrained_param_names(std::vector<std::string>& param_names,
                                                        bool include_tparams, bool include_gqs) const
{
  
  for ( size_t i = 0; i < _var_names.size(); ++i )
    param_names.push_back(_var_names[i]);
}

double themistan_model::log_prob(Eigen::VectorXd& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i);
  std::vector<int> vec_params_i;

  return log_prob(params, vec_params_i, msgs);
}

stan::math::var themistan_model::log_prob(Eigen::Matrix<stan::math::var, -1, 1>& params_r,
                         std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i).val();
  std::vector<int> vec_params_i;
  double lp = log_prob(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;
}

double themistan_model::log_prob_jacobian(Eigen::VectorXd& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i);
  std::vector<int> vec_params_i;
  return log_prob_jacobian(params, vec_params_i, msgs);
}

stan::math::var themistan_model::log_prob_jacobian(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i).val();
  std::vector<int> vec_params_i;
  double lp = log_prob_jacobian(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;
}

double themistan_model::log_prob_propto(Eigen::VectorXd& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i);
  std::vector<int> vec_params_i;
  return log_prob_propto(params, vec_params_i, msgs);
}

stan::math::var themistan_model::log_prob_propto(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i).val();
  std::vector<int> vec_params_i;
  double lp = log_prob_propto(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;
}

double themistan_model::log_prob_propto_jacobian(Eigen::VectorXd& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i);
  std::vector<int> vec_params_i;
  return log_prob_propto_jacobian(params, vec_params_i, msgs);
}

stan::math::var themistan_model::log_prob_propto_jacobian(Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params[i] = params_r(i).val();
  std::vector<int> vec_params_i;
  double lp = log_prob_propto_jacobian(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;

}

void themistan_model::transform_inits(const stan::io::var_context& context, Eigen::VectorXd& params_r, std::ostream* msgs) const
{
  std::vector<double> params;
  std::vector<int> params_i_vec;
  transform_inits(context, params_i_vec, params, msgs);
  params_r.resize(params.size());
  for ( int i = 0; i < params_r.size(); ++i )
    params_r(i) = params[i];
}

void themistan_model::write_array(boost::ecuyer1988& base_rng, 
                                          Eigen::VectorXd& params_r, 
                                          Eigen::VectorXd& vars, 
                                          bool emit_transformed_parameters, 
                                          bool emit_generated_quantities, std::ostream* msgs) const
{
  std::vector<double> params_r_vec(params_r.size(), 0.0);
  for ( int i = 0; i < params_r.size(); ++i )
    params_r_vec[i] = params_r(i);
  
  std::vector<double> vars_vec;
  std::vector<int> params_i_vec;
  write_array(base_rng, params_r_vec, params_i_vec, vars_vec, emit_transformed_parameters, emit_generated_quantities);
  vars.resize(vars_vec.size());
  for ( int i = 0; i < vars.size(); ++i )
    vars(i) = vars_vec[i];
}

double themistan_model::log_prob(std::vector<double>& params_r, std::vector<int>& params_i, std::ostream* msgs) const
{
  //Transform the HMC params to the ones Themis expects
  std::vector<double> params_t(params_r.size());
  for ( size_t i = 0; i < params_r.size(); ++i )
  {
    if ( (std::isfinite(_lower[i]) && std::isfinite(_upper[i])) )
      params_t[i] = stan::math::lub_constrain(params_r[i], _lower[i], _upper[i]);
    else if (std::isfinite(_lower[i]))
      params_t[i] = stan::math::lb_constrain(params_r[i], _lower[i]);
    else if (std::isfinite(_upper[i]))
      params_t[i] = stan::math::ub_constrain(params_r[i], _upper[i]);
    else
      params_t[i] = params_r[i];
  }

  // Distribute params_t across _comm
  //double *buff = new double[params_t.size()];
  //for (size_t i=0; i<params_t.size(); ++i)
  //  buff[i] = params_t[i];
  //MPI_Bcast(buff,params_t.size(),MPI_DOUBLE,0,_comm);
  //for (size_t i=0; i<params_t.size(); ++i)
  //  params_t[i] = buff[i];
  //delete[] buff;
  
  return _L->operator()(params_t);
}

stan::math::var themistan_model::log_prob(std::vector<stan::math::var>& params_r,
                                                  std::vector<int>& params_i,
                                                  std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( size_t i = 0; i < params_r.size(); ++i )
    params[i] = params_r[i].val();
  std::vector<int> vec_params_i;
  double lp = log_prob(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;
}



double themistan_model::log_prob_jacobian(
                                       std::vector<double>& params_r,
                                       std::vector<int>& params_i,
                                       std::ostream* msgs) const
{
  //Transform the HMC params to the ones Themis expects
  std::vector<double> params_t(params_r.size());
  double log_jacobian = 0.0;
  for ( size_t i = 0; i < params_r.size(); ++i )
  {
    double jac = 0.0;
    if ( (std::isfinite(_lower[i])&&std::isfinite(_upper[i])) ){
      params_t[i] = stan::math::lub_constrain(params_r[i], _lower[i], _upper[i], jac);
    } else if (std::isfinite(_lower[i])){ 
      params_t[i] = stan::math::lb_constrain(params_r[i], _lower[i], jac);
    }else if (std::isfinite(_upper[i])){
      params_t[i] = stan::math::ub_constrain(params_r[i], _upper[i], jac);
    }else{
      params_t[i] = params_r[i];
    }
    log_jacobian += jac;
    //std::cout << "PT: x_"<< i 
    //          << " = " << params_r[i] << std::endl;
  }

  // Distribute params_t across _comm
  //double *buff = new double[params_t.size()];
  //for (size_t i=0; i<params_t.size(); ++i)
  //  buff[i] = params_t[i];
  //MPI_Bcast(buff,params_t.size(),MPI_DOUBLE,0,_comm);
  //for (size_t i=0; i<params_t.size(); ++i)
  //  params_t[i] = buff[i];
  //delete[] buff;

  return _L->operator()(params_t) + log_jacobian;
}

stan::math::var themistan_model::log_prob_jacobian(
                                                std::vector<stan::math::var>& params_r,
                                                std::vector<int>& params_i,
                                                std::ostream* msgs) const
{
  std::vector<double> params(params_r.size(), 0.0);
  for ( size_t i = 0; i < params_r.size(); ++i )
    params[i] = params_r[i].val();
  std::vector<int> vec_params_i;
  double lp = log_prob_jacobian(params, vec_params_i, msgs);
  stan::math::var val(lp);
  return val;
}

double themistan_model::log_prob_propto(
                                     std::vector<double>& params_r,
                                     std::vector<int>& params_i,
                                     std::ostream* msgs) const
{
  return log_prob(params_r, params_i, msgs);
}


stan::math::var themistan_model::log_prob_propto(
                                     std::vector<stan::math::var>& params_r,
                                     std::vector<int>& params_i,
                                     std::ostream* msgs) const
{
  return log_prob(params_r, params_i, msgs);
}

double themistan_model::log_prob_propto_jacobian(
                                     std::vector<double>& params_r,
                                     std::vector<int>& params_i,
                                     std::ostream* msgs) const
{
  return log_prob_jacobian(params_r, params_i, msgs);
}


stan::math::var themistan_model::log_prob_propto_jacobian(
                                     std::vector<stan::math::var>& params_r,
                                     std::vector<int>& params_i,
                                     std::ostream* msgs) const
{
  return log_prob_jacobian(params_r, params_i, msgs);
}


void themistan_model::transform_inits(const stan::io::var_context& context,
                                   std::vector<int>& params_i,
                                   std::vector<double>& params_r,
                                   std::ostream* msgs) const
{
  params_r.resize(num_params_r__);
  //Move to unconstrained space for the initial step
  for ( size_t i = 0; i < num_params_r__; ++i )
  {
    //get the constrained initial position using the context.
    double param_scalar = context.vals_r(_var_names[i])[0];
    
    if ( (std::isfinite(_lower[i])&&std::isfinite(_upper[i])) )
      params_r[i] = stan::math::lub_free(param_scalar, _lower[i], _upper[i]);
    else if (std::isfinite(_lower[i]))
      params_r[i] = stan::math::lb_free(param_scalar, _lower[i]);
    else if (std::isfinite(_upper[i]))
      params_r[i] = stan::math::ub_free(param_scalar, _upper[i]);
    else
      params_r[i] = param_scalar;
  }
}

void themistan_model::write_array(boost::ecuyer1988& base_rng,
                               std::vector<double>& params_r,
                               std::vector<int>& params_i,
                               std::vector<double>& vars,
                               bool include_tparams, bool include_gqs,
                               std::ostream* msgs) const
{
  vars.resize(num_params_r__);
  for ( size_t i = 0; i < params_r.size(); ++i )
  {
    if ( (std::isfinite(_lower[i])&&std::isfinite(_upper[i])) )
      vars[i] = stan::math::lub_constrain(params_r[i], _lower[i], _upper[i]);
    else if (std::isfinite(_lower[i]))
      vars[i] = stan::math::lb_constrain(params_r[i], _lower[i]);
    else if (std::isfinite(_upper[i]))
      vars[i] = stan::math::ub_constrain(params_r[i], _upper[i]);
    else
      vars[i] = params_r[i];
  }
}
}//end Themis
