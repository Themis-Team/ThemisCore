/*!
  \file themistan_model.h
  \author Paul Tiede
  \brief Header file for hook of Themis model into Stan.  
*/

#include "base_hamiltonian_themis.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <fstream>
#include "likelihood.h"
#include <stan/model/model_header.hpp>
#include <stan/math.hpp>
#ifndef THEMIS_THEMISTAN_MODEL_H
#define THEMIS_THEMISTAN_MODEL_H

namespace Themis
{


//ThemiStan likelihood hook
class themistan_model : public stan::model::model_base
{
 public:
  themistan_model(likelihood& L, 
                  std::vector<std::string> var_names, 
                  std::ostream* msg_stream);

  ~themistan_model() {};

  // std::vector<double> get_themis_gradient(std::vector<double>& x) const {return _L->gradient(x);};
  std::vector<double> get_themis_gradient(std::vector<double>& x) const;
  std::vector<double> get_lower_bounds() const {return _lower;};
  std::vector<double> get_upper_bounds() const{return _upper;};

  void set_mpi_communicator(MPI_Comm comm){_comm = comm;};

  //All models we care about are themistan models.
  virtual std::string model_name() const; 

  //copies the var_names variable to names
  virtual void get_param_names(std::vector<std::string>& names) const;

  //this is just a vector of empty vectors since our params are all scalars
  virtual void get_dims(std::vector<std::vector<size_t> >& dimss) const;

  //Don't differentiate names for constrained and unconstrained params
  virtual void constrained_param_names(std::vector<std::string>& param_names,
                                           bool include_tparams = true,
                                           bool include_gqs = true) const;
  virtual void unconstrained_param_names(std::vector<std::string>& params_names, 
                                         bool include_tparams = true,
                                             bool include_gqs = true) const;


  //Now the vector definitions...
  //This is where the method is actually implemented for non-jacobian log_probs
  //Note we don't include the normalization constant because Themis doesn't care about them.
  virtual double log_prob(std::vector<double>& params_r,
                          std::vector<int>& params_i,
                          std::ostream* msgs) const;



  //This is where the method is actually implemented for jacobian probs
  //Note we don't include the normalization constant because Themis doesn't care about them.
  virtual double log_prob_jacobian(std::vector<double>& params_r,
                                   std::vector<int>& params_i,
                                   std::ostream* msgs) const;

      
  //The rest of the methods all default to the above two with some transformation so don't worry about editing these.
  virtual stan::math::var log_prob(std::vector<stan::math::var>& params_r,
                                   std::vector<int>& params_i,
                                   std::ostream* msgs) const;
  virtual stan::math::var log_prob_jacobian(std::vector<stan::math::var>& params_r,
                                            std::vector<int>& params_i,
                                                std::ostream* msgs) const;
    
  virtual double log_prob_propto(std::vector<double>& params_r,
                                 std::vector<int>& params_i,
                                 std::ostream* msgs) const;

  virtual stan::math::var log_prob_propto(std::vector<stan::math::var>& params_r,
                                          std::vector<int>& params_i,
                                          std::ostream* msgs) const;

      
  virtual double log_prob_propto_jacobian(std::vector<double>& params_r,
                                          std::vector<int>& params_i,
                                          std::ostream* msgs) const;

      
  virtual stan::math::var log_prob_propto_jacobian(std::vector<stan::math::var>& params_r,
                                                   std::vector<int>& params_i,
                                                   std::ostream* msgs) const;

      
  //Eigen methods all just switch to vector ones at this point
  virtual double log_prob(Eigen::VectorXd& params_r,
                          std::ostream* msgs) const;

  virtual stan::math::var log_prob(Eigen::Matrix<stan::math::var, -1, 1>& params_r,
                                   std::ostream* msgs) const;

  virtual double log_prob_jacobian(Eigen::VectorXd& params_r,
                                   std::ostream* msgs) const;

  virtual stan::math::var log_prob_jacobian(Eigen::Matrix<stan::math::var, -1, 1>& params_r, 
                                            std::ostream* msgs) const;

      
  virtual double log_prob_propto(Eigen::VectorXd& params_r,
                                 std::ostream* msgs) const;
  
  virtual stan::math::var log_prob_propto(Eigen::Matrix<stan::math::var, -1, 1>& params_r,
                                          std::ostream* msgs) const;
      
  virtual double log_prob_propto_jacobian(Eigen::VectorXd& params_r,
                                          std::ostream* msgs) const;

      
  virtual stan::math::var log_prob_propto_jacobian(
        Eigen::Matrix<stan::math::var, -1, 1>& params_r, std::ostream* msgs) const;


  template <bool propto, bool jacobian, typename T>
  inline T log_prob(std::vector<T>& params_r, std::vector<int>& params_i,
                    std::ostream* msgs) const {
    if (propto && jacobian)
      return log_prob_propto_jacobian(params_r, params_i, msgs);
    else if (propto && !jacobian)
      return log_prob_propto(params_r, params_i, msgs);
    else if (!propto && jacobian)
      return log_prob_jacobian(params_r, params_i, msgs);
    else  // if (!propto && !jacobian)
      return log_prob(params_r, params_i, msgs);
  }


  template <bool propto, bool jacobian, typename T>
  inline T log_prob(Eigen::Matrix<T, -1, 1>& params_r,
                    std::ostream* msgs) const {
    if (propto && jacobian)
      return log_prob_propto_jacobian(params_r, msgs);
    else if (propto && !jacobian)
      return log_prob_propto(params_r, msgs);
    else if (!propto && jacobian)
      return log_prob_jacobian(params_r, msgs);
    else  // if (!propto && !jacobian)
      return log_prob(params_r, msgs);
  }
      
  virtual void transform_inits(const stan::io::var_context& context,
                               Eigen::VectorXd& params_r,
                                   std::ostream* msgs) const;

  
  virtual void write_array(boost::ecuyer1988& base_rng,
                           Eigen::VectorXd& params_r,
                           Eigen::VectorXd& params_constrained_r,
                           bool include_tparams = true, bool include_gqs = true,
                           std::ostream* msgs = 0) const;

      
      
  virtual void transform_inits(const stan::io::var_context& context,
                               std::vector<int>& params_i,
                               std::vector<double>& params_r,
                               std::ostream* msgs) const;

      
      
  virtual void write_array(boost::ecuyer1988& base_rng,
                           std::vector<double>& params_r,
                           std::vector<int>& params_i,
                           std::vector<double>& params_r_constrained,
                           bool include_tparams = true, bool include_gqs = true,
                           std::ostream* msgs = 0) const;

  std::vector<double> _gradient;

 private:
  likelihood* _L;
  std::vector<std::string> _var_names;
  std::vector<double> _lower, _upper;
  MPI_Comm _comm;
}; //themistan_model



}//end Themis


#endif
