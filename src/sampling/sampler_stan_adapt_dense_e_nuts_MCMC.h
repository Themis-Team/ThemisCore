/*!
  \file sampler_stan_adapt_dense_e_nuts_MCMC.h
  \author Paul Tiede
  \brief Header file stan sampler used in Themis.  
*/


#ifndef THEMIS_SAMPLER_STAN_ADAPT_DENSE_E_NUTS_MCMC_H
#define THEMIS_SAMPLER_STAN_ADAPT_DENSE_E_NUTS_MCMC_H

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <fstream>
#include "likelihood.h"
#include <stan/model/model_header.hpp>
#include <stan/math.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/version.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/diagnose/diagnose.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <stan/services/util/create_rng.hpp>

#include "themistan_model.h"
#include "sampler_MCMC_base.h"
namespace Themis
{
  /*
    \class sampler_stan_adapt_dense_e_nuts_MCMC
    \brief Defines the interface to the Stan sampler which uses a adaptive HMC sampling method
    using NUTS.
  */
class sampler_stan_adapt_dense_e_nuts_MCMC : public sampler_MCMC_base
{
public:
  sampler_stan_adapt_dense_e_nuts_MCMC(int seed, likelihood& L, std::vector<std::string> var_names, size_t dimension);
  ~sampler_stan_adapt_dense_e_nuts_MCMC(){};


  /*
    \brief sets starting location of the sampler
    \param initial_parameters The initial location of the sampler. This must be set before calling run_sampler.
  */
  void set_initial_location(std::vector<double> initial_parameters);
  
  /*
    \brief Main run function that will run the sampler.
    \param nsteps Number steps to run the sampler.
    \param thin   Will save every thin steps.
    \param refresh Will save ouput every refresh steps. If =-1 will never save.
    \param verbosity Verbosity of the sampler. 1 means output 0 means restricted output. If -1 no information will be saved.
  */
  void run_sampler(int nsteps,
                   int thin, int refresh,
                   int verbosity = 1);
  
  

  //////////////////////////////////////////////////////////////////////////
  //   Stan adaptation settings

  /*!
   \brief sets the adaption stuff with the applicable window stuff for stan.
   \param num_warmup The number of adaptation steps to take. Default is 1000.
   \param save_adapt Whether to save the adaptation steps.
   \note Be aggressive with adaptation. If HMC is poorly tuned the performance tanks.
  */
  void set_adaptation_parameters(int num_adapt, bool save_adapt);

  
  //! Set the initial step size for the hmc algorithm. 
  //! [default is 0.1]
  void set_stepsize(double step_size);
  //! Sets the max tree depth (max number of leapfrog steps is 2^tree_depth) for the sampler 
  //! [default 10]
  void set_max_depth(int max_depth);
  //! Sets the target acceptance rate for nuts 
  //! [default of 0.8 which is the theoretical optimal result in high dim]
  void set_delta(double delta);
  //! Sets the adaptation regularization scale which must be greater than zero [default is 0.05]
  void set_gamma(double gamma);
  //! Sets the adaptation relaxation scale which must be greater than zero because it should damp. 
  //! [default is 0.75]
  void set_kappa(double kappa);
  //! Sets the adaptation iteration offset, which must be greater than 0 
  //! [default is 10]
  void set_t0(double t0);
  //! Sets the window parameters for STAN, these rarely need to be changed.
  void set_window_parameters(unsigned int init_buffer, unsigned int term_buffer, unsigned int window);


  //! Sets the initial metric, default is the identity metric.
  void set_initial_inverse_metric(Eigen::MatrixXd inverse_metric);



  /////////////////////////////////////////////////////////////////////
  //  Helper functions for the sampler. You don't need to use these

  /*
    \brief Access function that grabs the current state of the sampler, i.e. the chain, 
    and state of the sampler, i.e. the likelihood value of the chain and other pertinent information.
    Every sampler that inherits sampler_exploration_mcmc_base must define this.
    about the sampler.
    \param chain_state Vector containing the current parameter values of the chain. Note these are the transformed parameters.
    \param sampler_state Vector containing the current state of the sampler, i.e. likelihood value, and other pertinent information.
  */
  void get_chain_state(std::vector<double>& parameters, std::vector<double>& sample_state);


  void set_chain_state(std::vector<double> model_values, std::vector<double> state_values);

  void write_checkpoint(std::ostream& );
  void read_checkpoint(std::string ckpt_file){sampler_MCMC_base::read_checkpoint(ckpt_file);};
  void read_checkpoint(std::istream& ckpt_in);
  
  virtual double get_rand(){
    boost::random::uniform_real_distribution<double> gen(0.0, 1.0);
    return gen(_stan_rng);
  };

  virtual void reset_rng_seed(int seed);

private:
  //<!themistan model rng, and sampler
  themistan_model _stan_model;
  boost::ecuyer1988 _stan_rng;
  stan::mcmc::adapt_dense_e_nuts<themistan_model,boost::ecuyer1988> _stan_sampler;


  //<! Adaption parameters
  unsigned int _init_buffer, _term_buffer, _window;
  Eigen::MatrixXd _inverse_metric;
  
  //Write information.
  void write_state();
  void write_chain_header();
  void write_state_header();

  void write_sampler_header();


  /*<!
    \brief Provides for fine grained control over the sampler. Namely, it just specifies the type of 
    iteratation the total number to be run and whether to save, and if its warmup.
    \param niterations The number of iterations to run.
    \param start What the starting index of the iteration is.
    \param finish What the final index is, this does not have to be start+niterations. 
    \param nthin The thinning step of the chain.
    \param refresh How often to refresh the progress of the sampler.
    \param save Whether to save the output.
    \param verbosity Whether we are printing a lot of information.
  */
  void generate_transition(int step, int start, int finish, int nthin, int refresh,
                            bool adaptation,
                            bool save, int verbosity);
  
  //<!intitial location
  std::vector<double> _initial;
  stan::mcmc::sample _current_sample;

  //writer stuff during the STAN run and stan specific saves.
  stan::callbacks::stream_logger _logger;

  //adaptation counters
  bool _adaptation;
  bool _save_adaptation;
  int _nadapt, _start_step;
  //!< restart flag.
  bool _restart;
  
  
};//end sampler_stan_adapt_dense_e_nuts_MCMC


} //Themis namespace
#endif
