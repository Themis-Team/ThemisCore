/*!
  \file sampler_automated_factor_slice_sampler_MCMC.h
  \author Paul Tiede
  \brief Header file for a generic single factor slice sampler used in Themis. 
  This uses slice sampling to explore the distribution. The sampler will perform well 
  as long as there are not strong non-linear correlation between the parameters. One benefit
  is that the sampler is rejection free, albeit it only moves in one direction per sample.
  For more information see https://www.ncbi.nlm.nih.gov/pubmed/24955002 and Neal's introduction
  to slice sampling https://projecteuclid.org/euclid.aos/1056562461
*/


#ifndef THEMIS_SAMPLER_AUTOMATED_FACTOR_SLICE_SAMPLER_MCMC_H
#define THEMIS_SAMPLER_AUTOMATED_FACTOR_SLICE_SAMPLER_MCMC_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "likelihood.h"
#include "sampler_MCMC_base.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace Themis
{

/*!
  \class sampler_automated_factor_slice_sampler_MCMC
  \brief Defines the interface to an automated factor slice sampler, which isn't as efficient as Stan
  in terms of ESS per iteration, but doesn't require gradients and tends to do better than an adaptive 
  MH algorithm. Additionally it hardly uses any tuning parameters.
*/

class sampler_automated_factor_slice_sampler_MCMC : public sampler_MCMC_base
{
public:
  sampler_automated_factor_slice_sampler_MCMC(int seed, likelihood& L, std::vector<std::string> var_names, size_t dimension);
  ~sampler_automated_factor_slice_sampler_MCMC(){};

  /*!
    \brief Sets the starting location for the sampler.
    \param initial_parameters The initial location of the sampler. This must be set before calling run_sampler.
  */
  void set_initial_location(std::vector<double> initial);
  
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



  /*!
   \brief sets the adaption interval for the slice sampler. This will adapt the 
   directions of the slices and the windowing procedure for the problem. This is
   important for performance.
   \param num_warmup The number of adaptation steps to take. Default is 10,000.
   \param save_adapt Whether to save the adaptation steps.
   \note Be aggressive with adaptation.  
  */
  void set_adaptation_parameters(int num_warmup, bool save_adapt);

  /*!
    \brief Sets the initial widths to be used for slices.
    \param intitial_width The initial widths of the slices to be used.
    \note If you aren't sure what these should be make them too big. It is cheaper to 
    shrink a slice than to expand it.
  */
  void set_intial_widths(std::vector<double> intitial_width);

  /*!
    \brief Sets the window parameters for the adaptation. This is inspired by stan. 
    The initial window will be solely for building the initial covariance estimate
    and finding some reasonable initial widths for the slices. Window then says
    how long the initial covariance adaptation round will last. We then increase
    by factors of 2 until we have exhausted the adaptation stage of the sampler.
    \note at the the end of this we will have nadapt steps.
    \param init_buffer The initial length to adapt the slices.
    \param window The initial length of the covariance adaptation.
  */
  void set_window_parameters(int init_buffer, int window);

  /*!
    \brief Sets the initial covariance matrix for the slice sampler. This will then be 
    diagonalized to find the principal directions.
    \param covariance. The covariance matrix for the problem.
  */
  void set_initial_covariance(Eigen::MatrixXd covariance);



  /////////////////////////////////////////////////////////////////////
  //  Helper functions for the sampler. You don't need to use these

  /*
    \brief Access function that grabs the current state of the sampler, i.e. the chain, 
    and state of the sampler, i.e. the likelihood value of the chain and other pertinent information.
    about the sampler.
    \param chain_state Vector containing the current parameter values of the chain. Note these are the transformed parameters.
    \param sampler_state Vector containing the current state of the sampler, i.e. likelihood value, and other pertinent information.
  */
  void get_chain_state(std::vector<double>& parameters, std::vector<double>& sample_state);


  void set_chain_state(std::vector<double> model_values, std::vector<double> state_values);

  void write_checkpoint(std::ostream& );
  void read_checkpoint(std::string ckpt_file){sampler_MCMC_base::read_checkpoint(ckpt_file);};
  void read_checkpoint(std::istream& ckpt_in);

  void reset_sampler_step();

private:
  std::vector<double> _current_parameters, _current_state; 
  
  //adaptation counters
  bool _adaptation;
  bool _save_adaptation;
  int _nadapt, _start_step;
  double _width_tol, _cov_tol;
  int _width_adapt_stride, _cov_adapt_stride;
  int _adapt_width_count, _adapt_cov_count;
  int _number_slow_adapt;


  //!< restart flag.
  bool _restart;
  //Write information.
  void write_state();
  void write_chain_header();
  void write_state_header();

  void write_sampler_header();

  Eigen::MatrixXd _Gammak, _covariance; //collection of principal directions
  Eigen::VectorXd _Dk, _mean;
  std::vector<double> _initial_width; //initial interval lengths.
  std::vector<int> _nexpansion, _ncontraction; //number of expansion and contractions;
  std::vector<bool> _width_adapt;
  int _init_buffer, _window;
  std::vector<double> _lbound, _ubound;
  std::vector<Themis::transform_base*> _transforms;

  double loglklhd_transform(std::vector<double> cont_params);

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
                            bool save, int verbosity);

  
  void factor_slice_sample(const int k, const double h);

  //Finds the position in parameter space for the ray in the kth principal direction,
  //with parameter t. Where t is centered on the current parameter value.
  void ray_position(int k, double t, std::vector<double>& ray);
  //Does the step out procedure from Neal (2003) if the initial interval is 
  //too small and returns the number of expansions.
  int step_out(const int k, double logh,
                std::vector<double>& lower_ray, std::vector<double>& upper_ray, 
                double& tmin, double& tmax);
  
  //Does the shrink procedure if the sampler from the slice is outside the 
  //good area.
  void shrink(int k, std::vector<double>& lower_ray, std::vector<double>& upper_ray, double t );

  //updates the covariance matrix.
  void update_covariance(std::vector<double> sample);

  //sets the adaptation schedule.
  void set_adaptation_schedule();
  //updates the principal factors, i.e. directions.
  double factor_adapt();
  //updates the initial slice widths using a Robinson-Monroe recursion.
  void width_adapt();


  


}; //end sampler_automated_factor_slice_sampler_MCMC
}//end Themis

#endif
