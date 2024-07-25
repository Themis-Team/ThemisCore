/*!
  \file sampler_differential_evolution_deo_tempered_MCMC.h
  \author Mansour Karami & Paul Tiede
  \brief Header file for sampler_differential_evolution_deo_tempered_MCMC class
*/


#ifndef THEMIS_SAMPLER_DIFFERENTIAL_EVOLUTION_DEO_TEMPERED_MCMC_H
#define THEMIS_SAMPLER_DIFFERENTIAL_EVOLUTION_DEO_TEMPERED_MCMC_H

#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "../likelihood/likelihood.h"
#include "interpolator1D.h"


namespace Themis{

/*! 
  \class sampler_differential_evolution_deo_tempered_MCMC
  \brief Parallel tempered Markov Chain Monte Carlo sampler 
    
  \details Runs parallel tempered differential evolution ensemble sampling
  Markov Chain Monte Carlo chains to sample the likelihood
  surface. This routine uses parallel tempering on top of the 
  differential evolution method of Cajo J.F Ter Braak (2006).
  This implementation closely follows that of B. Nelson et. al (2013)
  used for analyzing radial velocity observations (RUN DMC code).
  We also use the DEO tempering and adaption scheme from Syed et. al (2019).  
  Given an object of type likelihood (which
  encompases the likelihood, priors and chi squared) the sampler
  explores and samples the likelihood surface over its dependent
  parameters. It will provide a sampling of the posterior probability
  distribution. The likelihood and the chi squared evaluated at the
  sampled points are also provided by the sampler.
*/
class sampler_differential_evolution_deo_tempered_MCMC
{
 public:
  
  /*!
    \brief Class constructor, accepts the integer seed for a random number generator as an argument
   */
  sampler_differential_evolution_deo_tempered_MCMC(int seed);
  ~sampler_differential_evolution_deo_tempered_MCMC();

  /*! 
    \brief Function to run the sampler, takes a likelihood object, name of output files, and tuning parameters for the sampler and returns the MCMC chain, likelihoods and chi squared values.
    
    \param _L An object of class likelihood.
    \param length Number of steps (stretch moves) taken by the ensemble sampler.
    \param thin Frequency that output is saved, e.g. if thin=10 then every 10 steps will be saved into the Chain and Lklhd file. Ideally this is set to the autocorrelation time of the sampler, but this is unknown before actually running the problem.
    \param temp_stride Number of steps between subsequent communication among chains of different temperatures.
    \param chi2_stride Number of steps between outputting Chi squared values.
    \param chain_file String variable holding the name of the output MCMC chain file.
    \param lklhd_file String variable holding the name of the output likelihood file, contains log-likelihood values for each MCMC step.
    \param chi2_file  String variable holding the name of the output chi squared file, contains chi squared values for the MCMC chain.
    \param annealing_file String variable holding the name of the output file that contains important stats for acceptance rates between tempering levels. Also produces the summary file which appends .summary to the annealing_file string.
    \param means Vector holding the mean values of parameters used for initializing the MCMC walkers.
    \param ranges Vector holding the standard deviation of parameters used for initializing the MCMC walkers.
    \param var_names A vector of strings. It holds the names for each sampled variable. The names are compiled as a header in the "chain_file". If the vector doesn't contain any names the header will not be generated. If the header is present The Themis analysis tools can use it to correctly label the generated diagnostics plots. 
    \param continue_flag Boolean variable. If set to "True" the sampler would use a checkpoint file to resume it's state and continue the run. If the output files exist the new data is appended to the same files. If set to "false" it will start a new chain using the provided "means" and "ranges" variables to initialize the chain. Note in the latter case existing output files will be overwitten by the new ones.
    \param output_precision Sets the output precision, the number of significant digits used to represent a number in the sampler output files. The defaul precision is 6.
    \param verbosity If set to one chain files will be produced for all tempering levels, otherwise only the lowest temperature will produce a chain file which is the deisred posterior probability distribution 
    \param inverse_temperatures Optional vector to set the \f$ \beta=1/T\f$ used for parallel tempering. 
  */
  void run_sampler(likelihood _L, int start_length, int thin, 
		   int temp_stride, int chi2_stride, std::string chain_file, std::string lklhd_file,
		   std::string chi2_file, std::string annealing_file, std::vector<double> means, 
		   std::vector<double> ranges, std::vector<std::string> var_names,
		   bool continue_flag, int output_precision = 6, int verbosity = 0, 
                   std::vector<double> inverse_temperatures=std::vector<double>(0));

  /*! 
    \brief Function to generate random numbers with a gaussian distribution.
  */
  double RndGaussian(double, double, bool);

  /*!
    \brief Function to generate real valued random numbers in an interval.
  */
  double RndUni(double, double);
  
  /*!
    \brief Function to generate integer valued random numbers in an interval.
  */
  int RndUnint(int, int);

  /*!
    \brief Function to set the distribution of processors in different layers of parallelization
    \param num_replicas Integer value (\f$ \geq 1 \f$). Number of replicas of the monte carlo process, i.e. number of temperatures. If set to one, the sampler will run without tempering.
    \param num_walkers Number of walkers used by ensemble sampler. This should be at least a few times the dimension of the parameter space.
    \param num_likelihood number of threads allocated for each likelihood calculation.


    The following plot shows how the sampler scales with different number of walkers per MPI process.
    The green line shows the ideal case of linear scaling where the run time is inversely proportional 
    to the number of MPI processes used. The purple line shows how the sampler scales with the number of
    MPI processes. As can be seen in the figure the scaling closely follows linear scaling and always 
    remains within \f$\%20\f$ of the ideal linear scaling. 
    \image html sampler_scaling2.png "Sampler scaling plot. The green line shows the linear scaling." 
  */
  void set_cpu_distribution(int num_replicas, int num_walkers, int num_likelihood);  


  /*!
    \brief Function to set the parameters of the tempering schedule. We now use the adaption scheme from Syed 2019. 
    We need the number of adaption rounds to use and the geometric increase factor for the number of rounds in each sample.
    \param num_rounds is the number of rounds to run
  */
  void set_annealing_schedule(int num_rounds, int geometric_increase, double initial_spacing=1.15);

  /*!
    \brief Function reads in an annealing.dat file and set the initial ladder for the sampler to the ladder from the final round 
    in the annealing file.
    \param annealing_file is the file containing the annealing information for the run. Namely the round, ladder, and rejection rates.
  */
  void read_initial_ladder(std::string annealing_file);
  
  /*!
    \brief Function to set the checkpoint/restart functionality.
    \param ckpt_stride Number of steps between writing a new checkpoint.
    \param ckpt_file String variable holding the name of the output checkpoint file.
   */
  void set_checkpoint(int ckpt_stride, std::string ckpt_file);

  /*!
    \brief Function to estimate the bayesian evidence using the Thermodynamic Integration method
    \param file_names Vector of strings holding the names of the likelihood files corresponding to each tempered level.
    \param temperatures Vector containing the temperature values for the corresponding log-likelihood sample files. The order of temperatures and likelihood file names in their vectors must be the same.
    \param burn_in Number of burn-in samples to exclude from the analysis.
   */
  void estimate_bayesian_evidence(std::vector<std::string> file_names, std::vector<double> temperatures, int burn_in);


  /*!
    \brief Finds the best fit within provided chain file and returns it.
  */
  std::vector<double> find_best_fit(std::string chain_file, std::string lklhd_file);

  /*!
   \brief Function that finds the communication barrier of the tempering problem 
  */
  
 private:
  
  int _nrounds, _b; //Adaption parameters.
  double _initial_spacing; //Initial geometric ladder spacing
  int TNum, ChNum, LKLHD_Num, dim, _ckpt_stride; //, size;
  bool default_cpu_distribution;
  bool _no_initial_ladder;
  std::string _old_annealing_file;
  std::string _ckpt_file;
  Ran2RNG _rng;
  GaussianRandomNumberGenerator<Ran2RNG> _grng;

  //<! Communicator stuff
  MPI_Comm E_COMM, T_COMM, L_COMM, C_COMM;
  int E_size, E_rank, T_size, T_rank, L_size, L_rank, C_size, C_rank;


  //<! There are a bunch of utility functions to aid me in construction.
  //<! Algorithm 2 of Syed 2019 that uses the previous tempering schedule and the rejection rates
  //<! of the MH swapping statistic to find the new optimized schedule.
  //<! Returns the estimated round trip rate for the previous round.
  double update_annealing_params(double *beta, double R[]);

  //<! Bisection to find the temperature ladder for rank k using the monotone interpolator.
  //<! and solving equation (32) of Syed (2019).
  double find_beta(Interpolator1D& fLambda, int k, double Lambda, double eps=1e-12, int MAX_ITR=1e8);






  
};
};

#endif 
