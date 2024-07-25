/*!
  \file sampler_affine_invariant_tempered_MCMC.h
  \author Mansour Karami
  \brief Header file for sampler_affine_invariant_tempered_MCMC class
*/


#ifndef THEMIS_SAMPLER_AFFINE_INVARIANT_TEMPERED_MCMC_H
#define THEMIS_SAMPLER_AFFINE_INVARIANT_TEMPERED_MCMC_H

#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "../likelihood/likelihood.h"


namespace Themis{

/*! 
  \class sampler_affine_invariant_tempered_MCMC
  \brief Parallel tempered Markov Chain Monte Carlo sampler 
    
  \details Runs parallel tempered affine invariant ensemble sampling
  Markov Chain Monte Carlo chains to sample the likelihood
  surface. This routine uses parallel tempering on top of the affine
  invariant MCMC method of Goodman, J. and Weare,
  J. (2010). Parallel tempering is optimized by dynamically
  adjusting the temperature ladder as described in W. D. Vousden
  et. al (2016).  Given an object of type likelihood (which
  encompases the likelihood, priors and chi squared) the sampler
  explores and samples the likelihood surface over its dependent
  parameters. It will provide a sampling of the posterior probability
  distribution. The likelihood and the chi squared evaluated at the
  sampled points are also provided by the sampler.
*/
class sampler_affine_invariant_tempered_MCMC
{
 public:
  /*!
    \brief Class constructor, accepts the integer seed for a random number generator as an argument
   */
  sampler_affine_invariant_tempered_MCMC(int seed);
  ~sampler_affine_invariant_tempered_MCMC();

  /*! 
    \brief Function to run the sampler, takes a likelihood object, name of output files, and tuning parameters for the sampler and returns the MCMC chain, likelihoods and chi squared values.
    
    \param _L An object of class likelihood.
    \param length Number of steps (stretch moves) taken by the ensemble sampler.
    \param temp_stride Number of steps between subsequent communication among chains of different temperatures.
    \param chi2_stride Number of steps between outputing Chi squared values.
    \param chain_file String variable holding the name of the output MCMC chain file.
    \param lklhd_file String variable holding the name of the output likelihood file, contains log-likelihood values for each MCMC step.
    \param chi2_file  String variable holding the name of the output chi squared file, contains chi squared values for the MCMC chain.
    \param means Vector holding the mean values of parameters used for initializing the MCMC walkers.
    \param ranges Vector holding the standard deviation of parameters used for initializing the MCMC walkers.
    \param var_names A vector of strings. It holds the names for each sampled variable. The names are compiled as a header in the "chain_file". If the vector doesn't contain any names the header will not be generated. If the header is present The Themis analysis tools can use it to correctly label the generated diagnostics plots. 
    \param continue_flag Boolean variable. If set to "True" the sampler would use a checkpoint file to resume it's state and continue the run. If the output files exist the new data is appended to the same files. If set to "false" it will start a new chain using the provided "means" and "ranges" variables to initialize the chain. Note in the latter case existing output files will be overwitten by the new ones.
    \param output_precision Sets the output precision, the number of significant digits used to represent a number in the sampler output files. The defaul precision is 6.
    \param verbosity If set to one chain files will be produced for all tempering levels, otherwise only the lowest temperature will produce a chain file which is the deisred posterior probability distribution 
    \param adaptive_temperature If set to "true" the code will iteratively adapt the temperature ladder to get optimize the parallel tempering. If set to "false" the temperatures would remain constant. The latter case can be useful if one needs to find the bayesian evidence from the output postriors/likihoods at fixed temperatures. The default setting is "true" which is the best choice for most cases. 
    \param temperatures Optional vector to set the temperatures used for parallel tempering. 
  */
  void run_sampler(likelihood _L, int length, 
		     int temp_stride, int chi2_stride, std::string chain_file, std::string lklhd_file,
		     std::string chi2_file, std::vector<double> means, 
		   std::vector<double> ranges, std::vector<std::string> var_names,
		   bool continue_flag, int output_precision = 6, int verbosity = 0, bool adaptive_temperature=true, std::vector<double> temperatures=std::vector<double>(0));

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
    \brief Function to generate random numbers with probability density g(z)  given by \f$ g(z) = 1/\sqrt{z} \f$, for \f$ (1/a) < z < a  \f$
  */
  double RndGz(double);
  
  /*!
    \brief Function to set the distribution of processors in different layers of parallelization
    \param num_temperatures Integer value (\f$ \geq 1 \f$). Number of temperatures used by the parallel tempering algorithm. If set to one, the sampler will run without tempering.
    \param num_walkers Number of walkers used by ensemble sampler. This should be at least a few times the dimension of the parameter space.
    \param num_likelihood number of threads allocated for each likelihood calculation.


    The following plot shows how the sampler scales with different number of walkers per MPI process.
    The green line shows the ideal case of linear scaling where the run time is inversely proportional 
    to the number of MPI processes used. The purple line shows how the sampler scales with the number of
    MPI processes. As can be seen in the figure the scaling closely follows linear scaling and always 
    remains within \f$\%20\f$ of the ideal linear scaling. 
    \image html sampler_scaling2.png "Sampler scaling plot. The green line shows the linear scaling." 
  */
  void set_cpu_distribution(int num_temperatures, int num_walkers, int num_likelihood);  
  
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

  
 private:
  
  double t0, nu;
  int TNum, ChNum, LKLHD_Num, dim, _ckpt_stride; //, size;
  bool default_cpu_distribution;  
  std::string _ckpt_file;
  //double **proc;
  //double a;
  Ran2RNG _rng;
  GaussianRandomNumberGenerator<Ran2RNG> _grng;
  
};
};

#endif 
