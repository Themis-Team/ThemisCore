/*!
  \file optimizer_kickout_powell.h
  \author Avery Broderick
  \brief Header file for optimizer_kickout_powell class
*/


#ifndef THEMIS_OPTIMIZER_KICKOUT_POWELL_H
#define THEMIS_OPTIMIZER_KICKOUT_POWELL_H

#include <mpi.h>
#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "../likelihood/likelihood.h"


namespace Themis
{

  /*! 
    \class optimizer_kickout_powell
    \brief Powell scheme optimizer with repeated kickouts for likelihoods
    
    \details Maximizes a given likelihood using the tempered_powell method as 
    described in Numerical Recipes in C (1992; Press, Teukoslky, Vetterling 
    and Flannery). An attempt to generate a global maximum is made by starting
    many instances at random/specified locations. Multiple rounds of trial 
    solutions of a fixed number of Powell iterations are generated and then 
    compared to an estimate of the expected value.  If too small, a new random 
    point is chosen. This drives the optimizer to try increasingly better
    locations. To avoid pathological behavior at the prior limits, if present, 
    the likelihood is "tempered" near the boundary by a function that drives 
    the likelihood to zero to generate smooth maxima. (This does not appear to
    be particularly important.)
    
    \todo
  */
class optimizer_kickout_powell
{
 public:
  
  /*!
    \brief Class constructor, accepts the integer seed for a random number generator as an argument
  */
  optimizer_kickout_powell(int seed);
  ~optimizer_kickout_powell();

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param dof_estimate An estimate of the number of expected degrees of freedom.  Better estimates will produce better quality checks, but even bad estimates are very useful.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_instances Number of independent realizations of the optimizer to run.  When set to 0 will run the maximum allowed by the number of processes being used.  Default is 0.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, int dof_estimate, std::string optimizer_results_filename="Opt.dat", size_t number_of_instances=0, size_t number_of_restarts=2, double tolerance=1e-10);

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param dof_estimate An estimate of the number of expected degrees of freedom.  Better estimates will produce better quality checks, but even bad estimates are very useful.
    \param start_parameter_values A vector of a single start point that we want to initialize a Powell optimization around.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_instances Number of independent realizations of the optimizer to run.  When set to 0 will run the maximum allowed by the number of processes being used.  Default is 0.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, int dof_estimate, std::vector<double> start_parameter_values, std::string optimizer_results_filename="Opt.dat", size_t number_of_instances=0, size_t number_of_restarts=2, double tolerance=1e-10);

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param dof_estimate An estimate of the number of expected degrees of freedom.  Better estimates will produce better quality checks, but even bad estimates are very useful.
    \param start_parameter_values A vector of start points at which to initialize every Powell optimization around.  Note that this sets the number of instances explicitly.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, int dof_estimate, std::vector< std::vector<double> > start_parameter_values, std::string optimizer_results_filename="Opt.dat", size_t number_of_restarts=2, double tolerance=1e-10);


  /*!
    \brief Function to set the distribution of processors in different layers of parallelization
    \param num_likelihood number of threads allocated for each likelihood calculation.
  */
  void set_cpu_distribution(int num_likelihood);  
 
  /*!
    \brief Function to set the meta-parameters associated with the kickout procedure.
    \param kickout_loglikelihood_reduction_factor Factor by which to multiply the negative 
    of the number of degrees of freedom above which a maximized likelihood is acceptable. Default 10.
    \param kickout_itermax Number of Powell iterations before kickout assessment.  Default 20.
    \param kickout_rounds Number of kickout rounds to perform.  Default 20.
  */
  void set_kickout_parameters(double kickout_loglikelihood_reduction_factor=10.0, size_t kickout_itermax=20, size_t kickout_rounds=20);

 
 private:
  likelihood* _Lptr; // Pointer to provide internal access to likelihood
  size_t _ndim; // Number of dimensions of parameter space
  int _num_likelihood; // Nuber of cores per likelihood
  Ran2RNG _rng; // Random number generator

  double _prior_edge_beta;
  int _dof_estimate;

  double _ko_ll_red_fac;
  size_t _ko_itermax, _ko_rounds;

  /*!
    \brief Function to generate a random starting point given the bounds on the prior.  Assumes an 
    arctan distribution if both bounds on the prior are infinity, an exponential distribution if the bound 
    on one side is infinity, and a uniform distribution if both bounds are finite.
    \param P vector of pointers to priors.
  */
  std::vector<double> generate_start_point(likelihood& L);

  /*!
    \brief Function to ncapsulate the process of optimization.  Returns the maximized likelihood and
    resets pvec to the optimal position.  
    \param pvec vector of start parameter position, reset to the optimal point.
    \param P vector of pointers to priors.
    \param tolerance numerical tolerance at which to terminate maximization.
  */
  double get_optimal_point(std::vector<double>& pvec, double tolerance, int itermax=0);


  // Negative of the log-likelihood evaluated at the unit-offset parameter array p[]
  double func(double p[]);
  double tempered_func(std::vector<double> p, double prior_edge_beta=1.0);

  // One-dimensional (along _xicom) version of func
  double *_pcom, *_xicom;
  double f1dim(double x);

  // NR routines
  void powell(double p[], double **xi, double ftol, int& iter, double& fret, int itermax=0);
  void linmin(double p[], double xi[], double& fret);
  double brent(double ax, double bx, double cx, double tol, double& xmin);
  void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc);

};
};

#endif 
