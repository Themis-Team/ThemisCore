/*!
  \file optimizer_powell.h
  \author Avery Broderick
  \brief Header file for optimizer_powell class
*/


#ifndef THEMIS_OPTIMIZER_POWELL_H
#define THEMIS_OPTIMIZER_POWELL_H

#include <mpi.h>
#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "../likelihood/likelihood.h"


namespace Themis
{

  /*! 
    \class optimizer_powell
    \brief Powell scheme optimizer for likelihoods
    
    \details Maximizes a given likelihood using the powell method as 
    described in Numerical Recipes in C (1992; Press, Teukoslky, Vetterling 
    and Flannery). An attempt to generate a global maximum is made by starting
    many powelles at random/specified locations. The maximum allowed iteratitons
    is set to 200.
    
    \todo
  */
class optimizer_powell
{
 public:
  
  /*!
    \brief Class constructor, accepts the integer seed for a random number generator as an argument
  */
  optimizer_powell(int seed);
  ~optimizer_powell();

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param P A vector of pointers to priors.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_instances Number of independent realizations of the optimizer to run.  When set to 0 will run the maximum allowed by the number of processes being used.  Default is 0.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, std::string optimizer_results_filename="Opt.dat", size_t number_of_instances=0, size_t number_of_restarts=1, double tolerance=1e-15);

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param P A vector of pointers to priors.
    \param start_parameter_values A vector of a single start point that we want to initialize a powell around.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_instances Number of independent realizations of the optimizer to run.  When set to 0 will run the maximum allowed by the number of processes being used.  Default is 0.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, std::vector<double> start_parameter_values, std::string optimizer_results_filename="Opt.dat", size_t number_of_instances=0, size_t number_of_restarts=1, double tolerance=1e-15);

  /*! 
    \brief Function to run the optimizer, takes a likelihood object, vector of priors, name of an optimizer output file, and tuning parameters for the optimizer.
    
    \param L An object of class likelihood.
    \param P A vector of pointers to priors.
    \param start_parameter_values A vector of start points at which to initialize every powell around.  Note that this sets the number of instances explicitly.
    \param optimizer_results_filename Name of file to which to write summary data.  Default is Opt.dat.
    \param number_of_restarts The number of times to restart each optimizer realization from the best point.  Default 2.
    \param tolerance Convergence tolerance.  Default 1e-15.
  */
  std::vector<double> run_optimizer(likelihood& L, std::vector< std::vector<double> > start_parameter_values, std::string optimizer_results_filename="Opt.dat", size_t number_of_restarts=1, double tolerance=1e-15);


  /*!
    \brief Function to set the distribution of processors in different layers of parallelization
    \param num_likelihood number of threads allocated for each likelihood calculation.
  */
  void set_cpu_distribution(int num_likelihood);  
  
 private:
  likelihood* _Lptr; // Pointer to provide internal access to likelihood
  size_t _ndim; // Number of dimensions of parameter space
  int _num_likelihood; // Nuber of cores per likelihood
  Ran2RNG _rng; // Random number generator

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
  double get_optimal_point(std::vector<double>& pvec, double tolerance);


  // Negative of the log-likelihood evaluated at the unit-offset parameter array p[]
  double func(double p[]);

  // One-dimensional (along _xicom) version of func
  double *_pcom, *_xicom;
  double f1dim(double x);

  // NR routines
  void powell(double p[], double **xi, double ftol, int& iter, double& fret);
  void linmin(double p[], double xi[], double& fret);
  double brent(double ax, double bx, double cx, double tol, double& xmin);
  void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc);

};
};

#endif 
