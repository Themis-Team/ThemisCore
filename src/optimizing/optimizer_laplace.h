/*!
  \file optimizer_laplace.h
  \author Paul Tiede
  \brief Header file for a class that does a Fisher-Matrix analysis of the likelihood.
*/

#ifndef THEMIS_OPTIMIZER_LAPLACE_H
#define THEMIS_OPTIMIZER_LAPLACE_H

#include <mpi.h>
#include <vector>
#include <string>
#include "../util/random_number_generator.h"
#include "likelihood.h"
#include <Eigen/Core>
#include <LBFGSB.h>

namespace Themis
{
  /*!
    \class optimizer_laplace.h
    \brief Optimizes the likelihood and then makes a Laplace approximation approximate the likelihood.
  */
class optimizer_laplace
{
public:
  optimizer_laplace(likelihood& L, 
      std::vector<std::string> var_names, 
      size_t dimension);

  //!Runs the optimizer and returns best parameters by reference
  int run_optimizer(std::vector<double>& parameters, double& MAP);
  int run_optimizer(Eigen::VectorXd& parameters, double& MAP);

  void set_start_points(std::vector<std::vector<double> > start_points);
  //! Parallel version of run_optimizer where we run number_of_instances copies of run_optimizer and find the best
  //! one.
  int parallel_optimizer(std::vector<double>& parameters, double& MAP, size_t number_of_instances, int seed=42, std::string optout="optimizer_parallel.out");

  //! Finds the inverse covariance matrix using the Laplace approximation around the peak defined by params.
  void find_precision(const Eigen::VectorXd& params, Eigen::MatrixXd& precision, std::string outname = "");

  //! Sets some tuning parameters for the BFGS algorithm
  //! \param epsilon : convergence tolerance for projected gradient. DEFAULT 1e-6
  //! \param ftol 
  //! \param max_iterations : The maximum number of iterations to run. If zero runs until convergence. DEFAULT 500
  //! \param max_iterations : The maximum number of trials for the linesearch to run. DEFAULT 200
  void set_parameters(double epsilon, size_t max_iterations, size_t max_linsearch); 

  //! Sets the scale for the loglikelihood
  void set_scale(double scale);

  void set_cpu_distribution(int num_likelihood);

private:
  
  //Stores a pointer to the likelihood function to be used for computation
  likelihood* _L;
  std::vector<std::string> _var_names;
  size_t _dimension;
  //Holds the bounds for the problem defined by the loglikelihood.
  Eigen::VectorXd _lower, _upper;
  std::vector<transform_base*> _transforms;
  double _epsilon;
  size_t _max_iterations;
  size_t _max_linesearch;
  double _scale; // Likelihood scale parmater. This should be the expected value for the likelihood
  size_t _num_likelihood;
  std::vector<std::vector<double> > _start_points;

  inline double step_size(double u)
  {
    static const double cbrt_epsilon = std::cbrt(std::numeric_limits<double>::epsilon());
    return cbrt_epsilon * std::max(1.0, std::fabs(u)); 
  }

  //!< Helper function for the loglikelihood so it is in the form 
  //LBGSpp wants.
  class _nlogprob
  {
    public: 
      _nlogprob(likelihood* L, size_t dimension):__L(L),__dimension(dimension),__scale(1.0){}
      void set_tranforms(std::vector<transform_base*> transforms){__transforms=transforms;}
      void set_scale(double scale);
      double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
      double loglikelihood_trans(const Eigen::VectorXd& x);
      double loglikelihood_trans(const std::vector<double>& x);
      
    private:
      likelihood* __L;
      std::vector<transform_base*> __transforms;
      size_t __dimension;
      double __scale;




  };



  _nlogprob _nln;


  std::vector<double> generate_start_point(Ran2RNG& rng);

};//end optimizer_laplace
 
};//end Themis

#endif
