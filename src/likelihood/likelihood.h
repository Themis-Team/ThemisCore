/*! 
  \file likelihood.h
  \authors 
  \date  April, 2017
  \brief Header file for the Likelihood class
  \internal This is just a paragraph of internal documentation
  which will not be displayed in the documentation
*/


#ifndef THEMIS_LKLHD_H_
#define THEMIS_LKLHD_H_

#include <string>
#include <vector>
#include "likelihood_base.h"
#include "likelihood_gaussian.h"
#include "likelihood_eggbox.h"
#include "likelihood_rosenbrock.h"
#include "likelihood_griewank.h"
#include "likelihood_cauchy.h"
#include "likelihood_lorentzian.h"
#include "likelihood_visibility_amplitude.h"
#include "likelihood_marginalized_visibility_amplitude.h"
#include "likelihood_closure_phase.h"
#include "likelihood_marginalized_closure_phase.h"
#include "likelihood_flux.h"
#include "transform_base.h"
#include "transform_fixed.h"
#include "transform_logarithmic.h"
#include "transform_none.h"
#include "prior.h"
#include <iostream>

#include <mpi.h>

namespace Themis
{

  /*! 
    \brief Defines the combined likelihood class
    \details Defines the combined likelihood class. This class combines different likelihoods into 
    a single likelihood used by the samplers. It also contains the priors on 
    all the parameters and the optional parameter transformations.
  */
  class likelihood
  {
    public:
      

      //! Likelihood class constructor
      likelihood(std::vector<prior_base*> P, std::vector<transform_base*> T, std::vector<likelihood_base*> L, std::vector<double>& W);

      //! Likelihood class constructor
      likelihood(std::vector<prior_base*> P, std::vector<likelihood_base*> L, std::vector<double>& W);
      
      //! Likelihood class destructor
      ~likelihood() {};


      //! Overloaded parenthesis operator that returns the log-likelihood 
      //! of a vector in the parameter space at which the likelihood is to be calculated
      virtual double operator() (std::vector<double>&);

      virtual double priorlognorm()
      {
          prior Pr(_P);
          return Pr.lognorm();
      }

      //! Gradient operator
      virtual std::vector<double> gradient(std::vector<double>&);

      //! Forward transformation function. It applies the parameter trasformations 
      //! supplied to the constructor on the corresponding parameters
      void forward_transform(std::vector<double>&);

      //! Returns the chi squared evaluated at a vector in the parameter space
      double chi_squared(std::vector<double>&);

      //! Returns a boolean flag, "true" indicates that coordinate transformations are supplied
      //! to be carried on and "false" indicates no parameter transformation is performed 
      bool transform_state();

      //! Set the likelihood to use finite differences in the gradient computation
      void use_finite_difference_gradients();

      //! Set the likelihood to use the intrinsic likelihood gradients in the gradient computation
      void use_intrinsic_likelihood_gradients();

      //! Returns access to the prior vector.
      std::vector<prior_base*> priors();
      //! Returns access to the transform vector.
      std::vector<transform_base*> transforms();
      //! Returns access to the base likelihood vector.
      std::vector<likelihood_base*> likelihoods();
      //! Returns access to the weights vector.
      std::vector<double>& weights();

      //! Defines a set of processors provided to the model for parallel
      //! computation via an MPI communicator.  Only facilates code 
      //! parallelization if the model computation is parallelized via MPI.
      void set_mpi_communicator(MPI_Comm comm);

      //! Output along a line
      void output_1d_slice(std::string fname, std::vector<double> p1, std::vector<double> p2, double xmin=-1, double xmax=2, size_t Nx=128);

      //! Output a surface
      void output_2d_slice(std::string fname, std::vector<double> p1, std::vector<double> p2, std::vector<double> p3, double xmin=-1, double xmax=2, size_t Nx=128, double ymin=-1, double ymax=2, size_t Ny=128);

      //! Output a chain of prior samples
      void write_prior_chain(std::string chainfile="prior_chain.dat", size_t nsamples=10000, int seed=42);

      //! A uniform prior check that does not evaluate the actual likelihood
      double prior_check(std::vector<double>& x);
	
    private:
      
      //! A vector of pointers to prior objects, these are the priors on
      //! each parameter in the parameter space
      std::vector<prior_base*> _P;
      
      //! A vector of pointers to transformation objects, these are the coordinate 
      //! transformations on each parameter in the parameter space
      std::vector<transform_base*> _T;
      
      //! A vector of pointers to likelihood objects, these are the individual 
      //! likelihoods that combine to make the total likelihood
      std::vector<likelihood_base*> _L;
      
      //! A vector of likelihood weights, these are the weights of individual 
      //! likelihoods in the final  combined likelihood
      std::vector<double> _W;
      
      //! A vector in the parameter space, this is the transformed parameter 
      //! vector if parameter transformations are supplied, otherwise it's the 
      //! original vector of parameters 
      std::vector<double> _X;
      
      //! Boolean variable,  "true" indicates that coordinate transformations 
      //! are supplied to be carried on and "false" indicates no parameter 
      //! transformation is performed 
      bool _is_transform_provided;

      //! Boolean switch to force full finite difference gradient computation at the
      //! combined likelihood level.  If model evaluation is the limiting factor
      //! in gradient times and there are not major efficiencies to be found in
      //! the individual likelihoods, this should be set.
      bool _use_finite_difference_gradients;
      

      inline double step_size(double u)
      {
	static const double cbrt_epsilon = std::cbrt(std::numeric_limits<double>::epsilon());
	return cbrt_epsilon * std::fabs(u); // Now u must be a non-zero scale!
      }

      
    protected:
      MPI_Comm _comm;
  };
}

#endif 
