/*! 
  \file likelihood_base.h
  \author 
  \date  April, 2017
  \brief Header file for the Base Likelihood class
*/

#ifndef THEMIS_BASE_LKLHD_H_
#define THEMIS_BASE_LKLHD_H_

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>

#include "prior.h"

#include <mpi.h>

#include <iostream>
#include <fstream>

namespace Themis
{

  /*! 
    \brief Defines an interface to calculate log-likelihoods given a set of parameters for an specific model, and assuming that the corresponding datatypes have Gaussian errors.
    
    \details 

    \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation. 
  */
  class likelihood_base
  {
    public:
      likelihood_base();
      
      virtual ~likelihood_base();
      
      //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
      virtual double operator() (std::vector<double>& x);

      //! Returns the gradient of the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
      //! The prior permits parameter checking if required during likelihood gradient evaluation, through the gradients of the prior is applied elsewhere.
      virtual std::vector<double> gradient(std::vector<double>& x, prior& Pr);

      virtual std::vector<double> gradient_uniproc(std::vector<double>& x, prior& Pr);      
      
      //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
      virtual double chi_squared(std::vector<double>& x);

      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);


      //! Defines the way in which CPUs are distributed among likelihood and model computation.  CPUs assigned to the likelihood will be spread across gradient computation.  Sets the model communicator.
      virtual void set_cpu_distribution(int num_model);

      
      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.  ASSUMES that
      //! only process 0 on _comm is outputting.
      void output_model_data_comparison(std::ostream& out);

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output file.  Useful for comparison later.
      void output_model_data_comparison(std::string filename);

      
    private:
      double _logLikelihood;
      std::vector<double> _xlast;

      inline double step_size(double u)
      {
	static const double cbrt_epsilon = std::cbrt(std::numeric_limits<double>::epsilon());
	return cbrt_epsilon * std::fabs(u); // Now u must be a non-zero scale!
	//return cbrt_epsilon * std::fmax(1.0, std::fabs(u));
      }
                  
    protected:
      MPI_Comm _comm;
      MPI_Comm _Mcomm, _Lcomm;
      int _L_rank, _L_size;
      int _N_model;
      
      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.  ASSUMES that
      //! only process 0 on _comm is outputting.
      virtual void output(std::ostream& out);

      //! Initialization function that generates the model and likelihood communicators
      void initialize_mpi();
      
      std::ofstream _procerr;

      int _local_size, _global_size;
      double *_grad_local_buff, *_grad_global_buff;
      size_t _size;
      
  };

};

#endif 
