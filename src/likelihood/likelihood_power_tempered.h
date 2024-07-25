/*! 
  \file likelihood_power_tempered.h
  \authors Paul Tiede
  \date  April, 2017
  \brief Modifies the likelihood according to a power tempering scheme.
  \internal This is just a paragraph of internal documentation
  which will not be displayed in the documentation
*/


#ifndef THEMIS_LIKELIHOOD_POWER_TEMPERED_H_
#define THEMIS_LIKELIHOOD_POWER_TEMPERED_H_

#include <string>
#include <vector>
#include "likelihood.h"
#include <iostream>

#include <mpi.h>

namespace Themis
{

  /*! 
    \brief Defines the power-tempered likelihood
    \details Defines the combined power_tempered likelihood class. This takes
    a generic combined likelihood function and then power tempers its value.
    This is useful when using a parallel tempering scheme.
  */
  class likelihood_power_tempered : public likelihood
  {
    public:
      
      //! Likelihood class constructor
      likelihood_power_tempered(likelihood& L);

      //! Likelihood class destructor
      ~likelihood_power_tempered() {};

      //! Sets the inverse temperature for the likelihood functions
      //! If not set, then beta = 1.
      void set_beta(double beta);

      void set_mpi_communicator(MPI_Comm comm){_L->set_mpi_communicator(comm);};

      //! get the likelihood without tempering.
      double get_lklhd_no_temp();





      //! get the reference normalization
      double referencelognorm()
      {
          std::vector<prior_base*> priors = _L->priors();
          double lnorm = 0.0;
          for (size_t i = 0; i < priors.size(); ++i)
          {
              double len = -std::log(priors[i]->upper_bound() - priors[i]->lower_bound());
              lnorm += len;
          }
          return lnorm;
      }

      likelihood* get_lklhd_unit_beta()
      {
          return _L;
      }

      //! Overloaded parenthesis operator that returns the log-likelihood multiplied by beta 
      //! of a vector in the parameter space at which the likelihood is to be calculated
      double operator() (std::vector<double>&);

      std::vector<double> gradient(std::vector<double>& x);

    private:
      likelihood* _L; 
      double _beta, _lklhd_no_temp;
  };
}

#endif 
