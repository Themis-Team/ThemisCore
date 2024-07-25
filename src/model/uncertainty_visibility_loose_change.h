/*!
  \file uncertainty_visibility_loose_change.h
  \author Avery E. Broderick
  \date  September, 2020
  \brief Header file for uncertainty model class approprirate for complex visibilities that adds a floor and fractional component.
  \details To be added
*/

#ifndef Themis_UNCERTAINTY_VISIBILITY_LOOSE_CHANGE_H_
#define Themis_UNCERTAINTY_VISIBILITY_LOOSE_CHANGE_H_

#include "data_visibility.h"
#include "uncertainty_visibility.h"
#include <vector>
#include <complex>
#include <mpi.h>

namespace Themis {

/*! 
  \brief Specifies an uncertainty model consisting of the original uncertainty plus a fixed floor and fixed fractional component, added in quadrature.

  \details The uncertainty is \f$ \sigma = \sqrt{ \Re(\sigma_V)^2 + t + f |V| } + i \sqrt{ \Im(\sigma_V)^2 + t + f |V| } \f$, corresponding to the "loose change" model.

  Parameter list:\n
    - parameters[0] ... Threshold to add in Jy.
    - parameters[1] ... Fraction of visibility amplitude to add.

  \warning 
*/
class uncertainty_visibility_loose_change : public uncertainty_visibility
{
 public:
  uncertainty_visibility_loose_change();
  virtual ~uncertainty_visibility_loose_change();

  //! A user-supplied one-time generate function that permits uncertainty model construction prior to calling the visibility for each datum.  Takes a vector of parameters.
  virtual void generate_uncertainty(std::vector<double> parameters);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const;

  //! Returns the complex visibility error in Jy computed given a datum_visibility object, containing all of the accoutrements.
  virtual std::complex<double> error(datum_visibility& d);

  //! Returns the complex visibility error in Jy computed given a datum_visibility object, containing all of the accoutrements.
  virtual double log_normalization(datum_visibility& d);
  
  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm);

 protected:
  MPI_Comm _comm;

 private:
  double _error_threshold, _error_fraction;
  
};

};
#endif
