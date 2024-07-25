/*!
  \file uncertainty_visibility.h
  \author Avery E. Broderick
  \date  March, 2022
  \brief Header file for uncertainty model class approprirate for crosshand visibilities.
  \details To be added
*/

#ifndef Themis_UNCERTAINTY_CROSSHAND_VISIBILITIES_H_
#define Themis_UNCERTAINTY_CROSSHAND_VISIBILITIES_H_

#include "data_crosshand_visibilities.h"
#include <vector>
#include <complex>
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines the interface for uncertainty models that generate modified uncertainties for crosshand visibilities.

  \details To enable analyses that reconstruct a model of the uncertainties in addition to model fits, here a fixed interface that may be assumed by likelihoods, etc.  All uncertainty models that produce crosshand visibilities uncertainties should be child classes of uncertainty_crosshand_visibilities, thereby inheriting this interface.  The key functions are error and log_normalization, where the latter adds a penalty term for large uncertainties.  Note that this will make the chi-squared metric redundant as the errors will be inflated until the reduced chi-squared is near unity.

  This base class returns a trivial transformation for use in the absence of any uncertainty.

  \warning 
*/
class uncertainty_crosshand_visibilities
{
 public:
  uncertainty_crosshand_visibilities();
  virtual ~uncertainty_crosshand_visibilities();

  //! A user-supplied one-time generate function that permits uncertainty model construction prior to calling the crosshand_visibilities for each datum.  Takes a vector of parameters.
  virtual void generate_uncertainty(std::vector<double> parameters);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const;

  //! Returns the crosshand_visibilities error in Jy computed given a datum_crosshand_visibilities object, containing all of the accoutrements.
  virtual std::vector< std::complex<double> >& error(datum_crosshand_visibilities& d);

  //! Returns the crosshand_visibilities error in Jy computed given a datum_crosshand_visibilities object, containing all of the accoutrements.
  virtual double log_normalization(datum_crosshand_visibilities& d);
  
  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm);

 protected:
  MPI_Comm _comm;

  std::vector< std::complex<double> > _err;
};

};
#endif
