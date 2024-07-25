/*!
  \file uncertainty_visibility_broken_power_change.h
  \author Avery E. Broderick, Roman Gold
  \date  September, 2020, Apr 2021 
  \brief Header file for uncertainty model class approprirate for complex visibilities that adds a floor and fractional component.
  \details To be added
*/

#ifndef Themis_UNCERTAINTY_VISIBILITY_BROKEN_POWER_CHANGE_H_
#define Themis_UNCERTAINTY_VISIBILITY_BROKEN_POWER_CHANGE_H_

#include "data_visibility.h"
#include "uncertainty_visibility.h"
#include <vector>
#include <complex>
#include <mpi.h>

namespace Themis {

/*! 
  \brief Specifies an uncertainty model consisting of the original uncertainty plus a fixed floor, fixed fractional, and broken broken_power-law component, added in quadrature.

  \details The uncertainty is \f$ \sigma = \sqrt{ \Re(\sigma_V)^2 + t^2 + f^2 |V|^2 + a^2/[1+(|u|/u_0)^b] } + i \sqrt{ \Im(\sigma_V)^2 + t^2 + f^2 |V| + a^2/[1+(|u|/u_0)^b] } \f$, corresponding to the "broken_power change" model.

  Parameter list:\n
    - parameters[0] ... Threshold to add in Jy, \f$t\f$.
    - parameters[1] ... Fraction of visibility amplitude to add, \f$f\f$.
    - parameters[2] ... Noise value in Jy, \f$a\f$ either at zero baseline or at 4 Glambda.
    - parameters[3] ... Baseline length of break in G\f$ lambda \f$, \f$u_0\f$.
    - parameters[4] ... Long-baseline broken_power law index, \f$b\f$.

  \warning 
*/
class uncertainty_visibility_broken_power_change : public uncertainty_visibility
{
 public:
  uncertainty_visibility_broken_power_change();
  virtual ~uncertainty_visibility_broken_power_change();

  //! A user-supplied one-time generate function that permits uncertainty model construction prior to calling the visibility for each datum.  Takes a vector of parameters.
  virtual void generate_uncertainty(std::vector<double> parameters);

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual size_t size() const;

  //! Returns the complex visibility error in Jy computed given a datum_visibility object, containing all of the accoutrements.
  virtual std::complex<double> error(datum_visibility& d);

  //! Returns the complex visibility error in Jy computed given a datum_visibility object, containing all of the accoutrements.
  virtual double log_normalization(datum_visibility& d);

  //! Specify bpl noise amplitude at 4 Glambda instead of break location u0
  void constrain_noise_at_fixed_baseline(double baseline);

  //! Specify bpl noise amplitude at 4 Glambda instead of break location u0
  void constrain_noise_at_4Glambda();

  //! Specify logarithmic range in sigma_T, f, a, up
  void logarithmic_ranges();
  
  
  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilitates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm);

 protected:
  MPI_Comm _comm;

 private:
  double _error_threshold, _error_fraction;
  double _error_zero_baseline, _error_baseline_break, _error_baseline_index, _error_short_baseline_index;
  bool _var_at_fixed_baseline;
  double _amplitude_baseline;

  bool _logarithmic_ranges;
};

};
#endif
