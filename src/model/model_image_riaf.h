/*!
  \file model_image_riaf.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for the extended RIAF image model class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_RIAF_H_
#define Themis_MODEL_IMAGE_RIAF_H_

#include "model_image.h"
#include "vrt2.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {
  
/*!
  \brief Defines an image model based on the RIAF models employed in Broderick, Fish, Doeleman and Loeb (2011) and Broderick et al. (2016), permitting greater freedom to modify the underlying semi-analytic accretion flow structure.

  \details Provides explicit implementation of the model_image object for the semi-analytic RIAF models.  There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_image_riaf.cpp for more.  It also provides a flux estimate for inclusion in later flux_model extensions (model_riaf).

  Parameter list:\n
  - parameters[0] ... Black hole spin parameter (-1 to 1).
  - parameters[1] ... Cosine of the spin inclination relative to line of sight.
  - parameters[2] ... "Thermal" electron population density normalization in \f${\rm cm}^{-3}\f$.  
  - parameters[3] ... "Thermal" electron population density radial power law.
  - parameters[4] ... "Thermal" electron population density \f$h/r\f$.
  - parameters[5] ... "Thermal" electron population temperature normalization in K.
  - parameters[6] ... "Thermal" electron population temperature radial power law.
  - parameters[7] ... Power-law electron population density normalization in \f${\rm cm}^{-3}\f$.
  - parameters[8] ... Power-law electron population density radial power law.
  - parameters[9] ... Power-law electron population density \f$h/r\f$.
  - parameters[10] .. Power-law electron population density spectral index.
  - parameters[11] .. Power-law electron population density minimum Lorentz factor.
  - parameters[12] .. Plasma beta.
  - parameters[13] .. Sub-keplerian parameter, i.e., sets \f$u_\phi/u_t =\f$ parameters[13]\f$\times\f$\<keplerian value\>.
  - parameters[14] .. Position angle (in model_image) in radians.

 \warning This defines a model_image object associated with a RIAF observed at single frequency.  Requires the VRT2 library to be installed.
*/
class model_image_riaf : public model_image
{
 private:
  //! Generates and returns rectalinear grid of intensities associated with the RIAF image model in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:
  //! Constructor to make a RIAF model.  Takes a frequency (Hz), mass (cm), distance (cm).
  model_image_riaf(double frequency=230e9, double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  virtual ~model_image_riaf() {};
  
  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return 15; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! Sets model_image_sed_fitted_riaf to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int Nray);
  
  //! Returns the frequency at which this model is created.
  inline double frequency() const { return _frequency; };

  //! Returns an estimate of the total flux, computed over a dynamically adjusted region, at the frequency at which this model is created.  An accuracy parameter (in Jy) is passed that defines the absolute accuracy at which the flux estimate must be computed.
  double generate_flux_estimate(double accuracy, std::vector<double> parameters);

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    _comm=comm;
    open_error_streams();
  };

 protected:
  MPI_Comm _comm;
  
 private:
  const double _M; //!< Internal mass in cm for the current model.
  const double _D; //!< Internal distance in cm for the current model.
  const double _frequency; //!< Internal frequency in Hz for the current model.
  const double _default_Rmax; //!< Default image sizes in M (note, not applied to flux estimates)

  int _Nray; //!< Image resolution
  
  //! Generates an image with all of the electron densities renormalized by density_factor, modeling a variation in the accretion rate.  Returns the integrated intensity.  Optionally returns I, alpha, beta.
  double generate_renormalized_image(double density_factor, int Nrays, double Rmax, std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

  //! Generates accuracy-assured flux estimate.  Note that there is some code duplication with generate_renormalized_image.
  double estimate_image_size_log_spiral(VRT2::PolarizationMap &pmap, double initial_r, double theta_step, 
					double percent_from_max_I, unsigned int max_iter, double safety_factor);


  //! Opens model-communicator-specific error streams for debugging output
  void open_error_streams();
  std::ofstream _merr; //!< Model-specific error stream

};

};
#endif
