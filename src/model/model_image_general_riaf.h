/*!
  \file model_image_general_riaf.h
  \author Avery E. Broderick, Paul Tiede
  \date  August 2019
  \brief Header file for the extended RIAF image model class with mass scaling.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_GENERAL_RIAF_H_
#define Themis_MODEL_IMAGE_GENERAL_RIAF_H_

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
  - parameters[0] ... Black hole mass in units solar mass units.
  - parameters[1] ... Black hole spin parameter (-1 to 1).
  - parameters[2] ... Cosine of the spin inclination relative to line of sight.
  - parameters[3] ... "Thermal" electron population density normalization in \f${\rm cm}^{-3}\f$.  
  - parameters[4] ... "Thermal" electron population density radial power law.
  - parameters[5] ... "Thermal" electron population density \f$h/r\f$.
  - parameters[6] ... "Thermal" electron population temperature normalization in K.
  - parameters[7] ... "Thermal" electron population temperature radial power law.
  - parameters[8] ... Power-law electron population density normalization in \f${\rm cm}^{-3}\f$.
  - parameters[9] ... Power-law electron population density radial power law.
  - parameters[10] ... Power-law electron population density \f$h/r\f$.
  - parameters[11] .. Infall parameter (0,1)
  - parameters[12] .. Sub-keplerian parameter, i.e., sets \f$u_\phi/u_t =\f$ parameters[13]\f$\times\f$\<keplerian value\>.
  - parameters[13] .. Position angle (in model_image) in radians.

 \warning This defines a model_image object associated with a RIAF observed at single frequency.  Requires the VRT2 library to be installed.
*/
class model_image_general_riaf : public model_image
{
 private:
  //! Generates and returns rectalinear grid of intensities associated with the RIAF image model in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:
  //! Constructor to make a RIAF model.  Takes a frequency (Hz), mass (cm), distance (cm).
  model_image_general_riaf(double frequency=230e9, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  virtual ~model_image_general_riaf() {};
  
  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return 14; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;
  
  //! Sets model_image_sed_fitted_riaf to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  //! number_of_refines set the number of refines to use when making the image
  //! defaults to zero. Refine, adaptively refines the underlying grid to
  //! roughly double the number of pixels in the image, while only send back more rays
  //! in interesting locations.
  void set_image_resolution(int Nray, int number_of_refines=0);
  
  //! Sets the screen size of the image in units M
  //! so that image has size Rmax x Rmax. The default is 15 which is
  //! probably larger than needed.
  void set_screen_size(double Rmax);
  
  //! Returns the frequency at which this model is created.
  inline double frequency() const { return _frequency; };

  //! Returns the estimate flux of the image
  inline double flux() const {return _flux;};

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
  const double _D; //!< Internal distance in cm for the current model.
  const double _frequency; //!< Internal frequency in Hz for the current model.
  
  double _Rmax; //!< Image sizes in M (note, not applied to flux estimates)
  int _Nray_base; //!< Base image resolution no refines
  int _Nray; //!< Image resolution after refines
  int _number_of_refines;//!< Refines the base image roughly doubling the resolution
  
  double _flux; //flux of image
  //! Opens model-communicator-specific error streams for debugging output
  void open_error_streams();
  std::ofstream _merr; //!< Model-specific error stream

  double estimate_image_size_log_spiral(VRT2::PolarizationMap &pmap, double initial_r, double theta_step, double percent_from_max_I, unsigned int max_iter, double safety_factor);

};

};
#endif
