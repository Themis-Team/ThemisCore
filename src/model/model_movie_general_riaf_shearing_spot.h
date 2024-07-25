/*!
  \file model_movie_general_riaf_shearing_spot.h
  \author Paul Tiede
  \date  April, 2017
  \brief Header file for creating a movie object for shearing spot based on model_image_shearing_spot.
  \details Creates a series of images i.e. a movie for the shearing spot model image class. Frames of the movie are selected by using nearest neighbor interpolation. For more details for how the shearing spot model image class works see model_image_shearing_spot.h.
*/

#ifndef Themis_MODEL_MOVIE_GENERAL_RIAF_SHEARING_SPOT_H_
#define Themis_MODEL_MOVIE_GENERAL_RIAF_SHEARING_SPOT_H_

#include "model_movie.h"
#include "model_image_general_riaf_shearing_spot.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

/*!
  \brief Defines an movie model based on the shearing spot models employed in VRT2.

  \details From a user defined time list this creates an image at each requested time to allow the creation of a shearing spot "movie".  There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_image_shearing_spot.cpp for more.

  Parameter list:
  - parameters[0] ... Black hole mass in solar mass units.
  - parameters[1] ... Black hole spin parameter (-1 to 1).
  - parameters[2] ... Cosine of the inclincation relative to the line of sight.
  - parameters[3] ... Spot electron density normalization \f$ n_0 \f$.
  - parameters[4] ... Standard deviation of the Gaussian in M of initial shearing spot \f$ R_s \f$.
  - parameters[5] ... Initial time (sec) of spot in observers time relative to start of observation.
  - parameters[6] ... Initial radius of spot in M.
  - parameters[7] ... Initial azimuthal angle of spot around accretion disk.
  - parameters[8] ... "Thermal" electron population density normalization in \f${\rm cm}^{-3}\f$.  
  - parameters[9] ... "Thermal" electron population density radial power law.
  - parameters[10] ... "Thermal" electron population density \f$h/r\f$.
  - parameters[11] ... "Thermal" electron population temperature normalization in K.
  - parameters[12] ... "Thermal" electron population temperature radial power law.
  - parameters[13] ... Power-law electron population density normalization in \f${\rm cm}^{-3}\f$.
  - parameters[14] ... Power-law electron population density radial power law.
  - parameters[15] ... Power-law electron population density \f$h/r\f$.
  - parameters[16] ... Infalling rate parameter \f$\alpha_r\f$ of accretion flow \f$u^r = u_K^r + \alpha_r(u_{ff}^r - u_K^r)\f$.
  - parameters[17] ... Subkeplerian factor \f$\alpha_{\Omega}\f$ of accretion flow \f$\Omega = u^{\phi}/u^t = \Omega_{ff} + \kappa(\Omega_{K} - \Omega_{ff})\f$.
  - parameters[18] ... Position angle (radians).

	\warning Requires the VRT2 library to be installed.
*/  
class model_movie_general_riaf_shearing_spot : public model_movie<model_image_general_riaf_shearing_spot>
{
 public:
  //! Constructor to make a shearing spot movie. Takes the start time of the observation (UTC), vector of times to create the movie at (UTC), frequerncy (Hz), mass (cm), distance (cm).
  model_movie_general_riaf_shearing_spot(double start_observation, 
					 std::vector<double> observation_times, 
					 double frequency=230e9, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  ~model_movie_general_riaf_shearing_spot();

  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return 19; };

  //! Sets model_image_shearing_spot to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int Nray);

  //! Sets the screen size of the image in units M
  //! so that image has size Rmax x Rmax. The default is 15 which is
  //! probably larger than needed.
  void set_screen_size(double Rmax);

  
};



};

#endif


