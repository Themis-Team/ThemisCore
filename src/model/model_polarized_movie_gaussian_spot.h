/*!
  \file model_polarized_movie_gaussian_spot.h
  \author Paul Tiede
  \date  April, 2017
  \brief Header file for creating a movie object for gaussian spot based on model_image_gaussian_spot.
  \details Creates a series of images i.e. a movie for the gaussian spot model image class. Frames of the movie are selected by using nearest neighbor interpolation. For more details for how the gaussian spot model image class works see model_image_gaussian_spot.h.
*/

#ifndef Themis_MODEL_POLARIZED_MOVIE_GAUSSIAN_SPOT_H_
#define Themis_MODEL_POLARIZED_MOVIE_GAUSSIAN_SPOT_H_

#include "model_polarized_movie.h"
#include "model_polarized_image_gaussian_spot.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

/*!
  \brief Defines an movie model based on the gaussian spot models employed in VRT2.

  \details From a user defined time list this creates an image at each requested time to allow the creation of a gaussian spot "movie".  There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_image_gaussian_spot.cpp for more.

  Parameter list:
  - parameters[0] ... Black hole spin parameter (-1 to 1).
  - parameters[1] ... Cosine of the inclincation relative to the line of sight.
  - parameters[2] ... Spot electron density normalization \f$ n_0 \f$.
  - parameters[3] ... Standard deviation of the gaussian in M of initial gaussian spot \f$ R_s \f$.
  - parameters[4] ... Initial time of spot in observers time relative to start of observation (M).
  - parameters[5] ... Initial radius of spot in M.
  - parameters[6] ... Initial azimuthal angle of spot around accretion disk.
  - parameters[7] ... Infalling rate parameter \f$\alpha_r\f$ of accretion flow \f$u^r = u_K^r + \alpha_r(u_{ff}^r - u_K^r)\f$.
  - parameters[8] ... Subkeplerian factor \f$\alpha_{\Omega}\f$ of accretion flow \f$\Omega = u^{\phi}/u^t = \Omega_{ff} + \kappa(\Omega_{K} - \Omega_{ff})\f$.
  # These are only needed if bkgrd_riaf is true in the constructor
  - parameters[9] ... thermal electron number density normalization
  - parameters[10] ... Thermal electron radial power-law index
  - parameters[11] ... Thermal electron H/R
  - parameters[12] ... Thermal electron temperature normalization
  - parameters[13] ... Thermal electron radial power-law index
  - parameters[14] ... Non-thermal electron number density normalization
  - parameters[15] ... non-thermal electron power-law index
  - parameters[16] ... Non-thermal electron H/R
  - parameters[17] ... Plasma beta (Pgas/Pmag)
	- parameters[9/18] ... Position angle (radians). (parameters 9 if no riaf, 18 if riaf)

	\warning Requires the VRT2 library to be installed.
*/  
class model_polarized_movie_gaussian_spot : public model_polarized_movie<model_polarized_image_gaussian_spot>
{
 public:
  //! Constructor to make a gaussian spot movie. Takes the start time of the observation (UTC), vector of times to create the movie at (UTC), frequerncy (Hz), mass (cm), distance (cm).
  model_polarized_movie_gaussian_spot(double start_observation,
											std::vector<double> observation_times,
                      bool bkgd_riaf = false,
			                double frequency=230e9, double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  ~model_polarized_movie_gaussian_spot();

  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return _movie_frames[0]->size(); };


  //! Sets model_image_gaussian_spot to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int Nray, int number_of_refines = 0);

  //! Sets the screen size of the image in units M
  //! so that image has size Rmax x Rmax. The default is 15 which is
  //! probably larger than needed.
  void set_screen_size(double Rmax);
  
};



};

#endif


