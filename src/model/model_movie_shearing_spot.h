/*!
  \file model_movie_shearing_spot.h
  \author Paul Tiede
  \date  April, 2017
  \brief Header file for creating a movie object for shearing spot based on model_image_shearing_spot.
  \details Creates a series of images i.e. a movie for the shearing spot model image class. Frames of the movie are selected by using nearest neighbor interpolation. For more details for how the shearing spot model image class works see model_image_shearing_spot.h.
*/

#ifndef Themis_MODEL_MOVIE_SHEARING_SPOT_H_
#define Themis_MODEL_MOVIE_SHEARING_SPOT_H_

#include "model_movie.h"
#include "model_image_shearing_spot.h"

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
    - parameters[0] ... Black hole spin parameter (-1 to 1).
    - parameters[1] ... Cosine of the inclincation relative to the line of sight.
    - parameters[2] ... Spot electron density normalization.
    - parameters[3] ... Standard deviation of the Gaussian in M of initial shearing spot.
    - parameters[4] ... Initial time of spot in observers time (M). Typically starts at -2000M for so that you see the start of the spot when ray-tracing back in time.
    - parameters[5] ... Initial radius of spot in M.
    - parameters[6] ... Initial azimuthal angle of spot around accretion disk.
    - parameters[7] ... Infalling rate parameter \f$\alpha_r\f$ of accretion flow \f$u^r = u_K^r + \alpha_r(u_{ff}^r - u_K^r)\f$.
    - parameters[8] ... Subkeplerian factor \f$\alpha_{\Omega}\f$ of accretion flow \f$\Omega u^{\phi}/u^t = \Omega_{ff} + \kappa(\Omega_K - \Omega_{ff})\f$.
    - parameters[9] ... Position angle in radians.

	\warning Requires the VRT2 library to be installed.
*/  
class model_movie_shearing_spot : public model_movie<model_image_shearing_spot>
{
 public:
  //! Constructor to make a shearing spot movie. Takes the start time of the observation (UTC), vector of times to create the movie at (UTC), frequerncy (Hz), mass (cm), distance (cm).
  model_movie_shearing_spot(double start_observation, 
											std::vector<double> observation_times, 
											std::string sed_fit_parameter_file="VRT2/DataFiles/2010_combined_fit_parameters.d",
			                double frequency=230e9, double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);
  ~model_movie_shearing_spot();

  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return 10; };

  //! Sets model_image_shearing_spot to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int Nray);

	//! Sets the screen size of the image in units M
	//! so that image has size Rmax x Rmax. The default is 15 which is
	//! probably larger than needed.
	void set_screen_size(double Rmax);

  //! Adds background sed fitted riaf to model as specified by sed_fit_parameters_file
  //! that was given in the constructor.
  void add_background_riaf();
  
};



};

#endif


