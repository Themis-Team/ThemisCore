/*!
  \file model_polarized_image_gaussian_spot.h
  author Paul Tiede
  \date  March, 2018
  \brief Header file for the gaussian spot model class.
  \details Creates an snapshot of a gaussian spot using the VRT2 semi-analytic gaussian spot class. Model uses a two-parameter vector field described below to model an infalling accretion flow. The spots initial densoty profile is taken to be a Gaussian spot with profile \f[ n_0\exp(-\Delta r^{\mu}\Delta r_{\mu}/2R_s - (u^{\mu}r_{mu})^2/2R_s)\f]. Currently the model uses the sed from the 2010 obsevation of Sgr A* for the spot's ectron population.
*/

#ifndef Themis_MODEL_POLARIZED_IMAGE_GAUSSIAN_SPOT_H_
#define Themis_MODEL_POLARIZED_IMAGE_GAUSSIAN_SPOT_H_


#include <string>
#include <vector>
#include <mpi.h>

#include "model_polarized_image.h"
#include "vrt2.h"
#include "utils.h"

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {
  
/*!
  \brief Defines an image model based on the semi-analytic gaussian spot models employed in VRT2.

  \details Provides explicit implementation of the model_polarized_image object for the semi-analytic gaussian spot models.  There are additional tuning parameters that may impact accuracy, e.g., distance to the image screen, number of rays to produce, etc.  See the appropriate parts of model_polarized_image_gaussian_spot.cpp for more.    
  Parameter list:\n
  - parameters[0] ... Black hole spin parameter (-1 to 1).
  - parameters[1] ... Cosine of the inclincation relative to the line of sight.
  - parameters[2] ... Spot electron density normalization \f$ n_0 \f$.
  - parameters[3] ... Standard deviation of the Gaussian in M of initial gaussian spot \f$ R_s \f$.
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

  \warning Model requires VRT2 to be installed and it explicitly time dependent. Model assumes one time is passed.
*/
 
class model_polarized_image_gaussian_spot : public model_polarized_image
{
 private:
  //! Generates and returns rectalinear grid of intensities associated with the gaussian spot image model in Jy/pixel located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.
  virtual void generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

 public:
  //! Constructor to make a gaussian spot model.  
  //! Takes the start time of the observation (UTC), 
  //! time you are observing the spot at (UTC), 
  //! whether to add a background riaf to the image 
  //! the frequency (Hz), mass (cm), distance (cm).
  model_polarized_image_gaussian_spot( double start_obs, double tobs, 
                             bool bkgd_riaf = false,
														 double frequency = 230e9,
														 double M=VRT2::VRT2_Constants::M_SgrA_cm, 
														 double D=VRT2::VRT2_Constants::D_SgrA_cm
													 );
  virtual ~model_polarized_image_gaussian_spot() {};
  
  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return _bkgd_riaf? 10 : 20; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! Sets model_polarized_image_gaussian_spot to generate production images with
  //! resolution Nray x Nray.  The default is 128x128, which is probably
  //! larger than required in practice.
  void set_image_resolution(int Nray, int number_of_refines = 0);

  //! Sets the screen size of the image in units M
  //! so that image has size Rmax x Rmax. The default is 15 which is
  //! probably larger than needed.
  void set_screen_size(double Rmax);
 
  //! Returns the frequency at which this model is created.
  inline double frequency() const { return _frequency; };
  
  //! Returns the frequency at which this model is created.
  inline double flux() const { return _flux; };

  //! Returns the start of the observation time in J2000 at which this model was created
  inline double obs_time() const {return _tobs; };

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    _comm=comm;
    open_error_streams();
  };

 protected:
  MPI_Comm _comm;
  
 private:
  const double _start_obs; //!< Start time of the simulation in J2000.
  const double _tobs; //!< Observation time of spot in J2000 in seconds.
  bool _bkgd_riaf; //!< Specifies whether to include riaf model in radiative transfer scheme.
  const double _frequency; //!< Internal frequency in Hz for the current model.
  const double _M; //!< Internal mass in cm.
  const double _D; //!< Internal distance in cm for the current model.

  int _Nray, _Nray_base, _number_of_refines; //!< Image resolution
	
  double _Rmax; //!< Default image sizes in M (note, not applied to flux estimates)
  double _flux; //!< Stores the image flux

  void open_error_streams();  //!< Opens model-communicator-specific error streams for debugging output

  std::ofstream _merr; //!< Model-specific error stream

};

};
#endif
