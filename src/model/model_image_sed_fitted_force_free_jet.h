/*!
  \file model_image_sed_fitted_force_free_jet.h
  \author Paul Tiede
  \date  April, 2017
  \brief Header file for SED-fitted force free jet model.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_SED_FITTED_FORCE_FREE_JET_H_
#define Themis_MODEL_IMAGE_SED_FITTED_FORCE_FREE_JET_H_

#include "utils.h"
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
    \brief Defines an image model associated with the force free jet model 
    models employed in Broderick & Loeb (2009).

    \details Provides explicit implementation of the model_image object 
    for the SED-fitted semi-analytic force free jet models from 2012 SED fits.  
    There are additional 
    tuning parameters that may impact accuracy, e.g., distance to the 
    image screen, number of rays to produce, etc.  See the appropriate 
    parts of model_image_sed_fitted_force_free_jet.cpp for more.  

    Parameter list:\n
    - parameters[0] ... Total image flux
    - parameters[1] ... Black hole spin parameter (-1 to 1).
    - parameters[2] ... Cosine of the spin inclination relative to line of sight.
    - parameters[3] ... jet electron radial power law index
    - parameters[4] ... jet opening angle in degrees
    - parameters[5] ... jet loading radius in M
    - parameters[6] ... jet max Lorentz factor
    - parameters[7] ... Position angle (in model_image) in radians.
      
    \warning Requires the VRT2 library to be installed and the SED fitted 
    disk parameter file, which may be found in 
    "Themis/src/VRT2/DataFiles/IRO_high_mass.d".
  */  
  class model_image_sed_fitted_force_free_jet : public model_image
  {
    private:
      
      //! Generates and returns rectalinear grid of intensities associated
      //! with the RIAF image model in Jy/str located at pixels centered 
      //! on angular positions alpha and beta, both specified in radians 
      //! and aligned with a fiducial direction.  Note that the parameter 
      //! vector has had the position removed.
      virtual void generate_image(vector<double> parameters, 
                                  vector<vector<double> >& I, 
                                  vector<vector<double> >& alpha, 
                                  vector<vector<double> >& beta);
  
    public:
      
      //! Constructor to make an SED-fitted RIAF model. Takes the disk 
      //! parameter fit file and frequency at which to generate models. 
      //! The mass of and distance to Sgr A* are hard-coded to be consistent
      //! with the SED fitting.
      model_image_sed_fitted_force_free_jet(std::string sed_fit_parameter_file=utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"), double M87_mass_cm=VRT2::VRT2_Constants::M_M87_cm, double M87_distance_cm=VRT2::VRT2_Constants::D_M87_cm, double frequency=230e9);
      virtual ~model_image_sed_fitted_force_free_jet() {};

      //! Returns the number of the parameters the model expects
      virtual inline size_t size() const { return 8; } ;
      
      //! Sets model_image_sed_fitted_force_free_jet to generate 32x32 images (via
      //! set_image_resolution).  This is usually used only for testing
      //! purposes.
      void use_small_images();

      //! Sets model_image_sed_fitted_force_free_jet to generate production images with
      //! resolution xNray x yNray.  The default is 128x128, which is probably
      //! larger than required in practice.
      void set_image_resolution(int xNray, int yNray);

      //! Sets image dimensions, since for a jet it is often beneficial to have a non-sqauare 
      //! image. The default values are -80, 40, -40, 80
      void set_image_dimensions( double xlow, double xhigh, double ylow, double yhigh );
      
      //! Defines a set of processors provided to the model for parallel
      //! computation via an MPI communicator.  Only facilates code 
      //! parallelization if the model computation is parallelized via MPI.
      //! The Following plot shows the scaling plot for the SED-fitted RIAF model.

      //!The green line shows the ideal case of linear scaling where the run time is inversely proportional 
      //!to the number of MPI processes used. The purple line shows how the sampler scales with the number of
      //!MPI processes. Up to \f$16\f$ MPI processes the scaling closely follows the linear scaling and
      //! deviation stays within \f$\%20\f$. Using \f$32\f$ MPI processes the deviation increases to \f$\%50\f$.
      //!\image html Lscale.png "SED-fitted RIAF scaling plot. The green line shows the linear scaling." width=10cm
      virtual void set_mpi_communicator(MPI_Comm comm) 
      {
        //std::cout << "model_image_sed_fitted_force_free_jet proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
        _comm=comm;
	open_error_streams();
      };

    protected:
      MPI_Comm _comm;
    
    
    private:
      VRT2::M87_PolintJetModelParameters2012 _sdmp; //!< Interpolator object to return 
      const double _M; //!< Mass of M87 in cm
      const double _D; //!< Distance to M87 in cm
      const double _frequency; //!< Frequency at which model images in Hz

  
      int _xNray, _yNray; //!< Image resolution
      int _xlow, _xhigh, _ylow, _yhigh; //!< Image dimensions


      //! Generates an image with all of the electron densities renormalized
      //! by density_factor, modeling a variation in the accretion rate. 
      //! Returns the integrated intensity.  Optionally returns I, alpha, beta.
      double generate_renormalized_image(double density_factor, 
                                        int Nxrays, int Nyrays, 
                                        vector<double> parameters, 
                                        vector<vector<double> >& I, 
                                        vector<vector<double> >& alpha, 
                                        vector<vector<double> >& beta);


      //! Opens model-communicator-specific error streams for debugging output
      void open_error_streams();
      std::ofstream _merr; //!< Model-specific error stream
  };
  
};
#endif
