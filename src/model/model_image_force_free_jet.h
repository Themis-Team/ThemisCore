/*!
  \file model_image_force_free_jet.h
  \author Paul Tiede
  \date  April, 2017
  \brief Header file for force free jet model.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_FORCE_FREE_JET_H_
#define Themis_MODEL_IMAGE_FORCE_FREE_JET_H_

#include "model_image.h"
#include "constants.h"
#include "vrt2.h"
#include <string>
#include <vector>
#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

  /*!
    \brief Defines an image model associated with the force free jet 
    models employed in Broderick & Loeb (2009).

    \details Provides explicit implementation of the model_image object 
    semi-analytic force free jet models.  
    There are additional 
    tuning parameters that may impact accuracy, e.g., distance to the 
    image screen, number of rays to produce, etc.  See the appropriate 
    parts of model_image_force_free_jet.cpp for more.  

    Parameter list:\n
    - parameters[0]  ... Black hole mass in solar mass.
    - parameters[1]  ... Distance to black hole in kpc.
    - parameters[2]  ... Black hole spin parameter (-1 to 1).
    - parameters[3]  ... Cosine of the spin inclination relative to line of sight.
    - parameters[4]  ... jet radial power law index
    - parameters[5]  ... jet opening angle in degrees
    - parameters[6]  ... jet loading radius in M
    - parameters[7]  ... jet max Lorentz factor
    - parameters[8]  ... jet electron density normalization
    - parameters[9]  ... jet synchrotron spectral index
    - parameters[10]  ... jet synchrotron magnetic field normalization
    - parameters[11] ... synchrotron emitting electrons minimum lorentz factor
    - parameters[12] ... edge of disk and jet in terms of ISCO.
    - parameters[13] ... Position angle (in model_image) in radians.
      

    \warning Requires the VRT2 library to be installed.
  */  
  class model_image_force_free_jet : public model_image
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
      model_image_force_free_jet(double frequency=230e9);
      virtual ~model_image_force_free_jet() {};

      //! Returns the number of the parameters the model expects
      virtual inline size_t size() const { return 14;};

      //! Returns the frequency the model was created at 
      inline double frequency() const { return _frequency; };

      //! Returns an estimate of the total flux, computed over a dynamically adjust region, at the
      //! frequency at which this model is created. An accuracy parameter (in Jy) is passed that defines
      //! the absolute accuracy at which the flux estimte must be computed.
      double generate_flux_estimate( double accuracy, std::vector<double> parameters );

      //! Sets model_image_force_free_jet to generate 32x32 images (via
      //! set_image_resolution).  This is usually used only for testing
      //! purposes.
      void use_small_images();

      //! Sets model_image_force_free_jet to generate production images with
      //! resolution xNray x yNray.  The default is 128x128, which is probably
      //! larger than required in practice.
      void set_image_resolution(int Nray, int number_of_refines=0);

      //! Sets image dimensions in units of M 
      void set_screen_size( double Rmax);
      
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
        //std::cout << "model_image_force_free_jet proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
        _comm=comm;
	open_error_streams();
      };

    protected:
      MPI_Comm _comm;
    
    
    private: 
      const double _frequency; //!< Frequency at which model images in Hz

  
      int _Nray_base; //!< Base image resolution before refining
      int _Nray; //!< Image resolution
      int _number_of_refines;//!< Refines the base image roughly doubling the resolution.
      int _Rmax; //!< Image dimensions



      //! Generates accuracy-assured flux estimate.  Note that there is some code duplication with generate_renormalized_image.
      double estimate_image_size_log_spiral(VRT2::PolarizationMap &pmap, double initial_r, double theta_step, 
					double percent_from_max_I, unsigned int max_iter, double safety_factor);



      //! Opens model-communicator-specific error streams for debugging output
      void open_error_streams();
      std::ofstream _merr; //!< Model-specific error stream
  };
  
};
#endif
