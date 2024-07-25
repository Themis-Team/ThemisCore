/*!
  \file model_image_sed_fitted_riaf.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for SED-fitted RIAF model class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_SED_FITTED_RIAF_H_
#define Themis_MODEL_IMAGE_SED_FITTED_RIAF_H_

#include <string>
#include <vector>
#include <mpi.h>
#include "model_image.h"
#include "vrt2.h"
#include "utils.h"

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

  /*!
    \brief Defines an image model associated with the SED-fitted RIAF 
    models employed in Broderick, Fish, Doeleman and Loeb (2011) and 
    Broderick et al. (2016).

    \details Provides explicit implementation of the model_image object 
    for the SED-fitted semi-analytic RIAF models.  There are additional 
    tuning parameters that may impact accuracy, e.g., distance to the 
    image screen, number of rays to produce, etc.  See the appropriate 
    parts of model_image_sed_fitted_riaf.cpp for more.  

    Parameter list:\n
    - parameters[0] ... Black hole spin parameter (-1 to 1).
    - parameters[1] ... Cosine of the spin inclination relative to line of sight.
    - parameters[2] ... Position angle (in model_image) in radians.
      

    \warning Requires the VRT2 library to be installed and the SED fitted 
    disk parameter file, which may be found in 
    "Themis/src/VRT2/DataFiles/2010_combined_fit_parameters.d".
  */  
  class model_image_sed_fitted_riaf : public model_image
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
      model_image_sed_fitted_riaf(std::string sed_fit_parameter_file=Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"), double frequency=230e9);
      virtual ~model_image_sed_fitted_riaf() {};

      //! Returns the number of the parameters the model expects
      virtual inline size_t size() const { return 3;} ;

      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const;

      //! Sets model_image_sed_fitted_riaf to generate 32x32 images (via
      //! set_image_resolution).  This is usually used only for testing
      //! purposes.
      void use_small_images();

      //! Sets model_image_sed_fitted_riaf to generate production images with
      //! resolution Nray x Nray.  The default is 128x128, which is probably
      //! larger than required in practice.
      void set_image_resolution(int Nray);
      
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
        //std::cout << "model_image_sed_fitted_riaf proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
        _comm=comm;
	open_error_streams();
      };

    protected:
      MPI_Comm _comm;
    
    
    private:
      VRT2::SgrA_PolintDiskModelParameters2010 _sdmp; //!< Interpolator object to return 
      const double _M; //!< Mass of Sgr A* in cm
      const double _D; //!< Distance to Sgr A* in cm
      const double _frequency; //!< Frequency at which model images in Hz

  
      int _Nray; //!< Image resolution

      //! Generates an image with all of the electron densities renormalized
      //! by density_factor, modeling a variation in the accretion rate. 
      //! Returns the integrated intensity.  Optionally returns I, alpha, beta.
      double generate_renormalized_image(double density_factor, int Nrays, 
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
