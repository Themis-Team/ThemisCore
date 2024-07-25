/*!
  \file model_image_vae_interpolated_riaf.h
  \author Ali Sarertoosi, Avery E. Broderick
  \date  January, 2023
  \brief Header file for variational auto-encoder interpolated RIAF model class.
  \details To be added
  \warning Requires Torch to be installed and TORCH_DIR to be specified in Makefile.config.
*/

#ifndef Themis_MODEL_IMAGE_VAE_INTERPOLATED_RIAF_H_
#define Themis_MODEL_IMAGE_VAE_INTERPOLATED_RIAF_H_

#ifdef ENABLE_TORCH // Only available if Torch is configured

#include <torch/script.h>
#include <string>
#include <vector>
#include <complex>
#include <mpi.h>
#include "model_image.h"
#include "vrt2.h"
#include "utils.h"
#include <fftw3.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

  /*!
    CHANGE DOCUMENTATION DETAILS THROUGHOUT!
    \brief Defines an image model associated with a trained neural network
    model.  Parameters are the latent variables, and may require subsequent
    interpretation.

    \details Provides an explicit implementation of the model_image object 
    for a trained neural network of images.  Parameters are the latent 
    variables, which are fed to a decoder from which trial images are 
    generated.  Additional parameters that are provided are a flux
    renormalization, a mass-to-distance ratio renormalization, and a
    position angle (via the model_image interface).

    Parameter list:\n
    - parameters[0] ....... First latent parameter
    - parameters[zsize-1] . Last latent parameter
    - parameters[zsize] ... Flux renormalization, which multiplies the image intensity.
    - parameters[zsize+1] . Mass renormalization, which multiplies the image size.
    - parameters[zsize+2] . Position angle (in model_image) in radians.

    \warning Requires Torch to be installed and TORCH_DIR to be specified in Makefile.config.
  */  
  class model_image_vae_interpolated_riaf : public model_image
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
      model_image_vae_interpolated_riaf(std::string modeldir, double frequency=230e9,
					double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);

      model_image_vae_interpolated_riaf(std::string metafile, std::string rangefile, std::string modelfile, double frequency=230e9,
					double M=VRT2::VRT2_Constants::M_SgrA_cm, double D=VRT2::VRT2_Constants::D_SgrA_cm);
      
      virtual ~model_image_vae_interpolated_riaf();

      //! Returns the number of the parameters the model expects
      virtual inline size_t size() const { return _size;} ;

      //! Returns the number of the latent parameters the model expects
      virtual inline size_t latent_size() const { return _zsize;} ;
      
      //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

      //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

      //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double closure_phase(datum_closure_phase& d, double accuracy);

      //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

      
      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const;

      //!The green line shows the ideal case of linear scaling where the run time is inversely proportional 
      //!to the number of MPI processes used. The purple line shows how the sampler scales with the number of
      //!MPI processes. Up to \f$16\f$ MPI processes the scaling closely follows the linear scaling and
      //! deviation stays within \f$\%20\f$. Using \f$32\f$ MPI processes the deviation increases to \f$\%50\f$.
      //!\image html Lscale.png "SED-fitted RIAF scaling plot. The green line shows the linear scaling." width=10cm
      virtual void set_mpi_communicator(MPI_Comm comm);
      /* { */
      /*   //std::cout << "model_image_sed_fitted_riaf proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl; */
      /*   _comm=comm; */
      /* 	//open_error_streams(); */
      /* }; */

      //! Set the target image size dimensions, which will result in an attempt at image size reduction before FFT
      void set_image_size(size_t target_Nx=0, size_t target_Ny=0);

    protected:
      MPI_Comm _comm;
    
      //! Utility function to numerically computes the complex visibilities from the image data after _alpha, _beta, and _I have been set.  Currently pads the intensity array by a factor of 8.
      virtual void compute_raw_visibilities();

      //! 2D FFT utility function based on FFTW
      virtual std::vector<std::vector<std::complex<double> > > fft_2d(const std::vector<std::vector<double> > &I);		//fft2 (from fftw)

      //! 2D shift utility function to center the zero-baseline modes at the center of the array
      virtual std::vector<std::vector<std::complex<double> > > fft_shift(const std::vector<std::vector<std::complex<double> > > &V);	//shift

    private:
      const double _frequency; //!< Frequency at which model images in Hz
      const double _M; //!< Mass of Sgr A* in cm
      const double _D; //!< Distance to Sgr A* in cm
      size_t _zsize;
      size_t _size;
      double _xfov,_yfov;
      int _Nx,_Ny;
      int _reduction_factor_x, _reduction_factor_y;
      
      const unsigned int _Npad; // Default padding factor of 8, MUST BE AN EVEN NUMBER

      
      double _mass_rescale; //!< Rescaling factor with M/D
      double _flux_rescale; //!< Rescaling factor for total flux
      
      std::string _metafile, _rangefile, _modelfile;
      torch::jit::script::Module _module;
      torch::Tensor _maxmins;
      
      void read_metadata(std::string fname);
      void read_range_limits(std::string fname);
      void read_model(std::string fname);

      std::vector<double> _current_latent_parameters;

      std::vector<std::vector<double> > _padded_I;
      
      double _fft_norm;

      fftw_complex *_in, *_out;
      fftw_plan _p;

      void construct_ffts(size_t Npad);
      void cleanup_ffts();      
  };
  
};
#endif
#endif
