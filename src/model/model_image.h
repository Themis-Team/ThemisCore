/*!
  \file model_image.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for image model class.
*/

#ifndef Themis_MODEL_IMAGE_H_
#define Themis_MODEL_IMAGE_H_

#include <vector>
#include <complex>

#include "model_visibility.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "interpolator2D.h"

#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines an interface for models that generate image data with a collection of utility functions to compute visibility amplitudes, closure phases, etc.

  \details Many EHT models provide primarily images.  This class provides both an interface to the inteferometric data models (model_visibility_amplitude, model_closure_phase, etc.) and utility functions for computing the appropriate data types.  Because images can be trivially rotated, model_image interprets the last parameter as the position angle (radians, E of N) and separately implements it removing the need to apply rotations in the generate_image function.  Expects that images will be provided as they appear on the sky, e.g., oriented with E on the left and the E-W coordinate increasing rightward.

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.  Uses the FFTW library to compute the FFT to generate visibilities.
*/
  class model_image : public model_visibility, public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
{
  //private:
 protected:

  //! A user-supplied function that generates and returns rectalinear grid of intensities in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta) = 0;


 public:
  model_image();
  virtual ~model_image();

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return 1; };

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);

  //! A one-time generate function that will generate the complex visibilities and store them. This must be called after generate_model has been called.
  virtual void generate_complex_visibilities();


  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  //! Outputs the generated image using the pmap format.
  //! If rotate is true it will rotate the image according to
  //! the position angle in the parameter list using a bicubic spline.
  //! Line 1 gives the limits in x direction (xmin, xmax, Nx)
  //! Line 2 gives the limits in y direction (ymin, ymax, Ny)
  //! Line 3 is the columns information
  //! Then the image is outputted with first two columns
  //! being the pixel numbers and then intensity in Jy/pixel
  void output_image(std::string fname, bool rotate=false);

  //! Provides direct access to the constructed image.  Sets a 2D grid of angles (alpha, beta) in radians and intensities in Jy per steradian.
  void get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;

  //! Provides direct access to the complex visibilities.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& V) const;

  //! Provides direct access to the visibility amplitudes.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibility_amplitudes(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<double> >& V) const;
  

  //! Provides ability to use bicubic spline interpolator (true) instead of regular bicubic. Code defaults to false.
  void use_spline_interp( bool use_spline );

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_image proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
    _comm=comm;
  };


  //! Write a unique identifying tag for use with the ThemisPy plotting features. This calls the overloaded version with the outstream, which is the only function that need be rewritten in child classes.
  void write_model_tag_file(std::string tagfilename="model_image.tag") const;

  //! Write a unique identifying tag for use with the ThemisPy plotting features. For most child classes, the default implementation is suffcient.  However, should that not be the case, this is the only function that need be rewritten in child classes.
  virtual void write_model_tag_file(std::ofstream& tagout) const;

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
  virtual std::string model_tag() const
  {
    return "UNDEFINED";
  };
  
  
 protected:
  MPI_Comm _comm;

  bool _generated_model; //!< True when a model is generated with generate_model.
  bool _generated_visibilities; //!< True when model visibilities have been computed.
  bool _use_spline; //!< True when want to use bicubic spline interpolator

  //! Position angle of the image, assumed to be the last element of the parameter list passed to generate_model.  Assumed to be in radians and defined E of N.
  double _position_angle; 

  //! Last set of parameters passed to generate_image, useful to determine if it is necessary to recompute the model.  Useful, e.g., if we are varying only position angle, or if recomputing for a number of different data sets at the same set of parameters.
  std::vector<double> _current_parameters;

  


 protected:
  // Space for image
  std::vector<std::vector<double> > _alpha; //!< 2D grid of horizonal pixel locations in radians, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<double> > _beta; //!< 2D grid of vertical pixel locations in radians, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<double> > _I; //!< 2D grid of intensities at pixel locations in Jy/str.

 //private:
 protected:
  std::vector<std::vector<double> > _u; //!< 2D grid of horizontal baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<double> > _v; //!< 2D grid of vertical baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<std::complex<double> > > _V; //!< 2D grid of complex visibilities at pixel locations in Jy.
  std::vector<std::vector<double> > _V_magnitude; //!< 2D grid of visibility amplitudes at pixel locations in Jy.

  // Define the interpolator object
  std::valarray<double> _i2du; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dv; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dV_r; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dV_i; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dV_M; //!< Arrays for defining the 2D interpolator objects.
  Interpolator2D _i2D_VM; //!< 2D interpolator object to estimate visibility amplitudes at arbitrary u-v locations.  model_image uses bicubic interpolation.
  Interpolator2D _i2D_Vr; //!< 2D interpolator object to estimate real component of the complex visibilities at arbitrary u-v locations.  model_image uses bicubic interpolation.
  Interpolator2D _i2D_Vi; //!< 2D interpolator object to estimate imaginary component of the complex visibilities at arbitrary u-v locations.  model_image uses bicubic interpolation.

  //! Utility function to numerically computes the complex visibilities from the image data after _alpha, _beta, and _I have been set.  Currently pads the intensity array by a factor of 8.
  virtual void compute_raw_visibilities();

  //! 2D FFT utility function based on FFTW
  virtual std::vector<std::vector<std::complex<double> > > fft_2d(const std::vector<std::vector<double> > &I);		//fft2 (from fftw)

  //! 2D shift utility function to center the zero-baseline modes at the center of the array
  virtual std::vector<std::vector<std::complex<double> > > fft_shift(const std::vector<std::vector<std::complex<double> > > &V);	//shift
};

};
#endif
