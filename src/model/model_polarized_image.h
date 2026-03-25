/*!
  \file model_polarized_image.h
  \author Avery E. Broderick
  \date  March, 2020
  \brief Header file for polarized image model class.
*/

#ifndef Themis_MODEL_POLARIZED_IMAGE_H_
#define Themis_MODEL_POLARIZED_IMAGE_H_

#include <vector>
#include <complex>
#include <array>

#include "model_crosshand_visibilities.h"
#include "model_polarization_fraction.h"
#include "model_visibility.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "interpolator2D.h"

#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines an interface for models that generate polarized image data with a collection of utility functions to compute visibility amplitudes, closure phases, etc.

  \details Many EHT models provide primarily polarized images.  This class provides both an interface to the inteferometric data models (model_crosshand_visibility, model_visibility, model_visibility_amplitude, model_closure_phase, etc.) and utility functions for computing the appropriate data types.  Unless otherwise stated, the quantities are Stokes I.  

  Because images can be trivially rotated, model_polarized_image interprets the last parameter as the position angle (radians, E of N) and separately implements it removing the need to apply rotations in the generate_image function.  Expects that images will be provided as they appear on the sky, e.g., oriented with E on the left and the E-W coordinate increasing rightward.  

  Because polarization must be calibrated, model_polarized_images have the option to postpend D-terms.  To do this, an array of stations must be provided.  An additional set of four times the size of the station array will be postpended to the parameters, corresponding to the real and imaginary components of the R-hand and L-hand D-terms for each station are added. 

  \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.  Uses the FFTW library to compute the FFT to generate visibilities.

  \todo Add the field rotation angle stuff to the model_polarization_fraction hooks, if required.
*/
  class model_polarized_image : public model_crosshand_visibilities, public model_polarization_fraction, public model_visibility, public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
{
 private:

  //! A user-supplied function that generates and returns rectalinear grid of intensities in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

  //! A user-supplied function that generates and returns rectalinear grid of Stokes parameters in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
  virtual void generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta) = 0;


 public:
  model_polarized_image();
  virtual ~model_polarized_image();

  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _size; };

  //! Add D-terms, increases the number of parameters by four times the number of station codes passed.  Will fail if a visibility is requested for a station for which the code has not been provided.
  void model_Dterms(std::vector<std::string> station_codes);

  //! Boolean to determine if this is fitting D-terms
  bool modeling_Dterms() const { return _modeling_Dterms; };

  //! Number of D-terms (each complex number)
  size_t number_of_Dterms() const { return _Dterms.size(); };

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);

  //! A one-time generate function that will generate the complex visibilities and store them. This must be called after generate_model has been called.
  virtual void generate_complex_visibilities();

  //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image given a datum_crosshand_visibilities_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  // virtual std::vector< std::complex<double> > crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy);

  using model_crosshand_visibilities::crosshand_visibilities;

  virtual std::vector< std::complex<double> >
  crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy);

  virtual std::vector< std::complex<double> >
  crosshand_visibilities(size_t d_idx,
                         datum_crosshand_visibilities& d,
                         double accuracy) override
  {
    return model_crosshand_visibilities::crosshand_visibilities(d_idx, d, accuracy);
  }


  void apply_Dterms_linear(datum_crosshand_visibilities& d, std::complex<double>* io)
  {
    std::vector<std::complex<double>> tmp(4);
    tmp[0] = io[0];
    tmp[1] = io[1];
    tmp[2] = io[2];
    tmp[3] = io[3];
    apply_Dterms(d, tmp);
    io[0] = tmp[0];
    io[1] = tmp[1];
    io[2] = tmp[2];
    io[3] = tmp[3];
  }

  void apply_Dterms_parameter_jacobian_linear(
      const datum_crosshand_visibilities& d,
      const std::complex<double>* in,
      std::complex<double> deriv[8][4]) const
  {
    for (int q = 0; q < 8; ++q)
      for (int h = 0; h < 4; ++h)
        deriv[q][h] = std::complex<double>(0.0, 0.0);

    if (!_modeling_Dterms)
      return;

    const size_t is1 = get_index_from_station_code(d.Station1);
    const size_t is2 = get_index_from_station_code(d.Station2);

    const std::complex<double> I(0.0,1.0);
    const std::complex<double> ei2p1 = std::exp(std::complex<double>(0.0,2.0)*d.phi1);
    const std::complex<double> ei2p2 = std::exp(std::complex<double>(0.0,2.0)*d.phi2);

    const std::complex<double> DR1  = _Dterms[2*is1]   * ei2p1;
    const std::complex<double> DL1  = _Dterms[2*is1+1] * std::conj(ei2p1);
    const std::complex<double> DR2c = std::conj(_Dterms[2*is2]   * ei2p2);
    const std::complex<double> DL2c = std::conj(_Dterms[2*is2+1] * std::conj(ei2p2));

    const std::complex<double> c0 = in[0];
    const std::complex<double> c1 = in[1];
    const std::complex<double> c2 = in[2];
    const std::complex<double> c3 = in[3];

    const std::complex<double> dDR1_re  = ei2p1;
    const std::complex<double> dDR1_im  = I * ei2p1;
    const std::complex<double> dDL1_re  = std::conj(ei2p1);
    const std::complex<double> dDL1_im  = I * std::conj(ei2p1);
    const std::complex<double> dDR2c_re = std::conj(ei2p2);
    const std::complex<double> dDR2c_im = -I * std::conj(ei2p2);
    const std::complex<double> dDL2c_re = ei2p2;
    const std::complex<double> dDL2c_im = -I * ei2p2;

    // q=0: Re DR1
    deriv[0][0] = dDR1_re * (DR2c*c1 + c3);
    deriv[0][1] = 0.0;
    deriv[0][2] = dDR1_re * (c1 + DL2c*c3);
    deriv[0][3] = 0.0;

    // q=1: Im DR1
    deriv[1][0] = dDR1_im * (DR2c*c1 + c3);
    deriv[1][1] = 0.0;
    deriv[1][2] = dDR1_im * (c1 + DL2c*c3);
    deriv[1][3] = 0.0;

    // q=2: Re DL1
    deriv[2][0] = 0.0;
    deriv[2][1] = dDL1_re * (DL2c*c0 + c2);
    deriv[2][2] = 0.0;
    deriv[2][3] = dDL1_re * (c0 + DR2c*c2);

    // q=3: Im DL1
    deriv[3][0] = 0.0;
    deriv[3][1] = dDL1_im * (DL2c*c0 + c2);
    deriv[3][2] = 0.0;
    deriv[3][3] = dDL1_im * (c0 + DR2c*c2);

    // q=4: Re DR2
    deriv[4][0] = dDR2c_re * (DR1*c1 + c2);
    deriv[4][1] = 0.0;
    deriv[4][2] = 0.0;
    deriv[4][3] = dDR2c_re * (c1 + DL1*c2);

    // q=5: Im DR2
    deriv[5][0] = dDR2c_im * (DR1*c1 + c2);
    deriv[5][1] = 0.0;
    deriv[5][2] = 0.0;
    deriv[5][3] = dDR2c_im * (c1 + DL1*c2);

    // q=6: Re DL2
    deriv[6][0] = 0.0;
    deriv[6][1] = dDL2c_re * (DL1*c0 + c3);
    deriv[6][2] = dDL2c_re * (c0 + DR1*c3);
    deriv[6][3] = 0.0;

    // q=7: Im DL2
    deriv[7][0] = 0.0;
    deriv[7][1] = dDL2c_im * (DL1*c0 + c3);
    deriv[7][2] = dDL2c_im * (c0 + DR1*c3);
    deriv[7][3] = 0.0;
  }
  /*
  void apply_Dterms_linear(datum_crosshand_visibilities& d,
                         std::complex<double>* io)
  {
    std::vector<std::complex<double>> tmp(4);
    tmp[0] = io[0];
    tmp[1] = io[1];
    tmp[2] = io[2];
    tmp[3] = io[3];
    apply_Dterms(d, tmp);
    io[0] = tmp[0];
    io[1] = tmp[1];
    io[2] = tmp[2];
    io[3] = tmp[3];
  }
  */
  void fill_Dterm_matrix_derivatives(
    const datum_crosshand_visibilities& d,
    const std::vector<size_t>& params,
    std::vector<std::array<std::complex<double>,16>>& dMdp) const;
  
  //! Returns complex visibility in Jy computed from the image given a datum_visibility object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  //! A user-supplied function that returns the closure amplitudes.  Takes a datum_polarization_fraction to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
  virtual double polarization_fraction(datum_polarization_fraction& d, double accuracy);

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

  //! Provides direct access to the constructed image in all Stokes parameters.  Sets a 2D grid of angles (alpha, beta) in radians and intensities in Jy per steradian.
  void get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V) const;

  //! Provides direct access to the complex visibilities.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& VI) const;

  //! Provides direct access to the crosshand complex visibilities.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& RR, std::vector<std::vector<std::complex<double> > >& LL, std::vector<std::vector<std::complex<double> > >& RL, std::vector<std::vector<std::complex<double> > >& LR) const;

  //! Provides direct access to the visibility amplitudes.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibility_amplitudes(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<double> >& V) const;

  //! Provides ability to use bicubic spline interpolator (true) instead of regular bicubic. Code defaults to false.
  void use_spline_interp( bool use_spline );

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) {
    //std::cout << "model_polarized_image proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
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
  std::vector<std::vector<double> > _I, _Q, _U, _V; //!< 2D grid of intensities at pixel locations in Jy/str.

  //! Applies the D-terms.  Checks if modeling inside and must be called in crosshand_visibilities if redefined by child classes.
  void apply_Dterms(const datum_crosshand_visibilities& d, std::vector< std::complex<double> >& crosshand_vector) const;

  //! Reads and remove Dterm parameters from the parameter list.  Checks if modeling inside and must be called in crosshand_visibilities if redefined by child classes.
  void read_and_strip_Dterm_parameters(std::vector<double>& parameters);


  size_t _size;

 private:
  std::vector<std::vector<double> > _u; //!< 2D grid of horizontal baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<double> > _v; //!< 2D grid of vertical baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
  std::vector<std::vector<std::complex<double> > > _VI, _VQ, _VU, _VV; //!< 2D grid of complex visibilities at pixel locations in Jy.

  // Define the interpolator object
  std::valarray<double> _i2du; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dv; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVI_r; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVI_i; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVQ_r; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVQ_i; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVU_r; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVU_i; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVV_r; //!< Arrays for defining the 2D interpolator objects.
  std::valarray<double> _i2dVV_i; //!< Arrays for defining the 2D interpolator objects.
  Interpolator2D _i2D_VIr, _i2D_VQr, _i2D_VUr, _i2D_VVr; //!< 2D interpolator object to estimate real component of the complex visibilities at arbitrary u-v locations.  model_polarized_image uses bicubic interpolation.
  Interpolator2D _i2D_VIi, _i2D_VQi, _i2D_VUi, _i2D_VVi; //!< 2D interpolator object to estimate imaginary component of the complex visibilities at arbitrary u-v locations.  model_polarized_image uses bicubic interpolation.

  //! Utility function to numerically computes the complex visibilities from the image data after _alpha, _beta, and _I have been set.  Currently pads the intensity array by a factor of 8.
  void compute_raw_visibilities();

  //! 2D FFT utility function based on FFTW
  std::vector<std::vector<std::complex<double> > > fft_2d(const std::vector<std::vector<double> > &I);		//fft2 (from fftw)

  //! 2D shift utility function to center the zero-baseline modes at the center of the array
  std::vector<std::vector<std::complex<double> > > fft_shift(const std::vector<std::vector<std::complex<double> > > &V);	//shift


  //! Quick bisection search to get index into Dterm array for a given station
  size_t inline get_index_from_station_code(std::string station_code) const
  {
    return ( std::lower_bound(_station_codes.begin(),_station_codes.end(),station_code)-_station_codes.begin() );
  }
  std::vector<size_t> _station_code_index_hash_table;
  std::vector< std::complex<double> > _Dterms;

 protected:
  bool _modeling_Dterms;
  std::vector<std::string> _station_codes;
};

};
#endif
