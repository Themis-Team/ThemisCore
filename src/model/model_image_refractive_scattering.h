/*!
  \file model_image_refractive_scattering.h   
  \author Paul Tiede 
  \date Oct 2018
  \brief Header file for refractive scattering interface.
  \details Implements full scattering model for Sgr A*, following Johnson 2016 and ehtim.
  The values for the ensemble average scattering screen are taken from Johnson et al. 2018.
  For the ensemble average itself, there are three models for the wandering magnetic field.
  The default value is dipole following ehtim and Johnson et al. (2018). Adds nModes*nModes-1 
  to the parameter list since we will fit out the explicit realization of the scattering screen./
*/

#ifndef Themis_MODEL_IMAGE_REFRACTIVE_SCATTERING_H_
#define Themis_MODEL_IMAGE_REFRACTIVE_SCATTERING_H_


#include "model_image.h"
#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "quadrature.h"
#include "interpolator1D.h"
#include "utils.h"
#include <vector>
#include <string>
#include <fstream>
#include <fftw3.h>
#include <iomanip>
#include <iostream>

namespace Themis {

  /*!
    \class model_image_refractive_scattering
    \author Paul Tiede
    \date Oct. 2018
    \brief Defines the interface for models that generates refractive scattered images
    \details Scattering implementation assumes we are in average strong-scattering regime where
    diffractive scintillation has been averaged over.
    Parameter list: \n
    - parameters[0...n-1]   ... Model parameters for the source sans position angle, i.e. if a RIAF then spin inclination and so on. 
    - parameters[n]         ... position angle for the image.
    - parameters[n+1...m]   ... nModes^2-1=m-n-1 normalized Fourier modes of the scattering screen where nModes is passed in the constructor.
                            ... note the normalized means the distribution of these modes is given by a Gaussian with zero mean and unit variance.

  */
  class model_image_refractive_scattering : public model_image
  {
    
    private:
    
      virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

    public:
      /*!
        Constructor for refractive scattering class.
        Depends on whether the source you want to scatter is a model image class
        or one of the eht observables model classes e.g. model_visibility_amplitude.

        nModes defines the number of modes to use when constructing the scattering screen.
        The rest of the parameters define the ensemble average scattering screen, where the default values
        follow Johnson et al. 2018.
      */
      model_image_refractive_scattering(model_image& model, size_t nModes, 
                                  double tobs, double frequency=230e9,
                                  std::string scattering_model="dipole",
                                  double observer_screen_distance=2.82*3.086e21, double source_screen_distance=5.53*3.086e21,
                                  double theta_maj_mas_cm=1.38, double theta_min_ma_cm=0.703, double POS_ANG=81.9, 
                                  double scatt_alpha=1.38, double r_in=800e5, double r_out=1e20,
                                  double vs_ss_kms = 50.0, double vy_ss_kms = 0.0);

      virtual ~model_image_refractive_scattering() {};

      //! Size of the supplied model image plus the number of modes to include in the screen.
      //! Gives the number of parameters to fit.
      virtual inline size_t size() const { return _model->size() + _nModes*_nModes-1; };

      //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
      virtual std::string model_tag() const;

      //! Takes model parameters and generates scattered observables e.g. VA
      virtual void generate_model(std::vector<double> parameters);

      //! Sets the image scattered image resolution to be used
      //! resolution of the image is nrayxnray, the default is 128,128 which is probably too much
      void set_image_resolution(size_t nray);

      //! Sets the fov size of the image in units of radians
      //! The current default is 100uas.
      void set_screen_size(double fov);


      /*
        \brief Provide access to unscattered image.
      
        \details Provide access to *unscattered* image, note that the 
        scattered image is never defined, as it need not.
          
        \param alpha coordinate 1 in image plane
        \param beta coordinate 2 in image plane
        \param I Intensity in Jy
      */

      void get_unscattered_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;
      
      /*
        \brief Provide access to ensemble average image.
      
        \details Provide access to *ensemble average* image.
        Note that the image will have been padded by a factor of 8.
          
        \param alpha coordinate 1 in image plane
        \param beta coordinate 2 in image plane
        \param I Intensity in Jy
      */
      void get_ensemble_average_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;

      /*
        \brief Provide access to refractive scattered image.
      
        \details Provide access to *refractive scattered* image.
          
        \param alpha coordinate 1 in image plane
        \param beta coordinate 2 in image plane
        \param I Intensity in Jy
      */
      void get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;


      //! Defines a set of processors provided to the model for parallel
      //! computation via an MPI communicator.  Only facilates code 
      //! parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
  
    private:
      //model_visibility_amplitude& _modelVA;
      //model_closure_phase& _modelCP;
      //model_closure_amplitude& _modelCA;
      model_image* _model;
      std::vector<double> _current_model_params; //!< Current parameters for the intrinsic model.
      
      const size_t _nModes; //!< Number of modes to include in scattering screen.
      const double _tobs; //!<Observing time since start in seconds
      const double _frequency; //!< frequency of observation
      const double _wavelength; //<! wavelength of observation in cm

      const std::string _scattering_model; //!< Which scattering model to use
      const double _observer_screen_distance, _source_screen_distance;//<! distance of observer and source to screen in cm.
      const double _theta_maj_mas_cm, _theta_min_mas_cm; //!< Semi-major and minor axis length of Gaussian component of scattering screen in mas at reference wavelength of 1 cm.
      //const double _POS_ANG;//!< Position angle of scattering Gaussian in degrees. Note we use the opposite convention of ehtim 
      const double _phi0;//!< (90-POS_ANG)
      const double _M; //!< Magnification factor
      const double _rF; //!< Fresnel scale

      const double _scatt_alpha;//!< Scattering turbulence and dissipation scale power law parameter. Kolmogorov scattering has alpha=5/3
      const double _rin; //!< Inner scale for the turbulence in the scattering screen. 
      //const double _rout; //!< Outer scale for the turbulence in the scattering screen. 
      const double _vx_ss_kms, _vy_ss_kms; //!< Velocity of scattering screen with respect to earth in km/s

      std::vector<std::vector<std::complex<double> > > _epsilon;
      
      size_t _nray; //number of pixels in the image created
      double _fov; //fov for the image stored in radians.
      
      
      double _zeta0;
      double _Qbar;//!< Power spectrum normalization
      double _C_scatt_0;//!< Scattering coefficent for phase structure function
      double _kZeta; //!< Scattering kernel phase structure coefficient for different wandering magnetic field models

      double _Bmaj,_Bmin; //Bmaj,Bmin coefficients for scattering kernel in Psaltis et. al. 2018
      
      //!< Blurs the input model image using the ensemble averaged blurring kernel.
      void ensemble_blur_image();
      //!< Utility function. Generates the gradient of the phase screen since this is what we need.
      void compute_kphase_screen(std::vector<double> screen_params);
      std::valarray<double> _i2drxK, _i2dryK, _i2dDxPhi, _i2dDyPhi;
      Interpolator2D _i2D_Dxphi, _i2D_Dyphi; //!< 2D interpolator object for the gradient of the phase screen.

      std::vector<std::vector<double> > _Iea; //!< 2D grid of intensities of ensemble averaged image at pixel locations in Jy/str.
      //<! Generate the model visibilities
      void generate_model_visibilities();
      std::vector<std::vector<double> > _u; //!< 2D grid of horizontal baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
      std::vector<std::vector<double> > _v; //!< 2D grid of vertical baseline locations in lambda, relative to the fiducial direction of the image (i.e., unrotated by the position angle).
      std::vector<std::vector<std::complex<double> > > _Vsrc; //!< complex visibilities of intrinsic source
      
      //!< Arrays for the interpolator objects
      std::valarray<double> _i2drx, _i2dry, _i2dIea;
      Interpolator2D _i2D_Iea; //!< 2D interpolator object to estimate intensity of ensemble average position at arbitrary alpha,beta coordinates.  model_image uses bicubic interpolation.
      
      //!< Dphi function for the ensemble averaged image.
      double Dphi(double r, double phi) const;

      //!< Power spectrum of phase screen.
      double Q(double qx, double qy) const;

      //!< Generate random Gaussian epsilon screen from parameter list
      void make_epsilon_screen(std::vector<double> screen_params);

      //<! Utiliy function that takes the complex visibilities and then performs
      // an inverse fourier transform to give back the intensities.
      std::vector<std::vector<double> > ifft_2d(const std::vector<std::vector<std::complex<double> > > &V);

      //<! Utiliy function that shifts the fft so that the DC components is at (0,0).
      //<! Note this is needed since for odd dimensions there is an offset.
      std::vector<std::vector<std::complex<double> > > ifft_shift(const std::vector<std::vector<std::complex<double> > > &V);
      //Class that defines the integrand for Bmaj and Bmin and P_phi
      class _P_phi : public Integrand
      {
       public:
         _P_phi(std::string axis,std::string model, double alpha, double phi0);
         inline void set_kZeta(double kZeta){_kZeta = kZeta; };
         double operator()(double x) const;
       private:
         const std::string _axis; //Whether using major or minor axis
         const std::string _model; //scattering model
         const double _alpha, _phi0;
         double _kZeta;
         double _angPow;
         double _phaseshift;
      };

      //!< Wandering magnetic field angular model.
      _P_phi _P_phi_func;
      //!< Normalization for wandering magnetic field model
      double _P_phi_norm;
      


    
  };

};
#endif

