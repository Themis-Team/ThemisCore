/*!
  \file model_image_diffractive_scattering_screen.h   
  \author Avery Broderick
  \date June 2020
  \brief Header file for diffractive scattering interface.
  \details Implements the diffractive scattering model, following Psaltis 2018 and ehtim with variable parameters.
  Based on model_visibility_galactic_center_scattering_screen.
*/

#ifndef Themis_MODEL_IMAGE_DIFFRACTIVE_SCATTERING_SCREEN_H_
#define Themis_MODEL_IMAGE_DIFFRACTIVE_SCATTERING_SCREEN_H_

#include "model_image.h"
#include "quadrature.h"
#include "interpolator1D.h"
#include "interpolator2D.h"
#include "utils.h"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <valarray>


namespace Themis {
  /*!
    \class model_visibility_galactic_center_diffractive_scattering_screen
    \author Paul Tiede
    \date Aug. 2019
    \brief Blurs the visibilities of the input model according to the diffractive scattering kernel from Johnson 2018.
    \details The diffractive scattering kernel parameters are appended to the end of the model parameter list:

    Parameter list:\n
    - parameters[0] ..... First model parameter.
    ...
    - parameters[N-1] ... Last model parameter where N is the size of the model.
    - parameters[N] ..... Major scattering axis FWHM in mas at a wavelength of 1 cm.
    - parameters[N+1].... Minor scattering axis FWHM in mas at a wavelength of 1 cm.
    - parameters[N+2] ... Position angle of the major axis in radians.
    - parameters[N+3] ... Scattering power law index.
    - parameters[N+4] ... log10(inner radius in km).
    - parameters[N+5] ... Magnification factor (ratio of the distance to the screen to the distance from the screen to the source).

    \warning
  */
  class model_image_diffractive_scattering_screen : public model_image 
  {

    public:
      model_image_diffractive_scattering_screen(model_image& model, 
						double frequency=230e9,
						std::string scattering_model="dipole");
						/* double observer_screen_distance=2.82*3.086e21, double source_screen_distance=5.53*3.086e21, */
						/* double theta_maj_mas_cm=1.38, double theta_min_ma_cm=0.703, double POS_ANG=81.9,  */
						/* double scatt_alpha=1.38, double r_in=800e5, double r_out=1e20); */

      virtual ~model_image_diffractive_scattering_screen() {};

      //! Size of the supplied model image, i.e., number of parameters expected.
      virtual inline size_t size() const { return _size; };

      //! Takes model parameters and generates the model with the diffractive scattering.
      virtual void generate_model(std::vector<double> parameters);

      //! Redefines the visibility function since we need to multiply by the scattering kernel
      virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
      
      //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

      //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double closure_phase(datum_closure_phase& d, double accuracy);

      //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
      virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);


      //! Sets the mpi communicator for generating images
      virtual void set_mpi_communicator(MPI_Comm comm);
    
    
    private:

      //!< Model visibility amplitudes
      model_image& _model;

      size_t _size;
      
      //const double _frequency; //!< frequency of observation in Hz
      const double _wavelength; //<! wavelength of observatoin in cm

      const std::string _scattering_model; //!< Which scattering model to use. Have three options, dipole, boxcar, vonMises
      //const double _observer_screen_distance; //<! Distance of observer to screen in cm.
      //const double _source_screen_distance; //<! Distance of source to screen in cm.

      std::vector<double> _screen_parameters;

      
      double _theta_maj_mas_cm, _theta_min_mas_cm; //!< Semi-major and minor axis length of Gaussian component of scattering screen in mas at refernce wavelength of 1 cm.
      //const double _POS_ANG; //!< Position angle of scattering Gaussian in degrees. Note we use the opposite convention of ehtim.
      double _phi0; //!< (90-_POS_ANG)
      double _M; //!< Magnification factor

      double _scatt_alpha; //!< Scattering turulence and dissipation scale power law parameter. Kolmogorov scattering has alpha=5/3
      double _rin; //!< Inner scale for the turbulence in the scattering screen.

      // Interpolators for special functions
      Interpolator1D _kZetainterp_vonMises, _kZetainterp_boxcar;
      Interpolator2D _kZetainterp_dipole;

      // Generate a new scattering screen
      void generate_scattering_screen(std::vector<double> screen_parameters);

      // Scattering special parameters
      //
      double _zeta0;
      double _Qbar;//!< Power spectrum normalization
      double _C_scatt_0;//!< Scattering coefficent for phase structure function
      double _kZeta; //!< Scattering kernel phase structure coefficient for different wandering magnetic field models

      double _Bmaj,_Bmin; //Bmaj,Bmin coefficients for scattering kernel in Psaltis et. al. 2018

      // Interpolation objects for int_maj, int_min
      void generate_minmajnorm(std::string scattering_model, double kzeta, double alpha, double phi0, double& int_min, double& int_maj, double& P_phi_norm);
      Interpolator2D _int_min_table;
      Interpolator2D _int_maj_table;
      Interpolator1D _Ppn_table;
      
      //!< Dphi function for the ensemble averaged image.
      double Dphi(double r, double phi) const;

      //!< Power spectrum of phase screen.
      double Q(double qx, double qy) const;

      //Class that defines the integrand for Bmaj and Bmin and P_phi
      class _P_phi : public Integrand
      {
       public:
         _P_phi(std::string axis,std::string model, double alpha, double phi0);
         inline void set_kZeta(double kZeta){_kZeta = kZeta; };
	 inline void set_alpha(double alpha){_alpha = alpha; };
	 inline void set_phi0(double phi0){_phi0 = phi0; };
         double operator()(double x) const;
       private:
         const std::string _axis; //Whether using major or minor axis
         const std::string _model; //scattering model
         double _alpha, _phi0;
         double _kZeta;
         double _angPow;
         double _phaseshift;
      };

      //!< Wandering magnetic field angular model.
      _P_phi _P_phi_func;
      //!< Normalization for wandering magnetic field model
      double _P_phi_norm;
      

      //! A user-supplied function that generates and returns rectalinear grid of intensities in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
      virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
      

  };

};

#endif

