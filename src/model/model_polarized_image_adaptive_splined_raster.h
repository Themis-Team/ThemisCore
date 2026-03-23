/*!
  \file model_polarized_image_adaptive_splined_raster.h
  \author Avery Broderick
  \date March, 2020
  \brief Header file for the model_polarized_image_adaptive_splined_raster image class.
  \details To be added
*/

#ifndef Themis_MODEL_POLARIZED_IMAGE_ADAPTIVE_SPLINED_RASTER_H_
#define Themis_MODEL_POLARIZED_IMAGE_ADAPTIVE_SPLINED_RASTER_H_

#include "model_polarized_image.h"
#include <vector>

#include <chrono>
#include <array>
#include <cstdint>
#include <iostream>

namespace Themis {

  /*!
    \brief Defines an adaptive splined raster image for image reconstruction.  The
    raster defined control points for an approximate spline, imposed in
    the u-v plane.  A rectalinear, rotated fov is adaptively determined.

    \details Generates four rastered pixel arrays, on which the logarithm 
    of the intensity, logarithm of the polarization fraction, EVPA, and 
    muV = V/sqrt(Q^2+U^2+V^2) are specified at each pixel.  The dimensions
    of the pixel array are set at construction in the two directions.  The
    fields of view in the two directions and orientation of the array and
    the location of the origin are fit parameters.  Thus, the total number
    of parameters is \f$ 4 \times N_x \times N_y + 3 \f$.  An approximate 
    cubic spline is imposed on the constructed Stokes parameters (I,Q,U,V)
    with a control parameter \f$a\f$ that should usually be set to 
    -0.5 (default) to -0.75 to prevent excessive ringing.

    Parameter list:\n
    - parameters[0] ............. log(I) pixel 1,0
    - parameters[1] ............. log(I) pixel 2,0
    ...
    - parameters[Nx-2] .......... log(I) pixel Nx-3,0
    - parameters[Nx-1] .......... log(I) pixel 0,1
    - parameters[Nx] ............ log(I) pixel 1,1
    ...
    - parameters[Nx*Ny-1] ....... log(I) pixel Nx-1,Ny-1

    - parameters[Nx*Ny] ......... log(m) pixel 1,0
    ...
    - parameters[2*Nx*Ny-1] ..... log(m) pixel Nx-1,Ny-1

    - parameters[2*Nx*Ny] ......... EVPA pixel 1,0
    ...
    - parameters[3*Nx*Ny-1] ..... EVPA pixel Nx-1,Ny-1

    - parameters[3*Nx*Ny] ......... muV pixel 1,0
    ...
    - parameters[4*Nx*Ny-1] ..... muV pixel Nx-1,Ny-1

    - parameters[4*Nx*Ny] .... fov in the "x"-direction
    - parameters[4*Nx*Ny+1] .. fov in the "y"-direction
    - parameters[4*Nx*Ny+2] .. rotation angle

    \warning 
  */
  class model_polarized_image_adaptive_splined_raster : public model_polarized_image
  {
  private:
    //! Sets the image pixel values
    virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

    //! A user-supplied function that generates and returns rectalinear grid of Stokes parameters in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
    virtual void generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  
  public:
    //! Constructs a model_polarized_image_splined_raster object.  Takes the extents and number of pixels in each directions (xmin and xmax are the locations of the minimum and maximum pixel centers, etc.).
    model_polarized_image_adaptive_splined_raster(size_t Nx, size_t Ny, double a=-0.5);

    //! State switch to select numerically computed visibilities using the machinery in model_polarized_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
    void use_numerical_visibilities();
    //! State switch to select analytically computed visibilities using the the analytical Fourier transform of the Gaussian.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
    void use_analytical_visibilities();

    //! State switch to using the standard library exp, std::exp, in the construction of the DFT weights.  Once called, all future visibilities will be computed analytically until use_fast_exp_approx() is called.
    void use_exact_exp();

    //! State switch to using the fast polynomial approximation in utils::fast_img_exp, which exploits the fact that the DFT employs weights that use only exponentials with strictly imaginary arguments.  Once called, all future visibilities will be computed analytically until use_exact_exp() is called.
    void use_fast_exp_approx();
    
    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return _size; };
    
    //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);

    //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image given a datum_crosshand_visibilities_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual std::vector< std::complex<double> > crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy);



    size_t Nx() const { return _Nx; }
    size_t Ny() const { return _Ny; }
    
    bool phase_cache_valid() const { return phase_cache_valid_; }
    size_t cached_Nd() const { return cached_Nd_; }
    
    const std::vector<std::complex<double>>& phase_cache() const { return phase_cache_; }
    
    const std::vector<double>& spline_kernel_cache() const { return spline_kernel_cache_; }
    const std::vector<double>& spline_kernel_dfovx_cache() const { return spline_kernel_dfovx_cache_; }
    const std::vector<double>& spline_kernel_dfovy_cache() const { return spline_kernel_dfovy_cache_; }
    const std::vector<double>& spline_kernel_dpa_cache()   const { return spline_kernel_dpa_cache_; }
    
    const std::vector<double>& I_flat() const { return _I_flat; }
    const std::vector<double>& Q_flat() const { return _Q_flat; }
    const std::vector<double>& U_flat() const { return _U_flat; }
    const std::vector<double>& V_flat() const { return _V_flat; }


    
    //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
    
    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

    //! A user-supplied function that returns the closure amplitudes.  Takes a datum_fractional_polarization to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    //virtual double polarization_fraction(datum_polarization_fraction& d, double accuracy);

    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
    virtual std::string model_tag() const;

 private:
  double _xmin, _xmax;
  const size_t _Nx;
  double _ymin, _ymax;
  const size_t _Ny;
  const size_t _size;
  bool _defined_raster_grid;


  double _tpdx, _tpdy;
  double _cpa, _spa;
  const double _a;
  double cubic_spline_kernel_1d(double k) const;
  double cubic_spline_kernel(double u, double v) const;

  bool _use_analytical_visibilities;
  bool _use_fast_exp_approx = false;

    private:
  std::vector<double> _I_flat, _Q_flat, _U_flat, _V_flat;

  virtual void generate_polarized_image(
      std::vector<double> parameters,
      std::vector<std::vector<double>>& I,
      std::vector<std::vector<double>>& Q,
      std::vector<std::vector<double>>& U,
      std::vector<std::vector<double>>& V,
      std::vector<double>& I_flat,
      std::vector<double>& Q_flat,
      std::vector<double>& U_flat,
      std::vector<double>& V_flat,
      std::vector<std::vector<double>>& alpha,
      std::vector<std::vector<double>>& beta);

    // optimization:
    const data_crosshand_visibilities* _data = nullptr;
    
    std::vector<std::complex<double>> phase_cache_;
    std::vector<double> spline_kernel_cache_;
    std::vector<double> spline_kernel_dfovx_cache_;
    std::vector<double> spline_kernel_dfovy_cache_;
    std::vector<double> spline_kernel_dpa_cache_;
    
    bool phase_cache_valid_ = false;
    size_t cached_Nd_ = 0;
    
    double cubic_spline_kernel_1d_prime(double k) const;
    void update_phase_cache_all_data(const data_crosshand_visibilities& data);


    
virtual void fill_crosshand_visibilities(size_t d_idx,
                                         datum_crosshand_visibilities& d,
                                         double accuracy,
                                         std::complex<double>* out) override;
    
inline void cached_crosshand_into(size_t d_idx,
                                  datum_crosshand_visibilities& d,
                                  std::complex<double>* out)
{
  const size_t Npix   = _Nx * _Ny;
  const size_t offset = d_idx * Npix;

  std::complex<double> VI(0.0,0.0), VQ(0.0,0.0), VU(0.0,0.0), VV(0.0,0.0);

  for (size_t k = 0; k < Npix; ++k)
  {
    const std::complex<double>& ph = phase_cache_[offset + k];
    VI += _I_flat[k] * ph;
    VQ += _Q_flat[k] * ph;
    VU += _U_flat[k] * ph;
    VV += _V_flat[k] * ph;
  }

  const double K = spline_kernel_cache_[d_idx];
  VI *= K; VQ *= K; VU *= K; VV *= K;

  out[0] = VI + VV;
  out[1] = VI - VV;
  out[2] = VQ + std::complex<double>(0.0,1.0)*VU;
  out[3] = VQ - std::complex<double>(0.0,1.0)*VU;

  std::vector<std::complex<double>> tmp(4);
  tmp[0]=out[0]; tmp[1]=out[1]; tmp[2]=out[2]; tmp[3]=out[3];
  apply_Dterms(d, tmp);
  out[0]=tmp[0]; out[1]=tmp[1]; out[2]=tmp[2]; out[3]=tmp[3];
}


    
  public:
    virtual void print_timing_summary(int mpi_rank = -1) const;  

    void set_data(const data_crosshand_visibilities& data) override {
      _data = &data;
      phase_cache_valid_ = false;
      cached_Nd_ = 0;
    }
    bool _use_cached_exp = false;
    bool use_cached_exp() const override { return _use_cached_exp; }
    void use_cached_exp();
    
    std::vector<std::complex<double>>
    crosshand_visibilities(size_t d_idx, datum_crosshand_visibilities& d, double accuracy) override;



 // ---------- PROFILING ----------
  enum class TimerID {
		      GenerateModel,
		      GenerateImage,
		      GeneratePolarizedImage,
		      UpdatePhaseCache,
		      VisibilitySingle,
		      VisibilityCached,
		      CrosshandVisibilitySingle,
		      CrosshandVisibilityCached,

		      VisibilityCached_Rotation,
		      VisibilityCached_Loop,
		      VisibilityCached_Kernel,
		      VisibilityCached_Scale,

		      VisibilityNumerical,
		      ClosurePhase,
		      ClosureAmplitude,
		      COUNT
  };
  
  mutable std::array<std::uint64_t,(size_t)TimerID::COUNT> timer_ns_{};
  mutable std::array<std::uint64_t,(size_t)TimerID::COUNT> timer_calls_{};
  
  struct ScopedTimer {
    TimerID id;
    std::chrono::steady_clock::time_point t0;
    std::array<std::uint64_t,(size_t)TimerID::COUNT>& totals;
    std::array<std::uint64_t,(size_t)TimerID::COUNT>& calls;
    
    ScopedTimer(TimerID i,
		std::array<std::uint64_t,(size_t)TimerID::COUNT>& t,
		std::array<std::uint64_t,(size_t)TimerID::COUNT>& c)
      : id(i), t0(std::chrono::steady_clock::now()), totals(t), calls(c) {}
    
    ~ScopedTimer() {
      auto t1 = std::chrono::steady_clock::now();
      std::uint64_t ns =
	std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      totals[(size_t)id] += ns;
      calls[(size_t)id]  += 1;
    }
  };
 
    
    
  };
  
};
#endif
