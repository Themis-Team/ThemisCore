/*!
  \file model_image_adaptive_splined_raster.h
  \author Avery Broderick
  \date February, 2020
  \brief Header file for the model_image_adaptive_splined_raster image class.
  \details To be added
*/

#ifndef Themis_MODEL_IMAGE_ADAPTIVE_SPLINED_RASTER_H_
#define Themis_MODEL_IMAGE_ADAPTIVE_SPLINED_RASTER_H_

#include "model_image.h"
#include <vector>

//RG: WIP DEVELOPING CACHING AND OTHER SPEEDUPS
#include <chrono>
#include <array>
#include <cstdint>
// #include <unordered_map>
// #include <tuple>
// #include <cstring>   // for std::memcpy
#include <iostream>

namespace Themis {

/*!
  \brief Defines an adaptive splined raster image for image reconstruction.  The
  raster defined control points for an approximate spline, imposed in
  the u-v plane.  A rectalinear, rotated fov is adaptively determined.

  \details Generates a rastered pixel array, for which the parameters
  are the intensities at each pixel.  On construction the number of 
  pixels in the two directions are set, setting the resulting number
  of parameters.  An approximate cubic spline is imposed with a control
  parameter \f$a\f$ that should usually be set to -0.5 (default) to -0.75 to 
  prevent excessive ringing.

  Parameter list:\n
  - parameters[0] ........ pixel 0,0
  - parameters[1] ........ pixel 1,0
  ...
  - parameters[Nx-1] ..... pixel Nx-1,0
  - parameters[Nx] ....... pixel 0,1
  - parameters[Nx+1] ..... pixel 1,1
  ...
  - parameters[Nx*Ny-1] .. pixel Nx-1,Ny-1
  - parameters[Nx*Ny] .... fov in the x-direction
  - parameters[Nx*Ny+1] .. fov in the y-direction
  - parameters[Nx*Ny+2] .. rotation angle
*/
class model_image_adaptive_splined_raster : public model_image
{
 public:
  using model_visibility::visibility;

 private:
  std::vector<double> _I_flat; // turns the compiler really likes flattening _I into a 1d structure 
  //! Sets the image pixel values
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
  virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<double>& I_flat, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

  // for cached exponential phase
  const std::vector<datum_visibility>* _data = nullptr;
  // for precomputed spline kernel
  bool kernel_valid_ = false;
  std::vector<double> spline_kernel_cache_;
  
public:
  //! Constructs a model_image_splined_raster object.  Takes the extents and number of pixels in each directions (xmin and xmax are the locations of the minimum and maximum pixel centers, etc.).
  model_image_adaptive_splined_raster(size_t Nx, size_t Ny, double a=-0.5);
  virtual ~model_image_adaptive_splined_raster();

  virtual void print_timing_summary(int mpi_rank = -1) const;  

  void set_data(const std::vector<datum_visibility>& data) {
    _data = &data;              // COPY ONCE
    phase_cache_valid_ = false;
    // std::cerr << "[CACHE DEBUG] set_data() called on model @" << this << " Nd=" << data.size() << "\n";
  }

  //! State switch to select numerically computed visibilities using the machinery in model_image.  Once called, all future visibilities will be computed numerically until use_analytical_visibilities() is called.
  void use_numerical_visibilities();

  //! State switch to select analytically computed visibilities using the the analytical DFTs.  Once called, all future visibilities will be computed analytically until use_numerical_visibilities() is called.
  void use_analytical_visibilities();

  //! State switch to using the standard library exp, std::exp, in the construction of the DFT weights.  Once called, all future visibilities will be computed analytically until use_fast_exp_approx() is called.
  void use_exact_exp();

  //! State switch to using the fast polynomial approximation in utils::fast_img_exp, which exploits the fact that the DFT employs weights that use only exponentials with strictly imaginary arguments.  Once called, all future visibilities will be computed analytically until use_exact_exp() is called.
  void use_fast_exp_approx();
  void use_cached_exp();

  std::chrono::duration<double> t_lookup_only_{0};
  std::chrono::duration<double> t_visibility_total_{0};

  std::vector<std::complex<double>> phase_cache_; // size = _Nx * _Ny * Nd
  // std::vector<std::complex<float>> phase_cache_; // size = _Nx * _Ny * Nd ; [potential speedup if we get away with float, but not for our problem size and cache layout ...]
  bool phase_cache_valid_ = false;

  // Metadata
  size_t cached_Nd_; // Indexing convention: phase_cache_[k * Nd + d]

  void update_phase_cache_all_data(const std::vector<datum_visibility>& data);
  
  //! A user-supplied function that returns the number of the parameters the model expects
  virtual inline size_t size() const { return _size; };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
  virtual std::string model_tag() const;

  //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
  virtual void generate_model(std::vector<double> parameters);


  //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  std::complex<double> visibility(datum_visibility& d, double accuracy);
  std::complex<double> visibility(size_t d, datum_visibility& dv, double acc) override; // overloaded version for phase caching across full data set and pixels

  //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

  //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_phase(datum_closure_phase& d, double accuracy);

  //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
  virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

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
  bool _use_fast_exp_approx;
  bool _use_cached_exp;

  // ---------- PROFILING ----------
  enum class TimerID {
		      GenerateModel,
		      GenerateImage,
		      UpdatePhaseCache,
		      VisibilitySingle,
		      VisibilityCached,

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
