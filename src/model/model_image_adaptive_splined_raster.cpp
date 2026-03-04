/*!
  \file model_image_adaptive_splined_raster.cpp
  \author Avery Broderick
  \date  October, 2017
  \brief Implements the model_image_splined_raster image class.
  \details To be added
*/

#include "model_image_adaptive_splined_raster.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>


namespace Themis {

  // Profiling
  void model_image_adaptive_splined_raster::print_timing_summary(int mpi_rank) const
  {
    static const char* names[] = {
				  "GenerateModel",
				  "GenerateImage",
				  "UpdatePhaseCache",
				  "VisibilitySingle",
				  "VisibilityCached",
				  "VisibilityCached_Rotation",
				  "VisibilityCached_Loop",
				  "VisibilityCached_Kernel",
				  "VisibilityCached_Scale",
				  "VisibilityNumerical",
				  "ClosurePhase",
				  "ClosureAmplitude"
    };
    
    if (mpi_rank < 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    }
    
    std::cout << "\n===== Profiling summary (rank "
	      << mpi_rank << ") =====\n";
    
    for (size_t i = 0; i < (size_t)TimerID::COUNT; ++i) {
      double ms = timer_ns_[i] / 1.0e6;
      std::uint64_t n = timer_calls_[i];
      double avg = (n > 0) ? ms / n : 0.0;
      
      std::cout << std::setw(24) << names[i]
		<< " : total = " << ms << " ms"
		<< ", calls = " << n
		<< ", avg = " << avg << " ms/call\n";
    }
    std::cout << "=================================\n\n";
  }
  
  

  // Profiling end
  
  model_image_adaptive_splined_raster::model_image_adaptive_splined_raster(size_t Nx, size_t Ny, double a)
    : _Nx(Nx), _Ny(Ny), _size(_Nx*_Ny+3), _defined_raster_grid(false), _a(a), _use_analytical_visibilities(false), _use_fast_exp_approx(false), _use_cached_exp(false)
  {
  }
  model_image_adaptive_splined_raster::~model_image_adaptive_splined_raster()
  {
  }

  
  void model_image_adaptive_splined_raster::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }

  void model_image_adaptive_splined_raster::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = true;
  }

  void model_image_adaptive_splined_raster::use_exact_exp()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using std:exp in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = false;
  }

  void model_image_adaptive_splined_raster::use_fast_exp_approx()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using fast exponential approximation in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = true;
  }

  void model_image_adaptive_splined_raster::use_cached_exp()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using cached exponential phase in DFTs in rank " << world_rank << std::endl;

    _use_cached_exp = true;
  }

  void model_image_adaptive_splined_raster::generate_model(std::vector<double> parameters)
  {
    // profiling
    ScopedTimer T(TimerID::GenerateModel, timer_ns_, timer_calls_);

    if (_generated_model && parameters==_current_parameters)
      {
	return;
      }
    else // parameters have changed
      {
	static uint64_t inv_fovx=0, inv_fovy=0, inv_pa=0, inv_shiftx=0, inv_shifty=0, inv_other=0, calls=0; ++calls;
	
	// Invalidate phase cache ONLY when geometry changes (FOVx, FOVy, PA).
	// Intensity parameter changes do NOT invalidate cached phases.
	if (!_current_parameters.empty())
	  {
	    int world_rank;
	    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	    auto changed = [&](double a, double b) { return a != b; /* temporarily */ };
	    
	    const double old_fovx = _current_parameters[_size-3];
	    const double old_fovy = _current_parameters[_size-2];
	    const double old_pa   = _current_parameters[_size-1];
	    
	    const double new_fovx = parameters[_size-3];
	    const double new_fovy = parameters[_size-2];
	    const double new_pa   = parameters[_size-1];
	    
	    auto dump_change = [&](const char* name, double newv, double oldv)
	    {
	      if (world_rank==0) 
	      std::cerr << std::setprecision(17)
			<< name << " old=" << oldv
			<< " new=" << newv
			<< " diff=" << (newv - oldv)
			<< std::endl;
	    };

	    static int idx_fovx = _size-3;
	    static int idx_fovy = _size-2;
	    static int idx_pa = _size-1;
	    static int idx_shiftx = _size-5;
	    static int idx_shifty = _size-4;
	    if (changed(parameters[idx_fovx], _current_parameters[idx_fovx])) {
	      ++inv_fovx;
	      dump_change("fovx", parameters[idx_fovx], _current_parameters[idx_fovx]);
	    }
	    if (changed(parameters[idx_fovy], _current_parameters[idx_fovy])) {
	      ++inv_fovy;
	      dump_change("fovy", parameters[idx_fovy], _current_parameters[idx_fovy]);
	    }
	    if (changed(parameters[idx_pa], _current_parameters[idx_pa])) {
	      ++inv_pa;
	      dump_change("pa", parameters[idx_pa], _current_parameters[idx_pa]);
	    }  

	    if (debug_context_ == 2) std::cerr << "[RESTORE] ...";
	    if (debug_context_ == 1) std::cerr << "[FD] ...";

	    if (new_fovx != old_fovx || new_fovy != old_fovy || new_pa != old_pa) 
	      phase_cache_valid_ = false;
	  }
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	if ((calls % 1000)==0 && world_rank==0) std::cerr<<"counters: "<< inv_fovx<<","<<inv_fovy<<","<<inv_pa<<","<<inv_shiftx<<","<<inv_shifty<<","<<inv_other<<","<<calls<<std::endl;
	
	_current_parameters = parameters;
	
	// Set the fov
	_xmin = -0.5*parameters[_size-3];
	_xmax =  0.5*parameters[_size-3];
	_ymin = -0.5*parameters[_size-2];
	_ymax =  0.5*parameters[_size-2];
	_tpdx = 2.*M_PI*(_xmax-_xmin)/(_Nx-1);
	_tpdy = 2.*M_PI*(_ymax-_ymin)/(_Ny-1); 
	_cpa = std::cos(parameters[_size-1]);
	_spa = std::sin(parameters[_size-1]);
	
	
	// Generate the image using the user-supplied routine
	// generate_image(parameters,_I,_alpha,_beta); // 2d version
	generate_image(parameters,_I,_I_flat,_alpha,_beta); // 1d version (faster)
	
	// Set some boolean flags for what is and is not defined
	_generated_model = true;
	_generated_visibilities = false;
	
	// rebuild cache, some catch statements to guard memory errors
	if (_use_cached_exp && _data && !_data->empty() && !phase_cache_valid_)
	  if (cache_mode_ == VisibilityCacheMode::Global) {
	    update_phase_cache_all_data(*_data);
	    phase_cache_valid_ = true;
	  }
      }
  }

  std::string model_image_adaptive_splined_raster::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_adaptive_splined_raster " << _Nx << " " << _Ny << " " << _a;
    
    return tag.str();
  }

  void model_image_adaptive_splined_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    // Ensure flat buffer exists
    _I_flat.resize(_Nx * _Ny); // faster than 2d I[][]

    // generate_image(parameters, I, _I_flat, alpha, beta);
    model_image_adaptive_splined_raster::generate_image(parameters, I, _I_flat, alpha, beta);
  }
    void model_image_adaptive_splined_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<double>& I_flat, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    ScopedTimer T(TimerID::GenerateImage, timer_ns_, timer_calls_);
    // Allocate if necessary
   if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(_Nx))
    {
      alpha.resize(_Nx);
      beta.resize(_Nx);
      I.resize(_Nx);
      I_flat.resize(_Nx * _Ny);

      _defined_raster_grid=false;
      for (size_t j=0; j<alpha.size(); ++j)
      {
	if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(_Ny))
        {
	  alpha[j].resize(_Ny,0.0);
	  beta[j].resize(_Ny,0.0);
	  I[j].resize(_Ny,0.0);
	}
      }
    }
      
   if (_defined_raster_grid==false)
    {
      double dx = (_xmax-_xmin)/(int(_Nx)-1);
      double dy = (_ymax-_ymin)/(int(_Ny)-1);
      
      // Fill array with new image
      for (size_t j=0; j<alpha.size(); ++j)
      {
	for (size_t k=0; k<alpha[j].size(); ++k)
        {
	  alpha[j][k] = double(j)*dx + _xmin;
	  beta[j][k] = double(k)*dy  + _ymin;
	}
      }
      //_defined_raster_grid=true;
      _defined_raster_grid=false;
    }
    
    // Fill array with new image
    size_t k=0;
    for (size_t i=0; i<_Nx; ++i)
      for (size_t j=0; j<_Ny; ++j)
	{
	  I[i][j] = std::exp(parameters[k]); // RG: can we remove this now that I_flat is there?
	  I_flat[k++] = I[i][j];
	  // I[i][j] = std::exp(parameters[k]);
	  // I_flat[k] = std::exp(parameters[k++]);
	}
  }

  // Currently there is only no cache or global, but who knows mybe in the future epochlocal gets interesting too?!
  enum class CacheMode {Global, EpochLocal};
  CacheMode cache_mode_;
  // Then use as
  // if (cache_mode_ == CacheMode::EpochLocal)
  //   update_phase_cache_for_data(epoch_data);
  // else
  //   update_phase_cache_all_data(all_data);
  // Same for visibility

  void model_image_adaptive_splined_raster::prepare_visibility_cache(const data_visibility& data, const std::vector<size_t>& ids)
  {
    if (!_use_cached_exp) return;
    
    cache_mode_ = VisibilityCacheMode::EpochLocal;
    cached_Nd_  = ids.size();
    cached_ids_ = ids;
    
    const size_t Nd   = ids.size();
    const size_t Npix = _Nx * _Ny;
    
    phase_cache_.resize(Npix * Nd);
    spline_kernel_cache_.resize(Nd);
    
#ifndef NDEBUG
    cached_ids_ = ids;   // exact mapping epoch-local → global
#endif
    
    for (size_t i = 0; i < Nd; ++i)
      {
	const auto& d = data.datum(ids[i]);
	
	const double ur =  _cpa*d.u + _spa*d.v;
	const double vr = -_spa*d.u + _cpa*d.v;
	
	spline_kernel_cache_[i] = cubic_spline_kernel(ur, vr) * (_alpha[1][1] - _alpha[0][0]) * (_beta [1][1] - _beta [0][0]);
	
	size_t k = 0;
	const double twopi=2.0*M_PI;
	for (size_t ix = 0; ix < _Nx; ++ix)
	  for (size_t iy = 0; iy < _Ny; ++iy, ++k)
	    {
	      const double phi = /*2.0 * M_PI **/ (ur * _alpha[ix][iy] + vr * _beta[ix][iy]);
	      
	      phase_cache_[i * Npix + k] =
		_use_fast_exp_approx
		? utils::fast_img_exp7(-phi)
		: std::exp(-std::complex<double>(0.0, 1.0) * phi * twopi);
	    }
      }
    
    cached_Nd_ = Nd;
    phase_cache_valid_ = true;
  }




 void model_image_adaptive_splined_raster::update_phase_cache_all_data(const std::vector<datum_visibility>& data)
{
  ScopedTimer T(TimerID::UpdatePhaseCache, timer_ns_, timer_calls_);

  cache_mode_ = VisibilityCacheMode::Global;

  const size_t Npix = _Nx * _Ny;
  const size_t Nd   = data.size();

  phase_cache_.resize(Npix * Nd);
  spline_kernel_cache_.resize(Nd);

  // Only needed for mode-2 “correct-correct”
  spline_kernel_dfovx_cache_.resize(Nd);
  spline_kernel_dfovy_cache_.resize(Nd);
  spline_kernel_dpa_cache_.resize(Nd);

  // Grid spacings (exactly what the old code used via alpha/beta diffs)
  // Safe because Nx,Ny >= 2 for your use-cases.
  const double dx = (_xmax - _xmin) / double(_Nx - 1);
  const double dy = (_ymax - _ymin) / double(_Ny - 1);
  const double dxdy = dx * dy;

  // fovx,fovy used only for dx/dy derivatives (equivalently: dx = fovx/(Nx-1))
  const double fovx = (_xmax - _xmin);
  const double fovy = (_ymax - _ymin);

  const double inv_nx1 = 1.0 / double(_Nx - 1);
  const double inv_ny1 = 1.0 / double(_Ny - 1);

  // dxdy = fovx*fovy/((Nx-1)(Ny-1))
  const double d_dxdy_dfovx = (fovy * inv_nx1 * inv_ny1);
  const double d_dxdy_dfovy = (fovx * inv_nx1 * inv_ny1);

  // tpdx = 2π*fovx/(Nx-1), tpdy = 2π*fovy/(Ny-1)
  const double dtpdx_dfovx = 2.0 * M_PI * inv_nx1;
  const double dtpdy_dfovy = 2.0 * M_PI * inv_ny1;

  for (size_t d = 0; d < Nd; ++d)
  {
    const double u = data[d].u;
    const double v = data[d].v;

    // MUST match visibility() and phase construction convention
    const double ur =  _cpa*u + _spa*v;
    const double vr = -_spa*u + _cpa*v;

    // Kernel arguments: ku = ur*tpdx, kv = vr*tpdy  (NOT divide!)
    const double ku = ur * _tpdx;
    const double kv = vr * _tpdy;

    const double Ku  = cubic_spline_kernel_1d(ku);
    const double Kv  = cubic_spline_kernel_1d(kv);
    const double Kup = cubic_spline_kernel_1d_prime(ku);
    const double Kvp = cubic_spline_kernel_1d_prime(kv);

    const double kernel = Ku * Kv;
    const double K = dxdy * kernel;

    spline_kernel_cache_[d] = K;

    // dK/dfovx = d(dxdy)/dfovx * Ku*Kv + dxdy * (dKu/dku)*(dku/dfovx)*Kv
    // dku/dfovx = ur * d(tpdx)/dfovx
    spline_kernel_dfovx_cache_[d] =
      d_dxdy_dfovx * kernel + dxdy * (Kup * (ur * dtpdx_dfovx)) * Kv;

    // dK/dfovy similarly
    spline_kernel_dfovy_cache_[d] =
      d_dxdy_dfovy * kernel + dxdy * Ku * (Kvp * (vr * dtpdy_dfovy));

    // dK/dpa: dxdy, tpdx, tpdy independent of pa; only ur,vr depend on pa:
    // dur/dpa = vr, dvr/dpa = -ur
    // dku/dpa = tpdx*dur/dpa = tpdx*vr
    // dkv/dpa = tpdy*dvr/dpa = -tpdy*ur
    spline_kernel_dpa_cache_[d] =
      dxdy * ( (Kup * (_tpdx * vr)) * Kv + Ku * (Kvp * (_tpdy * (-ur))) );

    // Phase cache: exp(-i phi), phi = 2π(ur*alpha + vr*beta)
    size_t k = 0;
    const double twopi=2.*M_PI;
    for (size_t ix = 0; ix < _Nx; ++ix)
      for (size_t iy = 0; iy < _Ny; ++iy, ++k)
      {
        const double phi = /*2.0*M_PI **/ (ur * _alpha[ix][iy] + vr * _beta[ix][iy]);
        phase_cache_[d * Npix + k] =
          _use_fast_exp_approx ? utils::fast_img_exp7(-phi)
                               : std::exp(-std::complex<double>(0.0, 1.0) * phi * twopi);
      }
  }

  cached_Nd_ = Nd;
  phase_cache_valid_ = true;
}


  void model_image_adaptive_splined_raster::update_phase_cache_for_data(const std::vector<datum_visibility>& data)
  {
    ScopedTimer T(TimerID::UpdatePhaseCache, timer_ns_, timer_calls_);
    
    const size_t Npix = _Nx * _Ny;
    const size_t Nd   = data.size();
    
    // Resize cache to EXACTLY the provided data span
    phase_cache_.resize(Npix * Nd);
    spline_kernel_cache_.resize(Nd);
    spline_kernel_dfovx_cache_.resize(Nd);
    spline_kernel_dfovy_cache_.resize(Nd);
    spline_kernel_dpa_cache_.resize(Nd);

    const double fovx = (_xmax - _xmin);
    const double fovy = (_ymax - _ymin);
    
    const double inv_nx1 = 1.0 / double(_Nx - 1);
    const double inv_ny1 = 1.0 / double(_Ny - 1);
    
    const double dxdy = fovx * fovy * inv_nx1 * inv_ny1;
    
    // derivatives of dxdy wrt fovx/fovy (avoid dividing by tiny fov values)
    const double d_dxdy_dfovx = fovy * inv_nx1 * inv_ny1;
    const double d_dxdy_dfovy = fovx * inv_nx1 * inv_ny1;
    
    // derivatives of tpdx/tpdy wrt fovx/fovy (tpdx = 2π fovx/(Nx-1), etc.)
    const double dtpdx_dfovx = 2.0 * M_PI * inv_nx1;
    const double dtpdy_dfovy = 2.0 * M_PI * inv_ny1;
    
    for (size_t d = 0; d < Nd; ++d)
      {
	// Counter-rotate point
	const double ur =  _cpa*data[d].u + _spa*data[d].v;
	const double vr = -_spa*data[d].u + _cpa*data[d].v;
	
	// Cache spline kernel for THIS datum index

	/*
	spline_kernel_cache_[d] =
	  cubic_spline_kernel(ur, vr)
	  * (_alpha[1][1] - _alpha[0][0])
	  * (_beta [1][1] - _beta [0][0]);
	  */


	const double ku = ur * _tpdx;
	const double kv = vr * _tpdy;
	
	const double Ku  = cubic_spline_kernel_1d(ku);
	const double Kv  = cubic_spline_kernel_1d(kv);
	const double Kup = cubic_spline_kernel_1d_prime(ku);
	const double Kvp = cubic_spline_kernel_1d_prime(kv);
	
	const double kernel = Ku * Kv;
	
	// store the same prefactor you already use in cached visibility:
	// spline_kernel_cache_[d] = dxdy * kernel;
	// MUST match cached visibility path:
	// return spline_kernel_cache_[d_idx] * V;
	const double dx = (_alpha[1][1] - _alpha[0][0]);
	const double dy = (_beta [1][1] - _beta [0][0]);
	spline_kernel_cache_[d] = cubic_spline_kernel(ur, vr) * dx * dy;


	// dK/dfovx = d(dxdy)/dfovx * kernel + dxdy * dKu/dfovx * Kv
	// ku = ur * tpdx, tpdx = 2π fovx/(Nx-1)  => dku/dfovx = ur * dtpdx/dfovx
	spline_kernel_dfovx_cache_[d] =
	  d_dxdy_dfovx * kernel
	  + dxdy * (Kup * (ur * dtpdx_dfovx)) * Kv;
	
	// dK/dfovy similarly
	spline_kernel_dfovy_cache_[d] =
	  d_dxdy_dfovy * kernel
	  + dxdy * Ku * (Kvp * (vr * dtpdy_dfovy));
	
	// dK/dpa: tpdx,t pdy,dxdy independent of pa; only ur,vr depend on pa.
	// dur/dpa = vr ; dvr/dpa = -ur
	// dKu/dpa = Kup * dku/dpa = Kup * (tpdx * dur/dpa) = Kup * (tpdx * vr)
	// dKv/dpa = Kvp * dkv/dpa = Kvp * (tpdy * dvr/dpa) = Kvp * (tpdy * (-ur))
	spline_kernel_dpa_cache_[d] =
	  dxdy * ( (Kup * (_tpdx * vr)) * Kv
		   + Ku * (Kvp * (_tpdy * (-ur))) );
	
	/*
	
	// partials of kernel wrt u,v and tpdx,tpdy
	const double dk_du    = (Kup / _tpdx) * Kv;
	const double dk_dv    = Ku * (Kvp / _tpdy);
	
	const double dk_dtpdx = Kup * (-ur / (_tpdx * _tpdx)) * Kv;
	const double dk_dtpdy = Ku * (Kvp * (-vr / (_tpdy * _tpdy)));
	
	// K = dxdy * kernel
	// dK/dfovx = d(dxdy)/dfovx * kernel + dxdy * (dk/dtpdx) * d(tpdx)/dfovx
	spline_kernel_dfovx_cache_[d] = d_dxdy_dfovx * kernel + dxdy * (dk_dtpdx * dtpdx_dfovx);
	
	// dK/dfovy similarly
	spline_kernel_dfovy_cache_[d] = d_dxdy_dfovy * kernel + dxdy * (dk_dtpdy * dtpdy_dfovy);
	
	// dK/dpa: dxdy * (dk/du * du/dpa + dk/dv * dv/dpa)
	// with du/dpa = vr and dv/dpa = -ur in your rotation convention:
	spline_kernel_dpa_cache_[d] = dxdy * (dk_du * vr - dk_dv * ur);

	*/
	
	size_t k = 0;
	for (size_t i = 0; i < _Nx; ++i)
	  for (size_t j = 0; j < _Ny; ++j, ++k)
	    {
	      const double phi = 2.0 * M_PI *
		(ur * _alpha[i][j] + vr * _beta[i][j]);
	      
	      phase_cache_[d * Npix + k] =
		_use_fast_exp_approx
		? utils::fast_img_exp7(-phi)
		: std::exp(-std::complex<double>(0.0, 1.0) * phi);
	    }
      }
    
    cached_Nd_ = Nd;
    phase_cache_valid_ = true;
  }

  
  std::complex<double> model_image_adaptive_splined_raster::visibility(datum_visibility& d, double acc)
  {
    static bool once = false;
    if (!once) {
      std::cerr << "[CACHE DEBUG] ENTERED SINGLE visibility(datum, ...)\n";
      once = true;
    }
    ScopedTimer T_total(TimerID::VisibilitySingle, timer_ns_, timer_calls_);

    if (_use_analytical_visibilities)
    {      
      // Counter-rotate point
      double ur =  _cpa*d.u + _spa*d.v;
      double vr = -_spa*d.u + _cpa*d.v;

      std::complex<double> V(0.0,0.0);
      if (_use_fast_exp_approx)
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	    V += _I[i][j] * utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) ); // RG:CHECK 2PI
      }
      else
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	    V += _I[i][j] * std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );
      }	
      return ( cubic_spline_kernel(ur,vr)*V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]) );
    }
    else // NOT ROTATED
      return ( cubic_spline_kernel(d.u,d.v)*model_image::visibility(d, acc) );
  }

  std::complex<double> model_image_adaptive_splined_raster::visibility(size_t d_idx, datum_visibility& d, double acc)
  // std::complex<double> model_image_adaptive_splined_raster::visibility_epoch_local(size_t d_idx, datum_visibility& d, double acc)
  {
    if (!_use_cached_exp) {
      // Explicitly fall back to the polymorphic non-cached path
      return visibility(d, acc);
    }
    static bool once = false;
    if (!once) {
      std::cerr << "[CACHE DEBUG] ENTERED CACHED visibility(size_t, ...)\n";
      once = true;
    }

    ScopedTimer T(
    _use_cached_exp ? TimerID::VisibilityCached
                    : TimerID::VisibilitySingle,
    timer_ns_, timer_calls_);

    if (_use_cached_exp && !phase_cache_valid_) {
      std::cerr << "[CACHE ERROR] visibility called without cache. " << "d_idx=" << d_idx << std::endl;
      throw std::logic_error("Cached visibility called without a valid epoch cache");
    }

    if (_use_cached_exp)
      if (!phase_cache_valid_) 
	throw std::logic_error("Cached visibility requested but phase cache is invalid");
    
    if (_use_analytical_visibilities)
    {
      double ur,vr;
      {
      	ScopedTimer T_rot(TimerID::VisibilityCached_Rotation,
      			  timer_ns_, timer_calls_);
      	// Counter-rotate point
      	ur =  _cpa*d.u + _spa*d.v;
      	vr = -_spa*d.u + _cpa*d.v;
      }
      // double ur =  _cpa*d.u + _spa*d.v;
      // double vr = -_spa*d.u + _cpa*d.v;

      std::complex<double> V(0.0,0.0);
      if (_use_fast_exp_approx && !_use_cached_exp)
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	    V += _I[i][j] * utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) );
      }
      else if (_use_cached_exp) {
	{
	  ScopedTimer T_loop(TimerID::VisibilityCached_Loop,
			     timer_ns_, timer_calls_);
	  
	const size_t Npix = _Nx * _Ny;
	const size_t offset = d_idx * Npix; 

	//#pragma omp simd reduction(+:V)
	for (size_t k = 0; k < Npix; ++k)
	  V += _I_flat[k] * phase_cache_[offset + k];
	  //V += _I[i][j] * phase_cache_[offset + k];
	// NOTE: phase_cache_[k * cached_Nd_ + d_idx]; // slow memory layout
	}
	return spline_kernel_cache_[d_idx] * V;
      }
      else
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	      V += _I[i][j] * std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );

	return cubic_spline_kernel(ur, vr) * V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]);
      }
      std::complex<double> result;
      {
      	ScopedTimer T_kernel(TimerID::VisibilityCached_Kernel,
      			     timer_ns_, timer_calls_);
      	result = cubic_spline_kernel(ur, vr) * V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]);
      }

      return ( result );
      // return (cubic_spline_kernel(ur,vr)*V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]) );
    }
    else // NOT ROTATED
      return ( cubic_spline_kernel(d.u,d.v)*model_image::visibility(d, acc) );
  }

  
  double model_image_adaptive_splined_raster::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp(d.u,d.v,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      return std::abs(visibility(tmp,acc));
    }
    else
      return ( cubic_spline_kernel(d.u,d.v)*model_image::visibility_amplitude(d, acc) );
  }

  double model_image_adaptive_splined_raster::closure_phase(datum_closure_phase& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
      datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station1,d.Source);
      std::complex<double> V123 = visibility(tmp1,acc)*visibility(tmp2,acc)*visibility(tmp3,acc);
      
      return ( std::imag(std::log(V123))*180.0/M_PI );
    }
    else
      return ( model_image::closure_phase(d,acc) );
  }



  double model_image_adaptive_splined_raster::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
      datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
      datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);
      
      double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
      return ( V1234 );
    }
    else
      return ( model_image::closure_amplitude(d,acc) );
  }


  double model_image_adaptive_splined_raster::cubic_spline_kernel_1d(double k) const
  {
    if (std::fabs(k)<1e-2)
      return 1.0 - (2.0*_a-1.0)*(k*k)/15.0 + (16.0*_a+1.0)*(k*k*k*k)/560.0;
    else
    {
      double sk=std::sin(k), ck=std::cos(k);
      double c2k=ck*ck-sk*sk;

      double G = -4.0*sk*(2.0*_a*ck+(4.0*_a+3.0))/(k*k*k) + 12.0*(_a*(1.0-c2k)+2.0*(1.0-ck))/(k*k*k*k);

      return G;
    }
  }

  double model_image_adaptive_splined_raster::cubic_spline_kernel(double u0, double v0) const
  {
    return cubic_spline_kernel_1d(u0*_tpdx)*cubic_spline_kernel_1d(v0*_tpdy);
  }

  double model_image_adaptive_splined_raster::cubic_spline_kernel_1d_prime(double k) const
  {
    const double ak = std::fabs(k);
    if (ak < 1e-2) {
      // G(k) = 1 - (2a-1)k^2/15 + (16a+1)k^4/560
      // G'(k)= -(2a-1)*2k/15 + (16a+1)*k^3/140
      const double a = _a;
      return -(2.0*a - 1.0) * (2.0*k)/15.0 + (16.0*a + 1.0) * (k*k*k)/140.0;
    } else {
      const double a = _a;
      const double b = 4.0*a + 3.0;
      
      const double s  = std::sin(k);
      const double c  = std::cos(k);
      const double c2 = c*c - s*s;          // cos(2k)
      const double s2 = 2.0*s*c;            // sin(2k)
      
      const double kk  = k*k;
      const double k3  = kk*k;
      const double k4  = k3*k;
      const double k5  = k4*k;
      
      // A = -4 s (2 a c + b)
      const double A  = -4.0 * s * (2.0*a*c + b);
      // A' = -4 (2 a cos(2k) + b cos(k))
      const double Ap = -4.0 * (2.0*a*c2 + b*c);
      
      // B = 12*( a*(1 - cos2k) + 2*(1 - cosk) )
      const double B  = 12.0 * ( a*(1.0 - c2) + 2.0*(1.0 - c) );
      // B' = 24*(a sin2k + sin k) = 24*s*(2a c + 1)
      const double Bp = 24.0 * s * (2.0*a*c + 1.0);
      
      // G = A/k^3 + B/k^4
      // G' = (A' k^2 - 3 A k + B' k - 4 B) / k^5
      const double num = Ap*kk - 3.0*A*k + Bp*k - 4.0*B;
      return num / k5;
    }
  }
}
