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
    int world_rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    
    if (world_rank == 0)
      this->print_timing_summary();
    std::cout << std::flush;  // ensure output appears
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

    // if (_use_cached_exp && !phase_cache_valid_ && !_data->empty()) {
    if (_generated_model && parameters==_current_parameters)
      {
	return;
      }
    else // parameters have changed
    {
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
      // generate_image(parameters,_I,_alpha,_beta);
      generate_image(parameters,_I,_I_flat,_alpha,_beta);
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;

      // rebuild cache, some catch statements to guard memory errors
      if (_use_cached_exp && _data && !_data->empty() && !phase_cache_valid_)
	update_phase_cache_all_data(*_data);
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
    _I_flat.resize(_Nx * _Ny);

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
	  I[i][j] = std::exp(parameters[k]);
	  I_flat[k++] = I[i][j];
	  // I[i][j] = std::exp(parameters[k]);
	  // I_flat[k] = std::exp(parameters[k++]);
	}
  }

  void model_image_adaptive_splined_raster::update_phase_cache_all_data(const std::vector<datum_visibility>& data)
  {
    ScopedTimer T(TimerID::UpdatePhaseCache, timer_ns_, timer_calls_);
    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t Npix = _Nx * _Ny;
    const size_t Nd   = data.size();

    phase_cache_.resize(Npix * Nd);
    spline_kernel_cache_.resize(Nd);

    for (size_t d = 0; d < Nd; ++d) {
	// Counter-rotate point
	const double ur =  _cpa*data[d].u + _spa*data[d].v;
	const double vr = -_spa*data[d].u + _cpa*data[d].v;
	// caching splines
	spline_kernel_cache_[d] = cubic_spline_kernel(ur, vr) * (_alpha[1][1] - _alpha[0][0]) * (_beta[1][1] - _beta[0][0]);
	size_t k = 0;
      for (size_t i = 0; i < _Nx; ++i)
	for (size_t j = 0; j < _Ny; ++j, ++k) {
	  const double phi = 2.0 * M_PI *
	    (ur * _alpha[i][j] + vr * _beta[i][j]);	  
	  // phase_cache_[d * Npix + k] =
	  //   _use_fast_exp_approx
          //   ? utils::fast_img_exp7(-phi)
          //   : std::exp(std::complex<double>(0.0, -phi)); //phase_cache_[k * Nd + d] = // inefficient memory layout, likely breaks L2 caching and vectorization
	  phase_cache_[d * Npix + k] =
	    _use_fast_exp_approx
            ? utils::fast_img_exp7(-phi)
            : std::exp(-std::complex<double>(0.0, 1.0) * phi);
	}
    }    
    cached_Nd_ = Nd;
    phase_cache_valid_ = true;
    //auto t1 = std::chrono::high_resolution_clock::now();
    //t_recompute_ += std::chrono::duration<double>(t1 - t0).count();
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
	    V += _I[i][j] * utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) );
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
  {
    static bool once = false;
    if (!once) {
      std::cerr << "[CACHE DEBUG] ENTERED CACHED visibility(size_t, ...)\n";
      once = true;
    }
    ScopedTimer T(
    _use_cached_exp ? TimerID::VisibilityCached
                    : TimerID::VisibilitySingle,
    timer_ns_, timer_calls_);

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
      if (_use_fast_exp_approx)
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	    V += _I[i][j] * utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) );
      }
      else if (_use_cached_exp) {
	{
	  ScopedTimer T_loop(TimerID::VisibilityCached_Loop,
			     timer_ns_, timer_calls_);
	  
	size_t k=0;
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

};
