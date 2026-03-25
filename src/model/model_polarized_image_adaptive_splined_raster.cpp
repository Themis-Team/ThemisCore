/*!
  \file model_polarized_image_adaptive_splined_raster.cpp
  \author Avery Broderick
  \date  March, 2020
  \brief Implements the model_polarized_image_splined_raster image class.
  \details To be added
*/

#include "model_polarized_image_adaptive_splined_raster.h"
#include "utils.h"
#include <iostream>
#include <iomanip>

namespace Themis {

    void model_polarized_image_adaptive_splined_raster::print_timing_summary(int mpi_rank) const
  {
    static const char* names[] = {
				  "GenerateModel",
				  "GenerateImage",
				  "GeneratePolarizedImage",
				  "UpdatePhaseCache",
				  "VisibilitySingle",
				  "VisibilityCached",
				  "CrosshandVisibilitySingle",
				  "CrosshandVisibilityCached",
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

  
  model_polarized_image_adaptive_splined_raster::model_polarized_image_adaptive_splined_raster(size_t Nx, size_t Ny, double a)
    : _Nx(Nx), _Ny(Ny), _size(4*_Nx*_Ny+3), _defined_raster_grid(false), _a(a), _use_analytical_visibilities(true),  _use_fast_exp_approx(false),
    _use_cached_exp(false)

  {
  }

  void model_polarized_image_adaptive_splined_raster::use_numerical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = false;
  }

  void model_polarized_image_adaptive_splined_raster::use_analytical_visibilities()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;

    _use_analytical_visibilities = true;
  }

  void model_polarized_image_adaptive_splined_raster::use_exact_exp()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using std:exp in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = false;
  }

  void model_polarized_image_adaptive_splined_raster::use_fast_exp_approx()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
    std::cout << "Using fast exponential approximation in DFTs in rank " << world_rank << std::endl;

    _use_fast_exp_approx = true;
  }

  void model_polarized_image_adaptive_splined_raster::use_cached_exp()
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    std::cout << "Using cached exponential phase in DFTs in rank "
	      << world_rank << std::endl;
    
    _use_cached_exp = true;
  }
  
  void model_polarized_image_adaptive_splined_raster::generate_model(std::vector<double> parameters)
  {
    ScopedTimer T(TimerID::GenerateModel, timer_ns_, timer_calls_);

    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      const int idx_fovx = _size - 3;
      const int idx_fovy = _size - 2;
      const int idx_pa   = _size - 1;

      if (!_current_parameters.empty()) {
	const bool geom_changed =
	  (parameters[idx_fovx] != _current_parameters[idx_fovx]) ||
	  (parameters[idx_fovy] != _current_parameters[idx_fovy]) ||
	  (parameters[idx_pa]   != _current_parameters[idx_pa]);
	
	if (geom_changed) {
	  phase_cache_valid_   = false;
	  _defined_raster_grid = false;
	}
      }
      /*
      if (!_current_parameters.empty()) {
	if (parameters[idx_fovx] != _current_parameters[idx_fovx] ||
	    parameters[idx_fovy] != _current_parameters[idx_fovy] ||
	    parameters[idx_pa]   != _current_parameters[idx_pa]) {
	  phase_cache_valid_ = false;
	}
      }
      */
      
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
      // generate_polarized_image(parameters,_I,_Q,_U,_V,_alpha,_beta);
      generate_polarized_image(parameters, _I, _Q, _U, _V, _I_flat, _Q_flat, _U_flat, _V_flat, _alpha, _beta);

      if (_use_cached_exp && _data && _data->size() > 0 && !phase_cache_valid_) {
	update_phase_cache_all_data(*_data);
	phase_cache_valid_ = true;
      }
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }

  void model_polarized_image_adaptive_splined_raster::generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    generate_polarized_image(parameters, I, Q, U, V, _I_flat, _Q_flat, _U_flat, _V_flat, alpha, beta);
  }

  void model_polarized_image_adaptive_splined_raster::generate_polarized_image(
    std::vector<double> parameters,
    std::vector<std::vector<double> >& I,
    std::vector<std::vector<double> >& Q,
    std::vector<std::vector<double> >& U,
    std::vector<std::vector<double> >& V,
    std::vector<double>& I_flat,
    std::vector<double>& Q_flat,
    std::vector<double>& U_flat,
    std::vector<double>& V_flat,
    std::vector<std::vector<double> >& alpha,
    std::vector<std::vector<double> >& beta)
  {
    ScopedTimer T(TimerID::GeneratePolarizedImage, timer_ns_, timer_calls_);
    if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(_Nx))
      {
	alpha.resize(_Nx);
	beta.resize(_Nx);
	I.resize(_Nx);
	Q.resize(_Nx);
	U.resize(_Nx);
	V.resize(_Nx);
	_defined_raster_grid = false;
	
	for (size_t j=0; j<alpha.size(); ++j)
	  {
	    if (alpha[j].size()!=beta[j].size() ||
		beta[j].size()!=I[j].size() ||
		I[j].size()!=size_t(_Ny))
	      {
		alpha[j].resize(_Ny,0.0);
		beta[j].resize(_Ny,0.0);
		I[j].resize(_Ny,0.0);
		Q[j].resize(_Ny,0.0);
		U[j].resize(_Ny,0.0);
		V[j].resize(_Ny,0.0);
	      }
	  }
      }
    
    if (_defined_raster_grid==false)
      {
	const double dx = (_xmax-_xmin)/(double(_Nx)-1.0);
	const double dy = (_ymax-_ymin)/(double(_Ny)-1.0);
	
	for (size_t j=0; j<alpha.size(); ++j)
	  for (size_t k=0; k<alpha[j].size(); ++k)
	    {
	      alpha[j][k] = double(j)*dx + _xmin;
	      beta[j][k]  = double(k)*dy + _ymin;
	    }
	
	_defined_raster_grid = true;
      }
    
    const size_t Npix = _Nx * _Ny;
    I_flat.resize(Npix);
    Q_flat.resize(Npix);
    U_flat.resize(Npix);
    V_flat.resize(Npix);
    
    size_t k = 0;
    for (size_t i = 0; i < _Nx; ++i)
      for (size_t j = 0; j < _Ny; ++j, ++k)
	{
	  const double Ii   = std::exp(parameters[k]);
	  const double m    = std::exp(parameters[k + Npix]);
	  const double EVPA = parameters[k + 2*Npix];
	  const double muV  = parameters[k + 3*Npix];
	  
	  const double pol = m * std::sqrt(1.0 - muV*muV);
	  
	  const double Qi = pol * std::cos(2.0*EVPA) * Ii;
	  const double Ui = pol * std::sin(2.0*EVPA) * Ii;
	  const double Vi = m * muV * Ii;
	  
	  I[i][j] = Ii;
	  Q[i][j] = Qi;
	  U[i][j] = Ui;
	  V[i][j] = Vi;
	  
	  I_flat[k] = Ii;
	  Q_flat[k] = Qi;
	  U_flat[k] = Ui;
	  V_flat[k] = Vi;
	}
  }
  
  void model_polarized_image_adaptive_splined_raster::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    ScopedTimer T(TimerID::GenerateImage, timer_ns_, timer_calls_);

    // Allocate if necessary
   if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(_Nx))
    {
      alpha.resize(_Nx);
      beta.resize(_Nx);
      I.resize(_Nx);
      _defined_raster_grid=false;
      for (size_t j=0; j<alpha.size(); j++)
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
      for (size_t j=0; j<alpha.size(); j++)
      {
	for (size_t k=0; k<alpha[j].size(); k++)
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
    for (size_t i=0; i<_Nx; i++)
      for (size_t j=0; j<_Ny; j++)
	I[i][j] = std::exp(parameters[k++]);

  }

  std::string model_polarized_image_adaptive_splined_raster::model_tag() const
  {
    std::stringstream tag;
    tag << "model_polarized_image_adaptive_splined_raster " << _Nx << " " << _Ny << " " << _a
	<< " " <<  _modeling_Dterms;
    if (_modeling_Dterms)
      for (size_t j=0; j<_station_codes.size(); ++j)
	tag << " " << _station_codes[j];
    
    return tag.str();
  }


  /*
  void model_polarized_image_adaptive_splined_raster::fill_crosshand_visibilities(
    size_t d_idx,
    datum_crosshand_visibilities& d,
    double accuracy,
    std::complex<double>* out)
{
  ScopedTimer T(
    _use_cached_exp ? TimerID::CrosshandVisibilityCached
                    : TimerID::CrosshandVisibilitySingle,
    timer_ns_, timer_calls_);

  (void)accuracy;

  if (!_use_cached_exp || !phase_cache_valid_ || d_idx >= cached_Nd_) {
    std::vector<std::complex<double>> tmp = crosshand_visibilities(d, accuracy);
    out[0] = tmp[0];
    out[1] = tmp[1];
    out[2] = tmp[2];
    out[3] = tmp[3];
    return;
  }

  if (_use_analytical_visibilities)
  {
    const size_t Npix   = _Nx * _Ny;
    const size_t offset = d_idx * Npix;

    std::complex<double> VI(0.0,0.0);
    std::complex<double> VQ(0.0,0.0);
    std::complex<double> VU(0.0,0.0);
    std::complex<double> VV(0.0,0.0);

    for (size_t k = 0; k < Npix; ++k)
    {
      const std::complex<double>& ph = phase_cache_[offset + k];
      VI += _I_flat[k] * ph;
      VQ += _Q_flat[k] * ph;
      VU += _U_flat[k] * ph;
      VV += _V_flat[k] * ph;
    }

    const double K = spline_kernel_cache_[d_idx];
    VI *= K;
    VQ *= K;
    VU *= K;
    VV *= K;

    out[0] = VI + VV;                                // RR
    out[1] = VI - VV;                                // LL
    out[2] = VQ + std::complex<double>(0.0,1.0)*VU;  // RL
    out[3] = VQ - std::complex<double>(0.0,1.0)*VU;  // LR

    // apply_Dterms currently wants a vector, so convert only here
    std::vector<std::complex<double>> tmp(4);
    tmp[0] = out[0];
    tmp[1] = out[1];
    tmp[2] = out[2];
    tmp[3] = out[3];
    apply_Dterms(d, tmp);
    out[0] = tmp[0];
    out[1] = tmp[1];
    out[2] = tmp[2];
    out[3] = tmp[3];
  }
  else
  {
    std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::fill_crosshand_visibilities :"
              << " numerical visibilities have not been implemented.\n\n";
    std::exit(1);
  }
}
  */




  
void model_polarized_image_adaptive_splined_raster::fill_crosshand_visibilities(
    size_t d_idx,
    datum_crosshand_visibilities& d,
    double accuracy,
    std::complex<double>* out)
{
  ScopedTimer T(
    _use_cached_exp ? TimerID::CrosshandVisibilityCached
                    : TimerID::CrosshandVisibilitySingle,
    timer_ns_, timer_calls_);

  (void)accuracy;

  if (!_use_cached_exp || !phase_cache_valid_ || d_idx >= cached_Nd_) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cerr
      << "[XH-FILL-FALLBACK] rank=" << rank
      << " use_cached_exp=" << _use_cached_exp
      << " phase_cache_valid=" << phase_cache_valid_
      << " d_idx=" << d_idx
      << " cached_Nd=" << cached_Nd_
      << " phase_cache_size=" << phase_cache_.size()
      << " spline_kernel_cache_size=" << spline_kernel_cache_.size()
      << " Nx=" << _Nx
      << " Ny=" << _Ny
      << " u=" << d.u
      << " v=" << d.v
      << " source=" << d.Source
      << " stations=" << d.Station1 << "-" << d.Station2
      << "\n";

    std::cerr
      << "ERROR: model_polarized_image_adaptive_splined_raster::fill_crosshand_visibilities\n"
      << "       attempted to fall back to old datum-only crosshand path.\n"
      << "       This should not happen in the cached crosshand gain implementation.\n";

    std::exit(1);
  }

  if (_use_analytical_visibilities)
  {
    const size_t Npix   = _Nx * _Ny;
    const size_t offset = d_idx * Npix;

    std::complex<double> VI(0.0,0.0);
    std::complex<double> VQ(0.0,0.0);
    std::complex<double> VU(0.0,0.0);
    std::complex<double> VV(0.0,0.0);

    for (size_t k = 0; k < Npix; ++k)
    {
      const std::complex<double>& ph = phase_cache_[offset + k];
      VI += _I_flat[k] * ph;
      VQ += _Q_flat[k] * ph;
      VU += _U_flat[k] * ph;
      VV += _V_flat[k] * ph;
    }

    const double K = spline_kernel_cache_[d_idx];
    VI *= K;
    VQ *= K;
    VU *= K;
    VV *= K;

    out[0] = VI + VV;
    out[1] = VI - VV;
    out[2] = VQ + std::complex<double>(0.0,1.0)*VU;
    out[3] = VQ - std::complex<double>(0.0,1.0)*VU;

    std::vector<std::complex<double>> tmp(4);
    tmp[0] = out[0];
    tmp[1] = out[1];
    tmp[2] = out[2];
    tmp[3] = out[3];
    apply_Dterms(d, tmp);
    out[0] = tmp[0];
    out[1] = tmp[1];
    out[2] = tmp[2];
    out[3] = tmp[3];
  }
  else
  {
    std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::fill_crosshand_visibilities :"
              << " numerical visibilities have not been implemented.\n\n";
    std::exit(1);
  }
}






std::vector<std::complex<double>>
model_polarized_image_adaptive_splined_raster::crosshand_visibilities(
    size_t d_idx,
    datum_crosshand_visibilities& d,
    double accuracy)
{
  // utils::ScopedTimer T(
  // 		       _use_cached_exp ? utils::TimerID::CrosshandVisibilityCached
  // 		       : utils::TimerID::CrosshandVisibilitySingle,
  //     timer_ns_, timer_calls_);

  (void)accuracy;

  if (!_use_cached_exp || !phase_cache_valid_ || d_idx >= cached_Nd_)
    return crosshand_visibilities(d, accuracy);

  if (_use_analytical_visibilities)
  {
    const size_t Npix   = _Nx * _Ny;
    const size_t offset = d_idx * Npix;

    std::complex<double> VI(0.0,0.0);
    std::complex<double> VQ(0.0,0.0);
    std::complex<double> VU(0.0,0.0);
    std::complex<double> VV(0.0,0.0);

    for (size_t k = 0; k < Npix; ++k)
    {
      const std::complex<double>& ph = phase_cache_[offset + k];
      VI += _I_flat[k] * ph;
      VQ += _Q_flat[k] * ph;
      VU += _U_flat[k] * ph;
      VV += _V_flat[k] * ph;
    }

    const double spline_factor = spline_kernel_cache_[d_idx];
    VI *= spline_factor;
    VQ *= spline_factor;
    VU *= spline_factor;
    VV *= spline_factor;

    std::vector<std::complex<double> > crosshand_vector(4);
    crosshand_vector[0] = VI + VV;                                // RR
    crosshand_vector[1] = VI - VV;                                // LL
    crosshand_vector[2] = VQ + std::complex<double>(0.0,1.0)*VU;  // RL
    crosshand_vector[3] = VQ - std::complex<double>(0.0,1.0)*VU;  // LR

    apply_Dterms(d, crosshand_vector);

    return crosshand_vector;
  }
  else
  {
    std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::crosshand_visibilities(size_t,...) : "
              << "numerical visibilities have not been implemented.\n";
    std::exit(1);
  }
}
  
  
  /*
std::vector<std::complex<double>>
model_polarized_image_adaptive_splined_raster::crosshand_visibilities(
    size_t d_idx,
    datum_crosshand_visibilities& d,
    double accuracy)
{

  // ScopedTimer T(
  //   _use_cached_exp ? TimerID::CrosshandVisibilityCached
  //                   : TimerID::CrosshandVisibilitySingle,
  //   timer_ns_, timer_calls_);

  std::complex<double> out[4];
  fill_crosshand_visibilities(d_idx, d, accuracy, out);

  std::vector<std::complex<double>> v(4);
  v[0] = out[0];
  v[1] = out[1];
  v[2] = out[2];
  v[3] = out[3];
  return v;
}
  */

  /* 
  std::vector< std::complex<double> >
  model_polarized_image_adaptive_splined_raster::crosshand_visibilities(
    size_t d_idx,
    datum_crosshand_visibilities& d,
    double accuracy)
{
  ScopedTimer T(
    _use_cached_exp ? TimerID::CrosshandVisibilityCached
                    : TimerID::CrosshandVisibilitySingle,
    timer_ns_, timer_calls_);

  static bool did_idx = false;
  if (!did_idx) {
    did_idx = true;
    std::cerr << "[TIMERCHK] entered indexed crosshand_visibilities\n";
  }

  
  (void)accuracy;

  if (!_use_cached_exp || !phase_cache_valid_ || d_idx >= cached_Nd_)
    return crosshand_visibilities(d, accuracy);

  if (_use_analytical_visibilities)
  {
    const size_t Npix   = _Nx * _Ny;
    const size_t offset = d_idx * Npix;

    std::complex<double> VI(0.0,0.0);
    std::complex<double> VQ(0.0,0.0);
    std::complex<double> VU(0.0,0.0);
    std::complex<double> VV(0.0,0.0);

    for (size_t k = 0; k < Npix; ++k)
    {
      const std::complex<double>& ph = phase_cache_[offset + k];
      VI += _I_flat[k] * ph;
      VQ += _Q_flat[k] * ph;
      VU += _U_flat[k] * ph;
      VV += _V_flat[k] * ph;
    }

    const double spline_factor = spline_kernel_cache_[d_idx];
    VI *= spline_factor;
    VQ *= spline_factor;
    VU *= spline_factor;
    VV *= spline_factor;

    std::vector< std::complex<double> > crosshand_vector(4);
    crosshand_vector[0] = VI + VV;                                // RR
    crosshand_vector[1] = VI - VV;                                // LL
    crosshand_vector[2] = VQ + std::complex<double>(0.0,1.0)*VU;  // RL
    crosshand_vector[3] = VQ - std::complex<double>(0.0,1.0)*VU;  // LR

    apply_Dterms(d, crosshand_vector);





  static bool did_check = false;
if (!did_check) {
  did_check = true;

  std::vector<std::complex<double>> oldv = crosshand_visibilities(d, accuracy);

  auto rel = [](std::complex<double> a, std::complex<double> b) {
    double den = std::max(1.0, std::abs(b));
    return std::abs(a - b) / den;
  };

  std::cerr << std::setprecision(17);
  std::cerr << "[XH-CHECK] d_idx=" << d_idx << "\n";
  std::cerr << "  RR new=" << crosshand_vector[0] << " old=" << oldv[0]
            << " abs=" << std::abs(crosshand_vector[0]-oldv[0])
            << " rel=" << rel(crosshand_vector[0], oldv[0]) << "\n";
  std::cerr << "  LL new=" << crosshand_vector[1] << " old=" << oldv[1]
            << " abs=" << std::abs(crosshand_vector[1]-oldv[1])
            << " rel=" << rel(crosshand_vector[1], oldv[1]) << "\n";
  std::cerr << "  RL new=" << crosshand_vector[2] << " old=" << oldv[2]
            << " abs=" << std::abs(crosshand_vector[2]-oldv[2])
            << " rel=" << rel(crosshand_vector[2], oldv[2]) << "\n";
  std::cerr << "  LR new=" << crosshand_vector[3] << " old=" << oldv[3]
            << " abs=" << std::abs(crosshand_vector[3]-oldv[3])
            << " rel=" << rel(crosshand_vector[3], oldv[3]) << "\n";
}





    return crosshand_vector;
  }
  else
  {
    std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::crosshand_visibilities :"
              << " numerical visibilities have not been implemented.\n\n";
    std::exit(1);
  }  
}

  */

  
  
  std::vector< std::complex<double> > model_polarized_image_adaptive_splined_raster::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
  {
    if (_use_analytical_visibilities)
    {
      // Counter-rotate point
      double ur =  _cpa*d.u + _spa*d.v;
      double vr = -_spa*d.u + _cpa*d.v;
    
      std::complex<double> VI(0.0,0.0), VQ(0.0,0.0), VU(0.0,0.0), VV(0.0,0.0), phase_factor;

      if (_use_fast_exp_approx)
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	  {
	    phase_factor = utils::fast_img_exp7( -(ur*_alpha[i][j]+vr*_beta[i][j]) );
	    VI += _I[i][j] * phase_factor;
	    VQ += _Q[i][j] * phase_factor;
	    VU += _U[i][j] * phase_factor;
	    VV += _V[i][j] * phase_factor;
	  }
      }
      else
      {
	for (size_t i=0; i<_Nx; ++i)
	  for (size_t j=0; j<_Ny; ++j)
	  {
	    phase_factor = std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );
	    VI += _I[i][j] * phase_factor;
	    VQ += _Q[i][j] * phase_factor;
	    VU += _U[i][j] * phase_factor;
	    VV += _V[i][j] * phase_factor;
	  }
      }
      
      // Apply cubic spline
      double spline_factor = cubic_spline_kernel(ur,vr) * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]);
      VI *= spline_factor;
      VQ *= spline_factor;
      VU *= spline_factor;
      VV *= spline_factor;

      // Convert to RR, LL, RL, LR
      std::vector< std::complex<double> > crosshand_vector(4);
      crosshand_vector[0] = VI+VV; // RR
      crosshand_vector[1] = VI-VV; // LL 
      crosshand_vector[2] = VQ+std::complex<double>(0.0,1.0)*VU; // RL
      crosshand_vector[3] = VQ-std::complex<double>(0.0,1.0)*VU; // LR

      // Apply Dterms
      apply_Dterms(d,crosshand_vector);
      
      return ( crosshand_vector );
    }
    else // NOT IMPLEMENTED
    {
      std::cerr << "ERROR: model_polarized_image_adaptive_splined_raster::crosshand_visiblities :"
		<< "       Numerical visibilities have not been properly implemented yet.\n"
		<< '\n';
      std::exit(1);
    }
  }
  


  double model_polarized_image_adaptive_splined_raster::cubic_spline_kernel_1d_prime(double k) const
{
  const double ak = std::fabs(k);

  if (ak < 1e-2)
  {
    const double a = _a;
    return -(2.0*a - 1.0) * (2.0*k)/15.0
           + (16.0*a + 1.0) * (k*k*k)/140.0;
  }
  else
  {
    const double a = _a;
    const double b = 4.0*a + 3.0;

    const double s  = std::sin(k);
    const double c  = std::cos(k);
    const double c2 = c*c - s*s;
    const double s2 = 2.0*s*c;
    (void)s2;

    const double kk = k*k;
    const double k3 = kk*k;
    const double k4 = k3*k;
    const double k5 = k4*k;

    const double A  = -4.0 * s * (2.0*a*c + b);
    const double Ap = -4.0 * (2.0*a*c2 + b*c);

    const double B  = 12.0 * ( a*(1.0 - c2) + 2.0*(1.0 - c) );
    const double Bp = 24.0 * s * (2.0*a*c + 1.0);

    const double num = Ap*kk - 3.0*A*k + Bp*k - 4.0*B;
    return num / k5;
  }
}



  void model_polarized_image_adaptive_splined_raster::update_phase_cache_all_data(
    const data_crosshand_visibilities& data)
{
  ScopedTimer T(TimerID::UpdatePhaseCache, timer_ns_, timer_calls_);

  if (!_use_cached_exp)
    return;

  const size_t Nd   = data.size();
  const size_t Npix = _Nx * _Ny;

  phase_cache_.resize(Nd * Npix);
  spline_kernel_cache_.resize(Nd);

  spline_kernel_dfovx_cache_.resize(Nd);
  spline_kernel_dfovy_cache_.resize(Nd);
  spline_kernel_dpa_cache_.resize(Nd);

  const double dx = (_xmax - _xmin) / double(_Nx - 1);
  const double dy = (_ymax - _ymin) / double(_Ny - 1);
  const double dxdy = dx * dy;

  const double fovx = (_xmax - _xmin);
  const double fovy = (_ymax - _ymin);

  const double inv_nx1 = 1.0 / double(_Nx - 1);
  const double inv_ny1 = 1.0 / double(_Ny - 1);

  const double d_dxdy_dfovx = fovy * inv_nx1 * inv_ny1;
  const double d_dxdy_dfovy = fovx * inv_nx1 * inv_ny1;

  const double dtpdx_dfovx = 2.0 * M_PI * inv_nx1;
  const double dtpdy_dfovy = 2.0 * M_PI * inv_ny1;

  for (size_t d_idx = 0; d_idx < Nd; ++d_idx)
  {
    const datum_crosshand_visibilities& d = data.datum(d_idx);

    const double ur =  _cpa*d.u + _spa*d.v;
    const double vr = -_spa*d.u + _cpa*d.v;

    const double ku = ur * _tpdx;
    const double kv = vr * _tpdy;

    const double Ku  = cubic_spline_kernel_1d(ku);
    const double Kv  = cubic_spline_kernel_1d(kv);
    const double Kup = cubic_spline_kernel_1d_prime(ku);
    const double Kvp = cubic_spline_kernel_1d_prime(kv);

    const double kernel = Ku * Kv;
    const double K = dxdy * kernel;

    spline_kernel_cache_[d_idx] = K;

    spline_kernel_dfovx_cache_[d_idx] =
      d_dxdy_dfovx * kernel + dxdy * (Kup * (ur * dtpdx_dfovx)) * Kv;

    spline_kernel_dfovy_cache_[d_idx] =
      d_dxdy_dfovy * kernel + dxdy * Ku * (Kvp * (vr * dtpdy_dfovy));

    spline_kernel_dpa_cache_[d_idx] =
      dxdy * ( (Kup * (_tpdx * vr)) * Kv
             + Ku * (Kvp * (_tpdy * (-ur))) );


    /*
size_t k = 0;
const double twopi = 2.0 * M_PI;

for (size_t iy = 0; iy < _Ny; ++iy)
  for (size_t ix = 0; ix < _Nx; ++ix, ++k)
  {
    const double phi = ur * _alpha[ix][iy] + vr * _beta[ix][iy];

    phase_cache_[d_idx * Npix + k] =
      _use_fast_exp_approx
        ? utils::fast_img_exp7(-phi)
        : std::exp(-std::complex<double>(0.0,1.0) * twopi * phi);
  }
    */


    size_t k = 0;
    const double twopi = 2.0 * M_PI;

    for (size_t ix = 0; ix < _Nx; ++ix)
      for (size_t iy = 0; iy < _Ny; ++iy, ++k)
      {
        const double phi = ur * _alpha[ix][iy] + vr * _beta[ix][iy];

        phase_cache_[d_idx * Npix + k] =
          _use_fast_exp_approx
            ? utils::fast_img_exp7(-phi)
            : std::exp(-std::complex<double>(0.0,1.0) * twopi * phi);
      }

  }

  cached_Nd_ = Nd;
  phase_cache_valid_ = true;
}

  
  
  std::complex<double> model_polarized_image_adaptive_splined_raster::visibility(datum_visibility& d, double acc)
  {
    ScopedTimer T_total(TimerID::VisibilitySingle, timer_ns_, timer_calls_);

    if (_use_analytical_visibilities)
    {
      // Counter-rotate point
      double ur =  _cpa*d.u + _spa*d.v;
      double vr = -_spa*d.u + _cpa*d.v;
    
      std::complex<double> V(0.0,0.0);
      for (size_t i=0; i<_Nx; ++i)
	for (size_t j=0; j<_Ny; ++j)
	  V += _I[i][j] * std::exp( - std::complex<double>(0.0,1.0) * 2.0*M_PI * (ur*_alpha[i][j]+vr*_beta[i][j]) );
      return ( cubic_spline_kernel(ur,vr)*V * (_alpha[1][1]-_alpha[0][0]) * (_beta[1][1]-_beta[0][0]) );
    }
    else // NOT ROTATED
      return ( cubic_spline_kernel(d.u,d.v)*model_polarized_image::visibility(d, acc) );
  }

  double model_polarized_image_adaptive_splined_raster::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    if (_use_analytical_visibilities)
    {
      datum_visibility tmp(d.u,d.v,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
      return std::abs(visibility(tmp,acc));
    }
    else
      return ( cubic_spline_kernel(d.u,d.v)*model_polarized_image::visibility_amplitude(d, acc) );
  }

  double model_polarized_image_adaptive_splined_raster::closure_phase(datum_closure_phase& d, double acc)
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
      return ( model_polarized_image::closure_phase(d,acc) );
  }



  double model_polarized_image_adaptive_splined_raster::closure_amplitude(datum_closure_amplitude& d, double acc)
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
      return ( model_polarized_image::closure_amplitude(d,acc) );
  }


  double model_polarized_image_adaptive_splined_raster::cubic_spline_kernel_1d(double k) const
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

  double model_polarized_image_adaptive_splined_raster::cubic_spline_kernel(double u0, double v0) const
  {
    return cubic_spline_kernel_1d(u0*_tpdx)*cubic_spline_kernel_1d(v0*_tpdy);
  }

};
