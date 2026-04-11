/*! 
  \file likelihood_visibility.cpp
  \author Avery E Broderick, Roman Gold
  \date  February, 2020, February, 2026
  \brief Implementation file for the Visibility Likelihood class
*/


#include "likelihood_visibility.h"
#include "model_image_adaptive_splined_raster.h"
#include "model_image_sum.h"
#include "model_image_asymmetric_gaussian.h"
#include <limits>
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty)
  {
    _use_cached_exp = _model.use_cached_exp();
    bind_model_data_();
  }

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model,
					       uncertainty_visibility& uncertainty)
    : _data(data), _model(model), _uncertainty(uncertainty)
  {
    _use_cached_exp = _model.use_cached_exp();
    bind_model_data_();
  }

  likelihood_visibility::likelihood_visibility(size_t d_idx,
					       data_visibility& data,
					       model_visibility& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _use_cached_exp(false)
  {
    _use_cached_exp = _model.use_cached_exp();
    bind_model_data_();
  }

  likelihood_visibility::~likelihood_visibility()
  {
  }

  void likelihood_visibility::bind_model_data_()
  {
    _datum_cache.clear();
    _datum_cache.reserve(_data.size());
    for (size_t i = 0; i < _data.size(); ++i)
      _datum_cache.push_back(_data.datum(i));
    
    if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(&_model))
      {
	r->set_data(_datum_cache);
      }
    else if (auto* s = dynamic_cast<model_image_sum*>(&_model))
      {
	s->set_data(_datum_cache);
      }
  }
  
  void likelihood_visibility::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
    _uncertainty.set_mpi_communicator(comm);
  }

  double likelihood_visibility::operator()(std::vector<double>& x)
  {
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    //#pragma omp parallel for schedule(static) reduction(+:sum) // careful with race condition affecting sum!
    for(i = 0; i < _data.size(); ++i)
    {
      datum_visibility& d = _data.datum(i);
      std::complex<double> V = d.V;
      std::complex<double> err = _uncertainty.error(d); // RG: revisit race condition with noise modeling ... no mutable in uncertainty ...
      double acc = 0.25 * std::abs(err);


      std::complex<double> Vm;
      if (_use_cached_exp) {
	Vm = _model.visibility(i, d, acc);
      } else {
	Vm = _model.visibility(d, acc);
      }
      // std::complex<double> Vm = _model.visibility(i, d, acc);  // technically should guard that with _use_exp_cache ...
      
      sum += - 0.5*( std::pow( (V.real()-Vm.real())/err.real(), 2)
		     +
		     std::pow( (V.imag()-Vm.imag())/err.imag(), 2) );

      sum += _uncertainty.log_normalization(_data.datum(i)); // RG: revisit potential race condition with noise modeling ... no mutable in uncertainty ...
    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_visibility::chi_squared(std::vector<double>& x)
  {
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    for(i = 0; i < _data.size(); ++i)
    {
      datum_visibility& d = _data.datum(i);
      std::complex<double> V = d.V;
      std::complex<double> err = _uncertainty.error(d);

      std::complex<double> Vm;
      if (_use_cached_exp) {
	Vm = _model.visibility(i, d, 0.25*std::abs(err));
      }
      else {
	Vm = _model.visibility(d, 0.25*std::abs(err));
      }

      sum += 0.5*( std::pow( (V.real()-Vm.real())/err.real(), 2)
		   +
		   std::pow( (V.imag()-Vm.imag())/err.imag(), 2) );

    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }

  void likelihood_visibility::restore_basepoint(const std::vector<double>& x)
  {
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t ii = 0;
    for (size_t j = 0; j < _model.size(); ++j)       mx[j] = x[ii++];
    for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
  }
  
  std::vector<double> likelihood_visibility::gradient(std::vector<double>& x, prior& Pr)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientTotal, timer_ns_, timer_calls_);

    static bool once=false;
    if(!once){
      std::cerr << "[GRAD] likelihood_visibility::gradient entered (mode=" << int(gradient_mode()) << ")\n";
      once=true;
    }
    return gradient_dispatch_(x, Pr);
  }
  
  std::vector<double> likelihood_visibility::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    // Keep behavior consistent no matter what calls gradient_uniproc()
    return gradient_dispatch_(x, Pr);
  }


  std::vector<double> likelihood_visibility::gradient_dispatch_(std::vector<double>& x, prior& Pr)
  {
    switch (gradient_mode())
      {
      case GradientMode::FD_ALL:
	{
	  Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);

	  // Pure FD through base helper (this calls operator()(y) repeatedly)
	  std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);	  
	  restore_basepoint(x); // restore model/uncertainty to basepoint x after FD loop
	  return g;
	}

      case GradientMode::HYBRID_INTENSITY:
	return gradient_hybrid(x, Pr /*intensity_only=*/);  // see next section
	
      case GradientMode::HYBRID_INTENSITY_GEOM:
      default:
	return gradient_hybrid(x, Pr /*intensity+geom*/);
      }
  }


  std::vector<double> likelihood_visibility::gradient_hybrid(std::vector<double>& x, prior& Pr)
  {
    const GradientMode mode = gradient_mode();
    const bool do_geom = (mode == GradientMode::HYBRID_INTENSITY_GEOM);
    
    const double Lx = this->operator()(x);
    (void)Lx;
    
    auto restore_basepoint = [&]() {
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;
      for (size_t j = 0; j < _model.size(); ++j)       mx[j] = x[ii++];
      for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
      _model.generate_model(mx);
      _uncertainty.generate_uncertainty(ux);
    };
    
    model_image_adaptive_splined_raster* direct_raster =
      dynamic_cast<model_image_adaptive_splined_raster*>(&_model);
    model_image_asymmetric_gaussian* direct_ag =
      dynamic_cast<model_image_asymmetric_gaussian*>(&_model);
    model_image_sum* sum_top =
      dynamic_cast<model_image_sum*>(&_model);
    
    enum class Kind { Raster, AsymGauss };
    
    struct CompInfo {
      Kind kind = Kind::Raster;
      
      model_image* img = nullptr;
      model_image_adaptive_splined_raster* r = nullptr;
      model_image_asymmetric_gaussian* ag = nullptr;
      
      size_t p0 = 0;
      size_t pend = 0;
      
      double xoff = 0.0;
      double yoff = 0.0;
      
      size_t Nx = 0;
      size_t Ny = 0;
      size_t Npix = 0;
      
      size_t idx_fovx = 0;
      size_t idx_fovy = 0;
      size_t idx_pa   = 0;
      size_t idx_xoff = 0;
      size_t idx_yoff = 0;
      
      bool analytic = false;
      bool analytic_requires_cache = false;
      bool use_analytic_now = false;
      
      std::vector<double> xfrac;
      std::vector<double> yfrac;
    };
    
    std::vector<CompInfo> comps;
    const size_t Npar = x.size();
    
    auto fill_pixel_fracs = [](CompInfo& c) {
      c.xfrac.resize(c.Npix);
      c.yfrac.resize(c.Npix);
      for (size_t ix = 0; ix < c.Nx; ++ix) {
	const double xf = (c.Nx > 1) ? (double(ix) / double(c.Nx - 1) - 0.5) : 0.0;
	for (size_t iy = 0; iy < c.Ny; ++iy) {
	  const double yf = (c.Ny > 1) ? (double(iy) / double(c.Ny - 1) - 0.5) : 0.0;
	  const size_t k = ix * c.Ny + iy;
	  c.xfrac[k] = xf;
	  c.yfrac[k] = yf;
	}
      }
    };
    
    if (direct_raster)
      {
	CompInfo c;
	c.kind = Kind::Raster;
	c.img = direct_raster;
	c.r = direct_raster;
	c.p0 = 0;
	c.pend = direct_raster->size();
	c.xoff = 0.0;
	c.yoff = 0.0;
	c.Nx = direct_raster->Nx();
	c.Ny = direct_raster->Ny();
	c.Npix = c.Nx * c.Ny;
	c.idx_fovx = c.p0 + c.Npix;
	c.idx_fovy = c.p0 + c.Npix + 1;
	c.idx_pa   = c.p0 + c.Npix + 2;
	c.analytic = true;
	c.analytic_requires_cache = true;
	c.use_analytic_now = true;
	fill_pixel_fracs(c);
	comps.push_back(c);
      }
    else if (direct_ag)
      {
	CompInfo c;
	c.kind = Kind::AsymGauss;
	c.img = direct_ag;
	c.ag = direct_ag;
	c.p0 = 0;
	c.pend = direct_ag->size();
	c.xoff = 0.0;
	c.yoff = 0.0;
	c.analytic = true;
	c.analytic_requires_cache = false;
	c.use_analytic_now = true;
	comps.push_back(c);
      }
    else if (sum_top)
      {
	size_t p = 0;
	const auto& imgs = sum_top->components();
	const auto& xs   = sum_top->x_offsets();
	const auto& ys   = sum_top->y_offsets();
	
	for (size_t j = 0; j < imgs.size(); ++j)
	  {
	    CompInfo c;
	    c.img  = imgs[j];
	    c.p0   = p;
	    c.pend = p + imgs[j]->size();
	    c.idx_xoff = c.pend;
	    c.idx_yoff = c.pend + 1;
	    c.xoff = xs[j];
	    c.yoff = ys[j];
	    
	    if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(imgs[j]))
	      {
		c.kind = Kind::Raster;
		c.r = r;
		c.Nx = r->Nx();
		c.Ny = r->Ny();
		c.Npix = c.Nx * c.Ny;
		c.idx_fovx = c.p0 + c.Npix;
		c.idx_fovy = c.p0 + c.Npix + 1;
		c.idx_pa   = c.p0 + c.Npix + 2;
		c.analytic = true;
		c.analytic_requires_cache = true;
		c.use_analytic_now = true;
		fill_pixel_fracs(c);
	      }
	    else if (auto* ag = dynamic_cast<model_image_asymmetric_gaussian*>(imgs[j]))
	      {
		c.kind = Kind::AsymGauss;
		c.ag = ag;
		c.analytic = true;
		c.analytic_requires_cache = false;
		c.use_analytic_now = true;
	      }
	    else
	      {
		c.analytic = false;
		c.analytic_requires_cache = false;
		c.use_analytic_now = false;
	      }
	    
	    comps.push_back(c);
	    p += imgs[j]->size();
	    p += 2;
	  }
      }
    else
      {
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	return g;
      }
    
    if (comps.empty())
      {
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	return g;
      }
    
    int local_ok = 1;
    for (auto& c : comps)
      {
	if (!c.use_analytic_now)
	  continue;
	
	if (!c.analytic_requires_cache)
	  continue;
	
	const model_image_adaptive_splined_raster* rc = c.r;
	
	if (!rc->use_cached_exp()) c.use_analytic_now = false;
	if (!rc->phase_cache_valid()) c.use_analytic_now = false;
	if (rc->cached_Nd() < _data.size()) c.use_analytic_now = false;
	
	if (rc->phase_cache().size() < _data.size() * c.Npix) c.use_analytic_now = false;
	if (rc->spline_kernel_cache().size() < _data.size()) c.use_analytic_now = false;
	if (rc->I_flat().size() < c.Npix) c.use_analytic_now = false;
	
	if (do_geom) {
	  if (rc->spline_kernel_dfovx_cache().size() < _data.size()) c.use_analytic_now = false;
	  if (rc->spline_kernel_dfovy_cache().size() < _data.size()) c.use_analytic_now = false;
	  if (rc->spline_kernel_dpa_cache().size()   < _data.size()) c.use_analytic_now = false;
	}
	
	if (!c.use_analytic_now)
	  local_ok = 0;
      }
    
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, _Lcomm);
    
    static bool dbg_cache = false;
    if (!dbg_cache && _L_rank == 0) {
      dbg_cache = true;
      std::cerr << "[I-GRAD] local_ok=" << local_ok << "\n";
      for (size_t ci = 0; ci < comps.size(); ++ci) {
	if (comps[ci].kind != Kind::Raster || comps[ci].r == nullptr)
	  continue;
	
	const model_image_adaptive_splined_raster* rc = comps[ci].r;
	std::cerr << "  comp " << ci
		  << " cached="      << rc->use_cached_exp()
		  << " phase_valid=" << rc->phase_cache_valid()
		  << " cached_Nd="   << rc->cached_Nd() << "/" << _data.size()
		  << " phase_sz="    << rc->phase_cache().size() << "/" << (_data.size() * comps[ci].Npix)
		  << " K_sz="        << rc->spline_kernel_cache().size() << "/" << _data.size()
		  << " I_sz="        << rc->I_flat().size() << "/" << comps[ci].Npix;
	
	if (do_geom) {
	  std::cerr << " dKx_sz=" << rc->spline_kernel_dfovx_cache().size() << "/" << _data.size()
		    << " dKy_sz=" << rc->spline_kernel_dfovy_cache().size() << "/" << _data.size()
		    << " dKp_sz=" << rc->spline_kernel_dpa_cache().size()   << "/" << _data.size();
	}
	std::cerr << "\n";
      }
      std::cerr << "[I-GRAD] global_ok=" << global_ok << "\n";
    }
    
    if (!global_ok)
      {
	Themis::utils::ScopedTimer Tfd(Themis::utils::TimerID::GradientFiniteDiff,
				       timer_ns_, timer_calls_);
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	return g;
      }
    
    std::vector<double> grad_local(Npar, 0.0);
    std::vector<char> analytic_mask(Npar, 0);
    
    for (const auto& c : comps)
      {
	if (!c.use_analytic_now)
	  continue;
	
	if (c.kind == Kind::Raster)
	  {
	    for (size_t k = 0; k < c.Npix; ++k)
	      analytic_mask[c.p0 + k] = 1;
	    
	    if (do_geom) {
	      analytic_mask[c.idx_fovx] = 1;
	      analytic_mask[c.idx_fovy] = 1;
	      analytic_mask[c.idx_pa]   = 1;
	    }
	  }
	else if (c.kind == Kind::AsymGauss)
	  {
	    analytic_mask[c.p0 + 0] = 1;
	    analytic_mask[c.p0 + 1] = 1;
	    analytic_mask[c.p0 + 2] = 1;
	    analytic_mask[c.p0 + 3] = 1;
	  }
	
	if (sum_top) {
	  analytic_mask[c.idx_xoff] = 1;
	  analytic_mask[c.idx_yoff] = 1;
	}
      }
    
    const std::complex<double> Iunit(0.0, 1.0);
    const std::complex<double> minus_i(0.0, -1.0);
    const double two_pi = 2.0 * M_PI;
    
    {
      Themis::utils::ScopedTimer Tana(Themis::utils::TimerID::GradientAnalytic,
				      timer_ns_, timer_calls_);
      
      for (size_t i = 0; i < _data.size(); ++i)
	{
	  if ((i % size_t(_L_size)) != size_t(_L_rank))
	    continue;
	  
	  datum_visibility& d = _data.datum(i);
	  const std::complex<double> err = _uncertainty.error(d);
	  const double er = err.real();
	  const double ei = err.imag();
	  if (er == 0.0 || ei == 0.0)
	    continue;
	  
	  const double u = d.u;
	  const double v = d.v;
	  const double acc = 0.25 * std::abs(err);
	  
	  std::complex<double> Vm_full;
	  if (_use_cached_exp) {
	    Vm_full = _model.visibility(i, d, acc);
	  } else {
	    Vm_full = _model.visibility(d, acc);
	  }
	  
	  const double dr = d.V.real() - Vm_full.real();
	  const double di = d.V.imag() - Vm_full.imag();
	  const double coeff_r = dr / (er * er);
	  const double coeff_i = di / (ei * ei);
	  
	  struct RasterScratch {
	    const CompInfo* c = nullptr;
	    std::complex<double> E{0.0, 0.0};
	    double Ki = 0.0;
	    std::complex<double> S0{0.0, 0.0};
	    std::complex<double> Sx{0.0, 0.0};
	    std::complex<double> Sy{0.0, 0.0};
	    std::complex<double> Vc{0.0, 0.0};
	    double ur = 0.0;
	    double vr = 0.0;
	  };
	  
	  std::vector<RasterScratch> raster_scratch;
	  raster_scratch.reserve(comps.size());
	  
	  for (const auto& c : comps)
	    {
	      if (!c.use_analytic_now)
		continue;
	      
	      if (c.kind == Kind::Raster)
		{
		  RasterScratch s;
		  s.c = &c;
		  
		  const auto& phase = c.r->phase_cache();
		  const auto& K     = c.r->spline_kernel_cache();
		  const auto& Iflat = c.r->I_flat();
		  
		  const double pa   = x[c.idx_pa];
		  const double cpa  = std::cos(pa);
		  const double spa  = std::sin(pa);
		  const double fovx = x[c.idx_fovx];
		  const double fovy = x[c.idx_fovy];
		  (void)fovx;
		  (void)fovy;
		  
		  s.ur =  cpa * u + spa * v;
		  s.vr = -spa * u + cpa * v;
		  
		  const size_t off = i * c.Npix;
		  s.Ki = K[i];
		  
		  for (size_t k = 0; k < c.Npix; ++k)
		    {
		      const std::complex<double> ph = phase[off + k];
		      const double Ik = Iflat[k];
		      const std::complex<double> Ikph = Ik * ph;
		      s.S0 += Ikph;
		      if (do_geom) {
			s.Sx += Ikph * c.xfrac[k];
			s.Sy += Ikph * c.yfrac[k];
		      }
		    }
		  
		  if (sum_top) {
		    const double shift_phase = -2.0 * M_PI * (u * c.xoff + v * c.yoff);
		    s.E = std::exp(Iunit * shift_phase);
		  } else {
		    s.E = std::complex<double>(1.0, 0.0);
		  }
		  
		  s.Vc = s.E * (s.Ki * s.S0);
		  raster_scratch.push_back(s);
		}
	      else if (c.kind == Kind::AsymGauss)
		{
		  std::complex<double> Vag, dVag[4];
		  c.ag->visibility_and_derivatives(d, Vag, dVag);
		  
		  const std::complex<double> E =
		    (sum_top ? std::exp(Iunit * (-2.0 * M_PI * (u * c.xoff + v * c.yoff)))
                     : std::complex<double>(1.0, 0.0));
		  
		  const std::complex<double> Vc = E * Vag;
		  
		  for (int q = 0; q < 4; ++q)
		    {
		      const std::complex<double> dz = E * dVag[q];
		      grad_local[c.p0 + size_t(q)] += coeff_r * dz.real() + coeff_i * dz.imag();
		    }
		  
		  if (sum_top)
		    {
		      const std::complex<double> dV_dx = (minus_i * two_pi * u) * Vc;
		      const std::complex<double> dV_dy = (minus_i * two_pi * v) * Vc;
		      
		      grad_local[c.idx_xoff] += coeff_r * dV_dx.real() + coeff_i * dV_dx.imag();
		      grad_local[c.idx_yoff] += coeff_r * dV_dy.real() + coeff_i * dV_dy.imag();
		    }
		}
	    }
	  
	  for (const auto& s : raster_scratch)
	    {
	      const CompInfo& c = *s.c;
	      const auto& phase = c.r->phase_cache();
	      const auto& Iflat = c.r->I_flat();
	      const size_t off = i * c.Npix;
	      
	      for (size_t k = 0; k < c.Npix; ++k)
		{
		  const std::complex<double> z = s.E * (s.Ki * phase[off + k]);
		  grad_local[c.p0 + k] += Iflat[k] * (coeff_r * z.real() + coeff_i * z.imag());
		}
	      
	      if (do_geom)
		{
		  const auto& dK_fovx = c.r->spline_kernel_dfovx_cache();
		  const auto& dK_fovy = c.r->spline_kernel_dfovy_cache();
		  const auto& dK_pa   = c.r->spline_kernel_dpa_cache();
		  
		  const double fovx = x[c.idx_fovx];
		  const double fovy = x[c.idx_fovy];
		  
		  const std::complex<double> dS0_dfovx = (minus_i * (two_pi * s.ur)) * s.Sx;
		  const std::complex<double> dS0_dfovy = (minus_i * (two_pi * s.vr)) * s.Sy;
		  const std::complex<double> dS0_dpa   = (minus_i * two_pi) * ((s.vr * fovx) * s.Sx - (s.ur * fovy) * s.Sy);
		  
		  const std::complex<double> dV_dfovx = s.E * (dK_fovx[i] * s.S0 + s.Ki * dS0_dfovx);
		  const std::complex<double> dV_dfovy = s.E * (dK_fovy[i] * s.S0 + s.Ki * dS0_dfovy);
		  const std::complex<double> dV_dpa   = s.E * (dK_pa[i]   * s.S0 + s.Ki * dS0_dpa);
		  
		  grad_local[c.idx_fovx] += coeff_r * dV_dfovx.real() + coeff_i * dV_dfovx.imag();
		  grad_local[c.idx_fovy] += coeff_r * dV_dfovy.real() + coeff_i * dV_dfovy.imag();
		  grad_local[c.idx_pa]   += coeff_r * dV_dpa.real()   + coeff_i * dV_dpa.imag();
		}
	      
	      if (sum_top)
		{
		  const std::complex<double> dV_dx = (minus_i * two_pi * u) * s.Vc;
		  const std::complex<double> dV_dy = (minus_i * two_pi * v) * s.Vc;
		  
		  grad_local[c.idx_xoff] += coeff_r * dV_dx.real() + coeff_i * dV_dx.imag();
		  grad_local[c.idx_yoff] += coeff_r * dV_dy.real() + coeff_i * dV_dy.imag();
		}
	    }
	}
    }
    
    MPI_Allreduce(MPI_IN_PLACE, grad_local.data(), int(Npar), MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    {
      Themis::utils::ScopedTimer Tfd(Themis::utils::TimerID::GradientFiniteDiff,
				     timer_ns_, timer_calls_);
      
      std::vector<double> y = x;
      for (size_t p = 0; p < Npar; ++p)
	{
	  if (analytic_mask[p])
	    continue;
	  
	  double h = step_size(std::fabs(Pr.upper_bound(p) - Pr.lower_bound(p)));
	  if (!(h > 0.0))
	    h = 1e-6 * std::max(1.0, std::fabs(x[p]));
	  
	  y[p] = x[p] + h;
	  const double Lp = std::isfinite(Pr(y))
	    ? this->operator()(y)
	    : -std::numeric_limits<double>::infinity();
	  
	  y[p] = x[p] - h;
	  const double Lm = std::isfinite(Pr(y))
	    ? this->operator()(y)
	    : std::numeric_limits<double>::infinity();
	  
	  y[p] = x[p];
	  grad_local[p] = (Lp - Lm) / (2.0 * h);
	}
    }
    
    restore_basepoint();
    return grad_local;
  }
  

  void likelihood_visibility::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_visibility output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "V.r (Jy)"
	  << std::setw(15) << "err.r (Jy)"
	  << std::setw(15) << "model V.r (Jy)"
	  << std::setw(15) << "residual.r (Jy)"
	  << std::setw(15) << "V.i (Jy)"
	  << std::setw(15) << "err.i (Jy)"
	  << std::setw(15) << "model V.i (Jy)"
	  << std::setw(15) << "residual.i (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      datum_visibility& d = _data.datum(i);
      std::complex<double> V;
      if (_use_cached_exp) {
	V = _model.visibility(i, d, 0.25*std::abs(_data.datum(i).err));
      } else {
	V = _model.visibility(d, 0.25*std::abs(_data.datum(i).err));
      }
      // std::complex<double> V = _model.visibility(_data.datum(i),0.25*std::abs(_data.datum(i).err));
      std::complex<double> err = _uncertainty.error(_data.datum(i));
      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).V.real()
	    << std::setw(15) << err.real()
	    << std::setw(15) << V.real()
	    << std::setw(15) << (_data.datum(i).V-V).real()
	    << std::setw(15) << _data.datum(i).V.imag()
	    << std::setw(15) << err.imag()
	    << std::setw(15) << V.imag()
	    << std::setw(15) << (_data.datum(i).V-V).imag()
	    << '\n';
    }
  }



     // Profiling                                                                                                  
  void likelihood_visibility::print_timing_summary(int mpi_rank) const
  {
    // static const char* names[] = {
    static constexpr std::array<const char*, (size_t)Themis::utils::TimerID::COUNT> names = {{
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
      "ClosureAmplitude",
      "GradientTotal",
      "GradientEnsureGains",
      "GradientAnalytic",
      "GradientFiniteDiff",
      "GainsDistributeTotal",
      "GainsMPIAllreduce",
      "GainsSolveTotal",
      "GainsSolveTrial",
      "GainsSolveLogTrial",
      "matrix_determinant",
      "gaussj",
      "mrqcof",
      "LikelihoodEpochTotal",
      "LikelihoodMultiprocTotal",
      "LikelihoodModelVisBuild",
      "LikelihoodVectorPack",
      "LikelihoodDirectTerm",
      "LikelihoodScalarAllreduce",
      "GradientBatchedBarrier",
      "GradientBatchedAllreduce"
      }};
    static_assert(names.size() == (size_t)Themis::utils::TimerID::COUNT);

    if (mpi_rank < 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    }

    std::cout << "\n===== Profiling summary (rank "
              << mpi_rank << ") =====\n";

    for (size_t i = 0; i < (size_t)utils::TimerID::COUNT; ++i) {
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


  
  
};
