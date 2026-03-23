/*! 
  \file likelihood_crosshand_visibilities.cpp
  \author Avery E Broderick
  \date  March, 2020
  \brief Implementation file for the crosshand visibilities likelihood class
*/


#include "likelihood_crosshand_visibilities.h"
#include "model_polarized_image_adaptive_splined_raster.h"
#include "model_polarized_image_sum.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>
#include "utils.h"
#include <chrono>

namespace Themis{

  likelihood_crosshand_visibilities::likelihood_crosshand_visibilities(data_crosshand_visibilities& data,
								       model_crosshand_visibilities& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty)
  {
    _model.set_data(_data);
  }

  likelihood_crosshand_visibilities::likelihood_crosshand_visibilities(data_crosshand_visibilities& data,
								       model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty)
    : _data(data), _model(model), _uncertainty(uncertainty)
  {
    _model.set_data(_data);
  }

  void likelihood_crosshand_visibilities::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  double likelihood_crosshand_visibilities::operator()(std::vector<double>& x)
  {
    //_model.generate_model(x);
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double sum = 0.0;
    for(size_t i = 0; i < _data.size(); ++i)
    {
      std::complex<double> RR = _data.datum(i).RR;
      //std::complex<double> RRerr = _data.datum(i).RRerr;
      std::complex<double> LL = _data.datum(i).LL;
      //std::complex<double> LLerr = _data.datum(i).LLerr;
      std::complex<double> RL = _data.datum(i).RL;
      //std::complex<double> RLerr = _data.datum(i).RLerr;
      std::complex<double> LR = _data.datum(i).LR;
      //std::complex<double> LRerr = _data.datum(i).LRerr;

      std::vector<std::complex<double> > err = _uncertainty.error(_data.datum(i));

      // removes one heap allocation per likelihood datum evaluation
      std::complex<double> cvo[4];
      _model.fill_crosshand_visibilities(i,_data.datum(i),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])),cvo);
      // std::vector<std::complex<double> > cvo = _model.crosshand_visibilities(i,_data.datum(i),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));

      // RR
      sum += - 0.5*( std::pow( (RR.real()-cvo[0].real())/err[0].real(), 2)
		     +
		     std::pow( (RR.imag()-cvo[0].imag())/err[0].imag(), 2) );

      // LL
      sum += - 0.5*( std::pow( (LL.real()-cvo[1].real())/err[1].real(), 2)
		     +
		     std::pow( (LL.imag()-cvo[1].imag())/err[1].imag(), 2) );

      // RL
      sum += - 0.5*( std::pow( (RL.real()-cvo[2].real())/err[2].real(), 2)
		     +
		     std::pow( (RL.imag()-cvo[2].imag())/err[2].imag(), 2) );

      // LR
      sum += - 0.5*( std::pow( (LR.real()-cvo[3].real())/err[3].real(), 2)
		     +
		     std::pow( (LR.imag()-cvo[3].imag())/err[3].imag(), 2) );

    }
    // the factor 0.25 accounts for finite accuracy of the model prediction;
    // it currently gives an error of 3% in the reconstructed uncertainties
    
    return sum;
  }
  
  double likelihood_crosshand_visibilities::chi_squared(std::vector<double>& x)
  {
    return ( -2.0*operator()(x) );
  }

  std::vector<double> likelihood_crosshand_visibilities::gradient(std::vector<double>& x, prior& Pr)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientTotal, timer_ns_, timer_calls_);
    
    static bool once = false;
    if (!once) {
      std::cerr << "[GRAD] likelihood_crosshand_visibilities::gradient entered (mode="
		<< int(gradient_mode()) << ")\n";
      once = true;
    }
    return gradient_dispatch_(x, Pr);
  }

  std::vector<double> likelihood_crosshand_visibilities::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    return gradient_dispatch_(x, Pr);
  }

  std::vector<double> likelihood_crosshand_visibilities::gradient_dispatch_(std::vector<double>& x, prior& Pr)
  {
    switch (gradient_mode())
      {
      case GradientMode::FD_ALL:
	{
	  Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
	  
	  std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	  
	  std::vector<double> mx(_model.size()), ux(_uncertainty.size());
	  size_t ii = 0;
	  for (size_t j = 0; j < _model.size(); ++j) mx[j] = x[ii++];
	  for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
	  _model.generate_model(mx);
	  _uncertainty.generate_uncertainty(ux);
	  
	  return g;
	}
	
      case GradientMode::HYBRID_INTENSITY:
	return gradient_hybrid(x, Pr, /*do_geom=*/false);
	
      case GradientMode::HYBRID_INTENSITY_GEOM:
      default:
	return gradient_hybrid(x, Pr, /*do_geom=*/true);
      }
  }
  
  std::vector<double> likelihood_crosshand_visibilities::gradient_hybrid(std::vector<double>& x, prior& Pr, bool do_geom)
  {
    const double Lx = this->operator()(x);
    (void)Lx;
    
    auto restore_basepoint = [&]() {
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;
      for (size_t j = 0; j < _model.size(); ++j) mx[j] = x[ii++];
      for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
      _model.generate_model(mx);
      _uncertainty.generate_uncertainty(ux);
    };
    
    model_polarized_image_adaptive_splined_raster* direct_top = dynamic_cast<model_polarized_image_adaptive_splined_raster*>(&_model);
    
    model_polarized_image_sum* sum_top = dynamic_cast<model_polarized_image_sum*>(&_model);
  
    struct CompInfo {
      model_polarized_image_adaptive_splined_raster* r = nullptr;
      size_t p0 = 0;
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
      std::vector<double> xfrac;
      std::vector<double> yfrac;
    };
    
    std::vector<CompInfo> comps;
    const size_t Npar = x.size();
    const size_t Nm   = _model.size();
    
    if (direct_top)
      {
	CompInfo c;
	c.r = direct_top;
	c.p0 = 0;
	c.xoff = 0.0;
	c.yoff = 0.0;
	c.Nx = direct_top->Nx();
	c.Ny = direct_top->Ny();
	c.Npix = c.Nx * c.Ny;
	c.idx_fovx = c.p0 + 4*c.Npix;
	c.idx_fovy = c.p0 + 4*c.Npix + 1;
	c.idx_pa   = c.p0 + 4*c.Npix + 2;
	c.xfrac.resize(c.Npix);
	c.yfrac.resize(c.Npix);
	for (size_t ix = 0; ix < c.Nx; ++ix) {
	  const double xf = (c.Nx > 1) ? (double(ix)/double(c.Nx-1) - 0.5) : 0.0;
	  for (size_t iy = 0; iy < c.Ny; ++iy) {
	    const double yf = (c.Ny > 1) ? (double(iy)/double(c.Ny-1) - 0.5) : 0.0;
	    const size_t k = ix*c.Ny + iy;
	    c.xfrac[k] = xf;
	    c.yfrac[k] = yf;
	  }
	}
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
	    model_polarized_image_adaptive_splined_raster* r =
	      dynamic_cast<model_polarized_image_adaptive_splined_raster*>(imgs[j]);
	    
	    if (!r) {
	      std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	      restore_basepoint();
	      return g;
	    }
	    
	    CompInfo c;
	    c.r = r;
	    c.p0 = p;
	    c.xoff = xs[j];
	    c.yoff = ys[j];
	    c.Nx = r->Nx();
	    c.Ny = r->Ny();
	    c.Npix = c.Nx * c.Ny;
	    c.idx_fovx = c.p0 + 4*c.Npix;
	    c.idx_fovy = c.p0 + 4*c.Npix + 1;
	    c.idx_pa   = c.p0 + 4*c.Npix + 2;
	    c.idx_xoff = c.p0 + r->size();
	    c.idx_yoff = c.p0 + r->size() + 1;
	    c.xfrac.resize(c.Npix);
	    c.yfrac.resize(c.Npix);
	    for (size_t ix = 0; ix < c.Nx; ++ix) {
	      const double xf = (c.Nx > 1) ? (double(ix)/double(c.Nx-1) - 0.5) : 0.0;
	      for (size_t iy = 0; iy < c.Ny; ++iy) {
		const double yf = (c.Ny > 1) ? (double(iy)/double(c.Ny-1) - 0.5) : 0.0;
		const size_t k = ix*c.Ny + iy;
		c.xfrac[k] = xf;
		c.yfrac[k] = yf;
	      }
	    }
	    comps.push_back(c);
	    
	    p += r->size();
	    p += 2; // component offsets
	  }
      }
    else
      {
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	return g;
      }
    
    int local_ok = 1;
    
    for (const auto& c : comps)
      {
	const model_polarized_image_adaptive_splined_raster* rc = c.r;
	
	if (!rc->use_cached_exp()) local_ok = 0;
	if (!rc->phase_cache_valid()) local_ok = 0;
	if (rc->cached_Nd() < _data.size()) local_ok = 0;
	
	if (rc->phase_cache().size() < _data.size() * c.Npix) local_ok = 0;
	if (rc->spline_kernel_cache().size() < _data.size()) local_ok = 0;
	
	if (rc->I_flat().size() < c.Npix) local_ok = 0;
	if (rc->Q_flat().size() < c.Npix) local_ok = 0;
	if (rc->U_flat().size() < c.Npix) local_ok = 0;
	if (rc->V_flat().size() < c.Npix) local_ok = 0;
	
	if (do_geom) {
	  if (rc->spline_kernel_dfovx_cache().size() < _data.size()) local_ok = 0;
	  if (rc->spline_kernel_dfovy_cache().size() < _data.size()) local_ok = 0;
	  if (rc->spline_kernel_dpa_cache().size()   < _data.size()) local_ok = 0;
	}
      }
    
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, _Lcomm);
    
    if (!global_ok)
      {
	
	static bool dbg_fallback = false;
	if (!dbg_fallback && _L_rank == 0) {
	  dbg_fallback = true;
	  std::cerr << "[XH-GRAD] fallback: cache/model checks failed, using FD\n";
	}
	
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	return g;
      }
    
    auto apply_top_dterms = [&](datum_crosshand_visibilities& d, std::complex<double>* io)
    {
      if (sum_top) {
	sum_top->apply_Dterms_linear(d, io);
      } else {
	direct_top->apply_Dterms_linear(d, io);
      }
    };
    
    std::vector<double> grad_local(Npar, 0.0);
    std::vector<char> analytic_mask(Npar, 0);
  
    struct DtermWork {
      size_t datum_idx;
      std::array<std::complex<double>,4> pre;
      double rr[4];
      double ri[4];
      double err_re[4];
      double err_im[4];
    };
    
    std::vector<DtermWork> dterm_work;
    dterm_work.reserve((_data.size() + size_t(_L_size) - 1 - size_t(_L_rank)) / size_t(_L_size));
  
    for (const auto& c : comps)
      {
	for (size_t k = 0; k < 4*c.Npix; ++k)
	  analytic_mask[c.p0 + k] = 1;
	
	if (do_geom) {
	  analytic_mask[c.idx_fovx] = 1;
	  analytic_mask[c.idx_fovy] = 1;
	  analytic_mask[c.idx_pa]   = 1;
	}
	
	if (sum_top) {
	  analytic_mask[c.idx_xoff] = 1;
	  analytic_mask[c.idx_yoff] = 1;
	}	
      }
    
    const double two_pi = 2.0 * M_PI;
    const std::complex<double> Iunit(0.0, 1.0);
    const std::complex<double> minus_i(0.0, -1.0);

    {
      utils::ScopedTimer T(utils::TimerID::GradientAnalytic, timer_ns_, timer_calls_);
      
      for (size_t i = 0; i < _data.size(); ++i)
	{
	  if ((i % size_t(_L_size)) != size_t(_L_rank))
	    continue;
	  
	  datum_crosshand_visibilities& d = _data.datum(i);
	  std::vector<std::complex<double>> err = _uncertainty.error(d);
	  
	  bool bad_err = false;
	  for (int h = 0; h < 4; ++h)
	    if (err[h].real() == 0.0 || err[h].imag() == 0.0)
	      bad_err = true;
	  if (bad_err)
	    continue;
	  
	  const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
	  
	  std::complex<double> pred[4];
	  _model.fill_crosshand_visibilities(i, d, acc, pred);
	  
	  const std::complex<double> data_h[4] = { d.RR, d.LL, d.RL, d.LR };
	  double rr[4], ri[4];
	  for (int h = 0; h < 4; ++h) {
	    rr[h] = data_h[h].real()/err[h].real() - pred[h].real()/err[h].real();
	    ri[h] = data_h[h].imag()/err[h].imag() - pred[h].imag()/err[h].imag();
	  }

	  DtermWork dw;
	  dw.datum_idx = i;
	  for (int h = 0; h < 4; ++h)
	    {
	      dw.pre[h]    = std::complex<double>(0.0,0.0);
	      dw.rr[h]     = rr[h];
	      dw.ri[h]     = ri[h];
	      dw.err_re[h] = err[h].real();
	      dw.err_im[h] = err[h].imag();
	    }
	  
	  std::complex<double> Dmat[4][4];
	  {
	    const auto t0 = std::chrono::steady_clock::now();
	    
	    for (int col = 0; col < 4; ++col)
	      {
		std::complex<double> basis[4] = {
		  std::complex<double>(0.0,0.0),
		  std::complex<double>(0.0,0.0),
		  std::complex<double>(0.0,0.0),
		  std::complex<double>(0.0,0.0)
		};
		basis[col] = std::complex<double>(1.0,0.0);
		
		apply_top_dterms(d, basis);
		
		for (int row = 0; row < 4; ++row)
		  Dmat[row][col] = basis[row];
	      }
	    
	    const auto t1 = std::chrono::steady_clock::now();
	    dterm_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
	    ++dterm_calls_;
	  }
	  
	  auto apply_Dmat = [&](std::complex<double>* io)
	  {
	    std::complex<double> tmp[4];
	    for (int row = 0; row < 4; ++row)
	      {
		tmp[row] = Dmat[row][0]*io[0]
		  + Dmat[row][1]*io[1]
		  + Dmat[row][2]*io[2]
		  + Dmat[row][3]*io[3];
	      }
	    io[0] = tmp[0];
	    io[1] = tmp[1];
	    io[2] = tmp[2];
	    io[3] = tmp[3];
	  };
	  
	  const double u = d.u;
	  const double v = d.v;
	  
	  for (const auto& c : comps)
	    {
	      auto& r = *c.r;
	      const auto& phase = r.phase_cache();
	      const auto& Kc    = r.spline_kernel_cache();
	      const auto& Iflat = r.I_flat();
	      const auto& Qflat = r.Q_flat();
	      const auto& Uflat = r.U_flat();
	      const auto& Vflat = r.V_flat();
	      
	      const size_t off = i * c.Npix;
	      const double K   = Kc[i];
	      
	      const double pa  = x[c.idx_pa];
	      const double fovx = x[c.idx_fovx];
	      const double fovy = x[c.idx_fovy];
	      const double cpa = std::cos(pa);
	      const double spa = std::sin(pa);
	      const double ur =  cpa*u + spa*v;
	      const double vr = -spa*u + cpa*v;
	      
	      const std::complex<double> shift_phase = std::exp(-2.0*M_PI*Iunit * (c.xoff*(-u) + c.yoff*v));

	      std::complex<double> SI0(0.0,0.0), SQ0(0.0,0.0), SU0(0.0,0.0), SV0(0.0,0.0);
	      std::complex<double> SIx(0.0,0.0), SQx(0.0,0.0), SUx(0.0,0.0), SVx(0.0,0.0);
	      std::complex<double> SIy(0.0,0.0), SQy(0.0,0.0), SUy(0.0,0.0), SVy(0.0,0.0);
	      
	      for (size_t k = 0; k < c.Npix; ++k)
		{
		  const std::complex<double>& ph = phase[off + k];
		  const double xf = c.xfrac[k];
		  const double yf = c.yfrac[k];
		  
		  SI0 += Iflat[k] * ph;
		  SQ0 += Qflat[k] * ph;
		  SU0 += Uflat[k] * ph;
		  SV0 += Vflat[k] * ph;
		  
		  if (do_geom) {
		    SIx += Iflat[k] * ph * xf;
		    SQx += Qflat[k] * ph * xf;
		    SUx += Uflat[k] * ph * xf;
		    SVx += Vflat[k] * ph * xf;
		    
		    SIy += Iflat[k] * ph * yf;
		    SQy += Qflat[k] * ph * yf;
		    SUy += Uflat[k] * ph * yf;
		    SVy += Vflat[k] * ph * yf;
		  }
		}

	      const std::complex<double> VI0 = shift_phase * (K * SI0);
	      const std::complex<double> VQ0 = shift_phase * (K * SQ0);
	      const std::complex<double> VU0 = shift_phase * (K * SU0);
	      const std::complex<double> VV0 = shift_phase * (K * SV0);
	      
	      std::complex<double> basevec[4];
	      basevec[0] = VI0 + VV0;
	      basevec[1] = VI0 - VV0;
	      basevec[2] = VQ0 + Iunit*VU0;
	      basevec[3] = VQ0 - Iunit*VU0;
	      
	      dw.pre[0] += basevec[0];
	      dw.pre[1] += basevec[1];
	      dw.pre[2] += basevec[2];
	      dw.pre[3] += basevec[3];
       
	      if (do_geom)
		{
		  const auto& dKx = r.spline_kernel_dfovx_cache();
		  const auto& dKy = r.spline_kernel_dfovy_cache();
		  const auto& dKp = r.spline_kernel_dpa_cache();
		  
		  const std::complex<double> dSI_dfovx = (minus_i * (two_pi * ur)) * SIx;
		  const std::complex<double> dSQ_dfovx = (minus_i * (two_pi * ur)) * SQx;
		  const std::complex<double> dSU_dfovx = (minus_i * (two_pi * ur)) * SUx;
		  const std::complex<double> dSV_dfovx = (minus_i * (two_pi * ur)) * SVx;
		  
		  const std::complex<double> dSI_dfovy = (minus_i * (two_pi * vr)) * SIy;
		  const std::complex<double> dSQ_dfovy = (minus_i * (two_pi * vr)) * SQy;
		  const std::complex<double> dSU_dfovy = (minus_i * (two_pi * vr)) * SUy;
		  const std::complex<double> dSV_dfovy = (minus_i * (two_pi * vr)) * SVy;
		  
		  const std::complex<double> dSI_dpa = (minus_i * two_pi) * ( (vr * fovx) * SIx - (ur * fovy) * SIy );
		  const std::complex<double> dSQ_dpa = (minus_i * two_pi) * ( (vr * fovx) * SQx - (ur * fovy) * SQy );
		  const std::complex<double> dSU_dpa = (minus_i * two_pi) * ( (vr * fovx) * SUx - (ur * fovy) * SUy );
		  const std::complex<double> dSV_dpa = (minus_i * two_pi) * ( (vr * fovx) * SVx - (ur * fovy) * SVy );
		  
		  auto add_geom = [&](size_t pidx, double dK, const std::complex<double>& dSI, const std::complex<double>& dSQ, const std::complex<double>& dSU, const std::complex<double>& dSV)
		  {
		    std::complex<double> dvec[4];
		    
		    const std::complex<double> dVI = shift_phase * (dK * SI0 + K * dSI);
		    const std::complex<double> dVQ = shift_phase * (dK * SQ0 + K * dSQ);
		    const std::complex<double> dVU = shift_phase * (dK * SU0 + K * dSU);
		    const std::complex<double> dVV = shift_phase * (dK * SV0 + K * dSV);
		    
		    dvec[0] = dVI + dVV;
		    dvec[1] = dVI - dVV;
		    dvec[2] = dVQ + Iunit*dVU;
		    dvec[3] = dVQ - Iunit*dVU;
		    
		    // apply_top_dterms(d, dvec);
		    apply_Dmat(dvec);
		    
		    double contrib = 0.0;
		    for (int h = 0; h < 4; ++h) {
		      contrib += rr[h] * (dvec[h].real()/err[h].real())
			+  ri[h] * (dvec[h].imag()/err[h].imag());
		    }
		    grad_local[pidx] += contrib;
		  };
		  
		  add_geom(c.idx_fovx, dKx[i], dSI_dfovx, dSQ_dfovx, dSU_dfovx, dSV_dfovx);
		  add_geom(c.idx_fovy, dKy[i], dSI_dfovy, dSQ_dfovy, dSU_dfovy, dSV_dfovy);
		  add_geom(c.idx_pa,   dKp[i], dSI_dpa,   dSQ_dpa,   dSU_dpa,   dSV_dpa);
		}
	      
	      if (sum_top)
		{
		  std::complex<double> compvec[4];
		  compvec[0] = basevec[0];
		  compvec[1] = basevec[1];
		  compvec[2] = basevec[2];
		  compvec[3] = basevec[3];
		  
		  auto add_offset = [&](size_t pidx, const std::complex<double>& fac)
		  {
		    std::complex<double> dvec[4];
		    dvec[0] = fac * compvec[0];
		    dvec[1] = fac * compvec[1];
		    dvec[2] = fac * compvec[2];
		    dvec[3] = fac * compvec[3];
		    
		    apply_Dmat(dvec);
		    
		    double contrib = 0.0;
		    for (int h = 0; h < 4; ++h) {
		      contrib += rr[h] * (dvec[h].real()/err[h].real()) + ri[h] * (dvec[h].imag()/err[h].imag());
		    }
		    grad_local[pidx] += contrib;
		  };
		  
		  add_offset(c.idx_xoff, (+two_pi * Iunit * u));
		  add_offset(c.idx_yoff, (-two_pi * Iunit * v));
		}
	      
	      for (size_t k = 0; k < c.Npix; ++k)
		{
		  const std::complex<double> z = shift_phase * (K * phase[off + k]);
		  
		  const double Ik = Iflat[k];
		  const double Qk = Qflat[k];
		  const double Uk = Uflat[k];
		  const double Vk = Vflat[k];
		  
		  const size_t pI    = c.p0 + k;
		  const size_t pM    = c.p0 + c.Npix + k;
		  const size_t pEVPA = c.p0 + 2*c.Npix + k;
		  const size_t pMuV  = c.p0 + 3*c.Npix + k;
		  
		  auto add_pixel = [&](size_t pidx, double dI, double dQ, double dU, double dV)
		  {
		    std::complex<double> dvec[4];
		    dvec[0] = z * (dI + dV);
		    dvec[1] = z * (dI - dV);
		    dvec[2] = z * (dQ + Iunit*dU);
		    dvec[3] = z * (dQ - Iunit*dU);
		    
		    // apply_top_dterms(d, dvec);
		    apply_Dmat(dvec);
		    
		    double contrib = 0.0;
		    for (int h = 0; h < 4; ++h) {
		      contrib += rr[h] * (dvec[h].real()/err[h].real()) + ri[h] * (dvec[h].imag()/err[h].imag());
		    }
		    grad_local[pidx] += contrib;
		  };
		  
		  add_pixel(pI, Ik, Qk, Uk, Vk);
		  add_pixel(pM, 0.0, Qk, Uk, Vk);
		  add_pixel(pEVPA, 0.0, -2.0*Uk, 2.0*Qk, 0.0);
		  
		  const double mu  = x[pMuV];
		  const double Ii  = std::exp(x[pI]);
		  const double m   = std::exp(x[pM]);
		  const double omm = std::max(1.0e-300, 1.0 - mu*mu);
		  const double dQdmu = -(mu/omm) * Qk;
		  const double dUdmu = -(mu/omm) * Uk;
		  const double dVdmu = Ii * m;
		  
		  add_pixel(pMuV, 0.0, dQdmu, dUdmu, dVdmu);
		}
	    }
	  
	  dterm_work.push_back(dw);
	  
	}
    }

    std::vector<double> grad(Npar, 0.0);
    std::vector<double> y = x;
    
    {
      utils::ScopedTimer T(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
      
      // Remaining non-analytic model-side parameters are the D-term block.
      // Handle them analytically as dD/dp applied to the cached pre-D-term 4-vectors.
      std::vector<size_t> dterm_params;
      dterm_params.reserve(_model.size());
      
      for (size_t p = 0; p < _model.size(); ++p)
	if (!analytic_mask[p])
	  dterm_params.push_back(p);
      
      // These are handled below, so keep them out of the full-likelihood FD loop.
      for (size_t q = 0; q < dterm_params.size(); ++q)
	analytic_mask[dterm_params[q]] = 1;
      
      if (!dterm_params.empty())
	{
	  for (size_t w = 0; w < dterm_work.size(); ++w)
	    {
	      datum_crosshand_visibilities& dd = _data.datum(dterm_work[w].datum_idx);
	      
	      std::vector<std::array<std::complex<double>,16>> dMdp;
	      
	      if (sum_top) {
		sum_top->fill_Dterm_matrix_derivatives(dd, dterm_params, dMdp);
	      } else {
		direct_top->fill_Dterm_matrix_derivatives(dd, dterm_params, dMdp);
	      }
	      
	      for (size_t q = 0; q < dterm_params.size(); ++q)
		{
		  const size_t p = dterm_params[q];
		  const auto& J  = dMdp[q];
		  const auto& pre = dterm_work[w].pre;
		  
		  std::complex<double> dvec[4];
		  dvec[0] = J[ 0]*pre[0] + J[ 1]*pre[1] + J[ 2]*pre[2] + J[ 3]*pre[3];
		  dvec[1] = J[ 4]*pre[0] + J[ 5]*pre[1] + J[ 6]*pre[2] + J[ 7]*pre[3];
		  dvec[2] = J[ 8]*pre[0] + J[ 9]*pre[1] + J[10]*pre[2] + J[11]*pre[3];
		  dvec[3] = J[12]*pre[0] + J[13]*pre[1] + J[14]*pre[2] + J[15]*pre[3];
		  
		  double contrib = 0.0;
		  for (int hh = 0; hh < 4; ++hh)
		    {
		      contrib += dterm_work[w].rr[hh] * (dvec[hh].real() / dterm_work[w].err_re[hh])
			+ dterm_work[w].ri[hh] * (dvec[hh].imag() / dterm_work[w].err_im[hh]);
		    }
		  
		  grad_local[p] += contrib;
		}
	    }
	}
  
      MPI_Allreduce(grad_local.data(), grad.data(), int(Npar), MPI_DOUBLE, MPI_SUM, _Lcomm);
      
      for (size_t p = 0; p < Npar; ++p)
	{
	  if (analytic_mask[p])
	    continue;
	  
	  double h = step_size(std::fabs(Pr.upper_bound(p) - Pr.lower_bound(p)));
	  if (!(h > 0.0))
	    h = 1.0e-6 * std::max(1.0, std::fabs(x[p]));
	  
	  y[p] = x[p] + h;
	  const double Lp = std::isfinite(Pr(y))
	    ? this->operator()(y)
	    : -std::numeric_limits<double>::infinity();
	  
	  y[p] = x[p] - h;
	  const double Lm = std::isfinite(Pr(y))
	    ? this->operator()(y)
	    :  std::numeric_limits<double>::infinity();
	  
	  y[p] = x[p];
	  grad[p] = (Lp - Lm) / (2.0 * h);
	}
    }
    
    restore_basepoint();
    
    return grad;
  }

  void likelihood_crosshand_visibilities::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_crosshand_visibilities output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "phi1 (rad)"
	  << std::setw(15) << "phi2 (rad)"
	  << std::setw(15) << "RR.r (Jy)"
	  << std::setw(15) << "RRerr.r (Jy)"
	  << std::setw(15) << "mod RR.r (Jy)"
	  << std::setw(15) << "RRres.r (Jy)"
	  << std::setw(15) << "RR.i (Jy)"
	  << std::setw(15) << "RRerr.i (Jy)"
	  << std::setw(15) << "mod RR.i (Jy)"
	  << std::setw(15) << "RRres.i (Jy)"
	  << std::setw(15) << "LL.r (Jy)"
	  << std::setw(15) << "LLerr.r (Jy)"
	  << std::setw(15) << "mod LL.r (Jy)"
	  << std::setw(15) << "LLres.r (Jy)"
	  << std::setw(15) << "LL.i (Jy)"
	  << std::setw(15) << "LLerr.i (Jy)"
	  << std::setw(15) << "mod LL.i (Jy)"
	  << std::setw(15) << "LLres.i (Jy)"
	  << std::setw(15) << "RL.r (Jy)"
	  << std::setw(15) << "RLerr.r (Jy)"
	  << std::setw(15) << "mod RL.r (Jy)"
	  << std::setw(15) << "RLres.r (Jy)"
	  << std::setw(15) << "RL.i (Jy)"
	  << std::setw(15) << "RLerr.i (Jy)"
	  << std::setw(15) << "mod RL.i (Jy)"
	  << std::setw(15) << "RLres.i (Jy)"
	  << std::setw(15) << "LR.r (Jy)"
	  << std::setw(15) << "LRerr.r (Jy)"
	  << std::setw(15) << "mod LR.r (Jy)"
	  << std::setw(15) << "LRres.r (Jy)"
	  << std::setw(15) << "LR.i (Jy)"
	  << std::setw(15) << "LRerr.i (Jy)"
	  << std::setw(15) << "mod LR.i (Jy)"
	  << std::setw(15) << "LRres.i (Jy)"
	  << '\n';

      
    for (size_t i=0; i<_data.size(); ++i)
    {
      // std::complex<double> RRerr = _data.datum(i).RRerr;
      // std::complex<double> LLerr = _data.datum(i).LLerr;
      std::vector<std::complex<double> > err = _uncertainty.error(_data.datum(i));

      std::complex<double> cvo[4];
      const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
      _model.fill_crosshand_visibilities(i,_data.datum(i),acc,cvo);

      if (rank==0)
	out << std::setw(15) << _data.datum(i).u/1e9
	    << std::setw(15) << _data.datum(i).v/1e9
	    << std::setw(15) << _data.datum(i).phi1
	    << std::setw(15) << _data.datum(i).phi2
	  // RR
	    << std::setw(15) << _data.datum(i).RR.real()
	    << std::setw(15) << err[0].real()
	    << std::setw(15) << cvo[0].real()
	    << std::setw(15) << (_data.datum(i).RR-cvo[0]).real()
	    << std::setw(15) << _data.datum(i).RR.imag()
	    << std::setw(15) << err[0].imag()
	    << std::setw(15) << cvo[0].imag()
	    << std::setw(15) << (_data.datum(i).RR-cvo[0]).imag()
	  // LL
	    << std::setw(15) << _data.datum(i).LL.real()
	    << std::setw(15) << err[1].real()
	    << std::setw(15) << cvo[1].real()
	    << std::setw(15) << (_data.datum(i).LL-cvo[1]).real()
	    << std::setw(15) << _data.datum(i).LL.imag()
	    << std::setw(15) << err[1].imag()
	    << std::setw(15) << cvo[1].imag()
	    << std::setw(15) << (_data.datum(i).LL-cvo[1]).imag()
	  // RL
	    << std::setw(15) << _data.datum(i).RL.real()
	    << std::setw(15) << err[2].real()
	    << std::setw(15) << cvo[2].real()
	    << std::setw(15) << (_data.datum(i).RL-cvo[2]).real()
	    << std::setw(15) << _data.datum(i).RL.imag()
	    << std::setw(15) << err[2].imag()
	    << std::setw(15) << cvo[2].imag()
	    << std::setw(15) << (_data.datum(i).RL-cvo[2]).imag()
	  // LR
	    << std::setw(15) << _data.datum(i).LR.real()
	    << std::setw(15) << err[3].real()
	    << std::setw(15) << cvo[3].real()
	    << std::setw(15) << (_data.datum(i).LR-cvo[3]).real()
	    << std::setw(15) << _data.datum(i).LR.imag()
	    << std::setw(15) << err[3].imag()
	    << std::setw(15) << cvo[3].imag()
	    << std::setw(15) << (_data.datum(i).LR-cvo[3]).imag()
	    << '\n';
    }
  }

  void likelihood_crosshand_visibilities::print_timing_summary(int mpi_rank) const
  {
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
    
    std::cout << "\n===== Crosshand likelihood timing summary (rank "
	      << mpi_rank << ") =====\n";
    
    for (size_t i = 0; i < (size_t)Themis::utils::TimerID::COUNT; ++i) {
      const double ms = timer_ns_[i] / 1.0e6;
      const std::uint64_t n = timer_calls_[i];
      const double avg = (n > 0) ? ms / n : 0.0;
      
      if (n == 0) continue;
      
      std::cout << std::setw(24) << names[i]
		<< " : total = " << ms << " ms"
		<< ", calls = " << n
		<< ", avg = " << avg << " ms/call\n";
    }
    
    {
      const double ms = dterm_ns_ / 1.0e6;
      const double avg = (dterm_calls_ > 0) ? ms / double(dterm_calls_) : 0.0;
      std::cout << std::setw(24) << "GradientDTerms"
		<< " : total = " << ms << " ms"
		<< ", calls = " << dterm_calls_
		<< ", avg = " << avg << " ms/call\n";
    }
    
    std::cout << "====================================================\n\n";
  }

};
