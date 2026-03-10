/*! 
  \file likelihood_visibility.cpp
  \author Avery E Broderick, Roman Gold
  \date  February, 2020, February, 2026
  \brief Implementation file for the Visibility Likelihood class
*/


#include "likelihood_visibility.h"
#include <iostream>
#include <iomanip>

namespace Themis{

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty)
  {
  }

  likelihood_visibility::likelihood_visibility(data_visibility& data,
					       model_visibility& model,
					       uncertainty_visibility& uncertainty)
    : _data(data), _model(model), _uncertainty(uncertainty)
  {
  }

  likelihood_visibility::likelihood_visibility(size_t d_idx,
					       data_visibility& data,
					       model_visibility& model)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _use_cached_exp(false)
  {
  }

  likelihood_visibility::~likelihood_visibility()
  {
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

      std::complex<double> Vm = _model.visibility(i, d, acc);  // technically should guard that with _use_exp_cache ...
      
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


  
  std::vector<double> likelihood_visibility::gradient(std::vector<double>& x, prior& Pr)
  {
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
    static bool once=false;
    if (_L_rank==0 && !once) std::cerr << "gradient_mode=" << int(gradient_mode()) << "\n";
    switch (gradient_mode())
      {
      case GradientMode::FD_ALL:
	{
	  // Pure FD through base helper (this calls operator()(y) repeatedly)
	  std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	  
	  // restore model/uncertainty to basepoint x after FD loop
	  {
	    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
	    size_t ii = 0;
	    for (size_t j = 0; j < _model.size(); ++j) mx[j] = x[ii++];
	    for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
	    _model.generate_model(mx);
	    _uncertainty.generate_uncertainty(ux);
	  }
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
    
    // Basepoint once (puts model+uncertainty at x)
    const double Lx = this->operator()(x);
    (void)Lx;
    
    // ---- cache availability checks ----
    int local_ok = 1;
    
    const size_t Nx   = _model.Nx();
    const size_t Ny   = _model.Ny();
    const size_t Npix = Nx * Ny;
    
    if (!_model.use_cached_exp())    local_ok = 0;
    if (!_model.phase_cache_valid()) local_ok = 0;
    if (Npix == 0)                   local_ok = 0;
    
    const auto& phase = _model.phase_cache();
    const auto& K     = _model.spline_kernel_cache();
    const auto& Iflat = _model.I_flat();
    
    if (phase.size() < _data.size() * Npix) local_ok = 0;
    if (K.size()     < _data.size())        local_ok = 0;
    if (Iflat.size() < Npix)                local_ok = 0;
    
    // If mode2: require dK caches
    const auto& dK_fovx = _model.spline_kernel_dfovx_cache();
    const auto& dK_fovy = _model.spline_kernel_dfovy_cache();
    const auto& dK_pa   = _model.spline_kernel_dpa_cache();
    
    if (do_geom) {
      if (dK_fovx.size() < _data.size()) local_ok = 0;
      if (dK_fovy.size() < _data.size()) local_ok = 0;
      if (dK_pa.size()   < _data.size()) local_ok = 0;
    }
    
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, _Lcomm);
    
    if (!global_ok) {
      // FD fallback (and restore basepoint)
      std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
      
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;
      for (size_t j = 0; j < _model.size(); ++j)       mx[j] = x[ii++];
      for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
      _model.generate_model(mx);
      _uncertainty.generate_uncertainty(ux);
      
      return g;
    }
    
    // ---- index layout ----
    // fov/pa are the last 3 model params
    const size_t Nm = _model.size();
    const size_t idx_fovx = Nm - 3;
    const size_t idx_fovy = Nm - 2;
    const size_t idx_pa   = Nm - 1;
    
    const double fovx = x[idx_fovx];
    const double fovy = x[idx_fovy];
    const double pa   = x[idx_pa];
    
    const double cpa = std::cos(pa);
    const double spa = std::sin(pa);
    
    // ---- analytic accumulators ----
    std::vector<double> grad_I_local(Npix, 0.0);
    double grad_fovx_local = 0.0;
    double grad_fovy_local = 0.0;
    double grad_pa_local   = 0.0;
    
    // Precompute xfrac/yfrac per pixel index
    static std::vector<double> xfrac, yfrac;
    static size_t lastNx = 0, lastNy = 0;
    if (lastNx != Nx || lastNy != Ny || xfrac.size() != Npix) {
      lastNx = Nx;
      lastNy = Ny;
      xfrac.resize(Npix);
      yfrac.resize(Npix);
      for (size_t ix = 0; ix < Nx; ++ix) {
	const double xf = (Nx > 1) ? (double(ix) / double(Nx - 1) - 0.5) : 0.0;
	for (size_t iy = 0; iy < Ny; ++iy) {
	  const double yf = (Ny > 1) ? (double(iy) / double(Ny - 1) - 0.5) : 0.0;
	  const size_t k = ix * Ny + iy;
	  xfrac[k] = xf;
	  yfrac[k] = yf;
	}
      }
    }
    
    const std::complex<double> minus_i(0.0, -1.0);
    const double two_pi = 2.0 * M_PI;
    
    for (size_t i = 0; i < _data.size(); ++i)
      {
	if ((i % size_t(_L_size)) != size_t(_L_rank))
	  continue;
	
	datum_visibility& d = _data.datum(i);
	
	const double u = d.u;
	const double v = d.v;
	
	const double ur =  cpa * u + spa * v;
	const double vr = -spa * u + cpa * v;
	
	const std::complex<double> err = _uncertainty.error(d);
	const double er = err.real();
	const double ei = err.imag();
	if (er == 0.0 || ei == 0.0)
	  continue;
	
	const size_t off = i * Npix;
	const double Ki  = K[i];
	
	// S0  = Σ Ik * phase
	// Sx  = Σ Ik * phase * xfrac
	// Sy  = Σ Ik * phase * yfrac
	std::complex<double> S0(0.0, 0.0);
	std::complex<double> Sx(0.0, 0.0);
	std::complex<double> Sy(0.0, 0.0);
	
	for (size_t k = 0; k < Npix; ++k) {
	  const std::complex<double> ph = phase[off + k];
	  const double Ik = Iflat[k];
	  const std::complex<double> Ikph = Ik * ph;
	  S0 += Ikph;
	  if (do_geom) {
	    Sx += Ikph * xfrac[k];
	    Sy += Ikph * yfrac[k];
	  }
	}
	
	const std::complex<double> Vm = Ki * S0;
	
	const double dr = d.V.real() - Vm.real();
	const double di = d.V.imag() - Vm.imag();
	
	const double coeff_r = dr / (er * er);
	const double coeff_i = di / (ei * ei);
	
	// Pixel grads: dVm/dIk = Ki * phase
	for (size_t k = 0; k < Npix; ++k) {
	  const std::complex<double> z = Ki * phase[off + k];
	  grad_I_local[k] += coeff_r * z.real() + coeff_i * z.imag();
	}
	
	if (do_geom) {
	  const std::complex<double> dS0_dfovx = (minus_i * (two_pi * ur)) * Sx;
	  const std::complex<double> dS0_dfovy = (minus_i * (two_pi * vr)) * Sy;
	  const std::complex<double> dS0_dpa   =
	    (minus_i * two_pi) * ((vr * fovx) * Sx - (ur * fovy) * Sy);
	  
	  const std::complex<double> dVm_dfovx = dK_fovx[i] * S0 + Ki * dS0_dfovx;
	  const std::complex<double> dVm_dfovy = dK_fovy[i] * S0 + Ki * dS0_dfovy;
	  const std::complex<double> dVm_dpa   = dK_pa[i]   * S0 + Ki * dS0_dpa;
	  
	  grad_fovx_local += coeff_r * dVm_dfovx.real() + coeff_i * dVm_dfovx.imag();
	  grad_fovy_local += coeff_r * dVm_dfovy.real() + coeff_i * dVm_dfovy.imag();
	  grad_pa_local   += coeff_r * dVm_dpa.real()   + coeff_i * dVm_dpa.imag();
	}
      }
    
    // Reduce across ranks
    MPI_Allreduce(MPI_IN_PLACE, grad_I_local.data(), (int)Npix, MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    double geom_local[3]  = {grad_fovx_local, grad_fovy_local, grad_pa_local};
    double geom_global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(geom_local, geom_global, 3, MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    // Assemble full gradient
    const size_t Npar = x.size();
    std::vector<double> grad(Npar, 0.0);
    
    // dL/d(log I) = I * dL/dI
    for (size_t k = 0; k < Npix && k < Npar; ++k)
      grad[k] = Iflat[k] * grad_I_local[k];
        
    // Temporary surgical FD helper
    auto fd_param = [&](size_t p) -> double {
      double h = step_size(std::fabs(Pr.upper_bound(p) - Pr.lower_bound(p)));
      if (!(h > 0.0))
	h = 1e-6 * std::max(1.0, std::fabs(x[p]));
      
      std::vector<double> y = x;
      
      y[p] = x[p] + h;
      const double Lp = std::isfinite(Pr(y)) ? this->operator()(y) : -std::numeric_limits<double>::infinity();
      
      y[p] = x[p] - h;
      const double Lm = std::isfinite(Pr(y)) ? this->operator()(y) :  std::numeric_limits<double>::infinity();
      
      y[p] = x[p];
      return (Lp - Lm) / (2.0 * h);
    };

    if (do_geom) {
      // grad[idx_fovx] = geom_global[0];
      // grad[idx_fovy] = geom_global[1];
      // grad[idx_pa]   = geom_global[2];
      grad[idx_fovx] = geom_global[0];
      grad[idx_fovy] = geom_global[1];
      grad[idx_pa]   = fd_param(idx_pa); // instead geom_global[2] until pa gradients are fixed
    }
    
    // Temporary surgical fix for localized bad intensity component (something seems special about pixel (0,0) ...)
    if (Npix > 0 && Npar > 0)
      grad[0] = fd_param(0);
    
    // FD everything else
    std::vector<double> y = x;
    for (size_t p = Npix; p < Npar; ++p)
      {
	if (do_geom && (p == idx_fovx || p == idx_fovy || p == idx_pa))
	  continue;
	
	double h = step_size(std::fabs(Pr.upper_bound(p) - Pr.lower_bound(p)));
	if (!(h > 0.0))
	  h = 1e-6 * std::max(1.0, std::fabs(x[p]));
	
	y[p] = x[p] + h;
	const double Lp = std::isfinite(Pr(y)) ? this->operator()(y) : -std::numeric_limits<double>::infinity();
	
	y[p] = x[p] - h;
	const double Lm = std::isfinite(Pr(y)) ? this->operator()(y) :  std::numeric_limits<double>::infinity();
	
	y[p] = x[p];
	grad[p] = (Lp - Lm) / (2.0 * h);
      }
    
    // Restore basepoint model/uncertainty (NB: FD loop perturbed x state)
    {
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;
      for (size_t j = 0; j < _model.size(); ++j)       mx[j] = x[ii++];
      for (size_t j = 0; j < _uncertainty.size(); ++j) ux[j] = x[ii++];
      _model.generate_model(mx);
      _uncertainty.generate_uncertainty(ux);
    }
    
    return grad;
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
  
};
