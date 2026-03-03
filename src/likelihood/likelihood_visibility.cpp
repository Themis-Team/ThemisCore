/*! 
  \file likelihood_visibility.cpp
  \author Avery E Broderick
  \date  February, 2020
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

      sum += _uncertainty.log_normalization(_data.datum(i)); // RG: revisit race condition with noise modeling ... no mutable in uncertainty ...
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
	  
	  // IMPORTANT: restore model/uncertainty to basepoint x after FD loop
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
  static bool once=false;
  if(!once){
    std::cerr << "[GRAD] likelihood_visibility::gradient_hybrid entered\n";
    once=true;
  }

  const bool do_geom = (gradient_mode() == GradientMode::HYBRID_INTENSITY_GEOM);

  // Evaluate basepoint once so model + uncertainty are at x
  const double Lx = this->operator()(x);
  (void)Lx;

  const size_t Npar = x.size();
  std::vector<double> grad(Npar, 0.0);

  // --- Active parameter span for THIS likelihood ---
  const size_t Nm = _model.size();
  const size_t Nu = _uncertainty.size();
  const size_t Nactive = std::min(Nm + Nu, Npar);

  // --- Cache availability checks (keep them cheap and decisive) ---
  const size_t Nx = _model.Nx();
  const size_t Ny = _model.Ny();
  const size_t Npix = Nx * Ny;

  int local_ok = 1;
  if (!_model.use_cached_exp())    local_ok = 0;
  if (!_model.phase_cache_valid()) local_ok = 0;
  if (Npix == 0)                   local_ok = 0;

  const auto& phase = _model.phase_cache();
  const auto& sk    = _model.spline_kernel_cache();
  const auto& Iflat = _model.I_flat();

  if (phase.size() < _data.size() * Npix) local_ok = 0;
  if (sk.size()    < _data.size())        local_ok = 0;
  if (Iflat.size() < Npix)                local_ok = 0;

  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, _Lcomm);

  if (!global_ok) {
    // Fallback: FD gradient (expensive but correct)
    std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);

    // restore basepoint model/uncertainty
    std::vector<double> mx(Nm), ux(Nu);
    size_t ii = 0;
    for (size_t j=0; j<Nm && ii<x.size(); ++j) mx[j] = x[ii++];
    for (size_t j=0; j<Nu && ii<x.size(); ++j) ux[j] = x[ii++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    return g;
  }

  // --- Determine geom indices from the ACTUAL model layout ---
  size_t idx_fovx = size_t(-1), idx_fovy = size_t(-1), idx_pa = size_t(-1);

  if (Nm == Npix + 3) {
    // model_image_adaptive_splined_raster: pixels + (fovx,fovy,pa)
    idx_fovx = Npix + 0;
    idx_fovy = Npix + 1;
    idx_pa   = Npix + 2;
  }
  else if (Nm == Npix + 5) { // I don't think this is ever our situation, but just to be on the safe side
    // pixels + (shiftx,shifty,fovx,fovy,pa)
    idx_fovx = Npix + 2;
    idx_fovy = Npix + 3;
    idx_pa   = Npix + 4;
  }
  else {
    // Unknown layout: do NOT attempt analytic geom.
    // may still do HYBRID_INTENSITY by setting do_geom=false at dispatch-level.
    std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);

    std::vector<double> mx(Nm), ux(Nu);
    size_t ii = 0;
    for (size_t j=0; j<Nm && ii<x.size(); ++j) mx[j] = x[ii++];
    for (size_t j=0; j<Nu && ii<x.size(); ++j) ux[j] = x[ii++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    return g;
  }

  // Pull geom params from x (model portion)
  const double fovx = x[idx_fovx];
  const double fovy = x[idx_fovy];
  const double pa   = x[idx_pa];
  const double cpa  = std::cos(pa);
  const double spa  = std::sin(pa);

  // --- Analytic pixel grads + (optional) analytic geom grads ---
  std::vector<double> grad_I_local(Npix, 0.0);
  double grad_fovx_local = 0.0;
  double grad_fovy_local = 0.0;
  double grad_pa_local   = 0.0;

  const std::complex<double> minus_i(0.0, -1.0);
  const double two_pi = 2.0 * M_PI;

  for (size_t i = 0; i < _data.size(); ++i) {
    if ((i % size_t(_L_size)) != size_t(_L_rank))
      continue;

    datum_visibility& d = _data.datum(i);

    const double u  = d.u;
    const double v  = d.v;

    // MUST match cache builder rotation
    const double ur =  cpa * u + spa * v;
    const double vr = -spa * u + cpa * v;

    const std::complex<double> err = _uncertainty.error(d);
    const double er = err.real();
    const double ei = err.imag();
    if (er == 0.0 || ei == 0.0)
      continue;

    const size_t off = i * Npix;
    const double K   = sk[i];

    std::complex<double> S0(0.0, 0.0);
    std::complex<double> Sgx(0.0, 0.0);
    std::complex<double> Sgy(0.0, 0.0);

    for (size_t k = 0; k < Npix; ++k) {
      const size_t ix = k / Ny;       // flatten must match cache layout: k = ix*Ny + iy
      const size_t iy = k - ix * Ny;

      const double xfrac = (Nx > 1) ? (double(ix) / double(Nx - 1) - 0.5) : 0.0;
      const double yfrac = (Ny > 1) ? (double(iy) / double(Ny - 1) - 0.5) : 0.0;

      const std::complex<double> ph = phase[off + k];
      const double Ik = Iflat[k];

      S0  += Ik * ph;
      Sgx += Ik * ph * xfrac;
      Sgy += Ik * ph * yfrac;
    }

    const std::complex<double> Vm = K * S0;

    const double dr = d.V.real() - Vm.real();
    const double di = d.V.imag() - Vm.imag();

    const double coeff_r = dr / (er * er);
    const double coeff_i = di / (ei * ei);

    if (do_geom) {
      const std::complex<double> dS0_dfovx = (minus_i * (two_pi * ur)) * Sgx;
      const std::complex<double> dS0_dfovy = (minus_i * (two_pi * vr)) * Sgy;
      const std::complex<double> dS0_dpa   = (minus_i * two_pi) * ((vr * fovx) * Sgx - (ur * fovy) * Sgy);

      std::complex<double> dVm_dfovx = K * dS0_dfovx;
      std::complex<double> dVm_dfovy = K * dS0_dfovy;
      std::complex<double> dVm_dpa   = K * dS0_dpa;

      // area scaling term from dxdy ~ fovx*fovy/(Nx-1)/(Ny-1)
      if (fovx != 0.0) dVm_dfovx += Vm / fovx;
      if (fovy != 0.0) dVm_dfovy += Vm / fovy;

      grad_fovx_local += coeff_r * dVm_dfovx.real() + coeff_i * dVm_dfovx.imag();
      grad_fovy_local += coeff_r * dVm_dfovy.real() + coeff_i * dVm_dfovy.imag();
      grad_pa_local   += coeff_r * dVm_dpa.real()   + coeff_i * dVm_dpa.imag();
    }

    // pixel grads
    for (size_t k = 0; k < Npix; ++k) {
      const std::complex<double> z = K * phase[off + k];
      grad_I_local[k] += coeff_r * z.real() + coeff_i * z.imag();
    }
  }

  // Reduce pixel gradients
  MPI_Allreduce(MPI_IN_PLACE, grad_I_local.data(), (int)Npix, MPI_DOUBLE, MPI_SUM, _Lcomm);

  // Reduce geom grads
  if (do_geom) {
    double geom_local[3]  = {grad_fovx_local, grad_fovy_local, grad_pa_local};
    double geom_global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(geom_local, geom_global, 3, MPI_DOUBLE, MPI_SUM, _Lcomm);

    grad[idx_fovx] = geom_global[0];
    grad[idx_fovy] = geom_global[1];
    grad[idx_pa]   = geom_global[2];
  }

  // Convert dL/dI -> dL/d(log I)
  for (size_t k = 0; k < Npix && k < Npar; ++k)
    grad[k] = Iflat[k] * grad_I_local[k];

  // --- FD ONLY for remaining ACTIVE parameters this likelihood uses ---
  std::vector<double> y = x;
  for (size_t p = Npix; p < Nactive; ++p) {
    if (do_geom && (p == idx_fovx || p == idx_fovy || p == idx_pa))
      continue;

    const double h = step_size(std::fabs(Pr.upper_bound(p) - Pr.lower_bound(p)));

    y[p] = x[p] + h;
    const double Lp = std::isfinite(Pr(y)) ? this->operator()(y) : -std::numeric_limits<double>::infinity();

    y[p] = x[p] - h;
    const double Lm = std::isfinite(Pr(y)) ? this->operator()(y) :  std::numeric_limits<double>::infinity();

    y[p] = x[p];
    grad[p] = (Lp - Lm) / (2.0 * h);
  }

  // --- Restore basepoint model/uncertainty (important for cache correctness) ---
  {
    std::vector<double> mx(Nm), ux(Nu);
    size_t ii = 0;
    for (size_t j=0; j<Nm && ii<x.size(); ++j) mx[j] = x[ii++];
    for (size_t j=0; j<Nu && ii<x.size(); ++j) ux[j] = x[ii++];
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
