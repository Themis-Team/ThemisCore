/*! 
  \file likelihood_optimal_complex_gain_crosshand_visibilities.cpp
  \author Roman Gold
  \date  March 2026
  \brief Implementation file for the likelihood_optimal_complex_gain_crosshand_visibilities likelihood class.
*/


#include "random_number_generator.h"

#include "likelihood_optimal_complex_gain_crosshand_visibilities.h"
#include "model_polarized_image_adaptive_splined_raster.h"
#include "model_polarized_image_sum.h"
#include "model_polarized_image_constant_polarization.h"
#include "model_image_asymmetric_gaussian.h"
#include <limits>
#include <algorithm>
#include <cmath>

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include "utils.h"
#include <chrono>
#include <array>

namespace Themis
{  
  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(
  data_crosshand_visibilities& data, model_crosshand_visibilities& model, std::vector<std::string> station_codes, std::vector<double> sigma_g)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Loop over data and generate a list of unique scan times,
    // which becomes the times of the gain correction epochs.
    _tge.resize(0);
    bool unique_scan_time;
    for (size_t j=0; j<_data.size(); ++j)
    {
      unique_scan_time=true;
      for (size_t k=0; k<_tge.size() && unique_scan_time; ++k)
	if (data.datum(j).tJ2000==_tge[k])
	  unique_scan_time=false;
      if (unique_scan_time)
	_tge.push_back(data.datum(j).tJ2000);
    }
        
    // Move times to boundaries of epochs located at the midpoints between them.
    std::vector<double> tgeold = _tge;
    //_tge[0] = tgeold[0]-0.5*(tgeold[1]-tgeold[0]);
    _tge[0] = tgeold[0]-1.0; // Put the first bin before the beginning
    for (size_t k=1; k<_tge.size(); ++k)
      _tge[k] = 0.5*(tgeold[k]+tgeold[k-1]);
    //_tge.push_back(tgeold[tgeold.size()-1]+0.5*(tgeold[tgeold.size()-1]-tgeold[tgeold.size()-2]));
    _tge.push_back(tgeold[tgeold.size()-1]+1.0); // Put one last bin beyond the end


    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);

  }

  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);
  }

  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(max_g), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);
  }

  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(
  data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Loop over data and generate a list of unique scan times,
    // which becomes the times of the gain correction epochs.
    _tge.resize(0);
    bool unique_scan_time;
    for (size_t j=0; j<_data.size(); ++j)
    {
      unique_scan_time=true;
      for (size_t k=0; k<_tge.size() && unique_scan_time; ++k)
	if (data.datum(j).tJ2000==_tge[k])
	  unique_scan_time=false;
      if (unique_scan_time)
	_tge.push_back(data.datum(j).tJ2000);
    }
        
    // Move times to boundaries of epochs located at the midpoints between them.
    std::vector<double> tgeold = _tge;
    //_tge[0] = tgeold[0]-0.5*(tgeold[1]-tgeold[0]);
    _tge[0] = tgeold[0]-1.0; // Put the first bin before the beginning
    for (size_t k=1; k<_tge.size(); ++k)
      _tge[k] = 0.5*(tgeold[k]+tgeold[k-1]);
    //_tge.push_back(tgeold[tgeold.size()-1]+0.5*(tgeold[tgeold.size()-1]-tgeold[tgeold.size()-2]));
    _tge.push_back(tgeold[tgeold.size()-1]+1.0); // Put one last bin beyond the end


    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);
  }

  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);
  }

  likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_optimal_complex_gain_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(max_g), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100), _x_last(0), _L_last(0)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    _model.set_data(_data);
  }
  

  likelihood_optimal_complex_gain_crosshand_visibilities::~likelihood_optimal_complex_gain_crosshand_visibilities()
  {
    const int ma = 4 * int(_sigma_g.size());
    
    for (int j=0; j<=ma; ++j)
      delete[] _mrq_oneda[j];
    delete[] _mrq_oneda;
    
    delete[] _mrq_da;
    delete[] _mrq_beta;
    delete[] _mrq_atry;
    
    for (int i=1; i<=ma; ++i)
      {
	delete[] _covar[i];
	delete[] _alpha[i];
      }
    delete[] _covar;
    delete[] _alpha;
    
    delete[] _g;
    delete[] _og;
    
    delete[] _sig;
    delete[] _ogc_hand;
    delete[] _ogc_is2;
    delete[] _ogc_is1;
    delete[] _ogc_yb;
    delete[] _ogc_y;
    
    delete[] _dyda;
    delete[] _vv;
    delete[] _ipiv;
    delete[] _indxr;
    delete[] _indxc;
    delete[] _indx;
  }
  
  
  void likelihood_optimal_complex_gain_crosshand_visibilities::check_station_codes()
  {
    // Loop over data and check that all of the data station codes
    // are in the station code list, and report any station codes
    // in the list that are not in the data.
    std::vector<bool> station_code_in_data(_station_codes.size(),false);
    bool station1_in_station_codes, station2_in_station_codes;
    for (size_t j=0; j<_data.size(); ++j)
    {
      station1_in_station_codes=false;
      station2_in_station_codes=false;
      for (size_t k=0; k<_station_codes.size(); ++k)
      {
	if ( _data.datum(j).Station1==_station_codes[k] )
	{
	  station1_in_station_codes=true;
	  station_code_in_data[k]=true;
	}
	if ( _data.datum(j).Station2==_station_codes[k] )
	{
	  station2_in_station_codes=true;
	  station_code_in_data[k]=true;
	}
      }
      if ( station1_in_station_codes==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_crosshand_visibilities:\n"
		  << "    Station " << _data.datum(j).Station1 << " not in station_codes list.\n"
		  << '\n';
      if ( station2_in_station_codes==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_crosshand_visibilities:\n"
		  << "    Station " << _data.datum(j).Station2 << " not in station_codes list.\n"
		  << '\n';
    }
    for (size_t k=0; k<_station_codes.size(); ++k)
      if ( station_code_in_data[k]==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_crosshand_visibilities:\n"
		  << "    station code " << _station_codes[k] << " not used in data set.\n"
		  << '\n'; 

    if (_use_prior_gain_solutions)
      std::cerr << "WARNING: likelihood_optimal_complex_gain_crosshand_visibilities:\n"
		<< "     prior gain information is being used to solve for gains.  This potentially\n"
		<< "     can lead to non-deterministic, path-dependent behavior in the presence of\n"
		<< "     pathologically poorly defined gains.\n"
		<< '\n';

  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::allocate_memory()
  {
    require_independent_mode_only_(__func__);
    
    _G.resize(_tge.size()-1);
    for (size_t j=0; j<_tge.size()-1; ++j)
      {
	_G[j].resize(_sigma_g.size());
	for (size_t k=0; k<_sigma_g.size(); ++k)
	  {
	    _G[j][k].GR = std::complex<double>(1.0,0.0);
	    _G[j][k].GL = std::complex<double>(1.0,0.0);
	  }
      }
    
    _sqrt_detC.resize(_tge.size()-1);
    for (size_t j=0; j<_sqrt_detC.size(); ++j)
      _sqrt_detC[j] = 1.0;
    
    const int ma = 4 * int(_sigma_g.size());
    
    _mrq_atry = new double[ma+1];
    _mrq_beta = new double[ma+1];
    _mrq_da   = new double[ma+1];
    
    _mrq_oneda = new double*[ma+1];
    for (int j=0; j<=ma; ++j)
      _mrq_oneda[j] = new double[2];
    
    _indx  = new int[ma+1];
    _indxc = new int[ma+1];
    _indxr = new int[ma+1];
    _ipiv  = new int[ma+1];
    _vv    = new double[ma+1];
    _dyda  = new double[ma+1];
    
    const int ndatamax = 8 * int(_data.size());
    _ogc_y    = new double[ndatamax+1];
    _ogc_yb   = new double[ndatamax+1];
    _ogc_is1  = new size_t[ndatamax+1];
    _ogc_is2  = new size_t[ndatamax+1];
    _ogc_hand = new unsigned char[ndatamax+1];
    _sig      = new double[ndatamax+1];
    
    _covar = new double*[ma+1];
    _alpha = new double*[ma+1];
    for (int i=1; i<=ma; ++i)
      {
	_covar[i] = new double[ma+1];
	_alpha[i] = new double[ma+1];
      }
    
    _g  = new double[ma+1];
    _og = new double[ma+1];
  }

    
  void likelihood_optimal_complex_gain_crosshand_visibilities::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(_Mcomm);
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::set_iteration_limit(int itermax)
  {
    _itermax=itermax;
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::solve_for_gains()
  {
    _solve_for_gains = true;
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::fix_gains()
  {
    _solve_for_gains = false;
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::solve_for_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = true;
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::fix_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = false;
  }
  
  void likelihood_optimal_complex_gain_crosshand_visibilities::use_prior_gain_solutions()
  {
    _use_prior_gain_solutions = true;
  }
  
  void likelihood_optimal_complex_gain_crosshand_visibilities::assume_smoothly_varying_gains()
  {
    _smoothly_varying_gains = true;
    _use_prior_gain_solutions = false;
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::assume_independently_varying_gains()
  {
    _smoothly_varying_gains = false;
    _use_prior_gain_solutions = false;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::require_independent_mode_only_(const char* where) const
  {
    (void)where;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::set_hand_gain_mode(HandGainMode mode)
  {
    _hand_gain_mode = mode;
  }
  
  void likelihood_optimal_complex_gain_crosshand_visibilities::set_ratio_priors(double sigma_ratio_logamp, double sigma_ratio_phase)
  {
    _sigma_ratio_logamp = sigma_ratio_logamp;
    _sigma_ratio_phase  = sigma_ratio_phase;
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::wrap_phase_(double x) const
  {
    return std::atan2(std::sin(x), std::cos(x));
  }
  
  double likelihood_optimal_complex_gain_crosshand_visibilities::sigma_ratio_logamp_eff_() const
  {
    return (_sigma_ratio_logamp > 0.0 ? _sigma_ratio_logamp : 0.10);
  }
  
  double likelihood_optimal_complex_gain_crosshand_visibilities::sigma_ratio_phase_eff_() const
  {
    return (_sigma_ratio_phase > 0.0 ? _sigma_ratio_phase : 10.0*M_PI/180.0);
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::ratio_common_and_diff_(const HandGains& G, double& ac, double& pc, double& eta, double& delta) const
  {
    const double aR = std::log(std::abs(G.GR));
    const double aL = std::log(std::abs(G.GL));
    const double pR = std::arg(G.GR);
    const double pL = std::arg(G.GL);
    
    ac    = 0.5 * (aR + aL);
    pc    = 0.5 * wrap_phase_(pR + pL);
    eta   = aR - aL;
    delta = wrap_phase_(pR - pL);
  }

  void likelihood_optimal_complex_gain_crosshand_visibilities::encode_station_params_(const HandGains& G, double p4[]) const
  {
    if (_hand_gain_mode == HandGainMode::Independent)
      {
	p4[0] = std::log(std::abs(G.GR));
	p4[1] = std::arg(G.GR);
	p4[2] = std::log(std::abs(G.GL));
	p4[3] = std::arg(G.GL);
      }
    else
      {
	double ac, pc, eta, delta;
	ratio_common_and_diff_(G, ac, pc, eta, delta);
	p4[0] = ac;
	p4[1] = pc;
	p4[2] = eta;
	p4[3] = delta;
      }
  }
  
  likelihood_optimal_complex_gain_crosshand_visibilities::HandGains likelihood_optimal_complex_gain_crosshand_visibilities::decode_station_params_(const double g[], size_t s) const
  {
    const int j = 4*int(s) + 1;
    const std::complex<double> I(0.0,1.0);
    
    HandGains out;
    
    if (_hand_gain_mode == HandGainMode::Independent)
      {
	const double gmagR = std::exp(g[j]);
	const double gmagL = std::exp(g[j+2]);
	
	out.GR = gmagR * std::exp(I*g[j+1]);
	out.GL = gmagL * std::exp(I*g[j+3]);
      }
    else
      {
	const double ac    = g[j];
	const double pc    = g[j+1];
	const double eta   = g[j+2];
	const double delta = g[j+3];
	
	const std::complex<double> Gc = std::exp(ac) * std::exp(I*pc);
	
	const std::complex<double> R = std::exp(0.5*eta) * std::exp(I*(0.5*delta));
	
	out.GR = Gc * R;
	out.GL = Gc / R;
      }
    
    return out;
  }
  

  void likelihood_optimal_complex_gain_crosshand_visibilities::output(std::ostream& out)
  {
    require_independent_mode_only_(__func__);
    
    int rank;
    MPI_Comm_rank(_comm, &rank);
    
    distribute_gains();
    
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
	  << std::endl;
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	for (size_t jj=0; jj<_datum_index_list[epoch].size(); ++jj)
	  {
	    const size_t i = _datum_index_list[epoch][jj];
	    
	    std::vector<std::complex<double> > err = _uncertainty.error(_data.datum(i));
	    const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
	    
	    std::complex<double> cvo[4];
	    _model.fill_crosshand_visibilities(i, _data.datum(i), acc, cvo);
	    
	    const HandGains& g1 = _G[epoch][_is1_list[epoch][jj]];
	    const HandGains& g2 = _G[epoch][_is2_list[epoch][jj]];
	    
	    cvo[0] *= baseline_gain_rr(g1, g2);
	    cvo[1] *= baseline_gain_ll(g1, g2);
	    cvo[2] *= baseline_gain_rl(g1, g2);
	    cvo[3] *= baseline_gain_lr(g1, g2);
	    
	    if (rank==0)
	      out << std::setw(15) << _data.datum(i).u/1e9
		  << std::setw(15) << _data.datum(i).v/1e9
		  << std::setw(15) << _data.datum(i).phi1
		  << std::setw(15) << _data.datum(i).phi2
		  << std::setw(15) << _data.datum(i).RR.real()
		  << std::setw(15) << err[0].real()
		  << std::setw(15) << cvo[0].real()
		  << std::setw(15) << (_data.datum(i).RR-cvo[0]).real()
		  << std::setw(15) << _data.datum(i).RR.imag()
		  << std::setw(15) << err[0].imag()
		  << std::setw(15) << cvo[0].imag()
		  << std::setw(15) << (_data.datum(i).RR-cvo[0]).imag()
		  << std::setw(15) << _data.datum(i).LL.real()
		  << std::setw(15) << err[1].real()
		  << std::setw(15) << cvo[1].real()
		  << std::setw(15) << (_data.datum(i).LL-cvo[1]).real()
		  << std::setw(15) << _data.datum(i).LL.imag()
		  << std::setw(15) << err[1].imag()
		  << std::setw(15) << cvo[1].imag()
		  << std::setw(15) << (_data.datum(i).LL-cvo[1]).imag()
		  << std::setw(15) << _data.datum(i).RL.real()
		  << std::setw(15) << err[2].real()
		  << std::setw(15) << cvo[2].real()
		  << std::setw(15) << (_data.datum(i).RL-cvo[2]).real()
		  << std::setw(15) << _data.datum(i).RL.imag()
		  << std::setw(15) << err[2].imag()
		  << std::setw(15) << cvo[2].imag()
		  << std::setw(15) << (_data.datum(i).RL-cvo[2]).imag()
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
  }


  size_t likelihood_optimal_complex_gain_crosshand_visibilities::number_of_independent_gains()
  {
    /*
      Computes the number of independent gains (amplitudes and phases) in the data set associated
      with the likelihood.  This is done for each via the following strategy:
      1. Construct a unique list of baselines, i.e., remove repeated baselines.
      2. Identify "rings", i.e., collections of baselines that are connected by at least one station.
         Note that this does not mean that all baselines will have a common station, only that every
	 baseline in the ring can be connected to any other via a sequence.  E.g., baselines 01, 12,
	 and 23 are all in the same ring because the first has stations 0,1, and the second has 1,2,
	 sharing the common station 1, and the third has stations 2,3, sharing a common station 2 with
	 the second, and thus being connected to the first.
      3. For each ring, compute the number of independent gains as the minimum of:
         a. The number of crosshand_visibilities components (i.e., real and imaginary), twice the number of baselines (counting only RR, I think)
	 b. The number of stations * 2 - 1 where the 1 corresponds to the ring-specific arbitrary 
	    phase that set.  
    */

    int number_of_gains = 0;

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Unique baseline lists
      std::vector<size_t> uis1(0), uis2(0);
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	bool unique_baseline=true;
	for (size_t iu=0; iu<uis1.size(); ++iu)
	{
	  if ( (_is1_list[epoch][i]==uis1[iu] && _is2_list[epoch][i]==uis2[iu])
	       ||
	       (_is1_list[epoch][i]==uis2[iu] && _is2_list[epoch][i]==uis1[iu]) )
	    unique_baseline=false;
	}
	if (unique_baseline)
	{
	  // Order them so that uis1 is the lesser station index.
	  if (_is1_list[epoch][i]<_is2_list[epoch][i])
	  {
	    uis1.push_back(_is1_list[epoch][i]);
	    uis2.push_back(_is2_list[epoch][i]);
	  }
	  else
	  {
	    uis1.push_back(_is2_list[epoch][i]);
	    uis2.push_back(_is1_list[epoch][i]);
	  }
	}
      }

      // Go through the list of unique baselines and generate collections of interconnected baselines
      int number_of_gains_in_epoch=0;
      for (int iu=0; iu<int(uis1.size()); ++iu)
      {
	std::vector<size_t> uis1_ring(0), uis2_ring(0);
	
	// Grabbing first remaining element
	uis1_ring.push_back(uis1[iu]);
	uis2_ring.push_back(uis2[iu]);
	uis1.erase(uis1.begin()+iu);
	uis2.erase(uis2.begin()+iu);
	--iu; 
	
	// Loop through the rest of the baselines and add those that are connected to one of the 
	for (size_t ku=0; ku<uis1_ring.size(); ++ku)
	{
	  for (int ju=iu+1; ju<int(uis1.size()); ++ju)
	    if ( (uis1[ju]==uis1_ring[ku] || uis2[ju]==uis1_ring[ku] || uis1[ju]==uis2_ring[ku] || uis2[ju]==uis2_ring[ku]) )
	    {
	      uis1_ring.push_back(uis1[ju]);
	      uis2_ring.push_back(uis2[ju]);
	      uis1.erase(uis1.begin()+ju);
	      uis2.erase(uis2.begin()+ju);
	      --ju;
	    }
	}

	// Now figure out the number of gains
	std::vector<size_t> unique_station_list(0);
	for (size_t ju=0; ju<uis1_ring.size(); ++ju)
	{
	  bool unique_station=true;
	  for (size_t ku=0; ku<unique_station_list.size(); ++ku)
	    if (uis1_ring[ju]==unique_station_list[ku])
	      unique_station=false;
	  if (unique_station)
	    unique_station_list.push_back(uis1_ring[ju]);

	  unique_station=true;
	  for (size_t ku=0; ku<unique_station_list.size(); ++ku)
	    if (uis2_ring[ju]==unique_station_list[ku])
	      unique_station=false;
	  if (unique_station)
	    unique_station_list.push_back(uis2_ring[ju]);
	}

	number_of_gains_in_epoch += std::min(2*uis1_ring.size(),2*unique_station_list.size()-1);
      }
      number_of_gains += number_of_gains_in_epoch;
    }

    return ( number_of_gains );
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::organize_data_lists()
  {
    _datum_index_list.resize(_tge.size()-1);
    _yrr_list.resize(_tge.size()-1);
    _yll_list.resize(_tge.size()-1);
    _yrl_list.resize(_tge.size()-1);
    _ylr_list.resize(_tge.size()-1);
    _is1_list.resize(_tge.size()-1);
    _is2_list.resize(_tge.size()-1);

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector<size_t> id;
      std::vector< std::complex<double> > yrr, yll, yrl, ylr;
      std::vector<std::string> s1,s2;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_data.size(); ++i)
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Get the index
	  id.push_back(i);

	  // Data V/sigma
	  yrr.push_back(std::complex<double>(_data.datum(i).RR.real()/_data.datum(i).RRerr.real(),_data.datum(i).RR.imag()/_data.datum(i).RRerr.imag()));
	  yll.push_back(std::complex<double>(_data.datum(i).LL.real()/_data.datum(i).LLerr.real(),_data.datum(i).LL.imag()/_data.datum(i).LLerr.imag()));
	  yrl.push_back(std::complex<double>(_data.datum(i).RL.real()/_data.datum(i).RLerr.real(),_data.datum(i).RL.imag()/_data.datum(i).RLerr.imag()));
	  ylr.push_back(std::complex<double>(_data.datum(i).LR.real()/_data.datum(i).LRerr.real(),_data.datum(i).LR.imag()/_data.datum(i).LRerr.imag()));

	  // Station names
	  s1.push_back(_data.datum(i).Station1);
	  s2.push_back(_data.datum(i).Station2);

	  // Station indexes
	  size_t a=999, b=999; // Large number to facilitate the identification of failed baseline index determination
	  for (size_t c=0; c<_sigma_g.size(); ++c)
	  {
	    if (_data.datum(i).Station1==_station_codes[c])
	      a = c;	
	    if (_data.datum(i).Station2==_station_codes[c])
	      b = c;	
	  }
	  is1.push_back(a);
	  is2.push_back(b);	  
	}
      _datum_index_list[epoch]=id;
      _yrr_list[epoch]=yrr;
      _yll_list[epoch]=yll;
      _yrl_list[epoch]=yrl;
      _ylr_list[epoch]=ylr;
      _is1_list[epoch]=is1;
      _is2_list[epoch]=is2;

    }
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::operator()(std::vector<double>& x)
  {
    if (_parallelize_likelihood)
      return likelihood_multiproc(x);
    else
      return likelihood_uniproc(x);
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_multiproc(std::vector<double>& x)
  {
    require_independent_mode_only_(__func__);
    Themis::utils::ScopedTimer Ttot(Themis::utils::TimerID::LikelihoodMultiprocTotal, timer_ns_, timer_calls_);
   
    if (x==_x_last)
      return _L_last;
    _x_last = x;
    
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double L = 0.0;
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	if (epoch%_L_size!=size_t(_L_rank))
	  continue;
	
	const size_t n = _datum_index_list[epoch].size();
	
	std::vector< std::complex<double> > ybrr; ybrr.reserve(n);
	std::vector< std::complex<double> > ybll; ybll.reserve(n);
	std::vector< std::complex<double> > ybrl; ybrl.reserve(n);
	std::vector< std::complex<double> > yblr; yblr.reserve(n);
	
	std::vector< std::complex<double> > yrr;  yrr.reserve(n);
	std::vector< std::complex<double> > yll;  yll.reserve(n);
	std::vector< std::complex<double> > yrl;  yrl.reserve(n);
	std::vector< std::complex<double> > ylr;  ylr.reserve(n);
	
	std::vector< std::vector< std::complex<double> > > yb(4), y(4);
	std::vector<size_t> is1, is2;
	
	double lognorm = 0.0;
	
	for (size_t ii=0; ii<n; ++ii)
	  {
	    const size_t j = _datum_index_list[epoch][ii];
	    
	    std::vector< std::complex<double> > err = _uncertainty.error(_data.datum(j));
	    const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
	    
	    std::complex<double> cvo[4];
	    _model.fill_crosshand_visibilities(j, _data.datum(j), acc, cvo);
	    
	    ybrr.push_back( std::complex<double>(cvo[0].real()/err[0].real(), cvo[0].imag()/err[0].imag()) );
	    ybll.push_back( std::complex<double>(cvo[1].real()/err[1].real(), cvo[1].imag()/err[1].imag()) );
	    ybrl.push_back( std::complex<double>(cvo[2].real()/err[2].real(), cvo[2].imag()/err[2].imag()) );
	    yblr.push_back( std::complex<double>(cvo[3].real()/err[3].real(), cvo[3].imag()/err[3].imag()) );
	    
	    yrr.push_back( std::complex<double>(_data.datum(j).RR.real()/err[0].real(), _data.datum(j).RR.imag()/err[0].imag()) );
	    yll.push_back( std::complex<double>(_data.datum(j).LL.real()/err[1].real(), _data.datum(j).LL.imag()/err[1].imag()) );
	    yrl.push_back( std::complex<double>(_data.datum(j).RL.real()/err[2].real(), _data.datum(j).RL.imag()/err[2].imag()) );
	    ylr.push_back( std::complex<double>(_data.datum(j).LR.real()/err[3].real(), _data.datum(j).LR.imag()/err[3].imag()) );
	    
	    lognorm += _uncertainty.log_normalization(_data.datum(j));
	  }
	
	yb[0].swap(ybrr);
	yb[1].swap(ybll);
	yb[2].swap(ybrl);
	yb[3].swap(yblr);
	
	y[0].swap(yrr);
	y[1].swap(yll);
	y[2].swap(yrl);
	y[3].swap(ylr);
	
	is1 = _is1_list[epoch];
	is2 = _is2_list[epoch];
	
	double marg_term;
	if (_solve_for_gains)
	  {
	    if (epoch>0)
	      {
		if (_use_prior_gain_solutions==false)
		  {
		    if (_smoothly_varying_gains)
		      {
			_G[epoch] = _G[epoch-1];
		      }
		    else
		      {
			for (size_t a=0; a<_sigma_g.size(); ++a)
			  {
			    _G[epoch][a].GR = std::complex<double>(1.0,0.0);
			    _G[epoch][a].GL = std::complex<double>(1.0,0.0);
			  }
		      }
		  }
	      }
	    
	    const double marg = optimal_complex_gains(y, yb, is1, is2, _G[epoch]);
	    if (marg>0)
	      _sqrt_detC[epoch] = marg;
	  }
	marg_term = _sqrt_detC[epoch];

	double dL = 0.0;
	for (size_t ii=0; ii<y[0].size(); ++ii)
	  {
	    const HandGains& g1 = _G[epoch][is1[ii]];
	    const HandGains& g2 = _G[epoch][is2[ii]];
	    
	    const std::complex<double> GGybrr = baseline_gain_rr(g1, g2) * yb[0][ii];
	    const std::complex<double> GGybll = baseline_gain_ll(g1, g2) * yb[1][ii];
	    const std::complex<double> GGybrl = baseline_gain_rl(g1, g2) * yb[2][ii];
	    const std::complex<double> GGyblr = baseline_gain_lr(g1, g2) * yb[3][ii];
	    
	    dL += -0.5 * ( std::pow( y[0][ii].real() - GGybrr.real(), 2) + std::pow( y[0][ii].imag() - GGybrr.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[1][ii].real() - GGybll.real(), 2) + std::pow( y[1][ii].imag() - GGybll.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[2][ii].real() - GGybrl.real(), 2) + std::pow( y[2][ii].imag() - GGybrl.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[3][ii].real() - GGyblr.real(), 2) + std::pow( y[3][ii].imag() - GGyblr.imag(), 2) );
	  }
 	
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  {
	    if (_hand_gain_mode == HandGainMode::Independent)
	      {
		const double GR  = std::log(std::abs(_G[epoch][a].GR));
		const double phR = std::arg(_G[epoch][a].GR);
		dL += -0.5*GR*GR/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*phR*phR*_opi2;
		
		const double GL  = std::log(std::abs(_G[epoch][a].GL));
		const double phL = std::arg(_G[epoch][a].GL);
		dL += -0.5*GL*GL/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*phL*phL*_opi2;
	      }
	    else
	      {
		double ac, pc, eta, delta;
		ratio_common_and_diff_(_G[epoch][a], ac, pc, eta, delta);
		
		dL += -0.5*ac*ac/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*pc*pc*_opi2;
		dL += -0.5*eta*eta/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
		dL += -0.5*delta*delta/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	      }
	  }

	dL += std::log(marg_term);
	dL += lognorm;
	
	L += dL;
      }
    
    double Ltot = 0.0;
    MPI_Allreduce(&L, &Ltot, 1, MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    _L_last = Ltot;
    return Ltot;
  }
  

  double likelihood_optimal_complex_gain_crosshand_visibilities::likelihood_uniproc(std::vector<double>& x)
  {
    require_independent_mode_only_(__func__);
    Themis::utils::ScopedTimer Ttot(Themis::utils::TimerID::LikelihoodMultiprocTotal, timer_ns_, timer_calls_);
    
    if (x==_x_last)
      return _L_last;
    _x_last = x;
    
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double L = 0.0;
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	const size_t n = _datum_index_list[epoch].size();
	
	std::vector< std::complex<double> > ybrr; ybrr.reserve(n);
	std::vector< std::complex<double> > ybll; ybll.reserve(n);
	std::vector< std::complex<double> > ybrl; ybrl.reserve(n);
	std::vector< std::complex<double> > yblr; yblr.reserve(n);
	
	std::vector< std::complex<double> > yrr;  yrr.reserve(n);
	std::vector< std::complex<double> > yll;  yll.reserve(n);
	std::vector< std::complex<double> > yrl;  yrl.reserve(n);
	std::vector< std::complex<double> > ylr;  ylr.reserve(n);
	
	std::vector< std::vector< std::complex<double> > > yb(4), y(4);
	std::vector<size_t> is1, is2;
	
	double lognorm = 0.0;
	
	for (size_t ii=0; ii<n; ++ii)
	  {
	    const size_t j = _datum_index_list[epoch][ii];
	    
	    std::vector< std::complex<double> > err = _uncertainty.error(_data.datum(j));
	    const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
	    
	    std::complex<double> cvo[4];
	    _model.fill_crosshand_visibilities(j, _data.datum(j), acc, cvo);
	    
	    ybrr.push_back( std::complex<double>(cvo[0].real()/err[0].real(), cvo[0].imag()/err[0].imag()) );
	    ybll.push_back( std::complex<double>(cvo[1].real()/err[1].real(), cvo[1].imag()/err[1].imag()) );
	    ybrl.push_back( std::complex<double>(cvo[2].real()/err[2].real(), cvo[2].imag()/err[2].imag()) );
	    yblr.push_back( std::complex<double>(cvo[3].real()/err[3].real(), cvo[3].imag()/err[3].imag()) );
	    
	    yrr.push_back( std::complex<double>(_data.datum(j).RR.real()/err[0].real(), _data.datum(j).RR.imag()/err[0].imag()) );
	    yll.push_back( std::complex<double>(_data.datum(j).LL.real()/err[1].real(), _data.datum(j).LL.imag()/err[1].imag()) );
	    yrl.push_back( std::complex<double>(_data.datum(j).RL.real()/err[2].real(), _data.datum(j).RL.imag()/err[2].imag()) );
	    ylr.push_back( std::complex<double>(_data.datum(j).LR.real()/err[3].real(), _data.datum(j).LR.imag()/err[3].imag()) );
	    
	    lognorm += _uncertainty.log_normalization(_data.datum(j));
	  }
	
	yb[0].swap(ybrr);
	yb[1].swap(ybll);
	yb[2].swap(ybrl);
	yb[3].swap(yblr);
	
	y[0].swap(yrr);
	y[1].swap(yll);
	y[2].swap(yrl);
	y[3].swap(ylr);
	
	is1 = _is1_list[epoch];
	is2 = _is2_list[epoch];
	
	double marg_term;
	if (_solve_for_gains)
	  {
	    if (epoch>0)
	      {
		if (_use_prior_gain_solutions==false)
		  {
		    if (_smoothly_varying_gains)
		      {
			_G[epoch] = _G[epoch-1];
		      }
		    else
		      {
			for (size_t a=0; a<_sigma_g.size(); ++a)
			  {
			    _G[epoch][a].GR = std::complex<double>(1.0,0.0);
			    _G[epoch][a].GL = std::complex<double>(1.0,0.0);
			  }
		      }
		  }
	      }
	    
	    const double marg = optimal_complex_gains(y, yb, is1, is2, _G[epoch]);
	    if (marg>0)
	      _sqrt_detC[epoch] = marg;
	  }
	marg_term = _sqrt_detC[epoch];

	double dL = 0.0;
	for (size_t ii=0; ii<y[0].size(); ++ii)
	  {
	    const HandGains& g1 = _G[epoch][is1[ii]];
	    const HandGains& g2 = _G[epoch][is2[ii]];
	    
	    const std::complex<double> GGybrr = baseline_gain_rr(g1, g2) * yb[0][ii];
	    const std::complex<double> GGybll = baseline_gain_ll(g1, g2) * yb[1][ii];
	    const std::complex<double> GGybrl = baseline_gain_rl(g1, g2) * yb[2][ii];
	    const std::complex<double> GGyblr = baseline_gain_lr(g1, g2) * yb[3][ii];
	    
	    dL += -0.5 * ( std::pow( y[0][ii].real() - GGybrr.real(), 2) + std::pow( y[0][ii].imag() - GGybrr.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[1][ii].real() - GGybll.real(), 2) + std::pow( y[1][ii].imag() - GGybll.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[2][ii].real() - GGybrl.real(), 2) + std::pow( y[2][ii].imag() - GGybrl.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[3][ii].real() - GGyblr.real(), 2) + std::pow( y[3][ii].imag() - GGyblr.imag(), 2) );
	  }
	
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  {
	    if (_hand_gain_mode == HandGainMode::Independent)
	      {
		const double GR  = std::log(std::abs(_G[epoch][a].GR));
		const double phR = std::arg(_G[epoch][a].GR);
		dL += -0.5*GR*GR/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*phR*phR*_opi2;
		
		const double GL  = std::log(std::abs(_G[epoch][a].GL));
		const double phL = std::arg(_G[epoch][a].GL);
		dL += -0.5*GL*GL/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*phL*phL*_opi2;
	      }
	    else
	      {
		double ac, pc, eta, delta;
		ratio_common_and_diff_(_G[epoch][a], ac, pc, eta, delta);
		
		dL += -0.5*ac*ac/(_sigma_g[a]*_sigma_g[a]);
		dL += -0.5*pc*pc*_opi2;
		dL += -0.5*eta*eta/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
		dL += -0.5*delta*delta/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	      }
	  }

	dL += std::log(marg_term);
	dL += lognorm;
	
	L += dL;
      }
    
    _L_last = L;
    return L;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::distribute_gains()
  {
    require_independent_mode_only_(__func__);
    
    const size_t N = 4*(_tge.size()-1)*_sigma_g.size() + (_tge.size()-1);
    
    double *local_buff  = new double[N];
    double *global_buff = new double[N];
    memset(local_buff,  0.0, N*sizeof(double));
    memset(global_buff, 0.0, N*sizeof(double));
    
    int i=0;
    for (size_t j=0; j<_tge.size()-1; ++j)
      {
	if (j%_L_size==size_t(_L_rank))
	  {
	    for (size_t k=0; k<_sigma_g.size(); ++k)
	      {
		local_buff[i++] = _G[j][k].GR.real();
		local_buff[i++] = _G[j][k].GR.imag();
		local_buff[i++] = _G[j][k].GL.real();
		local_buff[i++] = _G[j][k].GL.imag();
	      }
	    local_buff[i++] = _sqrt_detC[j];
	  }
	else
	  {
	    i += 4*_sigma_g.size() + 1;
	  }
      }
    
    MPI_Allreduce(local_buff, global_buff, N, MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    i=0;
    for (size_t j=0; j<_tge.size()-1; ++j)
      {
	for (size_t k=0; k<_sigma_g.size(); ++k)
	  {
	    _G[j][k].GR = std::complex<double>(global_buff[i],   global_buff[i+1]); i += 2;
	    _G[j][k].GL = std::complex<double>(global_buff[i],   global_buff[i+1]); i += 2;
	  }
	_sqrt_detC[j] = global_buff[i++];
      }
    
    delete[] local_buff;
    delete[] global_buff;
  }

 
  std::vector<double> likelihood_optimal_complex_gain_crosshand_visibilities::gradient(std::vector<double>& x, prior& Pr)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientTotal, timer_ns_, timer_calls_);
    return gradient_dispatch_(x, Pr);
  }

  
  std::vector<double> likelihood_optimal_complex_gain_crosshand_visibilities::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    return gradient_dispatch_(x, Pr);
  }

  
  std::vector<double> likelihood_optimal_complex_gain_crosshand_visibilities::gradient_dispatch_(std::vector<double>& x, prior& Pr)
  {
    switch (gradient_mode())
      {
      case GradientMode::FD_ALL:
	{
	  Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
	  const double Lx = ((_x_last.empty() || x != _x_last) ? this->operator()(x) : _L_last);
	  
	  const bool solving_for_gains_prev = _solve_for_gains;
	  if (!_solve_for_gains_during_gradient)
	    fix_gains();
	  
	  std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	  
	  std::vector<double> mx(_model.size()), ux(_uncertainty.size());
	  size_t ii = 0;
	  for (size_t j=0; j<_model.size(); ++j) mx[j] = x[ii++];
	  for (size_t j=0; j<_uncertainty.size(); ++j) ux[j] = x[ii++];
	  _model.generate_model(mx);
	  _uncertainty.generate_uncertainty(ux);
	  _x_last = x;
	  _L_last = Lx;
	  
	  if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
	    solve_for_gains();
	  
	  return g;
	}
	
      case GradientMode::HYBRID_INTENSITY:
	return gradient_hybrid(x, Pr, /*do_geom=*/false);
	
      case GradientMode::HYBRID_INTENSITY_GEOM:
      default:
	return gradient_hybrid(x, Pr, /*do_geom=*/true);
      }
  }


  std::vector<double> likelihood_optimal_complex_gain_crosshand_visibilities::gradient_hybrid(std::vector<double>& x, prior& Pr, bool /*do_geom*/)
  {
    const double Lx = ((_x_last.empty() || x != _x_last) ? this->operator()(x) : _L_last);
    
    const bool solving_for_gains_prev = _solve_for_gains;
    if (!_solve_for_gains_during_gradient)
      fix_gains();
    
    std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
    
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t ii = 0;
    for (size_t j=0; j<_model.size(); ++j) mx[j] = x[ii++];
    for (size_t j=0; j<_uncertainty.size(); ++j) ux[j] = x[ii++];
    
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    _x_last = x;
    _L_last = Lx;
    
    if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
      solve_for_gains();
    
    return g;
  }
  

  double likelihood_optimal_complex_gain_crosshand_visibilities::chi_squared(std::vector<double>& x)
  {
    require_independent_mode_only_(__func__);
    distribute_gains();
    
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
    double L = 0.0;
    
    std::vector<double> true_sigma_g = _sigma_g;
    _sigma_g.assign(_sigma_g.size(),1000.0);
    
    const double true_sigma_ratio_logamp = _sigma_ratio_logamp;
    const double true_sigma_ratio_phase  = _sigma_ratio_phase;
    if (_hand_gain_mode == HandGainMode::Ratio)
      {
	_sigma_ratio_logamp = 1.0e6;
	_sigma_ratio_phase  = 1.0e6;
      }
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	std::vector< std::complex<double> > ybrr,ybll,ybrl,yblr;
	std::vector< std::complex<double> > yrr,yll,yrl,ylr;
	std::vector< std::vector< std::complex<double> > > yb, y;
	std::vector<size_t> is1, is2;
	
	for (size_t ii=0; ii<_datum_index_list[epoch].size(); ++ii)
	  {
	    const size_t j = _datum_index_list[epoch][ii];
	    
	    std::vector< std::complex<double> > err = _uncertainty.error(_data.datum(j));
	    std::vector< std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(j), 0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));
	    
	    ybrr.push_back( std::complex<double>(cvo[0].real()/err[0].real(), cvo[0].imag()/err[0].imag()) );
	    ybll.push_back( std::complex<double>(cvo[1].real()/err[1].real(), cvo[1].imag()/err[1].imag()) );
	    ybrl.push_back( std::complex<double>(cvo[2].real()/err[2].real(), cvo[2].imag()/err[2].imag()) );
	    yblr.push_back( std::complex<double>(cvo[3].real()/err[3].real(), cvo[3].imag()/err[3].imag()) );
	    
	    yrr.push_back( std::complex<double>(_data.datum(j).RR.real()/err[0].real(), _data.datum(j).RR.imag()/err[0].imag()) );
	    yll.push_back( std::complex<double>(_data.datum(j).LL.real()/err[1].real(), _data.datum(j).LL.imag()/err[1].imag()) );
	    yrl.push_back( std::complex<double>(_data.datum(j).RL.real()/err[2].real(), _data.datum(j).RL.imag()/err[2].imag()) );
	    ylr.push_back( std::complex<double>(_data.datum(j).LR.real()/err[3].real(), _data.datum(j).LR.imag()/err[3].imag()) );
	  }
	
	y.push_back(yrr);
	y.push_back(yll);
	y.push_back(yrl);
	y.push_back(ylr);
	
	yb.push_back(ybrr);
	yb.push_back(ybll);
	yb.push_back(ybrl);
	yb.push_back(yblr);
	
	is1 = _is1_list[epoch];
	is2 = _is2_list[epoch];
	
	if (_solve_for_gains)
	  {
	    if (epoch>0)
	      {
		if (_use_prior_gain_solutions==false)
		  {
		    if (_smoothly_varying_gains)
		      {
			_G[epoch] = _G[epoch-1];
		      }
		    else
		      {
			for (size_t a=0; a<_sigma_g.size(); ++a)
			  {
			    _G[epoch][a].GR = std::complex<double>(1.0,0.0);
			    _G[epoch][a].GL = std::complex<double>(1.0,0.0);
			  }
		      }
		  }
	      }
	    
	    optimal_complex_gains(y,yb,is1,is2,_G[epoch]);
	  }
	
	double dL = 0.0;
	for (size_t ii=0; ii<y[0].size(); ++ii)
	  {
	    const HandGains& g1 = _G[epoch][is1[ii]];
	    const HandGains& g2 = _G[epoch][is2[ii]];
	    
	    const std::complex<double> GGybrr = baseline_gain_rr(g1, g2) * yb[0][ii];
	    const std::complex<double> GGybll = baseline_gain_ll(g1, g2) * yb[1][ii];
	    const std::complex<double> GGybrl = baseline_gain_rl(g1, g2) * yb[2][ii];
	    const std::complex<double> GGyblr = baseline_gain_lr(g1, g2) * yb[3][ii];
	    
	    dL += -0.5 * ( std::pow( y[0][ii].real() - GGybrr.real(), 2) + std::pow( y[0][ii].imag() - GGybrr.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[1][ii].real() - GGybll.real(), 2) + std::pow( y[1][ii].imag() - GGybll.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[2][ii].real() - GGybrl.real(), 2) + std::pow( y[2][ii].imag() - GGybrl.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[3][ii].real() - GGyblr.real(), 2) + std::pow( y[3][ii].imag() - GGyblr.imag(), 2) );
	  }
	
	L += dL;
      }
    
    _sigma_g = true_sigma_g;
    _sigma_ratio_logamp = true_sigma_ratio_logamp;
    _sigma_ratio_phase  = true_sigma_ratio_phase;  
    
    return (-2.0*L);
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::matrix_determinant(double **a)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::matrix_determinant, timer_ns_, timer_calls_);
    
    const int n = 4 * int(_sigma_g.size());
    double d;
    
    ludcmp(a, n, _indx, d);
    
    for (int i=1; i<=n; ++i)
      d *= a[i][i];
    
    return d;
  }


#define TINY 1.0e-20;
  void likelihood_optimal_complex_gain_crosshand_visibilities::ludcmp(double **a, int n, int *indx, double &d)
  {
    int i,imax=0,j,k;
    double big,dum,sum,temp;
    
    d = 1.0;
    for (i=1; i<=n; ++i) {
      big = 0.0;
      for (j=1; j<=n; ++j)
	if ((temp=std::fabs(a[i][j])) > big)
	  big = temp;
      if (big == 0.0)
	std::cerr << "Singular matrix in routine ludcmp";
      _vv[i] = 1.0/big;
    }
    for (j=1; j<=n; ++j) {
      for (i=1; i<j; ++i) {
	sum = a[i][j];
	for (k=1; k<i; ++k)
	  sum -= a[i][k]*a[k][j];
	a[i][j] = sum;
      }
      big = 0.0;
      for (i=j; i<=n; ++i) {
	sum = a[i][j];
	for (k=1; k<j; ++k)
	  sum -= a[i][k]*a[k][j];
	a[i][j] = sum;
	if ((dum=_vv[i]*std::fabs(sum)) >= big) {
	  big = dum;
	  imax = i;
	}
      }
      if (j != imax) {
	for (k=1; k<=n; ++k) {
	  dum = a[imax][k];
	  a[imax][k] = a[j][k];
	  a[j][k] = dum;
	}
	d = -d;
	_vv[imax] = _vv[j];
      }
      indx[j] = imax;
      if (a[j][j] == 0.0)
	a[j][j] = TINY;
      if (j != n) {
	dum = 1.0/(a[j][j]);
	for (i=j+1; i<=n; ++i)
	  a[i][j] *= dum;
      }
    }
  }
#undef TINY


  std::vector<double> likelihood_optimal_complex_gain_crosshand_visibilities::get_gain_times()
  {
    return ( _tge );
  }


  std::vector< std::vector< likelihood_optimal_complex_gain_crosshand_visibilities::HandGains > > likelihood_optimal_complex_gain_crosshand_visibilities::get_gains()
  {
    return _G;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::read_gain_file(std::string gain_file_name)
  {
    require_independent_mode_only_(__func__);
    
    int rank;
    MPI_Comm_rank(_comm, &rank);
    
    if (rank==0)
      {
	std::ifstream in(gain_file_name);
	
	in.ignore(4096,'\n');
	in.ignore(4096,'\n');
	in.ignore(4096,'\n');
	
	double tmp, Gr, Gi;
	for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
	  {
	    in >> tmp;
	    in >> tmp;
	    for (size_t a=0; a<_sigma_g.size(); ++a)
	      {
		in >> Gr;
		in >> Gi;
		const std::complex<double> Gtmp(Gr,Gi);
		_G[epoch][a].GR = Gtmp;
		_G[epoch][a].GL = Gtmp;
	      }
	    
	    if (in.eof()==true)
	      {
		std::cerr << "ERROR: likelihood_optimal_complex_gain_crosshand_visibilities::read_gain_file too few gains in "
			  << gain_file_name << '\n';
		std::exit(1);
	      }
	  }
      }
    
    const size_t ngains = 4*_sigma_g.size()*(_tge.size()-1);
    double *buff = new double[ngains];
    
    for (size_t epoch=0, k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
	{
	  buff[k++] = _G[epoch][a].GR.real();
	  buff[k++] = _G[epoch][a].GR.imag();
	  buff[k++] = _G[epoch][a].GL.real();
	  buff[k++] = _G[epoch][a].GL.imag();
	}
    
    MPI_Bcast(buff, ngains, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for (size_t epoch=0, k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
	{
	  _G[epoch][a].GR = std::complex<double>(buff[k], buff[k+1]); k += 2;
	  _G[epoch][a].GL = std::complex<double>(buff[k], buff[k+1]); k += 2;
	}
    
    delete[] buff;
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      _sqrt_detC[epoch] = 1.0;
  }
  

  void likelihood_optimal_complex_gain_crosshand_visibilities::output_gains(std::ostream& out)
  {
    int nprec = out.precision();
    out.precision(20);
    out << "# First epoch time at " << _tge[0] << " s from J2000.  All subsequent times measured from this value.\n";
    out.precision(nprec);
    out << "# Number of independent gains: " << number_of_independent_gains() << '\n';
    out << "#" << std::setw(14) << "start time (s)"
	<< std::setw(15) << "end time (s)";
    
    for (size_t a=0; a<_sigma_g.size(); ++a)
      out << std::setw(15) << _station_codes[a]+".R.real"
	  << std::setw(15) << _station_codes[a]+".R.imag"
	  << std::setw(15) << _station_codes[a]+".L.real"
	  << std::setw(15) << _station_codes[a]+".L.imag";
    out << '\n';
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	out << std::setw(15) << _tge[epoch]-_tge[0]
	    << std::setw(15) << _tge[epoch+1]-_tge[0];
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  out << std::setw(15) << _G[epoch][a].GR.real()
	      << std::setw(15) << _G[epoch][a].GR.imag()
	      << std::setw(15) << _G[epoch][a].GL.real()
	      << std::setw(15) << _G[epoch][a].GL.imag();
	out << '\n';
      }
  }
  

  void likelihood_optimal_complex_gain_crosshand_visibilities::output_gains(std::string outname)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    
    distribute_gains();
    
    if (rank==0)
      {
	std::ofstream out(outname.c_str());
	output_gains(out);
	out.close();
      }
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::output_gain_corrections(std::ostream& out)
  {
    int nprec = out.precision();
    out.precision(20);
    out << "# First epoch time at " << _tge[0] << " s from J2000.  All subsequent times measured from this value.\n";
    out.precision(nprec);
    out << "# Number of independent gains: " << number_of_independent_gains() << '\n';
    out << "#" << std::setw(14) << "start time (s)"
	<< std::setw(15) << "end time (s)";
    
    for (size_t a=0; a<_sigma_g.size(); ++a)
      out << std::setw(15) << (_station_codes[a]+".R")
	  << std::setw(15) << (_station_codes[a]+".L");
    out << '\n';
    
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	out << std::setw(15) << _tge[epoch]-_tge[0]
	    << std::setw(15) << _tge[epoch+1]-_tge[0];
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  out << std::setw(15) << (std::abs(_G[epoch][a].GR)-1.0)
	      << std::setw(15) << (std::abs(_G[epoch][a].GL)-1.0);
	out << '\n';
      }
  }

  
  void likelihood_optimal_complex_gain_crosshand_visibilities::output_gain_corrections(std::string outname)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    
    distribute_gains();
    
    if (rank==0)
      {
	std::ofstream out(outname.c_str());
	output_gain_corrections(out);
	out.close();
      }
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::optimal_complex_gains(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< HandGains >& gest)
  {
    require_independent_mode_only_(__func__);
    
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveTotal, timer_ns_, timer_calls_);
    
    std::vector<HandGains> gest_best = gest;
    double marg_best = -1;
    double chisq_best = std::numeric_limits<double>::infinity();
    
    std::vector<HandGains> gest_try = gest;
    double marg_try, chisq_try;
    
    marg_try = optimal_complex_gains_trial(y,yb,is1,is2,gest_try,chisq_try);
    if (marg_try>=0)
      {
	gest_best = gest_try;
	marg_best = marg_try;
	chisq_best = chisq_try;
      }
    
    gest_try = gest;
    marg_try = optimal_complex_gains_log_trial(y,yb,is1,is2,gest_try,chisq_try);
    if (marg_try>=0)
      marg_try = optimal_complex_gains_trial(y,yb,is1,is2,gest_try,chisq_try);
    
    if (marg_try>0 && chisq_try<chisq_best)
      {
	gest_best = gest_try;
	marg_best = marg_try;
	chisq_best = chisq_try;
      }
    
    if (marg_best<0)
      {
	for (size_t j=0; j<gest.size(); ++j)
	  {
	    gest[j].GR = std::complex<double>(1.0,0.0);
	    gest[j].GL = std::complex<double>(1.0,0.0);
	  }
	return -1;
      }
    
    gest = gest_best;
    return marg_best;
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::optimal_complex_gains_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1,std::vector<size_t>& is2, std::vector< HandGains >& gest, double& chisq_opt)
  {
    require_independent_mode_only_(__func__);
    
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveTrial, timer_ns_, timer_calls_);
    
    const int ndata = int(2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()));
    if (ndata==0)
      return 1.0;
    
    for (size_t i=0, j=1; i<y[0].size(); ++i)
      {
	_ogc_y[j]   = y[0][i].real(); _ogc_yb[j]   = yb[0][i].real(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 0; ++j;
	_ogc_y[j]   = y[0][i].imag(); _ogc_yb[j]   = yb[0][i].imag(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 0; ++j;
	_ogc_y[j]   = y[1][i].real(); _ogc_yb[j]   = yb[1][i].real(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 1; ++j;
	_ogc_y[j]   = y[1][i].imag(); _ogc_yb[j]   = yb[1][i].imag(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 1; ++j;
	_ogc_y[j]   = y[2][i].real(); _ogc_yb[j]   = yb[2][i].real(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 2; ++j;
	_ogc_y[j]   = y[2][i].imag(); _ogc_yb[j]   = yb[2][i].imag(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 2; ++j;
	_ogc_y[j]   = y[3][i].real(); _ogc_yb[j]   = yb[3][i].real(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 3; ++j;
	_ogc_y[j]   = y[3][i].imag(); _ogc_yb[j]   = yb[3][i].imag(); _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; _ogc_hand[j] = 3; ++j;
      }
    
    const int ma = 4 * int(_sigma_g.size());

    for (int i=0, j=1; i<int(gest.size()); ++i)
      {
	double p4[4];
	encode_station_params_(gest[i], p4);
	_g[j++] = p4[0];
	_g[j++] = p4[1];
	_g[j++] = p4[2];
	_g[j++] = p4[3];
      }

    double alambda = -1.0;
    double chisq = 0.0, ochisq, dg2;
    double dg2limit = 0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += 2.0 * _sigma_g[i]*_sigma_g[i];
    dg2limit *= 1e-12;
    
    bool notconverged = true;
    for (int iteration=0; iteration<_itermax && notconverged; ++iteration)
      {
	for (int i=1; i<=ma; ++i)
	  _og[i] = _g[i];
	ochisq = chisq;
	
	if (mrqmin(_ogc_y, ndata, _g, ma, _covar, _alpha, &chisq, &alambda))
	  return -1;
	
	if (iteration>1 && chisq<ochisq)
	  {
	    dg2 = 0.0;
	    for (int i=1; i<=ma; ++i)
	      dg2 += std::pow((_g[i]-_og[i]),2);
	    
	    if (dg2<dg2limit || (ochisq-chisq)<1e-8*chisq)
	      notconverged = false;
	  }
      }
    
    alambda = 0.0;
    mrqmin(_ogc_y, ndata, _g, ma, _covar, _alpha, &chisq, &alambda);
    
    for (int i=0; i<int(gest.size()); ++i)
      gest[i] = decode_station_params_(_g, i);

    double detC = matrix_determinant(_covar);
    for (size_t a=0; a<_sigma_g.size(); ++a)
      {
	const double oSigma2 = 1.0/(_sigma_g[a]*_sigma_g[a]);
	
	if (_hand_gain_mode == HandGainMode::Independent)
	  {
	    const double pf = oSigma2 * _opi2;
	    detC *= pf * pf;
	  }
	else
	  {
	    const double oEta2 = 1.0/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
	    const double oDel2 = 1.0/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	    detC *= (oSigma2 * _opi2 * oEta2 * oDel2);
	  }
      }
 
    chisq_opt = chisq;
    return std::sqrt(detC);
  }


  double likelihood_optimal_complex_gain_crosshand_visibilities::optimal_complex_gains_log_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< HandGains >& gest, double& chisq_opt)
  {
    require_independent_mode_only_(__func__);
    
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveLogTrial, timer_ns_, timer_calls_);
    
    const int ndata = int(2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()));
    if (ndata==0)
      return 1.0;
    
    std::complex<double> tmp;
    for (size_t i=0, j=1; i<y[0].size(); ++i)
      {
	tmp = y[0][i]/yb[0][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[0][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=0; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[0][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=0; ++j;
	
	tmp = y[1][i]/yb[1][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[1][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=1; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[1][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=1; ++j;
	
	tmp = y[2][i]/yb[2][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[2][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=2; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[2][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=2; ++j;
	
	tmp = y[3][i]/yb[3][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[3][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=3; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[3][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; _ogc_hand[j]=3; ++j;
      }
    
    const int ma = 4 * int(_sigma_g.size());
    
    for (int i=0, j=1; i<int(gest.size()); ++i)
      {
	double p4[4];
	encode_station_params_(gest[i], p4);
	_g[j++] = p4[0];
	_g[j++] = p4[1];
	_g[j++] = p4[2];
	_g[j++] = p4[3];
      }

    double alambda = -1.0;
    double chisq = 0.0, ochisq, dg2;
    const double ch2limit = 1.0e-8;
    double dg2limit = 0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += 2.0 * _sigma_g[i]*_sigma_g[i];
    dg2limit *= 1e-12;
    
    bool notconverged = true;
    for (int iteration=0; iteration<_itermax && notconverged; ++iteration)
      {
	for (int i=1; i<=ma; ++i)
	  _og[i] = _g[i];
	ochisq = chisq;
	
	if (mrqmin_log(_ogc_y, _sig, ndata, _g, ma, _covar, _alpha, &chisq, &alambda))
	  return -1;
	
	if (iteration>1 && chisq<ochisq)
	  {
	    dg2 = 0.0;
	    for (int i=1; i<=ma; ++i)
	      dg2 += std::pow((_g[i]-_og[i]),2);
	    
	    if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
	      notconverged = false;
	  }
      }
    
    alambda = 0.0;
    mrqmin_log(_ogc_y, _sig, ndata, _g, ma, _covar, _alpha, &chisq, &alambda);

    for (int i=0; i<int(gest.size()); ++i)
      gest[i] = decode_station_params_(_g, i);

    double detC = matrix_determinant(_covar);
    for (size_t a=0; a<_sigma_g.size(); ++a)
      {
	const double oSigma2 = 1.0/(_sigma_g[a]*_sigma_g[a]);
	
	if (_hand_gain_mode == HandGainMode::Independent)
	  {
	    const double pf = oSigma2 * _opi2;
	    detC *= pf * pf;
	  }
	else
	  {
	    const double oEta2 = 1.0/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
	    const double oDel2 = 1.0/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	    detC *= (oSigma2 * _opi2 * oEta2 * oDel2);
	  }
      }

    chisq_opt = chisq;
    return std::sqrt(detC);
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    const size_t nsta = _sigma_g.size();
    for (size_t a=1; a<=4*nsta; ++a)
      dydg[a] = 0.0;
    
    const size_t s1 = _ogc_is1[i];
    const size_t s2 = _ogc_is2[i];
    
    const HandGains hg1 = decode_station_params_(g, s1);
    const HandGains hg2 = decode_station_params_(g, s2);
    
    std::complex<double> G1, G2;
    double sign1 = 0.0, sign2 = 0.0;
    
    switch (_ogc_hand[i])
      {
      case 0: // RR
	G1 = hg1.GR; G2 = hg2.GR; sign1 = +1.0; sign2 = +1.0; break;
      case 1: // LL
	G1 = hg1.GL; G2 = hg2.GL; sign1 = -1.0; sign2 = -1.0; break;
      case 2: // RL
	G1 = hg1.GR; G2 = hg2.GL; sign1 = +1.0; sign2 = -1.0; break;
      case 3: // LR
	G1 = hg1.GL; G2 = hg2.GR; sign1 = -1.0; sign2 = +1.0; break;
      default:
	std::cerr << "ERROR: invalid _ogc_hand in " << __func__ << '\n';
	std::exit(1);
      }
    
    const std::complex<double> I(0.0,1.0);
    
    std::complex<double> yb, yc;
    if (i%2==1)
      {
	yb = std::complex<double>(_ogc_yb[i], _ogc_yb[i+1]);
	yc = G1 * std::conj(G2) * yb;
	(*y) = yc.real();
      }
    else
      {
	yb = std::complex<double>(_ogc_yb[i-1], _ogc_yb[i]);
	yc = G1 * std::conj(G2) * yb;
	(*y) = yc.imag();
      }
    
    if (_hand_gain_mode == HandGainMode::Independent)
      {
	int ia1, ip1, ia2, ip2;
	switch (_ogc_hand[i])
	  {
	  case 0: ia1=idx_logamp_R(s1); ip1=idx_phase_R(s1); ia2=idx_logamp_R(s2); ip2=idx_phase_R(s2); break;
	  case 1: ia1=idx_logamp_L(s1); ip1=idx_phase_L(s1); ia2=idx_logamp_L(s2); ip2=idx_phase_L(s2); break;
	  case 2: ia1=idx_logamp_R(s1); ip1=idx_phase_R(s1); ia2=idx_logamp_L(s2); ip2=idx_phase_L(s2); break;
	  case 3: ia1=idx_logamp_L(s1); ip1=idx_phase_L(s1); ia2=idx_logamp_R(s2); ip2=idx_phase_R(s2); break;
	  default: std::exit(1);
	  }
	
	if (i%2==1)
	  {
	    dydg[ia1] = ( yc ).real();
	    dydg[ip1] = ( I*yc ).real();
	    dydg[ia2] = ( yc ).real();
	    dydg[ip2] = (-I*yc ).real();
	  }
	else
	  {
	    dydg[ia1] = ( yc ).imag();
	    dydg[ip1] = ( I*yc ).imag();
	    dydg[ia2] = ( yc ).imag();
	    dydg[ip2] = (-I*yc ).imag();
	  }
      }
    else
      {
	const int ia1 = 4*int(s1)+1;
	const int ip1 = ia1+1;
	const int ie1 = ia1+2;
	const int id1 = ia1+3;
	
	const int ia2 = 4*int(s2)+1;
	const int ip2 = ia2+1;
	const int ie2 = ia2+2;
	const int id2 = ia2+3;
	
	if (i%2==1)
	  {
	    dydg[ia1] = ( yc ).real();
	    dydg[ip1] = ( I*yc ).real();
	    dydg[ie1] = ( 0.5*sign1*yc ).real();
	    dydg[id1] = ( 0.5*sign1*I*yc ).real();
	    
	    dydg[ia2] = ( yc ).real();
	    dydg[ip2] = (-I*yc ).real();
	    dydg[ie2] = ( 0.5*sign2*yc ).real();
	    dydg[id2] = (-0.5*sign2*I*yc ).real();
	  }
	else
	  {
	    dydg[ia1] = ( yc ).imag();
	    dydg[ip1] = ( I*yc ).imag();
	    dydg[ie1] = ( 0.5*sign1*yc ).imag();
	    dydg[id1] = ( 0.5*sign1*I*yc ).imag();
	    
	    dydg[ia2] = ( yc ).imag();
	    dydg[ip2] = (-I*yc ).imag();
	    dydg[ie2] = ( 0.5*sign2*yc ).imag();
	    dydg[id2] = (-0.5*sign2*I*yc ).imag();
	  }
      }
  }
  

  void likelihood_optimal_complex_gain_crosshand_visibilities::gain_optimization_log_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    const size_t nsta = _sigma_g.size();
    for (size_t a=1; a<=4*nsta; ++a)
      dydg[a] = 0.0;
    
    const size_t s1 = _ogc_is1[i];
    const size_t s2 = _ogc_is2[i];
    
    if (_hand_gain_mode == HandGainMode::Independent)
      {
	int ia1, ip1, ia2, ip2;
	switch (_ogc_hand[i])
	  {
	  case 0: ia1=idx_logamp_R(s1); ip1=idx_phase_R(s1); ia2=idx_logamp_R(s2); ip2=idx_phase_R(s2); break;
	  case 1: ia1=idx_logamp_L(s1); ip1=idx_phase_L(s1); ia2=idx_logamp_L(s2); ip2=idx_phase_L(s2); break;
	  case 2: ia1=idx_logamp_R(s1); ip1=idx_phase_R(s1); ia2=idx_logamp_L(s2); ip2=idx_phase_L(s2); break;
	  case 3: ia1=idx_logamp_L(s1); ip1=idx_phase_L(s1); ia2=idx_logamp_R(s2); ip2=idx_phase_R(s2); break;
	  default:
	    std::cerr << "ERROR: invalid _ogc_hand in " << __func__ << '\n';
	    std::exit(1);
	  }
	
	if (i%2==1)
	  {
	    (*y) = g[ia1] + g[ia2];
	    dydg[ia1] =  1.0;
	    dydg[ia2] =  1.0;
	  }
	else
	  {
	    (*y) = g[ip1] - g[ip2];
	    dydg[ip1] =  1.0;
	    dydg[ip2] = -1.0;
	  }
      }
    else
      {
	double sign1 = 0.0, sign2 = 0.0;
	switch (_ogc_hand[i])
	  {
	  case 0: sign1=+1.0; sign2=+1.0; break; // RR
	  case 1: sign1=-1.0; sign2=-1.0; break; // LL
	  case 2: sign1=+1.0; sign2=-1.0; break; // RL
	  case 3: sign1=-1.0; sign2=+1.0; break; // LR
	  default:
	    std::cerr << "ERROR: invalid _ogc_hand in " << __func__ << '\n';
	    std::exit(1);
	  }
	
	const int ia1 = 4*int(s1)+1;
	const int ip1 = ia1+1;
	const int ie1 = ia1+2;
	const int id1 = ia1+3;
	
	const int ia2 = 4*int(s2)+1;
	const int ip2 = ia2+1;
	const int ie2 = ia2+2;
	const int id2 = ia2+3;
	
	if (i%2==1)
	  {
	    (*y) = (g[ia1] + 0.5*sign1*g[ie1]) + (g[ia2] + 0.5*sign2*g[ie2]);
	    dydg[ia1] = 1.0;
	    dydg[ie1] = 0.5*sign1;
	    dydg[ia2] = 1.0;
	    dydg[ie2] = 0.5*sign2;
	  }
	else
	  {
	    (*y) = (g[ip1] + 0.5*sign1*g[id1]) - (g[ip2] + 0.5*sign2*g[id2]);
	    dydg[ip1] =  1.0;
	    dydg[id1] =  0.5*sign1;
	    dydg[ip2] = -1.0;
	    dydg[id2] = -0.5*sign2;
	  }
      }
  }  


  int likelihood_optimal_complex_gain_crosshand_visibilities::gaussj(double **a, int n, double **b, int m)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::gaussj, timer_ns_, timer_calls_);
    
    int i,icol=0,irow=0,j,k,l,ll;
    double big,dum,pivinv;
    
    for (j=1; j<=n; ++j)
      _ipiv[j] = 0;
    
    for (i=1; i<=n; ++i) {
      big = 0.0;
      for (j=1; j<=n; ++j)
	if (_ipiv[j] != 1)
	  for (k=1; k<=n; ++k) {
	    if (_ipiv[k] == 0) {
	      if (std::fabs(a[j][k]) >= big) {
		big = std::fabs(a[j][k]);
		irow = j;
		icol = k;
	      }
	    } else if (_ipiv[k] > 1) {
	      std::cerr << "gaussj: Singular Matrix-1\n";
	      return 1;
	    }
	  }
      
      ++(_ipiv[icol]);
      
      if (irow != icol) {
	for (l=1; l<=n; ++l) std::swap(a[irow][l], a[icol][l]);
	for (l=1; l<=m; ++l) std::swap(b[irow][l], b[icol][l]);
      }
      
      _indxr[i] = irow;
      _indxc[i] = icol;
      
      if (a[icol][icol] == 0.0) {
	std::cerr << "gaussj: Singular Matrix-2\n";
	return 2;
      }
      
      pivinv = 1.0/a[icol][icol];
      a[icol][icol] = 1.0;
      for (l=1; l<=n; ++l) a[icol][l] *= pivinv;
      for (l=1; l<=m; ++l) b[icol][l] *= pivinv;
      
      for (ll=1; ll<=n; ++ll)
	if (ll != icol) {
	  dum = a[ll][icol];
	  a[ll][icol] = 0.0;
	  for (l=1; l<=n; ++l) a[ll][l] -= a[icol][l]*dum;
	  for (l=1; l<=m; ++l) b[ll][l] -= b[icol][l]*dum;
	}
    }
    
    for (l=n; l>=1; --l) {
      if (_indxr[l] != _indxc[l])
	for (k=1; k<=n; ++k)
	  std::swap(a[k][_indxr[l]], a[k][_indxc[l]]);
    }
    
    return 0;
  }

 
  int likelihood_optimal_complex_gain_crosshand_visibilities::cholesky_solve(double **a, int n, const double rhs[], double x[])
  {
    // In-place Cholesky factorization of symmetric positive definite matrix a:
    // on exit, lower triangle contains L with a = L L^T.
    //
    // Uses _mrq_oneda[][1] as temporary storage for the forward-substitution vector.
    //
    // Returns 0 on success, nonzero on failure.
    
    for (int i=1; i<=n; ++i)
      {
	for (int j=i; j<=n; ++j)
	  {
	    double sum = a[j][i];
	    for (int k=1; k<i; ++k)
	      sum -= a[i][k] * a[j][k];
	    
	    if (j == i)
	      {
		if (!(sum > 0.0) || !std::isfinite(sum))
		  return 1;
		
		a[i][i] = std::sqrt(sum);
	      }
	    else
	      {
		a[j][i] = sum / a[i][i];
	      }
	  }
      }
    
    // Forward solve: L y = rhs
    for (int i=1; i<=n; ++i)
      {
	double sum = rhs[i];
	for (int k=1; k<i; ++k)
	  sum -= a[i][k] * _mrq_oneda[k][1];
	
	_mrq_oneda[i][1] = sum / a[i][i];
      }
    
    // Backward solve: L^T x = y
    for (int i=n; i>=1; --i)
      {
	double sum = _mrq_oneda[i][1];
	for (int k=i+1; k<=n; ++k)
	  sum -= a[k][i] * x[k];
	
	x[i] = sum / a[i][i];
      }
    
    return 0;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,dy;
    
    for (j=1; j<=ma; ++j)
      {
	beta[j] = 0.0;
	for (k=1; k<=ma; ++k)
	  alpha[j][k] = 0.0;
      }
    *chisq = 0.0;
    
    for (j=1; j<=ma; ++j)
      a[j] = std::min(std::max(a[j],-1.0e2),1.0e2);
    
    for (i=1; i<=ndata; ++i)
      {
	gain_optimization_likelihood(i, a, &ymod, _dyda);
	dy = y[i] - ymod;

	std::vector<int> idx;
	std::vector<double> val;
	idx.reserve(8);
	val.reserve(8);
	
	for (int p=1; p<=ma; ++p)
	  {
	    if (_dyda[p] != 0.0)
	      {
		idx.push_back(p);
		val.push_back(_dyda[p]);
	      }
	  }
	
	for (size_t q=0; q<idx.size(); ++q)
	  {
	    wt = val[q];
	    for (size_t r=0; r<=q; ++r)
	      alpha[idx[q]][idx[r]] += wt * val[r];
	    beta[idx[q]] += dy * wt;
	  }
    
	*chisq += dy*dy;
      }
    
    for (j=2; j<=ma; ++j)
      for (k=1; k<j; ++k)
	alpha[k][j] = alpha[j][k];
    
    for (size_t s=0; s<_sigma_g.size(); ++s)
      {
	const int i0 = 4*int(s) + 1;
	
	if (_hand_gain_mode == HandGainMode::Independent)
	  {
	    const double oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	    
	    const int aR = idx_logamp_R(s);
	    const int pR = idx_phase_R(s);
	    const int aL = idx_logamp_L(s);
	    const int pL = idx_phase_L(s);
	    
	    beta[aR] -= a[aR] * oSigma2;
	    beta[pR] -= a[pR] * _opi2;
	    beta[aL] -= a[aL] * oSigma2;
	    beta[pL] -= a[pL] * _opi2;
	    
	    alpha[aR][aR] += oSigma2;
	    alpha[pR][pR] += _opi2;
	    alpha[aL][aL] += oSigma2;
	    alpha[pL][pL] += _opi2;
	    
	    *chisq += a[aR]*a[aR]*oSigma2 + a[pR]*a[pR]*_opi2;
	    *chisq += a[aL]*a[aL]*oSigma2 + a[pL]*a[pL]*_opi2;
	  }
	else
	  {
	    const double oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	    const double oEta2   = 1.0/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
	    const double oDel2   = 1.0/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	    
	    const int ac = i0;
	    const int pc = i0+1;
	    const int et = i0+2;
	    const int de = i0+3;
	    
	    beta[ac] -= a[ac] * oSigma2;
	    beta[pc] -= a[pc] * _opi2;
	    beta[et] -= a[et] * oEta2;
	    beta[de] -= a[de] * oDel2;
	    
	    alpha[ac][ac] += oSigma2;
	    alpha[pc][pc] += _opi2;
	    alpha[et][et] += oEta2;
	    alpha[de][de] += oDel2;
	    
	    *chisq += a[ac]*a[ac]*oSigma2;
	    *chisq += a[pc]*a[pc]*_opi2;
	    *chisq += a[et]*a[et]*oEta2;
	    *chisq += a[de]*a[de]*oDel2;
	  }
      }

    double alpha_diag_max = 0.0;
    for (j=1; j<=ma; ++j)
      alpha_diag_max = std::max(alpha[j][j], alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max, 1.0);
    for (j=1; j<=ma; ++j)
      alpha[j][j] += 1.0e-12 * alpha_diag_max;
  }

  
  void likelihood_optimal_complex_gain_crosshand_visibilities::mrqcof_log(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,sig2i,dy;
    
    for (j=1; j<=ma; ++j)
      {
	beta[j] = 0.0;
	for (k=1; k<=ma; ++k)
	  alpha[j][k] = 0.0;
      }
    *chisq = 0.0;
    
    for (j=1; j<=ma; ++j)
      a[j] = std::min(std::max(a[j],-1.0e2),1.0e2);
    
    for (i=1; i<=ndata; ++i)
      {
	gain_optimization_log_likelihood(i, a, &ymod, _dyda);
	sig2i = 1.0/(sig[i]*sig[i]);
	dy = y[i] - ymod;
	
	std::vector<int> idx;
	std::vector<double> val;
	idx.reserve(8);
	val.reserve(8);
	
	for (int p=1; p<=ma; ++p)
	  {
	    if (_dyda[p] != 0.0)
	      {
		idx.push_back(p);
		val.push_back(_dyda[p]);
	      }
	  }
	
	for (size_t q=0; q<idx.size(); ++q)
	  {
	    wt = val[q];
	    for (size_t r=0; r<=q; ++r)
	      alpha[idx[q]][idx[r]] += wt * val[r];
	    beta[idx[q]] += dy * wt;
	  }

	*chisq += dy*dy*sig2i;
      }
    
    for (j=2; j<=ma; ++j)
      for (k=1; k<j; ++k)
	alpha[k][j] = alpha[j][k];
    
    for (size_t s=0; s<_sigma_g.size(); ++s)
      {
	const int i0 = 4*int(s) + 1;
	
	if (_hand_gain_mode == HandGainMode::Independent)
	  {
	    const double oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	    
	    const int aR = idx_logamp_R(s);
	    const int pR = idx_phase_R(s);
	    const int aL = idx_logamp_L(s);
	    const int pL = idx_phase_L(s);
	    
	    beta[aR] -= a[aR] * oSigma2;
	    beta[pR] -= a[pR] * _opi2;
	    beta[aL] -= a[aL] * oSigma2;
	    beta[pL] -= a[pL] * _opi2;
	    
	    alpha[aR][aR] += oSigma2;
	    alpha[pR][pR] += _opi2;
	    alpha[aL][aL] += oSigma2;
	    alpha[pL][pL] += _opi2;
	    
	    *chisq += a[aR]*a[aR]*oSigma2 + a[pR]*a[pR]*_opi2;
	    *chisq += a[aL]*a[aL]*oSigma2 + a[pL]*a[pL]*_opi2;
	  }
	else
	  {
	    const double oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	    const double oEta2   = 1.0/(sigma_ratio_logamp_eff_()*sigma_ratio_logamp_eff_());
	    const double oDel2   = 1.0/(sigma_ratio_phase_eff_()*sigma_ratio_phase_eff_());
	    
	    const int ac = i0;
	    const int pc = i0+1;
	    const int et = i0+2;
	    const int de = i0+3;
	    
	    beta[ac] -= a[ac] * oSigma2;
	    beta[pc] -= a[pc] * _opi2;
	    beta[et] -= a[et] * oEta2;
	    beta[de] -= a[de] * oDel2;
	    
	    alpha[ac][ac] += oSigma2;
	    alpha[pc][pc] += _opi2;
	    alpha[et][et] += oEta2;
	    alpha[de][de] += oDel2;
	    
	    *chisq += a[ac]*a[ac]*oSigma2;
	    *chisq += a[pc]*a[pc]*_opi2;
	    *chisq += a[et]*a[et]*oEta2;
	    *chisq += a[de]*a[de]*oDel2;
	  }
      }
  
    double alpha_diag_max = 0.0;
    for (j=1; j<=ma; ++j)
      alpha_diag_max = std::max(alpha[j][j], alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max, 1.0);
    for (j=1; j<=ma; ++j)
      alpha[j][j] += 1.0e-12 * alpha_diag_max;
  }


  void likelihood_optimal_complex_gain_crosshand_visibilities::covsrt(double **covar, int ma, int mfit)
  {
    (void)mfit; // in this class, mrqmin/mrqmin_log use mfit = ma
    
    for (int i = 1; i <= ma; ++i)
      for (int j = 1; j < i; ++j)
	covar[j][i] = covar[i][j];
  }


  int likelihood_optimal_complex_gain_crosshand_visibilities::mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;
    
    if (*alamda < 0.0)
      {
	*alamda = 0.001;
	mrqcof(y, ndata, a, ma, alpha, _mrq_beta, chisq);
	_mrq_ochisq = (*chisq);
	for (j=1; j<=ma; ++j)
	  _mrq_atry[j] = a[j];
      }
    
    for (j=1; j<=mfit; ++j)
      {
	for (k=1; k<=mfit; ++k)
	  covar[j][k] = alpha[j][k];
	covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
      }
    
    if (*alamda == 0.0)
      {
	// Keep the old robust path for the final covariance/inverse.
	for (j=1; j<=mfit; ++j)
	  _mrq_oneda[j][1] = _mrq_beta[j];
	
	if (gaussj(covar, mfit, _mrq_oneda, 1))
	  return 1;
	
	for (j=1; j<=mfit; ++j)
	  _mrq_da[j] = _mrq_oneda[j][1];
	
	covsrt(covar, ma, mfit);
	return 0;
      }
    
    // Hot path: use Cholesky solve for the damped LM step.
    if (cholesky_solve(covar, mfit, _mrq_beta, _mrq_da))
      {
	// Fallback: rebuild the matrix and use the old Gauss-Jordan solver.
	for (j=1; j<=mfit; ++j)
	  {
	    for (k=1; k<=mfit; ++k)
	      covar[j][k] = alpha[j][k];
	    covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
	    _mrq_oneda[j][1] = _mrq_beta[j];
	  }
	
	if (gaussj(covar, mfit, _mrq_oneda, 1))
	  return 1;
	
	for (j=1; j<=mfit; ++j)
	  _mrq_da[j] = _mrq_oneda[j][1];
      }
    
    for (l=1; l<=ma; ++l)
      _mrq_atry[l] = a[l] + _mrq_da[l];
    
    mrqcof(y, ndata, _mrq_atry, ma, covar, _mrq_da, chisq);
    
    if (*chisq < _mrq_ochisq)
      {
	*alamda *= 0.1;
	_mrq_ochisq = (*chisq);
	for (j=1; j<=mfit; ++j)
	  {
	    for (k=1; k<=mfit; ++k)
	      alpha[j][k] = covar[j][k];
	    _mrq_beta[j] = _mrq_da[j];
	  }
	for (l=1; l<=ma; ++l)
	  a[l] = _mrq_atry[l];
      }
    else
      {
	*alamda *= 10.0;
	*chisq = _mrq_ochisq;
      }
    
    return 0;
  }


  int likelihood_optimal_complex_gain_crosshand_visibilities::mrqmin_log(double y[], double sig[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;
    
    if (*alamda < 0.0)
      {
	*alamda = 0.001;
	mrqcof_log(y, sig, ndata, a, ma, alpha, _mrq_beta, chisq);
	_mrq_ochisq = (*chisq);
	for (j=1; j<=ma; ++j)
	  _mrq_atry[j] = a[j];
      }
    
    for (j=1; j<=mfit; ++j)
      {
	for (k=1; k<=mfit; ++k)
	  covar[j][k] = alpha[j][k];
	covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
      }
    
    if (*alamda == 0.0)
      {
	// Keep the old robust path for the final covariance/inverse.
	for (j=1; j<=mfit; ++j)
	  _mrq_oneda[j][1] = _mrq_beta[j];
	
	if (gaussj(covar, mfit, _mrq_oneda, 1))
	  return 1;
	
	for (j=1; j<=mfit; ++j)
	  _mrq_da[j] = _mrq_oneda[j][1];
	
	covsrt(covar, ma, mfit);
	return 0;
      }
    
    // Hot path: use Cholesky solve for the damped LM step.
    if (cholesky_solve(covar, mfit, _mrq_beta, _mrq_da))
      {
	// Fallback: rebuild the matrix and use the old Gauss-Jordan solver.
	for (j=1; j<=mfit; ++j)
	  {
	    for (k=1; k<=mfit; ++k)
	      covar[j][k] = alpha[j][k];
	    covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
	    _mrq_oneda[j][1] = _mrq_beta[j];
	  }
	
	if (gaussj(covar, mfit, _mrq_oneda, 1))
	  return 1;
	
	for (j=1; j<=mfit; ++j)
	  _mrq_da[j] = _mrq_oneda[j][1];
      }
    
    for (l=1; l<=ma; ++l)
      _mrq_atry[l] = a[l] + _mrq_da[l];
    
    mrqcof_log(y, sig, ndata, _mrq_atry, ma, covar, _mrq_da, chisq);
    
    if (*chisq < _mrq_ochisq)
      {
	*alamda *= 0.1;
	_mrq_ochisq = (*chisq);
	for (j=1; j<=mfit; ++j)
	  {
	    for (k=1; k<=mfit; ++k)
	      alpha[j][k] = covar[j][k];
	    _mrq_beta[j] = _mrq_da[j];
	  }
	for (l=1; l<=ma; ++l)
	  a[l] = _mrq_atry[l];
      }
    else
      {
	*alamda *= 10.0;
	*chisq = _mrq_ochisq;
      }
    
    return 0;
  }

  
  void likelihood_optimal_complex_gain_crosshand_visibilities::print_timing_summary(int mpi_rank) const
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
    
    std::cout << "\n===== Unconstrained crosshand gain likelihood timing summary (rank "
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
      std::cout << std::setw(24) << "DtermTotal"
		<< " : total = " << ms << " ms"
		<< ", calls = " << dterm_calls_
		<< ", avg = " << avg << " ms/call\n";
    }

    {
      const double ms = ogc_pack_trial_ns_ / 1.0e6;
      const double avg = (ogc_pack_trial_calls_ > 0) ? ms / double(ogc_pack_trial_calls_) : 0.0;
      std::cout << std::setw(24) << "OGCPackTrial"
		<< " : total = " << ms << " ms"
		<< ", calls = " << ogc_pack_trial_calls_
		<< ", avg = " << avg << " ms/call\n";
    }
    
    {
      const double ms = ogc_pack_log_ns_ / 1.0e6;
      const double avg = (ogc_pack_log_calls_ > 0) ? ms / double(ogc_pack_log_calls_) : 0.0;
      std::cout << std::setw(24) << "OGCPackLog"
		<< " : total = " << ms << " ms"
		<< ", calls = " << ogc_pack_log_calls_
		<< ", avg = " << avg << " ms/call\n";
    }
  
    std::cout << "===============================================================\n\n";
  }

  
};
