/*! 
  \file likelihood_optimal_complex_gain_constrained_crosshand_visibilities.cpp
  \author Avery E. Broderick, Roman Gold
  \date  March, 2020, March 2026
  \brief Implementation file for the likelihood_optimal_complex_gain_constrained_crosshand_visibilities likelihood class.
*/


#include "random_number_generator.h"

#include "likelihood_optimal_complex_gain_constrained_crosshand_visibilities.h"
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
  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(
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

  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
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

  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
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

  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(
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

  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
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

  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(data_crosshand_visibilities& data, model_crosshand_visibilities& model, uncertainty_crosshand_visibilities& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
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
  
  
  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::~likelihood_optimal_complex_gain_constrained_crosshand_visibilities()
  {
    for (size_t j=0; j<=2*_sigma_g.size(); ++j)
      delete[] _mrq_oneda[j];
    delete[] _mrq_oneda;
    
    delete[] _mrq_da;
    delete[] _mrq_beta;
    delete[] _mrq_atry;

        for (int i=1; i<=2*int(_sigma_g.size()); ++i)
    {
      delete[] _covar[i];
      delete[] _alpha[i];
    }
    delete[] _covar;
    delete[] _alpha;

    delete[] _g;
    delete[] _og;

    delete[] _sig;
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

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::check_station_codes()
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
	std::cerr << "WARNING: likelihood_optimal_complex_gain_constrained_crosshand_visibilities:\n"
		  << "    Station " << _data.datum(j).Station1 << " not in station_codes list.\n"
		  << '\n';
      if ( station2_in_station_codes==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_constrained_crosshand_visibilities:\n"
		  << "    Station " << _data.datum(j).Station2 << " not in station_codes list.\n"
		  << '\n';
    }
    for (size_t k=0; k<_station_codes.size(); ++k)
      if ( station_code_in_data[k]==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_constrained_crosshand_visibilities:\n"
		  << "    station code " << _station_codes[k] << " not used in data set.\n"
		  << '\n'; 

    if (_use_prior_gain_solutions)
      std::cerr << "WARNING: likelihood_optimal_complex_gain_constrained_crosshand_visibilities:\n"
		<< "     prior gain information is being used to solve for gains.  This potentially\n"
		<< "     can lead to non-deterministic, path-dependent behavior in the presence of\n"
		<< "     pathologically poorly defined gains.\n"
		<< '\n';

  }
  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::allocate_memory()
  {
    // Allocate space for marginalized gain corrections
    _G.resize(_tge.size()-1);
    for (size_t j=0; j<_tge.size()-1; ++j)
    {
      _G[j].resize(_sigma_g.size());

      for (size_t k=0; k<_sigma_g.size(); ++k)
	_G[j][k] = std::complex<double>(1.0,0.0);
    }
    _sqrt_detC.resize(_tge.size()-1);
    for (size_t j=0; j<_sqrt_detC.size(); ++j)
      _sqrt_detC[j] = 1.0;
	   
    int ma = 2*_sigma_g.size();
    _mrq_atry = new double[ma+1];
    _mrq_beta = new double[ma+1];
    _mrq_da = new double[ma+1];
    _mrq_oneda = new double*[ma+1];
    for (int j=0; j<=ma; j++)
      _mrq_oneda[j] = new double[2];

        _indx  = new int[ma+1];
    _indxc = new int[ma+1];
    _indxr = new int[ma+1];
    _ipiv  = new int[ma+1];
    _vv    = new double[ma+1];
    _dyda  = new double[ma+1];

    const int ndatamax = 8 * int(_data.size());
    _ogc_y   = new double[ndatamax+1];
    _ogc_yb  = new double[ndatamax+1];
    _ogc_is1 = new size_t[ndatamax+1];
    _ogc_is2 = new size_t[ndatamax+1];
    _sig     = new double[ndatamax+1];

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
 

  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(_Mcomm);
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::set_iteration_limit(int itermax)
  {
    _itermax=itermax;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::solve_for_gains()
  {
    _solve_for_gains = true;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::fix_gains()
  {
    _solve_for_gains = false;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::solve_for_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = true;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::fix_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = false;
  }
  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::use_prior_gain_solutions()
  {
    _use_prior_gain_solutions = true;
  }
  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::assume_smoothly_varying_gains()
  {
    _smoothly_varying_gains = true;
    _use_prior_gain_solutions = false;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::assume_independently_varying_gains()
  {
    _smoothly_varying_gains = false;
    _use_prior_gain_solutions = false;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::output(std::ostream& out)
  {
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
	    
	    const std::complex<double> G12 = _G[epoch][_is1_list[epoch][jj]] * std::conj(_G[epoch][_is2_list[epoch][jj]]);
	    
	    for (size_t k=0; k<4; ++k)
	      cvo[k] = G12 * cvo[k];
	    
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
  }

  size_t likelihood_optimal_complex_gain_constrained_crosshand_visibilities::number_of_independent_gains()
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


  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::organize_data_lists()
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


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::operator()(std::vector<double>& x)
  {
    if (_parallelize_likelihood)
      return likelihood_multiproc(x);
    else
      return likelihood_uniproc(x);
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_multiproc(std::vector<double>& x)
  {
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
		    for (size_t a=0; a<_sigma_g.size(); ++a)
		      _G[epoch][a] = std::complex<double>(1.0,0.0);
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
	    const std::complex<double> G12 =
	      _G[epoch][is1[ii]] * std::conj(_G[epoch][is2[ii]]);
	    
	    const std::complex<double> GGybrr = G12 * yb[0][ii];
	    const std::complex<double> GGybll = G12 * yb[1][ii];
	    const std::complex<double> GGybrl = G12 * yb[2][ii];
	    const std::complex<double> GGyblr = G12 * yb[3][ii];
	    
	    dL += -0.5 * ( std::pow( y[0][ii].real() - GGybrr.real(), 2) + std::pow( y[0][ii].imag() - GGybrr.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[1][ii].real() - GGybll.real(), 2) + std::pow( y[1][ii].imag() - GGybll.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[2][ii].real() - GGybrl.real(), 2) + std::pow( y[2][ii].imag() - GGybrl.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[3][ii].real() - GGyblr.real(), 2) + std::pow( y[3][ii].imag() - GGyblr.imag(), 2) );
	  }
	
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  {
	    const double G  = std::log(std::abs(_G[epoch][a]));
	    const double ph = std::arg(_G[epoch][a]);
	    dL += -0.5*G*G/(_sigma_g[a]*_sigma_g[a]);
	    dL += -0.5*ph*ph*_opi2;
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
  

  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_uniproc(std::vector<double>& x)
  {
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
			  _G[epoch][a] = std::complex<double>(1.0,0.0);
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
	    const std::complex<double> G12 = _G[epoch][is1[ii]] * std::conj(_G[epoch][is2[ii]]);
	    
	    const std::complex<double> GGybrr = G12 * yb[0][ii];
	    const std::complex<double> GGybll = G12 * yb[1][ii];
	    const std::complex<double> GGybrl = G12 * yb[2][ii];
	    const std::complex<double> GGyblr = G12 * yb[3][ii];
	    
	    dL += -0.5 * ( std::pow( y[0][ii].real() - GGybrr.real(), 2) + std::pow( y[0][ii].imag() - GGybrr.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[1][ii].real() - GGybll.real(), 2) + std::pow( y[1][ii].imag() - GGybll.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[2][ii].real() - GGybrl.real(), 2) + std::pow( y[2][ii].imag() - GGybrl.imag(), 2) );
	    dL += -0.5 * ( std::pow( y[3][ii].real() - GGyblr.real(), 2) + std::pow( y[3][ii].imag() - GGyblr.imag(), 2) );
	  }
	
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  {
	    const double G  = std::log(std::abs(_G[epoch][a]));
	    const double ph = std::arg(_G[epoch][a]);
	    dL += -0.5*G*G/(_sigma_g[a]*_sigma_g[a]);
	    dL += -0.5*ph*ph*_opi2;
	  }
	
	dL += std::log(marg_term);
	dL += lognorm;
	
	L += dL;
      }
    
    _L_last = L;
    return L;
  }

  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::distribute_gains()
  {
    size_t N = 2*(_tge.size()-1)*_sigma_g.size() + (_tge.size()-1);
    double *local_buff = new double[N];
    double *global_buff = new double[N];
    memset(local_buff,0.0,N*sizeof(double));
    memset(global_buff,0.0,N*sizeof(double));
    int i=0;
    for (size_t j=0; j<_tge.size()-1; ++j)
    {
      if (j%_L_size==size_t(_L_rank))
      {
	for (size_t k=0; k<_sigma_g.size(); ++k)
	{
	  local_buff[i++] = _G[j][k].real();
	  local_buff[i++] = _G[j][k].imag();
	}
	local_buff[i++] = _sqrt_detC[j];
      }
      else
	i += 2*_sigma_g.size() + 1;
    }

    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
    MPI_Comm_size(MPI_COMM_WORLD,&wsize);
    int lrank, lsize;
    MPI_Comm_rank(_Lcomm,&lrank);
    MPI_Comm_size(_Lcomm,&lsize);
    int rank, size;
    MPI_Comm_rank(_comm,&rank);
    MPI_Comm_size(_comm,&size);
    
    MPI_Allreduce(local_buff,global_buff,N,MPI_DOUBLE,MPI_SUM,_Lcomm);

    
    i=0;
    for (size_t j=0; j<_tge.size()-1; ++j)
    {
      for (size_t k=0; k<_sigma_g.size(); ++k)
      {
	_G[j][k] = std::complex<double>(global_buff[i], global_buff[i+1]);
	i+=2;
      }
      _sqrt_detC[j] = global_buff[i++];
    }
    
    delete[] local_buff;
    delete[] global_buff;
  }


  std::vector<double> likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gradient(std::vector<double>& x, prior& Pr)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientTotal, timer_ns_, timer_calls_);
    return gradient_dispatch_(x, Pr);
  }

  
  std::vector<double> likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    return gradient_dispatch_(x, Pr);
  }

  
  std::vector<double> likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gradient_dispatch_(std::vector<double>& x, prior& Pr)
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

  
  std::vector<double> likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gradient_hybrid(std::vector<double>& x, prior& Pr, bool do_geom)
  {
    const double Lx = ((_x_last.empty() || x != _x_last) ? this->operator()(x) : _L_last);
    
    const bool solving_for_gains_prev = _solve_for_gains;
    if (!_solve_for_gains_during_gradient)
      fix_gains();
    
    auto restore_basepoint = [&]() {
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;
      for (size_t j=0; j<_model.size(); ++j) mx[j] = x[ii++];
      for (size_t j=0; j<_uncertainty.size(); ++j) ux[j] = x[ii++];
      _model.generate_model(mx);
      _uncertainty.generate_uncertainty(ux);
      _x_last = x;
      _L_last = Lx;
    };
    
    model_polarized_image_adaptive_splined_raster* direct_top = dynamic_cast<model_polarized_image_adaptive_splined_raster*>(&_model);
    
    model_polarized_image_sum* sum_top = dynamic_cast<model_polarized_image_sum*>(&_model);



    enum class Kind { Raster, ConstPolAsymGauss, Unsupported };


        struct CompInfo {
      model_polarized_image* img = nullptr;
      model_polarized_image_adaptive_splined_raster* r = nullptr;
      model_polarized_image_constant_polarization* cp = nullptr;

      bool analytic = false;
      bool analytic_requires_cache = false;
      bool use_analytic_now = false;

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

      std::vector<double> xfrac;
      std::vector<double> yfrac;
    };
	
    
    std::vector<CompInfo> comps;
    const size_t Npar = x.size();
    
    if (direct_top)
      {
	CompInfo c;
	c.img = direct_top;
	c.r = direct_top;
	c.cp = nullptr;
	c.analytic = true;
	c.analytic_requires_cache = true;
	c.use_analytic_now = true;
	
	c.p0 = 0;
	c.pend = direct_top->size();
	
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
	    CompInfo c;
	    c.img  = imgs[j];
            c.p0   = p;
            c.pend = p + imgs[j]->size();
            c.idx_xoff = c.pend;
            c.idx_yoff = c.pend + 1;
            c.xoff = xs[j];
            c.yoff = ys[j];
	    
            c.r  = dynamic_cast<model_polarized_image_adaptive_splined_raster*>(imgs[j]);
            c.cp = dynamic_cast<model_polarized_image_constant_polarization*>(imgs[j]);

            if (c.r != nullptr)
            {
              c.analytic = true;
              c.analytic_requires_cache = true;
              c.use_analytic_now = true;

              c.Nx = c.r->Nx();
              c.Ny = c.r->Ny();
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
            }
            else if (c.cp != nullptr && c.cp->analytic_asym_gaussian_supported())
            {
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
            p += imgs[j]->size() + 2;
	  }
      }
    else
      {
	std::vector<double> g = likelihood_base::gradient_uniproc(x, Pr);
	restore_basepoint();
	if (!_solve_for_gains_during_gradient && solving_for_gains_prev) solve_for_gains();
	return g;
      }


    for (auto& c : comps)
      {
	if (!c.use_analytic_now)
	  continue;
	
	if (!c.analytic_requires_cache)
	  continue;
	
	const model_polarized_image_adaptive_splined_raster* rc = c.r;
	
	if (!rc->use_cached_exp()) c.use_analytic_now = false;
	if (!rc->phase_cache_valid()) c.use_analytic_now = false;
	if (rc->cached_Nd() < _data.size()) c.use_analytic_now = false;
	
	if (rc->phase_cache().size() < _data.size() * c.Npix) c.use_analytic_now = false;
	if (rc->spline_kernel_cache().size() < _data.size()) c.use_analytic_now = false;
	
	if (rc->I_flat().size() < c.Npix) c.use_analytic_now = false;
	if (rc->Q_flat().size() < c.Npix) c.use_analytic_now = false;
	if (rc->U_flat().size() < c.Npix) c.use_analytic_now = false;
	if (rc->V_flat().size() < c.Npix) c.use_analytic_now = false;
	
	if (do_geom) {
	  if (rc->spline_kernel_dfovx_cache().size() < _data.size()) c.use_analytic_now = false;
	  if (rc->spline_kernel_dfovy_cache().size() < _data.size()) c.use_analytic_now = false;
	  if (rc->spline_kernel_dpa_cache().size()   < _data.size()) c.use_analytic_now = false;
	}
      }
    
    auto apply_top_dterms = [&](datum_crosshand_visibilities& d, std::complex<double>* io)
    {
      if (sum_top) {
	sum_top->apply_Dterms_linear(d, io);
      } else {
	direct_top->apply_Dterms_linear(d, io);
      }
    };
  
    auto apply_top_dterms_param_jac = [&](const datum_crosshand_visibilities& d, const std::complex<double>* in, std::complex<double> deriv[8][4])
    {
      if (sum_top) {
	sum_top->apply_Dterms_parameter_jacobian_linear(d, in, deriv);
      } else {
	direct_top->apply_Dterms_parameter_jacobian_linear(d, in, deriv);
      }
    };

    const bool have_model_dterms = (sum_top ? sum_top->modeling_Dterms() : direct_top->modeling_Dterms());
    
    const size_t dterm_p0 = (have_model_dterms ? (_model.size() - 4*_station_codes.size()) : 0);

    auto station_index = [&](const std::string& s) -> size_t
    {
      return size_t(std::find(_station_codes.begin(), _station_codes.end(), s) - _station_codes.begin());
    };

    std::vector<double> grad_local(Npar, 0.0);
    std::vector<char> analytic_mask(Npar, 0);

    for (const auto& c : comps)
      {
        if (!c.use_analytic_now)
          continue;

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

        if (c.cp != nullptr)
	  {
	    analytic_mask[c.p0 + 0] = 1; // Itotal
	    analytic_mask[c.p0 + 1] = 1; // sigma
	    analytic_mask[c.p0 + 2] = 1; // A
	    analytic_mask[c.p0 + 3] = 1; // PA
	    analytic_mask[c.p0 + 4] = 1; // pfrac
	    analytic_mask[c.p0 + 5] = 1; // EVPA
	    analytic_mask[c.p0 + 6] = 1; // mu
	    
	      if (sum_top) {
		analytic_mask[c.idx_xoff] = 1;
		analytic_mask[c.idx_yoff] = 1;
	      }
	  }
      }
        
    if (have_model_dterms)
      for (size_t q = 0; q < 4*_station_codes.size(); ++q)
	analytic_mask[dterm_p0 + q] = 1;
    
    const double two_pi = 2.0 * M_PI;
    const std::complex<double> Iunit(0.0, 1.0);
    const std::complex<double> minus_i(0.0, -1.0);
    
    {
      Themis::utils::ScopedTimer T(Themis::utils::TimerID::GradientAnalytic, timer_ns_, timer_calls_);
      
      const size_t Nep = _tge.size() - 1;
      for (size_t epoch = 0; epoch < Nep; ++epoch)
	{
	  if (epoch % _L_size != size_t(_L_rank))
	    continue;
	  
	  const auto& idx_list = _datum_index_list[epoch];
	  const auto& is1_list = _is1_list[epoch];
	  const auto& is2_list = _is2_list[epoch];
	  
	  for (size_t ii = 0; ii < idx_list.size(); ++ii)
	    {
	      const size_t j = idx_list[ii];
	      datum_crosshand_visibilities& d = _data.datum(j);
	      
	      std::vector<std::complex<double>> err = _uncertainty.error(d);
	      
	      bool bad_err = false;
	      for (int h = 0; h < 4; ++h)
		if (err[h].real() == 0.0 || err[h].imag() == 0.0)
		  bad_err = true;
	      if (bad_err)
		continue;
	      
	      const double acc = 0.25 * std::sqrt(std::abs(err[0]*err[0]) + std::abs(err[1]*err[1]));
	      
	      std::complex<double> base_pred[4];
	      _model.fill_crosshand_visibilities(j, d, acc, base_pred);
	      
	      const std::complex<double> G12 = _G[epoch][is1_list[ii]] * std::conj(_G[epoch][is2_list[ii]]);
	      
	      std::complex<double> pred[4];
	      for (int h = 0; h < 4; ++h)
		pred[h] = G12 * base_pred[h];
	      
	      const std::complex<double> data_h[4] = { d.RR, d.LL, d.RL, d.LR };
	      double rr[4], ri[4];
	      for (int h = 0; h < 4; ++h) {
		rr[h] = data_h[h].real()/err[h].real() - pred[h].real()/err[h].real();
		ri[h] = data_h[h].imag()/err[h].imag() - pred[h].imag()/err[h].imag();
	      }
	      
	      std::complex<double> base0_total[4] = {
		std::complex<double>(0.0,0.0),
		std::complex<double>(0.0,0.0),
		std::complex<double>(0.0,0.0),
		std::complex<double>(0.0,0.0)
	      };
	      
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
		  if (!c.use_analytic_now)
		    {
		      if (sum_top)
			{
			  const std::complex<double> shift_phase = std::exp(-2.0*M_PI*Iunit * (c.xoff*(-u) + c.yoff*v));

			  std::complex<double> tmp[4];
			  c.img->fill_crosshand_visibilities(j, d, acc, tmp);
			  
			  for (int h = 0; h < 4; ++h)
			    base0_total[h] += shift_phase * tmp[h];
			}
		      
		      continue;
		    }
		  
		  if (c.cp != nullptr)
		    {
		      const std::complex<double> shift_phase =
			std::exp(-2.0*M_PI*Iunit * (c.xoff*(-u) + c.yoff*v));
		      
		      std::complex<double> raw[4];
		      std::complex<double> draw[7][4];
		      c.cp->visibility_and_derivatives(d, raw, draw);
		      
		      std::complex<double> compvec[4];
		      for (int h = 0; h < 4; ++h) {
			compvec[h] = shift_phase * raw[h];
			base0_total[h] += compvec[h];
		      }
		      
		      auto add_comp_deriv = [&](size_t pidx, const std::complex<double>* d0)
		      {
			std::complex<double> dvec[4];
			for (int h = 0; h < 4; ++h)
			  dvec[h] = shift_phase * d0[h];
			
			apply_Dmat(dvec);
			for (int h = 0; h < 4; ++h)
			  dvec[h] *= G12;
			
			double contrib = 0.0;
			for (int h = 0; h < 4; ++h) {
			  contrib += rr[h] * (dvec[h].real()/err[h].real()) + ri[h] * (dvec[h].imag()/err[h].imag());
			}
			grad_local[pidx] += contrib;
		      };
		      
		      add_comp_deriv(c.p0 + 0, draw[0]);  // Itotal
		      add_comp_deriv(c.p0 + 4, draw[4]);  // pfrac
		      add_comp_deriv(c.p0 + 5, draw[5]);  // EVPA
		      add_comp_deriv(c.p0 + 6, draw[6]);  // mu
		      
		      if (do_geom) {
			add_comp_deriv(c.p0 + 1, draw[1]); // sigma
			add_comp_deriv(c.p0 + 2, draw[2]); // A
			add_comp_deriv(c.p0 + 3, draw[3]); // PA
		      }
		      
		      if (sum_top && do_geom)
			{
			  auto add_offset = [&](size_t pidx, const std::complex<double>& fac)
			  {
			    std::complex<double> dvec[4];
			    dvec[0] = fac * compvec[0];
			    dvec[1] = fac * compvec[1];
			    dvec[2] = fac * compvec[2];
			    dvec[3] = fac * compvec[3];
			    
			    apply_Dmat(dvec);
			    for (int h = 0; h < 4; ++h)
			      dvec[h] *= G12;
			    
			    double contrib = 0.0;
			    for (int h = 0; h < 4; ++h) {
			      contrib += rr[h] * (dvec[h].real()/err[h].real())	+ ri[h] * (dvec[h].imag()/err[h].imag());
			    }
			    grad_local[pidx] += contrib;
			  };
			  
			  add_offset(c.idx_xoff, (+two_pi * Iunit * u));
			  add_offset(c.idx_yoff, (-two_pi * Iunit * v));
			}
		      
		      continue;
		    }
		  
		  auto& r = *c.r;
		  const auto& phase = r.phase_cache();
		  const auto& Kc    = r.spline_kernel_cache();
		  const auto& Iflat = r.I_flat();
		  const auto& Qflat = r.Q_flat();
		  const auto& Uflat = r.U_flat();
		  const auto& Vflat = r.V_flat();
		  
		  const size_t off = j * c.Npix;
		  const double K   = Kc[j];
		  
		  const double pa   = x[c.idx_pa];
		  const double fovx = x[c.idx_fovx];
		  const double fovy = x[c.idx_fovy];
		  const double cpa  = std::cos(pa);
		  const double spa  = std::sin(pa);
		  const double ur   =  cpa*u + spa*v;
		  const double vr   = -spa*u + cpa*v;
		  
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
		      
		      const std::complex<double> dSI_dpa = (minus_i * two_pi) * ((vr * fovx) * SIx - (ur * fovy) * SIy);
		      const std::complex<double> dSQ_dpa = (minus_i * two_pi) * ((vr * fovx) * SQx - (ur * fovy) * SQy);
		      const std::complex<double> dSU_dpa = (minus_i * two_pi) * ((vr * fovx) * SUx - (ur * fovy) * SUy);
		      const std::complex<double> dSV_dpa = (minus_i * two_pi) * ((vr * fovx) * SVx - (ur * fovy) * SVy);
		      
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
			
			apply_Dmat(dvec);
			for (int h = 0; h < 4; ++h)
			  dvec[h] *= G12;
			
			double contrib = 0.0;
			for (int h = 0; h < 4; ++h) {
			  contrib += rr[h] * (dvec[h].real()/err[h].real()) + ri[h] * (dvec[h].imag()/err[h].imag());
			}
			grad_local[pidx] += contrib;
		      };
		      
		      add_geom(c.idx_fovx, dKx[j], dSI_dfovx, dSQ_dfovx, dSU_dfovx, dSV_dfovx);
		      add_geom(c.idx_fovy, dKy[j], dSI_dfovy, dSQ_dfovy, dSU_dfovy, dSV_dfovy);
		      add_geom(c.idx_pa,   dKp[j], dSI_dpa,   dSQ_dpa,   dSU_dpa,   dSV_dpa);
		    }
		  
		  const std::complex<double> VI0 = shift_phase * (K * SI0);
		  const std::complex<double> VQ0 = shift_phase * (K * SQ0);
		  const std::complex<double> VU0 = shift_phase * (K * SU0);
		  const std::complex<double> VV0 = shift_phase * (K * SV0);
		  
		  std::complex<double> compvec[4];
		  compvec[0] = VI0 + VV0;
		  compvec[1] = VI0 - VV0;
		  compvec[2] = VQ0 + Iunit*VU0;
		  compvec[3] = VQ0 - Iunit*VU0;
		  
		  for (int h = 0; h < 4; ++h)
		    base0_total[h] += compvec[h];
		  
		  if (sum_top)
		    {
		      auto add_offset = [&](size_t pidx, const std::complex<double>& fac)
		      {
			std::complex<double> dvec[4];
			dvec[0] = fac * compvec[0];
			dvec[1] = fac * compvec[1];
			dvec[2] = fac * compvec[2];
			dvec[3] = fac * compvec[3];
			
			apply_Dmat(dvec);
			for (int h = 0; h < 4; ++h)
			  dvec[h] *= G12;
			
			double contrib = 0.0;
			for (int h = 0; h < 4; ++h) {
			  contrib += rr[h] * (dvec[h].real()/err[h].real())	+ ri[h] * (dvec[h].imag()/err[h].imag());
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
			
			apply_Dmat(dvec);
			for (int h = 0; h < 4; ++h)
			  dvec[h] *= G12;
	    
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
	      
	      if (have_model_dterms)
		{
		  const size_t s1 = station_index(d.Station1);
		  const size_t s2 = station_index(d.Station2);
		  
		  const size_t pD[8] = {
		    dterm_p0 + 4*s1 + 0,
		    dterm_p0 + 4*s1 + 1,
		    dterm_p0 + 4*s1 + 2,
		    dterm_p0 + 4*s1 + 3,
		    dterm_p0 + 4*s2 + 0,
		    dterm_p0 + 4*s2 + 1,
		    dterm_p0 + 4*s2 + 2,
		    dterm_p0 + 4*s2 + 3
		  };
		  
		  std::complex<double> dD[8][4];
		  apply_top_dterms_param_jac(d, base0_total, dD);
		  
		  for (int q = 0; q < 8; ++q)
		    {
		      double contrib = 0.0;
		      for (int h = 0; h < 4; ++h)
			{
			  const std::complex<double> z = G12 * dD[q][h];
			  contrib += rr[h] * (z.real()/err[h].real()) + ri[h] * (z.imag()/err[h].imag());
			}
		      grad_local[pD[q]] += contrib;
		    }
		}
	    }
	}
      
    } // Scoped Timer
    
    std::vector<double> grad(Npar, 0.0);
    MPI_Allreduce(grad_local.data(), grad.data(), int(Npar), MPI_DOUBLE, MPI_SUM, _Lcomm);
    
    std::vector<double> y = x;
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
    
    restore_basepoint();
    
    if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
      solve_for_gains();
    
    return grad;
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::chi_squared(std::vector<double>& x)
  {
    distribute_gains();

    //_model.generate_model(x);
    // Make sure that model and uncertainty are properly generated
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);

    // Log-likelihood accumulator
    double L = 0;

    // Remove the prior?
    std::vector<double> true_sigma_g = _sigma_g;
    _sigma_g.assign(_sigma_g.size(),1000.0);


    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector< std::complex<double> > ybrr,ybll,ybrl,yblr;
      std::vector< std::complex<double> > yrr,yll,yrl,ylr;      
      std::vector< std::vector< std::complex<double> > > yb, y;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	// std::complex<double> RRerr = _data.datum(_datum_index_list[epoch][i]).RRerr;
	// std::complex<double> LLerr = _data.datum(_datum_index_list[epoch][i]).LLerr;
	// std::complex<double> RLerr = _data.datum(_datum_index_list[epoch][i]).RLerr;
	// std::complex<double> LRerr = _data.datum(_datum_index_list[epoch][i]).LRerr;
	// std::vector< std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(_datum_index_list[epoch][i]),0.25*std::sqrt(std::abs(RRerr*RRerr)+std::abs(LLerr*LLerr)));

	size_t j = _datum_index_list[epoch][i];
	
	std::vector< std::complex<double> > err = _uncertainty.error(_data.datum(j));
	std::vector< std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(j),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));

	ybrr.push_back( std::complex<double>(cvo[0].real()/err[0].real(), cvo[0].imag()/err[0].imag()) );
	ybll.push_back( std::complex<double>(cvo[1].real()/err[1].real(), cvo[1].imag()/err[1].imag()) );
	ybrl.push_back( std::complex<double>(cvo[2].real()/err[2].real(), cvo[2].imag()/err[2].imag()) );
	yblr.push_back( std::complex<double>(cvo[3].real()/err[3].real(), cvo[3].imag()/err[3].imag()) );

	yrr.push_back( std::complex<double>(_data.datum(j).RR.real()/err[0].real(), _data.datum(j).RR.imag()/err[0].imag()) );
	yll.push_back( std::complex<double>(_data.datum(j).LL.real()/err[1].real(), _data.datum(j).LL.imag()/err[1].imag()) );
	yrl.push_back( std::complex<double>(_data.datum(j).RL.real()/err[2].real(), _data.datum(j).RL.imag()/err[2].imag()) );
	ylr.push_back( std::complex<double>(_data.datum(j).LR.real()/err[3].real(), _data.datum(j).LR.imag()/err[3].imag()) );
      }

      yb.push_back( ybrr );
      yb.push_back( ybll );
      yb.push_back( ybrl );
      yb.push_back( yblr );

      y.push_back( yrr );
      y.push_back( yll );
      y.push_back( yrl );
      y.push_back( ylr );
      
      // y.push_back( _yrr_list[epoch] );
      // y.push_back( _yll_list[epoch] );
      // y.push_back( _yrl_list[epoch] );
      // y.push_back( _ylr_list[epoch] );

      is1 = _is1_list[epoch];
      is2 = _is2_list[epoch];

      if (_solve_for_gains)
      {
	// Determine the initial guess for the gains based on currently stated assumptions.
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
		_G[epoch][a] = std::complex<double>(1.0,0.0);
	    }
	  }
	}

	// Levenberg-Marquardt minimization of full likelihood
	optimal_complex_gains(y,yb,is1,is2,_G[epoch]);
      }

      // Add in the direct likelihood
      double dL = 0.0;
      for (size_t i=0; i<y[0].size(); ++i)
      {
	std::complex<double> GGybrr=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[0][i];
	std::complex<double> GGybll=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[1][i];
	std::complex<double> GGybrl=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[2][i];
	std::complex<double> GGyblr=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[3][i];
	dL += -0.5 * ( std::pow( y[0][i].real() - GGybrr.real(), 2) + std::pow( y[0][i].imag() - GGybrr.imag(), 2) );
	dL += -0.5 * ( std::pow( y[1][i].real() - GGybll.real(), 2) + std::pow( y[1][i].imag() - GGybll.imag(), 2) );
	dL += -0.5 * ( std::pow( y[2][i].real() - GGybrl.real(), 2) + std::pow( y[2][i].imag() - GGybrl.imag(), 2) );
	dL += -0.5 * ( std::pow( y[3][i].real() - GGyblr.real(), 2) + std::pow( y[3][i].imag() - GGyblr.imag(), 2) );
      }

      L += dL;
    }

    // Reset the prior
    _sigma_g = true_sigma_g;

    return (-2.0*L);
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::matrix_determinant(double **a)
  {
    const int n = int(_sigma_g.size());
    double d;
    
    ludcmp(a, n, _indx, d);
    
    for (int i=1; i<=n; ++i)
      d *= a[i][i];
    
    return d;
  }


#define TINY 1.0e-20;
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::ludcmp(double **a, int n, int *indx, double &d)
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


  std::vector<double> likelihood_optimal_complex_gain_constrained_crosshand_visibilities::get_gain_times()
  {
    return ( _tge );
  }

  std::vector< std::vector< std::complex<double> > > likelihood_optimal_complex_gain_constrained_crosshand_visibilities::get_gains()
  {
    return ( _G );
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::read_gain_file(std::string gain_file_name)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
    {
      std::ifstream in(gain_file_name);

      // Remove headers, which are not needed
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
	  _G[epoch][a] = std::complex<double>(Gr,Gi);
	}

	if (in.eof()==true)
	{
	  std::cerr << "ERROR: likelihood_optimal_complex_gain_constrained_crosshand_visibilities::read_gain_file too few gains in " << gain_file_name << '\n';
	  std::exit(1);
	}
      }
    }

    size_t ngains=2*_sigma_g.size()*(_tge.size()-1);
    double *buff = new double[ngains];
    for (size_t epoch=0,k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
      {
	buff[k++] = _G[epoch][a].real();
	buff[k++] = _G[epoch][a].imag();
      }
    MPI_Bcast(buff,ngains,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (size_t epoch=0,k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
	{
	  _G[epoch][a] = std::complex<double>(buff[k],buff[k+1]);
	  k+=2;
	}
    delete[] buff;

    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      _sqrt_detC[epoch] = 1.0;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::output_gains(std::ostream& out)
  {
    int nprec = out.precision();
    out.precision(20);    
    out << "# First epoch time at " << _tge[0] << " s from J2000.  All subsequent times measured from this value.\n";
    out.precision(nprec);
    out << "# Number of independent gains: " << number_of_independent_gains() << '\n';
    out << "#" << std::setw(14) << "start time (s)"
	<< std::setw(15) << "end time (s)";
    for (size_t a=0; a<_sigma_g.size(); ++a)
      out << std::setw(15) << _station_codes[a]+".real"
	  << std::setw(15) << _station_codes[a]+".imag";
    out << '\n';
      
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      out << std::setw(15) << _tge[epoch]-_tge[0]
	  << std::setw(15) << _tge[epoch+1]-_tge[0];
      for (size_t a=0; a<_sigma_g.size(); ++a)
	out << std::setw(15) << _G[epoch][a].real()
	    << std::setw(15) << _G[epoch][a].imag();
      out << '\n';
    } 
  }
  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::output_gains(std::string outname)
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

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::output_gain_corrections(std::ostream& out)
  {
    int nprec = out.precision();
    out.precision(20);    
    out << "# First epoch time at " << _tge[0] << " s from J2000.  All subsequent times measured from this value.\n";
    out.precision(nprec);
    out << "# Number of independent gains: " << number_of_independent_gains() << '\n';
    out << "#" << std::setw(14) << "start time (s)"
	<< std::setw(15) << "end time (s)";
    for (size_t a=0; a<_sigma_g.size(); ++a)
      out << std::setw(15) << _station_codes[a];
    out << '\n';
      
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      out << std::setw(15) << _tge[epoch]-_tge[0]
	  << std::setw(15) << _tge[epoch+1]-_tge[0];
      for (size_t a=0; a<_sigma_g.size(); ++a)
	out << std::setw(15) << std::abs(_G[epoch][a])-1.0;
      out << '\n';
    } 
  }
  
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::output_gain_corrections(std::string outname)
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


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::optimal_complex_gains(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveTotal, timer_ns_, timer_calls_);
    
    std::vector<std::complex<double> > gest_best = gest;
    double marg_best = -1;
    double chisq_best = std::numeric_limits<double>::infinity();
    
    std::vector<std::complex<double> > gest_try = gest;
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
    if (marg_try>0)
      if (chisq_try<chisq_best)
	{
	  gest_best = gest_try;
	  marg_best = marg_try;
	  chisq_best = chisq_try;
	}
    
    if (marg_best<0)
      {
	for (size_t j=0; j<gest.size(); ++j)
	  gest[j] = std::complex<double>(1.0,0.0);
	return -1;
      }
    
    gest = gest_best;
    return marg_best;
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::optimal_complex_gains_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveTrial, timer_ns_, timer_calls_);
    
    int ndata = int( 2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()) );
    if (ndata==0)
      return 1.0;
    
    for (size_t i=0, j=1; i<y[0].size(); ++i)
      {
	_ogc_y[j]   = y[0][i].real();   _ogc_yb[j]   = yb[0][i].real();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[0][i].imag();   _ogc_yb[j]   = yb[0][i].imag();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[1][i].real();   _ogc_yb[j]   = yb[1][i].real();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[1][i].imag();   _ogc_yb[j]   = yb[1][i].imag();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[2][i].real();   _ogc_yb[j]   = yb[2][i].real();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[2][i].imag();   _ogc_yb[j]   = yb[2][i].imag();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[3][i].real();   _ogc_yb[j]   = yb[3][i].real();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
	_ogc_y[j]   = y[3][i].imag();   _ogc_yb[j]   = yb[3][i].imag();   _ogc_is1[j] = is1[i]; _ogc_is2[j] = is2[i]; ++j;
      }
    
    const int ma = 2*_sigma_g.size();
    
    for (int i=0, j=1; i<int(gest.size()); ++i)
      {
	_g[j++] = std::log(std::abs(gest[i]));
	_g[j++] = std::arg(gest[i]);
      }
    
    double alambda = -1.0;
    double chisq = 0.0, ochisq, dg2;
    double dg2limit = 0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += _sigma_g[i]*_sigma_g[i];
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
    
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
      {
	const double gmag = std::exp(_g[j]);
	gest[i] = gmag * std::exp(std::complex<double>(0.0,1.0)*_g[j+1]);
      }
    
    double detC = matrix_determinant(_covar);
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;
    
    chisq_opt = chisq;
    return std::sqrt(detC);
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::optimal_complex_gains_log_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    Themis::utils::ScopedTimer T(Themis::utils::TimerID::GainsSolveLogTrial, timer_ns_, timer_calls_);
    
    int ndata = int( 2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()) );
    if (ndata==0)
      return 1.0;
    
    std::complex<double> tmp;
    for (size_t i=0, j=1; i<y[0].size(); ++i)
      {
	tmp = y[0][i]/yb[0][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[0][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[0][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	
	tmp = y[1][i]/yb[1][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[1][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[1][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	
	tmp = y[2][i]/yb[2][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[2][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[2][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	
	tmp = y[3][i]/yb[3][i]; if (std::abs(tmp)<1e-6) tmp = 1e-6;
	_ogc_y[j] = std::log(tmp).real(); _sig[j] = 1.0/std::abs(y[3][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
	_ogc_y[j] = std::log(tmp).imag(); _sig[j] = 1.0/std::abs(y[3][i]); _ogc_yb[j]=0.0; _ogc_is1[j]=is1[i]; _ogc_is2[j]=is2[i]; ++j;
      }
    
    const int ma = 2*_sigma_g.size();
    
    for (int i=0, j=1; i<int(gest.size()); ++i)
      {
	_g[j++] = std::log(std::abs(gest[i]));
	_g[j++] = std::arg(gest[i]);
      }
    
    double alambda = -1.0;
    double chisq = 0.0, ochisq, dg2;
    const double ch2limit = 1.0e-8;
    double dg2limit = 0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += _sigma_g[i]*_sigma_g[i];
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
    
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
      {
	const double gmag = std::exp(_g[j]);
	gest[i] = gmag * std::exp(std::complex<double>(0.0,1.0)*_g[j+1]);
      }
    
    double detC = matrix_determinant(_covar);
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;
    
    chisq_opt = chisq;
    return std::sqrt(detC);
  }


  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    // Gain corrected model value
    std::complex<double> e1 = std::exp(std::complex<double>(0.0,1.0)*g[2*_ogc_is1[i]+2]);
    std::complex<double> e2 = std::exp(std::complex<double>(0.0,1.0)*g[2*_ogc_is2[i]+2]);
    std::complex<double> G1 = std::exp(g[2*_ogc_is1[i]+1])*e1;
    std::complex<double> G2 = std::exp(g[2*_ogc_is2[i]+1])*e2;

    // Derivatives
    for (size_t a=1; a<=2*_sigma_g.size(); ++a) // Factor of 2 from real,imag
      dydg[a] = 0.0;

    // If i is odd, this is a real point, if i is even it is an imag point
    std::complex<double> yc, yb, ytest;
    if (i%2==1) // Real
    {
      yb = std::complex<double>( _ogc_yb[i], _ogc_yb[i+1] );
      ytest = std::complex<double>( _ogc_y[i], _ogc_y[i+1] );
      yc = G1*std::conj(G2)*yb;

      (*y) = yc.real();

      // Derivatives
      dydg[2*_ogc_is1[i]+1] = ( yc ).real(); // 1 g
      dydg[2*_ogc_is1[i]+2] = ( std::complex<double>(0.0,1.0)*yc ).real(); // 1 phase
      dydg[2*_ogc_is2[i]+1] = ( yc ).real(); // 2 g
      dydg[2*_ogc_is2[i]+2] = ( -std::complex<double>(0.0,1.0)*yc ).real(); // 2 phase
    }
    else // Imag
    {
      yb = std::complex<double>( _ogc_yb[i-1], _ogc_yb[i] );
      ytest = std::complex<double>( _ogc_y[i-1], _ogc_y[i] );
      yc = G1*std::conj(G2)*yb;

      (*y) = yc.imag();

      // Derivatives
      dydg[2*_ogc_is1[i]+1] = ( yc ).imag(); // 1 g
      dydg[2*_ogc_is1[i]+2] = ( std::complex<double>(0.0,1.0)*yc ).imag(); // 1 phase
      dydg[2*_ogc_is2[i]+1] = ( yc ).imag(); // 2 g
      dydg[2*_ogc_is2[i]+2] = ( -std::complex<double>(0.0,1.0)*yc ).imag(); // 2 phase
    }
  }


  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gain_optimization_log_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    // Gain corrected model value
    double g1 = g[2*_ogc_is1[i]+1];
    double p1 = g[2*_ogc_is1[i]+2];
    double g2 = g[2*_ogc_is2[i]+1];
    double p2 = g[2*_ogc_is2[i]+2];

    // Derivatives
    for (size_t a=1; a<=2*_sigma_g.size(); ++a) // Factor of 2 from real,imag
      dydg[a] = 0.0;

    // If i is odd, this is a real point, if i is even it is an imag point
    if (i%2==1) // Real
    {
      // Value
      (*y) = g1+g2;

      // Derivatives
      dydg[2*_ogc_is1[i]+1] = 1.0; // 1 g
      dydg[2*_ogc_is2[i]+1] = 1.0; // 2 g
    }
    else // Imag
    {
      // Value
      (*y) = p1-p2;

      // Derivatives
      dydg[2*_ogc_is1[i]+2] = 1.0; // 1 phase
      dydg[2*_ogc_is2[i]+2] = -1.0; // 2 phase
    }
  }

  
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::covsrt(double **covar, int ma, int mfit)
  {
    int i,j,k;
    double swap;
    
    for (i=mfit+1;i<=ma;i++)
      for (j=1;j<=i;j++)
	covar[i][j]=covar[j][i]=0.0;
    k=mfit;
    for (j=ma;j>=1;j--) {
      for (i=1;i<=ma;i++)
	SWAP(covar[i][k],covar[i][j]);
      for (i=1;i<=ma;i++)
	SWAP(covar[k][i],covar[j][i]);
      k--;
    }
  }
  

  int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::gaussj(double **a, int n, double **b, int m)
  {
    int i,icol=0,irow=0,j,k,l,ll;
    double big,dum,pivinv,swap;
    
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
	for (l=1; l<=n; ++l) SWAP(a[irow][l],a[icol][l]);
	for (l=1; l<=m; ++l) SWAP(b[irow][l],b[icol][l]);
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
	  SWAP(a[k][_indxr[l]],a[k][_indxc[l]]);
    }
    
    return 0;
  }


  int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::cholesky_solve(double **a, int n, const double rhs[], double x[])
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


  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
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
	
	int idx[4];
	double val[4];
	int nact = 0;
	
	const int cand[4] = {
	  int(2*_ogc_is1[i] + 1),
	  int(2*_ogc_is1[i] + 2),
	  int(2*_ogc_is2[i] + 1),
	  int(2*_ogc_is2[i] + 2)
	};
	
	for (int t=0; t<4; ++t)
	  {
	    const int p = cand[t];
	    const double v = _dyda[p];
	    if (v == 0.0) continue;
	    
	    bool found = false;
	    for (int q=0; q<nact; ++q)
	      if (idx[q] == p) {
		val[q] += v;
		found = true;
		break;
	      }
	    
	    if (!found) {
	      idx[nact] = p;
	      val[nact] = v;
	      ++nact;
	    }
	  }
	
	for (int q=0; q<nact; ++q)
	  {
	    wt = val[q];
	    for (int r=0; r<=q; ++r)
	      alpha[idx[q]][idx[r]] += wt * val[r];
	    beta[idx[q]] += dy * wt;
	  }
	
	*chisq += dy*dy;
      }
    
    for (j=2; j<=ma; ++j)
      for (k=1; k<j; ++k)
	alpha[k][j] = alpha[j][k];
    
    double oSigma2;
    for (size_t s=0, jj=1; s<_sigma_g.size(); ++s, jj+=2)
      {
	oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	
	beta[jj]   -= a[jj]   * oSigma2;
	beta[jj+1] -= a[jj+1] * _opi2;
	
	alpha[jj][jj]     += oSigma2;
	alpha[jj+1][jj+1] += _opi2;
	
	*chisq += a[jj]*a[jj]*oSigma2 + a[jj+1]*a[jj+1]*_opi2;
      }
    
    double alpha_diag_max = 0.0;
    for (j=1; j<=ma; ++j)
      alpha_diag_max = std::max(alpha[j][j], alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max, 1.0);
    for (j=1; j<=ma; ++j)
      alpha[j][j] += 1.0e-12 * alpha_diag_max;
  }


  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqcof_log(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
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
	
	int idx[2];
	double val[2];
	int nact = 0;
	
	if (i%2==1)
	  {
	    idx[nact] = int(2*_ogc_is1[i] + 1);
	    val[nact] = _dyda[idx[nact]];
	    ++nact;
	    
	    const int p2 = int(2*_ogc_is2[i] + 1);
	    const double v2 = _dyda[p2];
	    if (p2 == idx[0])
	      val[0] += v2;
	    else {
	      idx[nact] = p2;
	      val[nact] = v2;
	      ++nact;
	    }
	  }
	else
	  {
	    idx[nact] = int(2*_ogc_is1[i] + 2);
	    val[nact] = _dyda[idx[nact]];
	    ++nact;
	    
	    const int p2 = int(2*_ogc_is2[i] + 2);
	    const double v2 = _dyda[p2];
	    if (p2 == idx[0])
	      val[0] += v2;
	    else {
	      idx[nact] = p2;
	      val[nact] = v2;
	      ++nact;
	    }
	  }
	
	for (int q=0; q<nact; ++q)
	  {
	    wt = val[q] * sig2i;
	    for (int r=0; r<=q; ++r)
	      alpha[idx[q]][idx[r]] += wt * val[r];
	    beta[idx[q]] += dy * wt;
	  }
	
	*chisq += dy*dy*sig2i;
      }
    
    for (j=2; j<=ma; ++j)
      for (k=1; k<j; ++k)
	alpha[k][j] = alpha[j][k];
    
    double oSigma2;
    for (size_t s=0, jj=1; s<_sigma_g.size(); ++s, jj+=2)
      {
	oSigma2 = 1.0/(_sigma_g[s]*_sigma_g[s]);
	
	beta[jj]   -= a[jj]   * oSigma2;
	beta[jj+1] -= a[jj+1] * _opi2;
	
	alpha[jj][jj]     += oSigma2;
	alpha[jj+1][jj+1] += _opi2;
	
	*chisq += a[jj]*a[jj]*oSigma2 + a[jj+1]*a[jj+1]*_opi2;
      }
    
    double alpha_diag_max = 0.0;
    for (j=1; j<=ma; ++j)
      alpha_diag_max = std::max(alpha[j][j], alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max, 1.0);
    for (j=1; j<=ma; ++j)
      alpha[j][j] += 1.0e-12 * alpha_diag_max;
  }


  int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
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


int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqmin_log(
    double y[], double sig[], int ndata, double a[], int ma,
    double **covar, double **alpha, double *chisq, double *alamda)
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

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::print_timing_summary(int mpi_rank) const
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
    
    std::cout << "\n===== Constrained crosshand gain likelihood timing summary (rank "
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
    
    std::cout << "===============================================================\n\n";
  }

  
};
