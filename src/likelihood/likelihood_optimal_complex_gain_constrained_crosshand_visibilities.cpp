/*! 
  \file likelihood_optimal_complex_gain_constrained_crosshand_visibilities.cpp
  \author Avery E. Broderick
  \date  March, 2020
  \brief Implementation file for the likelihood_optimal_complex_gain_constrained_crosshand_visibilities likelihood class.
*/


#include "random_number_generator.h"

#include "likelihood_optimal_complex_gain_constrained_crosshand_visibilities.h"
#include <cmath>

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>

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
  }
  
  
  likelihood_optimal_complex_gain_constrained_crosshand_visibilities::~likelihood_optimal_complex_gain_constrained_crosshand_visibilities()
  {
    for (size_t j=0; j<=2*_sigma_g.size(); ++j)
      delete[] _mrq_oneda[j];
    delete[] _mrq_oneda;
    
    delete[] _mrq_da;
    delete[] _mrq_beta;
    delete[] _mrq_atry;
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

    // DEBUG
    //std::cerr << "Started in output\n";


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


    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      for (size_t j=0; j<_datum_index_list[epoch].size(); ++j)
      {

	// DEBUG
	//std::cerr << "  on " << epoch << " : " << j << '\n';

	size_t i = _datum_index_list[epoch][j];

	std::vector< std::complex<double> > err = _uncertainty.error(_data.datum(i));
	std::vector<std::complex<double> > cvo = _model.crosshand_visibilities(_data.datum(i),0.25*std::sqrt(std::abs(err[0]*err[0])+std::abs(err[1]*err[1])));

	for (size_t k=0; k<4; ++k) 
	  cvo[k] = _G[epoch][_is1_list[epoch][j]]*std::conj(_G[epoch][_is2_list[epoch][j]])*cvo[k];

	if (rank==0)
	  out << std::setw(15) << _data.datum(i).u/1e9
	      << std::setw(15) << _data.datum(i).v/1e9
	      << std::setw(15) << _data.datum(i).phi1
	      << std::setw(15) << _data.datum(i).phi2
	    // RR
	      << std::setw(15) << _data.datum(i).RR.real()
	    //<< std::setw(15) << _data.datum(i).RRerr.real()
	      << std::setw(15) << err[0].real()
	      << std::setw(15) << cvo[0].real()
	      << std::setw(15) << (_data.datum(i).RR-cvo[0]).real()
	      << std::setw(15) << _data.datum(i).RR.imag()
	    //<< std::setw(15) << _data.datum(i).RRerr.imag()
	      << std::setw(15) << err[0].imag()
	      << std::setw(15) << cvo[0].imag()
	      << std::setw(15) << (_data.datum(i).RR-cvo[0]).imag()
	    // LL
	      << std::setw(15) << _data.datum(i).LL.real()
	    //<< std::setw(15) << _data.datum(i).LLerr.real()
	      << std::setw(15) << err[1].real()
	      << std::setw(15) << cvo[1].real()
	      << std::setw(15) << (_data.datum(i).LL-cvo[1]).real()
	      << std::setw(15) << _data.datum(i).LL.imag()
	    //<< std::setw(15) << _data.datum(i).LLerr.imag()
	      << std::setw(15) << err[1].imag()
	      << std::setw(15) << cvo[1].imag()
	      << std::setw(15) << (_data.datum(i).LL-cvo[1]).imag()
	    // RL
	      << std::setw(15) << _data.datum(i).RL.real()
	    //<< std::setw(15) << _data.datum(i).RLerr.real()
	      << std::setw(15) << err[2].real()
	      << std::setw(15) << cvo[2].real()
	      << std::setw(15) << (_data.datum(i).RL-cvo[2]).real()
	      << std::setw(15) << _data.datum(i).RL.imag()
	    //<< std::setw(15) << _data.datum(i).RLerr.imag()
	      << std::setw(15) << err[2].imag()
	      << std::setw(15) << cvo[2].imag()
	      << std::setw(15) << (_data.datum(i).RL-cvo[2]).imag()
	    // LR
	      << std::setw(15) << _data.datum(i).LR.real()
	    //<< std::setw(15) << _data.datum(i).LRerr.real()
	      << std::setw(15) << err[3].real()
	      << std::setw(15) << cvo[3].real()
	      << std::setw(15) << (_data.datum(i).LR-cvo[3]).real()
	      << std::setw(15) << _data.datum(i).LR.imag()
	    //<< std::setw(15) << _data.datum(i).LRerr.imag()
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
    if (x==_x_last)
      return _L_last;
    _x_last = x;

    // _model.generate_model(x);
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

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      if (epoch%_L_size==size_t(_L_rank))
      {
	// Get vector of error-normed model and data visibilities once
	std::vector< std::complex<double> > ybrr,ybll,ybrl,yblr;
	std::vector< std::complex<double> > yrr,yll,yrl,ylr;
	std::vector< std::vector< std::complex<double> > > yb, y;
	std::vector<size_t> is1, is2;
      
	double lognorm = 0.0;
	for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
	{
	  size_t j = _datum_index_list[epoch][i];
	  
	  // std::complex<double> RRerr = _data.datum(_datum_index_list[epoch][i]).RRerr;
	  // std::complex<double> LLerr = _data.datum(_datum_index_list[epoch][i]).LLerr;
	  // std::complex<double> RLerr = _data.datum(_datum_index_list[epoch][i]).RLerr;
	  // std::complex<double> LRerr = _data.datum(_datum_index_list[epoch][i]).LRerr;
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

	  lognorm += _uncertainty.log_normalization(_data.datum(j));
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

	double marg_term;
	if (_solve_for_gains)
	{
	  // Determine the initial guess for the gains based on currently stated assumptions.
	  if (epoch>0)
	  {
	    if (_use_prior_gain_solutions==false)
	    {
	      for (size_t a=0; a<_sigma_g.size(); ++a)
		_G[epoch][a] = std::complex<double>(1.0,0.0);
	      // if (_smoothly_varying_gains)
	      // {
	      // 	_G[epoch] = _G[epoch-1];
	      // }
	      // else
	      // {
	      // 	for (size_t a=0; a<_sigma_g.size(); ++a)
	      // 	  _G[epoch][a] = std::complex<double>(1.0,0.0);
	      // }
	    }
	  }

	  // Levenberg-Marquardt minimization of full likelihood
	  double marg = optimal_complex_gains(y,yb,is1,is2,_G[epoch]);

	  if (marg>0)
	    _sqrt_detC[epoch] = marg;
	}
	marg_term = _sqrt_detC[epoch];

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
      
	// Add the Gaussian prior 
	for (size_t a=0; a<_sigma_g.size(); ++a)
	{
	  double G = std::log(std::abs(_G[epoch][a]));
	  dL += -0.5*G*G/(_sigma_g[a]*_sigma_g[a]);

	  double ph = std::arg(_G[epoch][a]);
	  dL += -0.5*ph*ph*_opi2;
	}

	// Add a quadratic approximation of the integral over the distribution about the best-fit gain corrections
	dL += std::log(marg_term);

	// Add error term
	dL += lognorm;

	// Accumulate contribution
	L += dL;
      }
    }

    double Ltot=0.0;
    MPI_Allreduce(&L,&Ltot,1,MPI_DOUBLE,MPI_SUM,_Lcomm);
    
    _L_last = Ltot;
    
    return Ltot;
  }

  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::likelihood_uniproc(std::vector<double>& x)
  {
    if (x==_x_last)
      return _L_last;
    _x_last = x;

    // _model.generate_model(x);
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

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // DEBUG
      //std::cerr << "On epoch " << epoch << '\n';

      // Get vector of error-normed model and data visibilities once
      std::vector< std::complex<double> > ybrr,ybll,ybrl,yblr;
      std::vector< std::complex<double> > yrr,yll,yrl,ylr;
      std::vector< std::vector< std::complex<double> > > yb, y;
      std::vector<size_t> is1, is2;

      double lognorm = 0.0;      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	// std::complex<double> RRerr = _data.datum(_datum_index_list[epoch][i]).RRerr;
	// std::complex<double> LLerr = _data.datum(_datum_index_list[epoch][i]).LLerr;
	// std::complex<double> RLerr = _data.datum(_datum_index_list[epoch][i]).RLerr;
	// std::complex<double> LRerr = _data.datum(_datum_index_list[epoch][i]).LRerr;

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

	lognorm += _uncertainty.log_normalization(_data.datum(j));
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

      double marg_term;
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
	double marg = optimal_complex_gains(y,yb,is1,is2,_G[epoch]);
	if (marg>0)
	  _sqrt_detC[epoch] = marg;
      }
      marg_term = _sqrt_detC[epoch];

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
      
      // Add the Gaussian prior 
      for (size_t a=0; a<_sigma_g.size(); ++a)
      {
	double G = std::log(std::abs(_G[epoch][a]));
	dL += -0.5*G*G/(_sigma_g[a]*_sigma_g[a]);

	double ph = std::arg(_G[epoch][a]);
	dL += -0.5*ph*ph*_opi2;
      }

      // Add a quadratic approximation of the integral over the distribution about the best-fit gain corrections
      dL += std::log(marg_term);

      // Add error term
      dL += lognorm;

      // Accumulate contribution
      L += dL;
    }

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
    // Make sure that gains are computed
    this->operator()(x);

    // Fix the gains (and remember the up to now state)
    bool solving_for_gains = _solve_for_gains;
    if (_solve_for_gains_during_gradient==false)
      fix_gains();

    // Compute the finite-difference gradient
    // std::vector<double> grad = likelihood_base::gradient(x,Pr);
    std::vector<double> grad = likelihood_base::gradient_uniproc(x,Pr);

    // Enable solving for gains again if we were doing so before
    if (_solve_for_gains_during_gradient==false && solving_for_gains) 
      solve_for_gains();
    
    // Return gradients
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
    int n = int(_sigma_g.size());
    double d;
    //double **a, d;
    int i,*indx;

    indx = new int[n+1];

    ludcmp(a,n,indx,d);

    // Find determinant of a
    for (i=1; i<=n; ++i)
      d *= a[i][i];

    // Clean up memory
    delete[] indx;

    return d;
  }
  

#define TINY 1.0e-20;
  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::ludcmp(double **a, int n, int *indx, double &d)
  {
    int i,imax=0,j,k;
    double big,dum,sum,temp;
    double *vv = new double[n+1];

    d=1.0;
    for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
	if ((temp=std::fabs(a[i][j])) > big)
	  big=temp;
      if (big == 0.0)
	std::cerr << "Singular matrix in routine ludcmp";
      vv[i]=1.0/big;
    }
    for (j=1;j<=n;j++) {
      for (i=1;i<j;i++) {
	sum=a[i][j];
	for (k=1;k<i;k++)
	  sum -= a[i][k]*a[k][j];
	a[i][j]=sum;
      }
      big=0.0;
      for (i=j;i<=n;i++) {
	sum=a[i][j];
	for (k=1;k<j;k++)
	  sum -= a[i][k]*a[k][j];
	a[i][j]=sum;
	if ( (dum=vv[i]*std::fabs(sum)) >= big) {
	  big=dum;
	  imax=i;
	}
      }
      if (j != imax) {
	for (k=1;k<=n;k++) {
	  dum=a[imax][k];
	  a[imax][k]=a[j][k];
	  a[j][k]=dum;
	}
	d = -(d);
	vv[imax]=vv[j];
      }
      indx[j]=imax;
      if (a[j][j] == 0.0)
	a[j][j]=TINY;
      if (j != n) {
	dum=1.0/(a[j][j]);
	for (i=j+1;i<=n;i++)
	  a[i][j] *= dum;
      }
    }
    delete[] vv;
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
    std::vector<std::complex<double> > gest_best = gest;
    double marg_best = -1;
    double chisq_best = std::numeric_limits<double>::infinity();

    std::vector<std::complex<double> > gest_try = gest;
    double marg_try, chisq_try;
    
    // Start at the passed guess
    marg_try = optimal_complex_gains_trial(y,yb,is1,is2,gest_try,chisq_try);
    if (marg_try>=0)
    {
      gest_best = gest_try;
      marg_best = marg_try;
      chisq_best = chisq_try;
    }

    // Log fit -- guaranteed to be convergent to the correct root, rapidly, but will
    // have the wrong covariance.
    gest_try = gest;
    marg_try = optimal_complex_gains_log_trial(y,yb,is1,is2,gest_try,chisq_try);
    if (marg_try>=0) // If successful, try the proper maximization
      marg_try = optimal_complex_gains_trial(y,yb,is1,is2,gest_try,chisq_try);
    if (marg_try>0) // If successful and successful, grab the best case
      if (chisq_try<chisq_best)
      {
	gest_best = gest_try;
	marg_best = marg_try;
	chisq_best = chisq_try;
      }

    // If failed all attempts, fix gains to unity and return fail code
    if (marg_best<0)
    {
      for (size_t j=0; j<gest.size(); ++j)
	gest[j] = std::complex<double>(1.0,0.0);
      return -1;
    }
    // Otherwise, return the best
    gest = gest_best;    
    return marg_best;
  }

    
  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::optimal_complex_gains_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    // Get the size of y (factor of 2 from real,imag)
    int ndata = int( 2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()) );

    if (ndata==0) {
      return 1.0;
    }

    // Make global pointers to avoid nightmares in rigging the NR stuff.
    _ogc_y = new double[ndata+1];
    _ogc_yb = new double[ndata+1];
    _ogc_is1 = new size_t[ndata+1];
    _ogc_is2 = new size_t[ndata+1];

    /*// DEBUG
    std::cerr << "ndata = " << ndata 
	      << " vs y[j].size() = " 
	      << y[0].size() << ", "
	      << y[1].size() << ", "
	      << y[2].size() << ", "
	      << y[3].size() << '\n';
    */

    for (size_t i=0, j=1; i<y[0].size(); ++i)
    {
      // RR
      _ogc_y[j] = y[0][i].real();
      _ogc_yb[j] = yb[0][i].real();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      _ogc_y[j] = y[0][i].imag();
      _ogc_yb[j] = yb[0][i].imag();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // LL
      _ogc_y[j] = y[1][i].real();
      _ogc_yb[j] = yb[1][i].real();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      _ogc_y[j] = y[1][i].imag();
      _ogc_yb[j] = yb[1][i].imag();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // RL
      _ogc_y[j] = y[2][i].real();
      _ogc_yb[j] = yb[2][i].real();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      _ogc_y[j] = y[2][i].imag();
      _ogc_yb[j] = yb[2][i].imag();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // LR
      _ogc_y[j] = y[3][i].real();
      _ogc_yb[j] = yb[3][i].real();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      _ogc_y[j] = y[3][i].imag();
      _ogc_yb[j] = yb[3][i].imag();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;

      // DEBUG
      //std::cerr << "    on j = " << j << '\n';
    }

    // Make space for mrqmin objects
    int ma = 2*_sigma_g.size(); // real,imag
    double **covar, **alpha;
    covar = new double*[ma+1];
    alpha = new double*[ma+1];
    for (int i=1; i<=ma; ++i)
    {
      covar[i] = new double[ma+1];
      alpha[i] = new double[ma+1];
    }


    // Start running mrqmin
    double *g = new double[ma+1]; // Internal gain representation is gain correction magnitude and phase, i.e., G = exp[ g - i phi ].
    double *og = new double[ma+1];
    for (int i=0, j=1; i<int(gest.size()); ++i)
    {
      g[j++] = std::log(std::abs(gest[i]));
      g[j++] = std::arg(gest[i]);
    }
    double alambda = -1.0;
    double chisq=0.0, ochisq, dg2;
    double dg2limit=0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += _sigma_g[i]*_sigma_g[i];
    dg2limit *= 1e-12; 


    bool notconverged = true;
    int iteration;
    for (iteration=0; iteration<_itermax && notconverged; ++iteration)
    {
      for (int i=1; i<=ma; ++i)
	og[i] = g[i];
      ochisq = chisq;

      if (mrqmin(_ogc_y,ndata,g,ma,covar,alpha,&chisq,&alambda))
	return -1;
      
      if (iteration>5 && chisq<ochisq)
      {
	dg2 = 0.0;
	for (int i=1; i<=ma; ++i)
	  dg2 += std::pow((g[i]-og[i]),2);

	if (dg2<dg2limit || (ochisq-chisq)<1e-8*chisq)
	  notconverged = false;
      }
    }
    alambda=0.0;
    mrqmin(_ogc_y,ndata,g,ma,covar,alpha,&chisq,&alambda);

    // Save output
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
    {
      double gmag = std::exp(g[j]);
      /*
      // Limit from below
      if (gmag<1.0/(1.0+_sigma_g[i]*_max_g[i]))
	gmag = 1.0/(1.0+_sigma_g[i]*_max_g[i]);
      // Limit from above
      if (gmag>(1.0+_sigma_g[i]*_max_g[i]))
	gmag = (1.0+_sigma_g[i]*_max_g[i]);
      */
      gest[i] = gmag * std::exp( std::complex<double>(0.0,1.0)*g[j+1] );
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;

    // Clean up
    delete[] og;
    delete[] g;
    for (int i=1; i<=ma; ++i)
    {
      delete[] covar[i];
      delete[] alpha[i];
    }
    delete[] covar;
    delete[] alpha;
    delete[] _ogc_is2;
    delete[] _ogc_is1;
    delete[] _ogc_yb;
    delete[] _ogc_y;

    chisq_opt = chisq;
    
    return std::sqrt(detC); // Success!
  }


  double likelihood_optimal_complex_gain_constrained_crosshand_visibilities::optimal_complex_gains_log_trial(std::vector< std::vector< std::complex<double> > >& y, std::vector< std::vector< std::complex<double> > >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    // Get the size of y (factor of 2 from real,imag)
    int ndata = int( 2*(y[0].size()+y[1].size()+y[2].size()+y[3].size()) );

    if (ndata==0) {
      return 1.0;
    }

    // Make global pointers to avoid nightmares in rigging the NR stuff.
    _ogc_y = new double[ndata+1];
    _ogc_yb = new double[ndata+1];
    _ogc_is1 = new size_t[ndata+1];
    _ogc_is2 = new size_t[ndata+1];
    double *sig = new double[ndata+1];
    std::complex<double> tmp;
    for (size_t i=0, j=1; i<y[0].size(); ++i)
    {
      // RR
      tmp = y[0][i]/yb[0][i];
      if (std::abs(tmp)<1e-6)
	tmp = 1e-6;
      //_ogc_y[j] = std::log((y[0][i]==0.0 ? 1e-15 : y[0][i])/yb[0][i]).real();
      _ogc_y[j] = std::log(tmp).real();
      sig[j] = 1.0/std::abs(y[0][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      //_ogc_y[j] = std::log((y[0][i]==0.0 ? 1e-15 : y[0][i])/yb[0][i]).imag();
      _ogc_y[j] = std::log(tmp).imag();
      sig[j] = 1.0/std::abs(y[0][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // LL
      tmp = y[1][i]/yb[1][i];
      if (std::abs(tmp)<1e-6)
	tmp = 1e-6;
      //_ogc_y[j] = std::log((y[1][i]==0.0 ? 1e-15 : y[1][i])/yb[1][i]).real();
      _ogc_y[j] = std::log(tmp).real();      
      sig[j] = 1.0/std::abs(y[1][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      //_ogc_y[j] = std::log((y[1][i]==0.0 ? 1e-15 : y[1][i])/yb[1][i]).imag();
      _ogc_y[j] = std::log(tmp).imag();      
      sig[j] = 1.0/std::abs(y[1][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // RL
      tmp = y[2][i]/yb[2][i];
      if (std::abs(tmp)<1e-6)
	tmp = 1e-6;
      //_ogc_y[j] = std::log((y[2][i]==0.0 ? 1e-15 : y[2][i])/yb[2][i]).real();
      _ogc_y[j] = std::log(tmp).real();      
      sig[j] = 1.0/std::abs(y[2][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      //_ogc_y[j] = std::log((y[2][i]==0.0 ? 1e-15 : y[2][i])/yb[2][i]).imag();
      _ogc_y[j] = std::log(tmp).imag();      
      sig[j] = 1.0/std::abs(y[2][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      // LR
      tmp = y[3][i]/yb[3][i];
      if (std::abs(tmp)<1e-6)
	tmp = 1e-6;
      //_ogc_y[j] = std::log((y[3][i]==0.0 ? 1e-15 : y[3][i])/yb[3][i]).real();
      _ogc_y[j] = std::log(tmp).real();      
      sig[j] = 1.0/std::abs(y[3][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      //_ogc_y[j] = std::log((y[3][i]==0.0 ? 1e-15 : y[3][i])/yb[3][i]).imag();
      _ogc_y[j] = std::log(tmp).imag();      
      sig[j] = 1.0/std::abs(y[3][i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
    }

    // Make space for mrqmin objects
    int ma = 2*_sigma_g.size(); // real,imag
    double **covar, **alpha;
    covar = new double*[ma+1];
    alpha = new double*[ma+1];
    for (int i=1; i<=ma; ++i)
    {
      covar[i] = new double[ma+1];
      alpha[i] = new double[ma+1];
    }


    // Start running mrqmin
    double *g = new double[ma+1]; // Internal gain representation is gain correction magnitude and phase, i.e., G = exp[ g - i phi ].
    double *og = new double[ma+1];
    for (int i=0, j=1; i<int(gest.size()); ++i)
    {
      g[j++] = std::log(std::abs(gest[i]));
      g[j++] = std::arg(gest[i]);
    }
    double alambda = -1.0;
    double chisq=0.0, ochisq, dg2;
    double ch2limit=1.0e-8;
    double dg2limit=0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += _sigma_g[i]*_sigma_g[i];
    dg2limit *= 1e-12; 


    bool notconverged = true;
    int iteration;
    for (iteration=0; iteration<_itermax && notconverged; ++iteration)
    {
      for (int i=1; i<=ma; ++i)
	og[i] = g[i];
      ochisq = chisq;
      
      if (mrqmin_log(_ogc_y,sig,ndata,g,ma,covar,alpha,&chisq,&alambda))
	return -1;
      
      if (iteration>5 && chisq<ochisq)
      {
	dg2 = 0.0;
	for (int i=1; i<=ma; ++i)
	  dg2 += std::pow((g[i]-og[i]),2);

	if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
	  notconverged = false;
      }
    }
    alambda=0.0;
    mrqmin_log(_ogc_y,sig,ndata,g,ma,covar,alpha,&chisq,&alambda);

    // Save output
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
    {
      double gmag = std::exp(g[j]);
      /*
      // Limit from below
      if (gmag<1.0/(1.0+_sigma_g[i]*_max_g[i]))
	gmag = 1.0/(1.0+_sigma_g[i]*_max_g[i]);
      // Limit from above
      if (gmag>(1.0+_sigma_g[i]*_max_g[i]))
	gmag = (1.0+_sigma_g[i]*_max_g[i]);
      */
      gest[i] = gmag * std::exp( std::complex<double>(0.0,1.0)*g[j+1] );
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;
    
    // Clean up
    delete[] og;
    delete[] g;
    for (int i=1; i<=ma; ++i)
    {
      delete[] covar[i];
      delete[] alpha[i];
    }
    delete[] covar;
    delete[] alpha;
    delete[] _ogc_is2;
    delete[] _ogc_is1;
    delete[] _ogc_yb;
    delete[] _ogc_y;
    delete[] sig;
    
    chisq_opt = chisq;
    
    return std::sqrt(detC); // Success!
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

    int *indxc = new int[n+1];
    int *indxr = new int[n+1];
    int *ipiv = new int[n+1];


    /*
    // DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double aorig[n+1][n+1];
    double borig[n+1][m+1];
    if (n>0) {
      for (int i2=1; i2<=n; i2++)
	for (int j2=1; j2<=n; j2++)
	  aorig[i2][j2] = a[i2][j2];
    }
    if (m>0) {
      for (int i2=1; i2<=n; i2++)
	for (int j2=1; j2<=m; j2++)
	  borig[i2][j2] = b[i2][j2];
    }
    */

    
    for (j=1;j<=n;j++)
      ipiv[j]=0;
    for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
	if (ipiv[j] != 1)
	  for (k=1;k<=n;k++) {
	    if (ipiv[k] == 0) {
	      if (std::fabs(a[j][k]) >= big) {
		big=std::fabs(a[j][k]);
		irow=j;
		icol=k;
	      }
	    } else if (ipiv[k] > 1) {
	      std::cerr << "gaussj: Singular Matrix-1\n";
	      /*// DEBUG
	      std::cerr << rank << " - aorig -----------------------------\n";
	      for (int i2=1; i2<=n; i2++) {
		for (int j2=1; j2<=n; j2++)
		  std::cerr << std::setw(15) << aorig[i2][j2];
		std::cerr << '\n';
	      }
	      std::cerr << rank << " - aorig -----------------------------\n";
	      std::cerr << rank << " - borig -----------------------------\n";
	      for (int i2=1; i2<=n; i2++) {
		for (int j2=1; j2<=m; j2++)
		  std::cerr << std::setw(15) << borig[i2][j2];
		std::cerr << '\n';
	      }
	      std::cerr << rank << " - borig -----------------------------\n";
	      std::cerr << rank << " - a -----------------------------\n";
	      for (int i2=1; i2<=n; i2++) {
		for (int j2=1; j2<=n; j2++)
		  std::cerr << std::setw(15) << a[i2][j2];
		std::cerr << '\n';
	      }
	      std::cerr << rank << " - a -----------------------------\n";
	      std::cerr << rank << " - b -----------------------------\n";
	      for (int i2=1; i2<=n; i2++) {
		for (int j2=1; j2<=m; j2++)
		  std::cerr << std::setw(15) << b[i2][j2];
		std::cerr << '\n';
	      }
	      std::cerr << rank << " - b -----------------------------\n";
	      */
	      delete[] ipiv;
	      delete[] indxr;
	      delete[] indxc;
	      return 1;
	    }
	  }
      ++(ipiv[icol]);
      if (irow != icol) {
	for (l=1;l<=n;l++)
	  SWAP(a[irow][l],a[icol][l]);
	for (l=1;l<=m;l++)
	  SWAP(b[irow][l],b[icol][l]);
      }
      indxr[i]=irow;
      indxc[i]=icol;
      if (a[icol][icol] == 0.0) {
	std::cerr << "gaussj: Singular Matrix-2\n";
	/*
	// DEBUG
	std::cerr << rank << " - aorig -----------------------------\n";
	for (int i2=1; i2<=n; i2++) {
	  for (int j2=1; j2<=n; j2++)
	    std::cerr << std::setw(15) << aorig[i2][j2];
	  std::cerr << '\n';
	}
	std::cerr << rank << " - aorig -----------------------------\n";
	std::cerr << rank << " - borig -----------------------------\n";
	for (int i2=1; i2<=n; i2++) {
	  for (int j2=1; j2<=m; j2++)
	    std::cerr << std::setw(15) << borig[i2][j2];
	  std::cerr << '\n';
	}
	std::cerr << rank << " - borig -----------------------------\n";
	std::cerr << rank << " - a -----------------------------\n";
	for (int i2=1; i2<=n; i2++) {
	  for (int j2=1; j2<=n; j2++)
	    std::cerr << std::setw(15) << a[i2][j2];
	  std::cerr << '\n';
	}
	std::cerr << rank << " - a -----------------------------\n";
	std::cerr << rank << " - b -----------------------------\n";
	for (int i2=1; i2<=n; i2++) {
	  for (int j2=1; j2<=m; j2++)
	    std::cerr << std::setw(15) << b[i2][j2];
	  std::cerr << '\n';
	}
	std::cerr << rank << " - b -----------------------------\n";
	*/
	delete[] ipiv;
	delete[] indxr;
	delete[] indxc;
	return 2;
      }
      pivinv=1.0/a[icol][icol];
      a[icol][icol]=1.0;
      for (l=1;l<=n;l++) a[icol][l] *= pivinv;
      for (l=1;l<=m;l++) b[icol][l] *= pivinv;
      for (ll=1;ll<=n;ll++)
	if (ll != icol) {
	  dum=a[ll][icol];
	  a[ll][icol]=0.0;
	  for (l=1;l<=n;l++) a[ll][l] -= a[icol][l]*dum;
	  for (l=1;l<=m;l++) b[ll][l] -= b[icol][l]*dum;
	}
    }
    for (l=n;l>=1;l--) {
      if (indxr[l] != indxc[l])
	for (k=1;k<=n;k++)
	  SWAP(a[k][indxr[l]],a[k][indxc[l]]);
    }

    delete[] ipiv;
    delete[] indxr;
    delete[] indxc;

    return 0;
  }
#undef SWAP

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,dy;

    double *dyda = new double[ma+1];

    // Gain limiter to avoid NaNs  Restricts gains to exp(-1e2) to exp(1e2)
    for (j=1;j<=ma; ++j)
      a[j] = std::min(std::max(a[j],-1.0e2),1.0e2);
    
    for (j=1;j<=ma;j++) {
      beta[j]=0.0;
      for (k=1;k<=ma;k++)
	alpha[j][k]=0.0;
    }
    *chisq=0.0;
    for (i=1;i<=ndata;i++) {
      gain_optimization_likelihood(i,a,&ymod,dyda);
      dy=y[i]-ymod;
      for (j=1;j<=ma;j++) {
	wt=dyda[j];
	for (k=1;k<=j;k++)
	  alpha[j][k] += wt*dyda[k];
	beta[j] += dy*wt;
      }

      /*
      for (int j=1; j<=2*int(_sigma_g.size()); j++)
	std::cerr << std::setw(15) << a[j];
      std::cerr << "\n----\n";
      std::cerr << std::setw(5) << i
		<< std::setw(15) << y[i]
		<< std::setw(15) << _ogc_is1[i]
		<< std::setw(15) << _ogc_is2[i]
		<< '\n';
      for (int j=1; j<=2*int(_sigma_g.size()); j++)
	std::cerr << std::setw(15) << y[j];
      std::cerr << "\n----\n";      
      std::cerr << "\n";
      for (int j=1; j<=2*int(_sigma_g.size()); j++)
	std::cerr << std::setw(15) << dyda[j];
      std::cerr << "FOO ----------------------------------------------------------\n";    
      for (int j=1; j<=2*int(_sigma_g.size()); j++)
	std::cerr << std::setw(15) << y[j];
      std::cerr << "\n----\n";
      for (int i=1; i<=2*int(_sigma_g.size()); i++)
      {
	for (int j=1; j<=2*int(_sigma_g.size()); j++)
	  std::cerr << std::setw(15) << alpha[i][j];
	std::cerr << '\n';
      }
      std::cerr << "OOF ----------------------------------------------------------\n";
      */
      *chisq += dy*dy;
    }
    for (j=2;j<=ma;j++)
      for (k=1;k<j;k++)
	alpha[k][j]=alpha[j][k];

    // Add priors to alpha and beta
    // In addition to the prior on g that is given, a weak prior on phi is provided to drive the solution toward G=1 in the absence of other information.
    double oSigma2;
    for (size_t i=0, j=1; i<_sigma_g.size(); i++, j+=2)
    {
      oSigma2 = 1.0/(_sigma_g[i]*_sigma_g[i]);

      beta[j] -= a[j]*oSigma2; // g^2/2 Sigma^2
      beta[j+1] -= a[j+1]*_opi2; // phi^2 / 2 varpi^2

      alpha[j][j] += oSigma2;
      alpha[j+1][j+1] += _opi2;
      
      //alpha[j][j] += std::max(1e-12*std::fabs(alpha[j][j]),oSigma2);
      //alpha[j+1][j+1] += std::max(1e-12*std::fabs(alpha[j+1][j+1]),_opi2);

      //alpha[j][j] += std::max(1e-12*std::fabs(alpha[j][j]),oSigma2);
      //alpha[j+1][j+1] += std::max(1e-12*std::fabs(alpha[j][j]),_opi2);

      (*chisq) += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
    }
    double alpha_diag_max = 0.0;
    for (int j=1; j<=2*int(_sigma_g.size()); j++)
      alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max,1.0);
    for (int j=1; j<=2*int(_sigma_g.size()); j++)    
      alpha[j][j] += 1.0e-12*alpha_diag_max;
    //alpha[j][j] = std::max(alpha[j][j],1.0e-10*alpha_diag_max);
    
    delete[] dyda;
  }

  int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;
    //int gaussj_err;
    
    if (*alamda < 0.0) {
      *alamda=0.001;
      mrqcof(y,ndata,a,ma,alpha,_mrq_beta,chisq);
      _mrq_ochisq=(*chisq);
      for (j=1;j<=ma;j++)
	_mrq_atry[j]=a[j];
    }
    for (j=1;j<=mfit;j++) {
      for (k=1;k<=mfit;k++)
	covar[j][k]=alpha[j][k];
      covar[j][j]=alpha[j][j]*(1.0+(*alamda));
      _mrq_oneda[j][1]=_mrq_beta[j];
    }
    if (gaussj(covar,mfit,_mrq_oneda,1))
      return 1;
    /*
    gaussj_err = gaussj(covar,mfit,_mrq_oneda,1);
    // DEBUGGING OUTPUT FOR GAUSSJ ERRORS
    if (gaussj_err>0) {
      std::cerr << "alambda = " << (*alamda) << " ====================== \n";
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << a[j];
      std::cerr << '\n';
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << _mrq_beta[j];
      std::cerr << "\n ====================== \n";      
      for (j=1;j<=mfit;j++) {
	for (k=1;k<=mfit;k++)
	  std::cerr << std::setw(15) << alpha[j][k];
	std::cerr << '\n';
      }
      std::cerr << "\n ====================== \n";
      mrqcof(y,ndata,a,ma,alpha,_mrq_beta,chisq);
      for (j=1;j<=ndata;j++)
	std::cerr << std::setw(15) << y[j];
      std::cerr << '\n';      
      for (j=1;j<=ndata;j++)
	std::cerr << std::setw(15) << _station_codes[_ogc_is1[j]]+_station_codes[_ogc_is2[j]];
      std::cerr << '\n';      
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << a[j];
      std::cerr << '\n';
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << _mrq_beta[j];
      std::cerr << "\n ---------------------- \n";            
      for (j=1;j<=mfit;j++) {
	for (k=1;k<=mfit;k++)
	  std::cerr << std::setw(15) << alpha[j][k];
	std::cerr << '\n';
      }
      std::cerr << "\n ====================== \n";
      for (j=1;j<=mfit; ++j)
	a[j] = std::min(std::max(a[j],-1e2),1e2);
      int i,j,k;
      double ymod,wt,dy;
      double *dyda = new double[ma+1];
      for (j=1;j<=ma;j++) {
	_mrq_beta[j]=0.0;
	for (k=1;k<=ma;k++)
	  alpha[j][k]=0.0;
      }
      *chisq=0.0;
      for (i=1;i<=ndata;i++) {
	gain_optimization_likelihood(i,a,&ymod,dyda);
	dy=y[i]-ymod;
	for (j=1;j<=ma;j++) {
	  wt=dyda[j];
	  for (k=1;k<=j;k++)
	    alpha[j][k] += wt*dyda[k];
	  _mrq_beta[j] += dy*wt;
      }
	*chisq += dy*dy;
      }
      for (j=2;j<=ma;j++)
	for (k=1;k<j;k++)
	  alpha[k][j]=alpha[j][k];
      // Add priors to alpha and beta
      // In addition to the prior on g that is given, a weak prior on phi is provided to drive the solution toward G=1 in the absence of other information.
      double oSigma2;
      for (size_t i=0, j=1; i<_sigma_g.size(); i++, j+=2) {
	oSigma2 = 1.0/(_sigma_g[i]*_sigma_g[i]);
	
	_mrq_beta[j] -= a[j]*oSigma2; // g^2/2 Sigma^2
	_mrq_beta[j+1] -= a[j+1]*_opi2; // phi^2 / 2 varpi^2
	
	(*chisq) += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
      }
      double alpha_diag_max = 0.0;
      for (int j=1; j<=2*int(_sigma_g.size()); j++)
	alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
      alpha_diag_max = std::max(alpha_diag_max,1.0);
      for (int j=1; j<=2*int(_sigma_g.size()); j++)    
	alpha[j][j] += 1.0e-12*alpha_diag_max;
      for (j=1;j<=ndata;j++)
	std::cerr << std::setw(15) << y[j];
      std::cerr << '\n';      
      for (j=1;j<=ndata;j++)
	std::cerr << std::setw(15) << _station_codes[_ogc_is1[j]]+_station_codes[_ogc_is2[j]];
      std::cerr << '\n';      
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << a[j];
      std::cerr << '\n';
      for (j=1;j<=mfit;j++)
	std::cerr << std::setw(15) << _mrq_beta[j];
      std::cerr << "\n ---------------------- \n";            
      for (j=1;j<=mfit;j++) {
	for (k=1;k<=mfit;k++)
	  std::cerr << std::setw(15) << alpha[j][k];
	std::cerr << '\n';
      }      
      std::exit(1);
    }
    // GUBED */
    for (j=1;j<=mfit;j++)
      _mrq_da[j]=_mrq_oneda[j][1];
    if (*alamda == 0.0) {
      covsrt(covar,ma,mfit);
      return 0;
    }
    for (l=1;l<=ma;l++)
      _mrq_atry[l]=a[l]+_mrq_da[l];
    mrqcof(y,ndata,_mrq_atry,ma,covar,_mrq_da,chisq);
    if (*chisq < _mrq_ochisq) {
      *alamda *= 0.1;
      _mrq_ochisq=(*chisq);
      for (j=1;j<=mfit;j++) {
	for (k=1;k<=mfit;k++)
	  alpha[j][k]=covar[j][k];
	_mrq_beta[j]=_mrq_da[j];
      }
      for (l=1;l<=ma;l++)
	a[l]=_mrq_atry[l];
    } else {
      *alamda *= 10.0;
      *chisq=_mrq_ochisq;
    }
    return 0;
  }

  void likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqcof_log(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,sig2i,dy;
   
    double *dyda = new double[ma+1];

    // Gain limiter to avoid NaNs  Restricts gains to exp(-1e2) to exp(1e2)
    for (j=1;j<=ma; ++j)
      a[j] = std::min(std::max(a[j],-1.0e2),1.0e2);
    
    for (j=1;j<=ma;j++) {
      beta[j]=0.0;
      for (k=1;k<=ma;k++)
	alpha[j][k]=0.0;
    }
    *chisq=0.0;
    for (i=1;i<=ndata;i++) {
      gain_optimization_log_likelihood(i,a,&ymod,dyda);
      sig2i=1.0/(sig[i]*sig[i]);
      dy=y[i]-ymod;
      for (j=1;j<=ma;j++) {
	wt=dyda[j]*sig2i;
	for (k=1;k<=j;k++)
	  alpha[j][k] += wt*dyda[k];
	beta[j] += dy*wt;
      }
      *chisq += dy*dy*sig2i;
    }
    for (j=2;j<=ma;j++)
      for (k=1;k<j;k++)
	alpha[k][j]=alpha[j][k];

    // Add priors to alpha and beta
    // In addition to the prior on g that is given, a weak prior on phi is provided to drive the solution toward G=1 in the absence of other information.
    double oSigma2;
    for (size_t i=0, j=1; i<_sigma_g.size(); i++, j+=2)
    {
      oSigma2 = 1.0/(_sigma_g[i]*_sigma_g[i]);

      beta[j] -= a[j]*oSigma2; // g^2/2 Sigma^2
      beta[j+1] -= a[j+1]*_opi2; // phi^2 / 2 varpi^2

      alpha[j][j] += oSigma2;
      alpha[j+1][j+1] += _opi2;

      (*chisq) += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
    }
    double alpha_diag_max = 0.0;
    for (int j=1; j<=2*int(_sigma_g.size()); j++)
      alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max,1.0);
    for (int j=1; j<=2*int(_sigma_g.size()); j++)    
      alpha[j][j] += 1.0e-12*alpha_diag_max;
    //alpha[j][j] = std::max(alpha[j][j],1.0e-10*alpha_diag_max);
    
    delete[] dyda;
  }

  int likelihood_optimal_complex_gain_constrained_crosshand_visibilities::mrqmin_log(double y[], double sig[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;

    if (*alamda < 0.0) {
      *alamda=0.001;
      mrqcof_log(y,sig,ndata,a,ma,alpha,_mrq_beta,chisq);
      _mrq_ochisq=(*chisq);
      for (j=1;j<=ma;j++)
	_mrq_atry[j]=a[j];
    }
    for (j=1;j<=mfit;j++) {
      for (k=1;k<=mfit;k++)
	covar[j][k]=alpha[j][k];
      covar[j][j]=alpha[j][j]*(1.0+(*alamda));
      _mrq_oneda[j][1]=_mrq_beta[j];
    }
    if (gaussj(covar,mfit,_mrq_oneda,1))
      return 1;
    for (j=1;j<=mfit;j++)
      _mrq_da[j]=_mrq_oneda[j][1];
    if (*alamda == 0.0) {
      covsrt(covar,ma,mfit);
      return 0;
    }
    for (l=1;l<=ma;l++)
      _mrq_atry[l]=a[l]+_mrq_da[l];
    mrqcof_log(y,sig,ndata,_mrq_atry,ma,covar,_mrq_da,chisq);
    if (*chisq < _mrq_ochisq) {
      *alamda *= 0.1;
      _mrq_ochisq=(*chisq);
      for (j=1;j<=mfit;j++) {
	for (k=1;k<=mfit;k++)
	  alpha[j][k]=covar[j][k];
	_mrq_beta[j]=_mrq_da[j];
      }
      for (l=1;l<=ma;l++)
	a[l]=_mrq_atry[l];
    } else {
      *alamda *= 10.0;
      *chisq=_mrq_ochisq;
    }
    return 0;
  }
  
  
};


