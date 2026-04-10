/*! 
  \file likelihood_optimal_complex_gain_visibility.cpp
  \author Avery E. Broderick, Roman Gold
  \date  February, 2020, February 2026
  \brief Implementation file for the likelihood_optimal_complex_gain_visibility likelihood class
*/


#include "random_number_generator.h"

#include "likelihood_optimal_complex_gain_visibility.h"
#include "model_image_adaptive_splined_raster.h"
#include "model_image_sum.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_xsringauss.h"

#include <cmath>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace Themis
{
  // Cholesky debug START
  long long g_chol_calls = 0;
  long long g_chol_failures = 0;
  long long g_chol_compares = 0;
  double g_chol_max_abs_da_diff = 0.0;
  double g_chol_max_rel_da_diff = 0.0;
  // Cholesky debug END
  

  
  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g)
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


    // Output times for inspection
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank==0) {
      std::cerr << " --- Gain epoch times ---- \n";
      for (size_t k=0; k<_tge.size(); ++k)
	std::cerr << "  _tge[" << k << "] " << _tge[k] << " " << _tge[k]-_tge[0] << '\n';
      std::cerr << " ------------------------- \n";
    }

    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    //_model.set_data(_data);
  }


  
  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(true), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    //_model.set_data(_data);
  }

  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
    : _data(data), _model(model), _uncertainty(_local_uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(max_g), _tge(t_ge), _smoothly_varying_gains(true), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    //_model.set_data(_data);
  }

  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _use_prior_gain_solutions(true), _smoothly_varying_gains(false), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100)
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

    //_model.set_data(_data);
  }

  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _tge(t_ge), _use_prior_gain_solutions(true), _smoothly_varying_gains(true), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();

    //_model.set_data(_data);
  }

  likelihood_optimal_complex_gain_visibility::likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& uncertainty, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
    : _data(data), _model(model), _uncertainty(uncertainty), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(max_g), _tge(t_ge), _smoothly_varying_gains(true), _solve_for_gains(true), _solve_for_gains_during_gradient(false), _parallelize_likelihood(true), _opi2(1e-8), _itermax(100)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();
    
    //_model.set_data(_data);
  }

  
  likelihood_optimal_complex_gain_visibility::~likelihood_optimal_complex_gain_visibility()
  {
    for (size_t j=0; j<=2*_sigma_g.size(); ++j)
      delete[] _mrq_oneda[j];
    delete[] _mrq_oneda;
    
    delete[] _mrq_da;
    delete[] _mrq_beta;
    delete[] _mrq_atry;

    // new
    delete[] _indx;
    delete[] _indxc;
    delete[] _indxr;
    delete[] _ipiv;
    delete[] _vv;
    delete[] _dyda;
    
    delete[] _ogc_y;
    delete[] _ogc_yb;
    delete[] _ogc_is1;
    delete[] _ogc_is2;
    delete[] _sig;
    
    for (size_t i=1; i<=2*_sigma_g.size(); ++i)
    {
      delete[] _covar[i];
      delete[] _alpha[i];
    }
    delete[] _covar;
    delete[] _alpha;    
    
    delete[] _g;
    delete[] _og;
  }

  void likelihood_optimal_complex_gain_visibility::check_station_codes()
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
	std::cerr << "WARNING: likelihood_optimal_complex_gain_visibility:\n"
		  << "    Station " << _data.datum(j).Station1 << " not in station_codes list.\n"
		  << '\n';
      if ( station2_in_station_codes==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_visibility:\n"
		  << "    Station " << _data.datum(j).Station2 << " not in station_codes list.\n"
		  << '\n';
    }
    for (size_t k=0; k<_station_codes.size(); ++k)
      if ( station_code_in_data[k]==false )
	std::cerr << "WARNING: likelihood_optimal_complex_gain_visibility:\n"
		  << "    station code " << _station_codes[k] << " not used in data set.\n"
		  << '\n'; 
  }
  
  void likelihood_optimal_complex_gain_visibility::allocate_memory()
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

    int ma = 2*_sigma_g.size();
    _mrq_atry = new double[ma+1];
    _mrq_beta = new double[ma+1];
    _mrq_da = new double[ma+1];
    _mrq_oneda = new double*[ma+1];
    for (int j=0; j<=ma; j++)
      _mrq_oneda[j] = new double[2];


    // new
    _indx = new int[ma+1];
    _indxc = new int[ma+1];
    _indxr = new int[ma+1];
    _ipiv = new int[ma+1];
    _vv = new double[ma+1];
    _dyda = new double[ma+1];

    int ndatamax = 2*_data.size();
    _ogc_y = new double[ndatamax+1];
    _ogc_yb = new double[ndatamax+1];
    _ogc_is1 = new size_t[ndatamax+1];
    _ogc_is2 = new size_t[ndatamax+1];

    _sig = new double[ndatamax+1];
    
    _covar = new double*[ma+1];
    _alpha = new double*[ma+1];
    for (int i=1; i<=ma; ++i)
    {
      _covar[i] = new double[ma+1];
      _alpha[i] = new double[ma+1];
    }
    
    _g = new double[ma+1];
    _og = new double[ma+1];
  }

  
  void likelihood_optimal_complex_gain_visibility::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(_Mcomm);
  }

  void likelihood_optimal_complex_gain_visibility::set_iteration_limit(int itermax)
  {
    _itermax=itermax;
  }

  void likelihood_optimal_complex_gain_visibility::solve_for_gains()
  {
    _solve_for_gains = true;
  }

  void likelihood_optimal_complex_gain_visibility::fix_gains()
  {
    _solve_for_gains = false;
  }

  void likelihood_optimal_complex_gain_visibility::solve_for_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = true;
  }

  void likelihood_optimal_complex_gain_visibility::fix_gains_during_gradient()
  {
    _solve_for_gains_during_gradient = false;
  }
  
  void likelihood_optimal_complex_gain_visibility::use_prior_gain_solutions()
  {
    _use_prior_gain_solutions = true;
  }

  void likelihood_optimal_complex_gain_visibility::assume_smoothly_varying_gains()
  {
    _smoothly_varying_gains = true;
    _use_prior_gain_solutions = false;
  }
  
  void likelihood_optimal_complex_gain_visibility::assume_independently_varying_gains()
  {
    _smoothly_varying_gains = false;
    _use_prior_gain_solutions = false;
  }



  std::vector<size_t> collect_owned_datum_indices(const std::vector<std::vector<size_t>>& datum_index_list, int lrank, int lsize)
  {
    std::vector<size_t> ids;
    size_t n = 0;
    for (size_t e = 0; e < datum_index_list.size(); ++e)
      if (e % size_t(lsize) == size_t(lrank))
	n += datum_index_list[e].size();
    
    ids.reserve(n);
    for (size_t e = 0; e < datum_index_list.size(); ++e)
      if (e % size_t(lsize) == size_t(lrank))
	ids.insert(ids.end(),
		   datum_index_list[e].begin(),
		   datum_index_list[e].end());
    return ids;
  }
  
  std::vector<size_t> collect_all_datum_indices(size_t n)
  {
    std::vector<size_t> ids(n);
    for (size_t i = 0; i < n; ++i)
      ids[i] = i;
    return ids;
  }

  static void set_raster_cache_mode_recursive(model_visibility& model,
                                              model_image_adaptive_splined_raster::VisibilityCacheMode mode)
  {
    if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(&model))
    {
      r->set_cache_mode(mode);
      return;
    }

    if (auto* s = dynamic_cast<model_image_sum*>(&model))
    {
      const auto& imgs = s->components();
      for (size_t j = 0; j < imgs.size(); ++j)
      {
        if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(imgs[j]))
          r->set_cache_mode(mode);
      }
    }
  }


  
  static model_image_adaptive_splined_raster*
  find_single_raster_component(model_visibility& model)
  {
    if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(&model))
      return r;

    if (auto* s = dynamic_cast<model_image_sum*>(&model))
    {
      model_image_adaptive_splined_raster* found = nullptr;
      const auto& imgs = s->components();

      for (size_t j = 0; j < imgs.size(); ++j)
      {
        if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(imgs[j]))
        {
          if (found != nullptr)
            return nullptr; // more than one raster: unsupported in this incremental step
          found = r;
        }
        else
        {
          return nullptr;   // non-raster component present: unsupported for this step
        }
      }
      return found;
    }

    return nullptr;
  }

  
  
  void likelihood_optimal_complex_gain_visibility::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    /*
    distribute_gains();
    
    if (rank==0)
      out << "# likelihood_visibility output file\n#"
    */
        distribute_gains();

    if (rank != 0)
      return;
    
    // Intentionally rebuild a full cache by passing all datum ids.
    // This still uses the OwnedGlobal builder, but with all global slots filled.
    _model.prepare_visibility_cache(_data, collect_all_datum_indices(_data.size()));
    
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

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	std::complex<double> err = _uncertainty.error(_data.datum(_datum_index_list[epoch][i]));
	std::complex<double> V = _model.visibility(_datum_index_list[epoch][i],_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));
	
	V = _G[epoch][_is1_list[epoch][i]]*std::conj(_G[epoch][_is2_list[epoch][i]])*V;
	
	if (rank==0)
	  out << std::setw(15) << _data.datum(_datum_index_list[epoch][i]).u/1e9
	      << std::setw(15) << _data.datum(_datum_index_list[epoch][i]).v/1e9
	      << std::setw(15) << _data.datum(_datum_index_list[epoch][i]).V.real()
	    //<< std::setw(15) << _data.datum(_datum_index_list[epoch][i]).err.real()
	      << std::setw(15) << err.real()
	      << std::setw(15) << V.real()
	      << std::setw(15) << (_data.datum(_datum_index_list[epoch][i]).V-V).real()
	      << std::setw(15) << _data.datum(_datum_index_list[epoch][i]).V.imag()
	    //<< std::setw(15) << _data.datum(_datum_index_list[epoch][i]).err.imag()
	      << std::setw(15) << err.imag()
	      << std::setw(15) << V.imag()
	      << std::setw(15) << (_data.datum(_datum_index_list[epoch][i]).V-V).imag()
	      << '\n';
      }
    }
  }

  
  size_t likelihood_optimal_complex_gain_visibility::number_of_independent_gains()
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
         a. The number of visibility components (i.e., real and imaginary), twice the number of baselines
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


  void likelihood_optimal_complex_gain_visibility::organize_data_lists()
  {
    _datum_index_list.resize(_tge.size()-1);
    _y_list.resize(_tge.size()-1);
    _is1_list.resize(_tge.size()-1);
    _is2_list.resize(_tge.size()-1);

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector<size_t> id;
      std::vector< std::complex<double> > y;
      std::vector<std::string> s1,s2;
      std::vector<size_t> is1, is2;

      for (size_t i=0; i<_data.size(); ++i)
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Get the index
	  id.push_back(i);
	  
	  // Data V/sigma
	  y.push_back(std::complex<double>(_data.datum(i).V.real()/_data.datum(i).err.real(),_data.datum(i).V.imag()/_data.datum(i).err.imag()));
	  
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
      _y_list[epoch]=y;
      _datum_index_list[epoch]=id;
      _is1_list[epoch]=is1;
      _is2_list[epoch]=is2;

    }
  }


  double likelihood_optimal_complex_gain_visibility::operator()(std::vector<double>& x)
  {
    if (_parallelize_likelihood)
      return likelihood_multiproc(x);
    else
      return likelihood_uniproc(x);
  }

    
  double likelihood_optimal_complex_gain_visibility::likelihood_multiproc(std::vector<double>& x)
  {
    _x_debug = x; // DEBUGGING

    if (x==_x_last)
      return _L_last;
    _x_last = x;

    
    // Make sure that model and uncertainty are properly generated
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];




    model_image_adaptive_splined_raster* Rcache = find_single_raster_component(_model);
    /*
    if (_L_rank == 0) {
      std::cerr << "[CACHEDBG] Rcache=" << (Rcache ? "yes" : "no") << "\n";
      std::cerr << "[CACHEDBG] owned_ids=" 
		<< collect_owned_datum_indices(_datum_index_list, _L_rank, _L_size).size()
		<< "\n";
    }
    if (!Rcache) {
      std::cerr << "[CACHEDBG] find_single_raster_component(_model) returned nullptr in likelihood_multiproc\n";
      std::cerr << "[CACHEDBG] typeid(_model) = " << typeid(_model).name() << "\n";
      std::exit(1);
    }
    */

    /*  
    if (Rcache)
      Rcache->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);
    // Rcache->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::OwnedGlobal);

    _model.generate_model(mx);

    if (Rcache)
      Rcache->prepare_visibility_cache(_data, collect_owned_datum_indices(_datum_index_list, _L_rank, _L_size));
    else
      _model.prepare_visibility_cache(_data, collect_owned_datum_indices(_datum_index_list, _L_rank, _L_size));
*/


    if (Rcache)
      Rcache->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);

    _model.generate_model(mx);

    const std::vector<size_t> all_ids = collect_all_datum_indices(_data.size());

    if (Rcache)
      Rcache->prepare_visibility_cache(_data, all_ids);
    else
      _model.prepare_visibility_cache(_data, all_ids);

    
    
    _uncertainty.generate_uncertainty(ux);


    
    // Log-likelihood accumulator
    double L = 0;

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      if (epoch%_L_size==size_t(_L_rank))
      {
	// Get vector of error-normed model and data visibilities once
	std::vector< std::complex<double> > yb, y;
	std::vector<size_t> is1, is2;
	
	double lognorm = 0.0;      
	for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
	{
	  //std::complex<double> err_orig = _data.datum(_datum_index_list[epoch][i]).err; // Data error
	  std::complex<double> err = _uncertainty.error(_data.datum(_datum_index_list[epoch][i]));
	  std::complex<double> Vd = _data.datum(_datum_index_list[epoch][i]).V;
	  std::complex<double> Vm = _model.visibility(_datum_index_list[epoch][i],_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

	  yb.push_back( std::complex<double>(Vm.real()/err.real(), Vm.imag()/err.imag()) );
	  y.push_back( std::complex<double>(Vd.real()/err.real(), Vd.imag()/err.imag()) );
	  
	  lognorm += _uncertainty.log_normalization(_data.datum(_datum_index_list[epoch][i]));
	}
	//y = _y_list[epoch];
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
	for (size_t i=0; i<y.size(); ++i)
	{
	  std::complex<double> GGyb=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[i];
	  dL += -0.5 * ( std::pow( y[i].real() - GGyb.real(), 2) + std::pow( y[i].imag() - GGyb.imag(), 2) );
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

  double likelihood_optimal_complex_gain_visibility::likelihood_uniproc(std::vector<double>& x)
  {
    _x_debug = x; // DEBUGGING

    if (x==_x_last)
      return _L_last;
    _x_last = x;

    
    // Make sure that model and uncertainty are properly generated
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];

    set_raster_cache_mode_recursive(_model, model_image_adaptive_splined_raster::VisibilityCacheMode::Global);    
    // if (auto* mr = dynamic_cast<model_image_adaptive_splined_raster*>(&_model)) {
    //   mr->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);
    // }



    model_image_adaptive_splined_raster* Rcache = find_single_raster_component(_model);

    if (Rcache)
      Rcache->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);

    _model.generate_model(mx);

    if (Rcache)
      Rcache->prepare_visibility_cache(_data, collect_all_datum_indices(_data.size()));
    else
      _model.prepare_visibility_cache(_data, collect_all_datum_indices(_data.size()));

    _uncertainty.generate_uncertainty(ux);
    
    // _model.generate_model(mx);
    // _uncertainty.generate_uncertainty(ux);



    
    // Log-likelihood accumulator
    double L = 0;

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector< std::complex<double> > yb, y;
      std::vector<size_t> is1, is2;

      double lognorm = 0.0;      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	//std::complex<double> err_orig = _data.datum(_datum_index_list[epoch][i]).err; // Data error
	std::complex<double> err = _uncertainty.error(_data.datum(_datum_index_list[epoch][i]));
	std::complex<double> Vd = _data.datum(_datum_index_list[epoch][i]).V;
	std::complex<double> Vm = _model.visibility(_datum_index_list[epoch][i],_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

	yb.push_back( std::complex<double>(Vm.real()/err.real(), Vm.imag()/err.imag()) );
	y.push_back( std::complex<double>(Vd.real()/err.real(), Vd.imag()/err.imag()) );

	lognorm += _uncertainty.log_normalization(_data.datum(_datum_index_list[epoch][i]));
      }
      //y = _y_list[epoch];
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
      for (size_t i=0; i<y.size(); ++i)
      {
	std::complex<double> GGyb=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[i];
	dL += -0.5 * ( std::pow( y[i].real() - GGyb.real(), 2) + std::pow( y[i].imag() - GGyb.imag(), 2) );
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
    _L_last = L;

    return L;
  }

  void likelihood_optimal_complex_gain_visibility::distribute_gains()
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
    // if (wrank==2) {
    //   std::cerr << "DEBUGGING distribute gains 1 (ALMA):"
    // 		<< " (" << _L_rank << "/" << _L_size << ")"
    // 		<< " (" << wrank << "/" << wsize << ")"
    // 		<< " (" << lrank << "/" << lsize << ")"
    // 		<< " (" << rank << "/" << size << ")"
    // 		<< "\n";
    //   i=0;
    //   for (size_t j=0; j<_tge.size()-1; ++j)
    //   {
    // 	std::cerr << std::setw(5) << j
    // 		  << std::setw(15) << _G[j][0].real()
    // 		  << std::setw(15) << _G[j][0].imag()
    // 		  << std::setw(15) << local_buff[i]
    // 		  << std::setw(15) << local_buff[i+1]
    // 		  << '\n';
    // 	i += 2*_sigma_g.size() + 1;
    //   }
    //   std::cerr << "----------------------------------\n";
    // }
    
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
  
  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient_fd_all(std::vector<double>& x, prior& Pr)
  {
    utils::ScopedTimer T(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
   
    static bool once = false;
    int wrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    if (!once && wrank == 0) {
      std::cerr << "[CHECK] likelihood_visibility::gradient_fd_all() entered\n" << std::flush;
      once = true;
    }
    
    // Make sure that gains are computed
    this->operator()(x);

    // bool parallelizing_likelihoods = _parallelize_likelihood;
    // if (_parallelize_likelihood)
    //   _parallelize_likelihood = false;
    
    // Fix the gains (and remember the up to now state)
    bool solving_for_gains = _solve_for_gains;
    if (_solve_for_gains_during_gradient==false)
      fix_gains();

    // Compute the finite-difference gradient (we parallelize on likelihood construction here)
    std::vector<double> grad = likelihood_base::gradient_uniproc(x,Pr);

    // Enable solving for gains again if we were doing so before
    if (_solve_for_gains_during_gradient==false && solving_for_gains) 
      solve_for_gains();

    // _parallelize_likelihood = parallelizing_likelihoods;
    
    // Return gradients
    return grad;
  }

  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient(std::vector<double>& x, prior& Pr)
  {
    utils::ScopedTimer T(utils::TimerID::GradientTotal, timer_ns_, timer_calls_);

    static bool once=false;
    if (!once) {
      std::cerr << "[GRAD] lvg::gradient entered (mode=" << int(gradient_mode()) << ")\n";
      once=true;
    }
    return gradient_dispatch_(x, Pr);
  }


  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient_dispatch_(std::vector<double>& x, prior& Pr)
  {
    switch (_grad_mode)
      {
      case GradientMode::FD_ALL:
	return gradient_fd_all(x, Pr);
	
      case GradientMode::HYBRID_INTENSITY:
	return gradient_hybrid(x, Pr, /*do_geom=*/false);
	
      case GradientMode::HYBRID_INTENSITY_GEOM:
	return gradient_hybrid(x, Pr, /*do_geom=*/true);
	
      default:
	return gradient_fd_all(x, Pr);
      }
  }
    

  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient_uniproc(std::vector<double>& x, prior& Pr)
  {
    return gradient_dispatch_(x, Pr);
  }



  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient_hybrid(std::vector<double>& x, prior& Pr, bool do_geom)
  {
    const double Lx = ((_x_last.empty() || x != _x_last) ? this->operator()(x) : _L_last);

    const bool solving_for_gains_prev = _solve_for_gains;
    if (!_solve_for_gains_during_gradient)
      fix_gains();

    auto restore_basepoint = [&]()
    {
      std::vector<double> mx(_model.size()), ux(_uncertainty.size());
      size_t ii = 0;

      for (size_t j = 0; j < _model.size(); ++j)
        mx[j] = x[ii++];

      for (size_t j = 0; j < _uncertainty.size(); ++j)
        ux[j] = x[ii++];

      model_image_adaptive_splined_raster* Rcache = find_single_raster_component(_model);
      if (Rcache)
        Rcache->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);

      _model.generate_model(mx);

      const std::vector<size_t> all_ids = collect_all_datum_indices(_data.size());
      if (Rcache)
        Rcache->prepare_visibility_cache(_data, all_ids);
      else
        _model.prepare_visibility_cache(_data, all_ids);

      _uncertainty.generate_uncertainty(ux);
      _x_last = x;
      _L_last = Lx;
    };

    model_image_adaptive_splined_raster* direct_top =
      dynamic_cast<model_image_adaptive_splined_raster*>(&_model);

    model_image_asymmetric_gaussian* direct_ag_top =
      dynamic_cast<model_image_asymmetric_gaussian*>(&_model);

    model_image_xsringauss* direct_xs_top =
      dynamic_cast<model_image_xsringauss*>(&_model);

    model_image_sum* sum_top =
      dynamic_cast<model_image_sum*>(&_model);

    struct CompInfo {
      model_image_adaptive_splined_raster* r = nullptr;
      size_t p0 = 0;
      size_t Nx = 0;
      size_t Ny = 0;
      size_t Npix = 0;
      size_t idx_fovx = 0;
      size_t idx_fovy = 0;
      size_t idx_pa   = 0;
      bool has_shift = false;
      size_t idx_xoff = 0;
      size_t idx_yoff = 0;
      std::vector<double> xfrac;
      std::vector<double> yfrac;
    };

    struct AGInfo {
      model_image_asymmetric_gaussian* g = nullptr;
      size_t p0 = 0;
      bool has_shift = false;
      size_t idx_flux  = 0;
      size_t idx_sigma = 0;
      size_t idx_A     = 0;
      size_t idx_pa    = 0;
      size_t idx_xoff  = 0;
      size_t idx_yoff  = 0;
    };

    struct XSInfo {
      model_image_xsringauss* xg = nullptr;
      size_t p0 = 0;
      bool has_shift = false;
      size_t idx_flux = 0;
      size_t idx_R    = 0;
      size_t idx_psi  = 0;
      size_t idx_eps  = 0;
      size_t idx_f    = 0;
      size_t idx_gax  = 0;
      size_t idx_aq   = 0;
      size_t idx_gq   = 0;
      size_t idx_pa   = 0;
      size_t idx_xoff = 0;
      size_t idx_yoff = 0;
    };

    std::vector<CompInfo> comps;
    std::vector<AGInfo> ags;
    std::vector<XSInfo> xss;
    const size_t Npar = x.size();

    auto fill_pixel_fracs = [](CompInfo& c)
    {
      c.xfrac.resize(c.Npix);
      c.yfrac.resize(c.Npix);

      for (size_t ix = 0; ix < c.Nx; ++ix)
      {
        const double xf = (c.Nx > 1) ? (double(ix) / double(c.Nx - 1) - 0.5) : 0.0;
        for (size_t iy = 0; iy < c.Ny; ++iy)
        {
          const double yf = (c.Ny > 1) ? (double(iy) / double(c.Ny - 1) - 0.5) : 0.0;
          const size_t k = ix * c.Ny + iy;
          c.xfrac[k] = xf;
          c.yfrac[k] = yf;
        }
      }
    };

    if (direct_top)
    {
      CompInfo c;
      c.r = direct_top;
      c.p0 = 0;
      c.Nx = direct_top->Nx();
      c.Ny = direct_top->Ny();
      c.Npix = c.Nx * c.Ny;
      c.idx_fovx = c.p0 + c.Npix;
      c.idx_fovy = c.p0 + c.Npix + 1;
      c.idx_pa   = c.p0 + c.Npix + 2;
      c.has_shift = false;
      fill_pixel_fracs(c);
      comps.push_back(c);
    }
    else if (direct_ag_top)
    {
      AGInfo a;
      a.g = direct_ag_top;
      a.p0 = 0;
      a.has_shift = false;
      a.idx_flux  = 0;
      a.idx_sigma = 1;
      a.idx_A     = 2;
      a.idx_pa    = 3;
      ags.push_back(a);
    }
    else if (direct_xs_top)
    {
      XSInfo xs;
      xs.xg = direct_xs_top;
      xs.p0 = 0;
      xs.has_shift = false;
      xs.idx_flux = 0;
      xs.idx_R    = 1;
      xs.idx_psi  = 2;
      xs.idx_eps  = 3;
      xs.idx_f    = 4;
      xs.idx_gax  = 5;
      xs.idx_aq   = 6;
      xs.idx_gq   = 7;
      xs.idx_pa   = 8;
      xss.push_back(xs);
    }
    else if (sum_top)
    {
      size_t p = 0;
      const auto& imgs = sum_top->components();

      for (size_t j = 0; j < imgs.size(); ++j)
      {
        if (auto* r = dynamic_cast<model_image_adaptive_splined_raster*>(imgs[j]))
        {
          CompInfo c;
          c.r = r;
          c.p0 = p;
          c.Nx = r->Nx();
          c.Ny = r->Ny();
          c.Npix = c.Nx * c.Ny;
          c.idx_fovx = c.p0 + c.Npix;
          c.idx_fovy = c.p0 + c.Npix + 1;
          c.idx_pa   = c.p0 + c.Npix + 2;
          c.has_shift = true;
          c.idx_xoff = c.p0 + r->size();
          c.idx_yoff = c.p0 + r->size() + 1;
          fill_pixel_fracs(c);
          comps.push_back(c);
        }
        else if (auto* g = dynamic_cast<model_image_asymmetric_gaussian*>(imgs[j]))
        {
          AGInfo a;
          a.g = g;
          a.p0 = p;
          a.has_shift = true;
          a.idx_flux  = p + 0;
          a.idx_sigma = p + 1;
          a.idx_A     = p + 2;
          a.idx_pa    = p + 3;
          a.idx_xoff  = p + g->size();
          a.idx_yoff  = p + g->size() + 1;
          ags.push_back(a);
        }
        else if (auto* xs = dynamic_cast<model_image_xsringauss*>(imgs[j]))
        {
          XSInfo xsi;
          xsi.xg = xs;
          xsi.p0 = p;
          xsi.has_shift = true;
          xsi.idx_flux = p + 0;
          xsi.idx_R    = p + 1;
          xsi.idx_psi  = p + 2;
          xsi.idx_eps  = p + 3;
          xsi.idx_f    = p + 4;
          xsi.idx_gax  = p + 5;
          xsi.idx_aq   = p + 6;
          xsi.idx_gq   = p + 7;
          xsi.idx_pa   = p + 8;
          xsi.idx_xoff = p + xs->size();
          xsi.idx_yoff = p + xs->size() + 1;
          xss.push_back(xsi);
        }

        p += imgs[j]->size();
        p += 2;
      }
    }
    else
    {
      std::vector<double> g;
      {
        utils::ScopedTimer Tfd(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
        g = likelihood_base::gradient_uniproc(x, Pr);
      }
      restore_basepoint();
      if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
        solve_for_gains();
      return g;
    }

    if (comps.empty() && ags.empty() && xss.empty())
    {
      std::vector<double> g;
      {
        utils::ScopedTimer Tfd(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
        g = likelihood_base::gradient_uniproc(x, Pr);
      }
      restore_basepoint();
      if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
        solve_for_gains();
      return g;
    }

    int local_ok = 1;
    for (const auto& c : comps)
    {
      const model_image_adaptive_splined_raster* rc = c.r;

      if (!rc->use_cached_exp_getter()) local_ok = 0;
      if (!rc->phase_cache_valid())     local_ok = 0;

      if (rc->phase_cache().size() < _data.size() * c.Npix) local_ok = 0;
      if (rc->spline_kernel_cache().size() < _data.size())  local_ok = 0;
      if (rc->I_flat().size() < c.Npix)                     local_ok = 0;

      if (do_geom)
      {
        if (rc->spline_kernel_dfovx_cache().size() < _data.size()) local_ok = 0;
        if (rc->spline_kernel_dfovy_cache().size() < _data.size()) local_ok = 0;
        if (rc->spline_kernel_dpa_cache().size()   < _data.size()) local_ok = 0;
      }
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, _Lcomm);

    if (!global_ok)
    {
      std::vector<double> g;
      {
        utils::ScopedTimer Tfd(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);
        g = likelihood_base::gradient_uniproc(x, Pr);
      }
      restore_basepoint();
      if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
        solve_for_gains();
      return g;
    }

    std::vector<double> grad_local(Npar, 0.0);
    std::vector<unsigned char> analytic_mask(Npar, 0);

    for (const auto& c : comps)
    {
      for (size_t k = 0; k < c.Npix; ++k)
        analytic_mask[c.p0 + k] = 1;

      if (c.has_shift)
      {
        analytic_mask[c.idx_xoff] = 1;
        analytic_mask[c.idx_yoff] = 1;
      }

      if (do_geom)
      {
        analytic_mask[c.idx_fovx] = 1;
        analytic_mask[c.idx_fovy] = 1;
        analytic_mask[c.idx_pa]   = 1;
      }
    }

    for (const auto& a : ags)
    {
      analytic_mask[a.idx_flux] = 1;

      if (a.has_shift)
      {
        analytic_mask[a.idx_xoff] = 1;
        analytic_mask[a.idx_yoff] = 1;
      }

      if (do_geom)
      {
        analytic_mask[a.idx_sigma] = 1;
        analytic_mask[a.idx_A]     = 1;
        analytic_mask[a.idx_pa]    = 1;
      }
    }

    for (const auto& xs : xss)
    {
      analytic_mask[xs.idx_flux] = 1;

      if (xs.has_shift)
      {
        analytic_mask[xs.idx_xoff] = 1;
        analytic_mask[xs.idx_yoff] = 1;
      }

      if (do_geom)
      {
        analytic_mask[xs.idx_R]   = 1;
        analytic_mask[xs.idx_psi] = 1;
        analytic_mask[xs.idx_eps] = 1;
        analytic_mask[xs.idx_f]   = 1;
        analytic_mask[xs.idx_gax] = 1;
        analytic_mask[xs.idx_aq]  = 1;
        analytic_mask[xs.idx_gq]  = 1;
        analytic_mask[xs.idx_pa]  = 1;
      }
    }

    const std::complex<double> minus_i(0.0, -1.0);
    const double two_pi = 2.0 * M_PI;

    auto BesselJ0 = [](double x) -> double
    {
      double ax, z;
      double xx, y, ans, ans1, ans2;

      if ((ax = std::fabs(x)) < 8.0) {
        y = x*x;
        ans1 = 57568490574.0 + y*(-13362590354.0 + y*(651619640.7
             + y*(-11214424.18 + y*(77392.33017 + y*(-184.9052456)))));
        ans2 = 57568490411.0 + y*(1029532985.0 + y*(9494680.718
             + y*(59272.64853 + y*(267.8532712 + y*1.0))));
        ans = ans1 / ans2;
      }
      else {
        z = 8.0 / ax;
        y = z*z;
        xx = ax - 0.785398164;
        ans1 = 1.0 + y*(-0.1098628627e-2 + y*(0.2734510407e-4
             + y*(-0.2073370639e-5 + y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1 + y*(0.1430488765e-3
             + y*(-0.6911147651e-5 + y*(0.7621095161e-6
             - y*0.934945152e-7)));
        ans = std::sqrt(0.636619772 / ax) * (std::cos(xx) * ans1 - z * std::sin(xx) * ans2);
      }
      return ans;
    };

    auto BesselJ1 = [](double x) -> double
    {
      double ax, z;
      double xx, y, ans, ans1, ans2;

      if ((ax = std::fabs(x)) < 8.0) {
        y = x*x;
        ans1 = x*(72362614232.0 + y*(-7895059235.0 + y*(242396853.1
             + y*(-2972611.439 + y*(15704.48260 + y*(-30.16036606))))));
        ans2 = 144725228442.0 + y*(2300535178.0 + y*(18583304.74
             + y*(99447.43394 + y*(376.9991397 + y*1.0))));
        ans = ans1 / ans2;
      } else {
        z = 8.0 / ax;
        y = z*z;
        xx = ax - 2.356194491;
        ans1 = 1.0 + y*(0.183105e-2 + y*(-0.3516396496e-4
             + y*(0.2457520174e-5 + y*(-0.240337019e-6))));
        ans2 = 0.04687499995 + y*(-0.2002690873e-3
             + y*(0.8449199096e-5 + y*(-0.88228987e-6
             + y*0.105787412e-6)));
        ans = std::sqrt(0.636619772 / ax) * (std::cos(xx) * ans1 - z * std::sin(xx) * ans2);
        if (x < 0.0) ans = -ans;
      }
      return ans;
    };

    auto BesselJ2 = [&](double x) -> double
    {
      if (std::fabs(x) < 1.0e-14)
        return 0.0;
      return (2.0 * BesselJ1(x) / x - BesselJ0(x));
    };

    auto xsring_component_visibility = [&](const double p[11], double u, double v) -> std::complex<double>
    {
      const double raw_V0  = p[0];
      const double raw_R   = p[1];
      const double raw_psi = p[2];
      const double raw_eps = p[3];
      const double raw_f   = p[4];
      const double raw_gax = p[5];
      const double raw_aq  = p[6];
      const double raw_gq  = p[7];
      const double pa      = p[8];
      const double xoff    = p[9];
      const double yoff    = p[10];

      const double V0    = std::max(1e-8,  raw_V0);
      const double Rext  = std::max(1e-20, raw_R);
      const double qpsi  = std::min(std::max(1.0 - raw_psi, 1e-4), 0.9999);
      const double Rint  = qpsi * Rext;
      const double epsc  = std::min(std::max(raw_eps, 1e-4), 0.9999);
      const double dcen  = epsc * (Rext - Rint);
      const double fc    = std::min(std::max(raw_f, 1e-4), 0.9999);
      const double gaxc  = std::max(raw_gax, 1e-4);
      const double sa    = gaxc * Rext;
      const double aqc   = std::max(raw_aq, 1e-4);
      const double sb    = aqc * sa;
      const double gqc   = std::min(std::max(raw_gq, 1e-4), 0.9999);

      const double cpa = std::cos(pa);
      const double spa = std::sin(pa);

      double ru = u*cpa + v*spa;
      double rv = -u*spa + v*cpa;

      ru *= -1.0;

      const double k = two_pi * std::sqrt(ru*ru + rv*rv);
      const std::complex<double> I(0.0, 1.0);

      std::complex<double> Vring(1.0, 0.0);
      std::complex<double> Vgauss(1.0, 0.0);

      if (k > 1.0e-14)
      {
        const double H = (2.0 / M_PI) /
          ((1.0 + fc) * (Rext*Rext - Rint*Rint) - (1.0 - fc) * dcen * Rint*Rint / Rext);

        const std::complex<double> exponent = -two_pi * I * dcen * ru;

        const double J0e = BesselJ0(k*Rext);
        const double J1e = BesselJ1(k*Rext);
        const double J2e = BesselJ2(k*Rext);

        const double J0i = BesselJ0(k*Rint);
        const double J1i = BesselJ1(k*Rint);
        const double J2i = BesselJ2(k*Rint);

        const std::complex<double> term1 =
          (M_PI * H / k) * (1.0 + fc) * Rext * J1e;

        const std::complex<double> term2 =
          -(M_PI * H / k) * ((1.0 + fc) - (1.0 - fc) * dcen / Rext) * std::exp(exponent) * Rint * J1i;

        const std::complex<double> term3 =
          -(I * M_PI * H / (2.0 * k * k)) * two_pi * ru * (1.0 - fc) *
          (Rext * J0e - Rext * J2e - 2.0 * J1e / k);

        const std::complex<double> term4 =
          +(I * M_PI * H / (2.0 * k * k)) * two_pi * ru * (1.0 - fc) *
          (Rint * J0i - Rint * J2i - 2.0 * J1i / k) *
          (Rint / Rext) * std::exp(exponent);

        Vring = term1 + term2 + term3 + term4;

        const double exponent_gauss_arg =
          -2.0 * M_PI * M_PI * (ru*ru*(sa*sa) + rv*rv*(sb*sb));
        const std::complex<double> exponent_gauss =
          -two_pi * I * (dcen - Rint) * ru;

        Vgauss = (exponent_gauss_arg < -200.0)
          ? std::complex<double>(0.0, 0.0)
          : std::exp(exponent_gauss + exponent_gauss_arg);
      }

      const std::complex<double> V0comp = V0 * ((1.0 - gqc) * Vring + gqc * Vgauss);

      if (xoff == 0.0 && yoff == 0.0)
        return V0comp;

      const double psi_shift = -two_pi * (u * xoff + v * yoff);
      const std::complex<double> Eshift = std::exp(std::complex<double>(0.0, psi_shift));
      return Eshift * V0comp;
    };

    {
      utils::ScopedTimer Tana(utils::TimerID::GradientAnalytic, timer_ns_, timer_calls_);

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
          const size_t d_idx = idx_list[ii];
          datum_visibility& d = _data.datum(d_idx);

          const std::complex<double> err = _uncertainty.error(d);
          const double er = err.real();
          const double ei = err.imag();
          if (er == 0.0 || ei == 0.0)
            continue;

          const double inv_er = 1.0 / er;
          const double inv_ei = 1.0 / ei;

          const std::complex<double> y(d.V.real() * inv_er, d.V.imag() * inv_ei);

          const std::complex<double> Vm_full =
            _model.visibility(d_idx, d, 0.25 * std::abs(err));

          const std::complex<double> yb_full(Vm_full.real() * inv_er,
                                             Vm_full.imag() * inv_ei);

          const std::complex<double> g =
            _G[epoch][is1_list[ii]] * std::conj(_G[epoch][is2_list[ii]]);

          const std::complex<double> pred(
            g.real() * yb_full.real() - g.imag() * yb_full.imag(),
            g.real() * yb_full.imag() + g.imag() * yb_full.real()
          );

          const double rr = y.real() - pred.real();
          const double ri = y.imag() - pred.imag();

          const double u = d.u;
          const double v = d.v;

          auto accum_from_dVm = [&](size_t idx, const std::complex<double>& dVm)
          {
            const std::complex<double> dyb(dVm.real() * inv_er, dVm.imag() * inv_ei);
            const std::complex<double> dp(
              g.real() * dyb.real() - g.imag() * dyb.imag(),
              g.real() * dyb.imag() + g.imag() * dyb.real()
            );
            grad_local[idx] += rr * dp.real() + ri * dp.imag();
          };

          for (const auto& c : comps)
          {
            const auto& phase = c.r->phase_cache();
            const auto& K     = c.r->spline_kernel_cache();
            const auto& Iflat = c.r->I_flat();

            const double pa   = x[c.idx_pa];
            const double cpa  = std::cos(pa);
            const double spa  = std::sin(pa);
            const double fovx = x[c.idx_fovx];
            const double fovy = x[c.idx_fovy];

            const double shiftx = c.has_shift ? x[c.idx_xoff] : 0.0;
            const double shifty = c.has_shift ? x[c.idx_yoff] : 0.0;

            const double ur =  cpa * u + spa * v;
            const double vr = -spa * u + cpa * v;

            const double psi = -two_pi * (u * shiftx + v * shifty);
            const std::complex<double> E =
              c.has_shift ? std::exp(std::complex<double>(0.0, psi))
                          : std::complex<double>(1.0, 0.0);

            const size_t off = d_idx * c.Npix;
            const double Ki  = K[d_idx];

            std::complex<double> S0(0.0, 0.0), Sx(0.0, 0.0), Sy(0.0, 0.0);
            for (size_t k = 0; k < c.Npix; ++k)
            {
              const std::complex<double> ph = phase[off + k];
              const double Ik = Iflat[k];
              const std::complex<double> Ikph = Ik * ph;
              S0 += Ikph;
              if (do_geom)
              {
                Sx += Ikph * c.xfrac[k];
                Sy += Ikph * c.yfrac[k];
              }
            }

            const std::complex<double> Vc = E * (Ki * S0);

            if (c.has_shift)
            {
              const std::complex<double> dVm_dshiftx = (minus_i * (two_pi * u)) * Vc;
              const std::complex<double> dVm_dshifty = (minus_i * (two_pi * v)) * Vc;

              const std::complex<double> dyb_dshiftx(dVm_dshiftx.real() * inv_er,
                                                     dVm_dshiftx.imag() * inv_ei);
              const std::complex<double> dyb_dshifty(dVm_dshifty.real() * inv_er,
                                                     dVm_dshifty.imag() * inv_ei);

              const std::complex<double> dp_dshiftx(
                g.real() * dyb_dshiftx.real() - g.imag() * dyb_dshiftx.imag(),
                g.real() * dyb_dshiftx.imag() + g.imag() * dyb_dshiftx.real()
              );
              const std::complex<double> dp_dshifty(
                g.real() * dyb_dshifty.real() - g.imag() * dyb_dshifty.imag(),
                g.real() * dyb_dshifty.imag() + g.imag() * dyb_dshifty.real()
              );

              grad_local[c.idx_xoff] += rr * dp_dshiftx.real() + ri * dp_dshiftx.imag();
              grad_local[c.idx_yoff] += rr * dp_dshifty.real() + ri * dp_dshifty.imag();
            }

            if (do_geom)
            {
              const auto& dKx = c.r->spline_kernel_dfovx_cache();
              const auto& dKy = c.r->spline_kernel_dfovy_cache();
              const auto& dKp = c.r->spline_kernel_dpa_cache();

              const std::complex<double> dS0_dfovx = (minus_i * (two_pi * ur)) * Sx;
              const std::complex<double> dS0_dfovy = (minus_i * (two_pi * vr)) * Sy;
              const std::complex<double> dS0_dpa   =
                (minus_i * two_pi) * ((vr * fovx) * Sx - (ur * fovy) * Sy);

              const std::complex<double> dVm_dfovx = E * (dKx[d_idx] * S0 + Ki * dS0_dfovx);
              const std::complex<double> dVm_dfovy = E * (dKy[d_idx] * S0 + Ki * dS0_dfovy);
              const std::complex<double> dVm_dpa   = E * (dKp[d_idx] * S0 + Ki * dS0_dpa);

              const std::complex<double> dyb_dfovx(dVm_dfovx.real() * inv_er,
                                                   dVm_dfovx.imag() * inv_ei);
              const std::complex<double> dyb_dfovy(dVm_dfovy.real() * inv_er,
                                                   dVm_dfovy.imag() * inv_ei);
              const std::complex<double> dyb_dpa(dVm_dpa.real() * inv_er,
                                                 dVm_dpa.imag() * inv_ei);

              const std::complex<double> dp_dfovx(
                g.real() * dyb_dfovx.real() - g.imag() * dyb_dfovx.imag(),
                g.real() * dyb_dfovx.imag() + g.imag() * dyb_dfovx.real()
              );
              const std::complex<double> dp_dfovy(
                g.real() * dyb_dfovy.real() - g.imag() * dyb_dfovy.imag(),
                g.real() * dyb_dfovy.imag() + g.imag() * dyb_dfovy.real()
              );
              const std::complex<double> dp_dpa(
                g.real() * dyb_dpa.real() - g.imag() * dyb_dpa.imag(),
                g.real() * dyb_dpa.imag() + g.imag() * dyb_dpa.real()
              );

              grad_local[c.idx_fovx] += rr * dp_dfovx.real() + ri * dp_dfovx.imag();
              grad_local[c.idx_fovy] += rr * dp_dfovy.real() + ri * dp_dfovy.imag();
              grad_local[c.idx_pa]   += rr * dp_dpa.real()   + ri * dp_dpa.imag();
            }

            for (size_t k = 0; k < c.Npix; ++k)
            {
              const std::complex<double> z = E * (Ki * phase[off + k]);
              const std::complex<double> dzb(z.real() * inv_er, z.imag() * inv_ei);
              const std::complex<double> dp(
                g.real() * dzb.real() - g.imag() * dzb.imag(),
                g.real() * dzb.imag() + g.imag() * dzb.real()
              );

              grad_local[c.p0 + k] += Iflat[k] * (rr * dp.real() + ri * dp.imag());
            }
          }

          for (const auto& a : ags)
          {
            const double raw_flux  = x[a.idx_flux];
            const double raw_sigma = x[a.idx_sigma];
            const double raw_A     = x[a.idx_A];
            const double pa        = x[a.idx_pa];

            const double flux  = std::fabs(raw_flux);
            const double sigma = std::fabs(raw_sigma);
            const double Acl   = std::min(std::max(raw_A, 0.0), 0.99);

            const double s_flux  = (raw_flux >= 0.0 ? 1.0 : -1.0);
            const double s_sigma = (raw_sigma >= 0.0 ? 1.0 : -1.0);
            const bool   A_active = (raw_A > 0.0 && raw_A < 0.99);

            const double cpa = std::cos(pa);
            const double spa = std::sin(pa);

            const double shiftx = a.has_shift ? x[a.idx_xoff] : 0.0;
            const double shifty = a.has_shift ? x[a.idx_yoff] : 0.0;

            const double ru = -two_pi * (u * cpa + v * spa);
            const double rv =  two_pi * (-u * spa + v * cpa);

            const double sa2 = sigma * sigma / (1.0 + Acl);
            const double sb2 = sigma * sigma / (1.0 - Acl);

            const double expo = -0.5 * (ru * ru * sa2 + rv * rv * sb2);
            const double amp0 = (expo < -200.0 ? 0.0 : std::exp(expo));

            const double psi = -two_pi * (u * shiftx + v * shifty);
            const std::complex<double> E =
              a.has_shift ? std::exp(std::complex<double>(0.0, psi))
                          : std::complex<double>(1.0, 0.0);

            const std::complex<double> Vag = E * (flux * amp0);

            if (amp0 > 0.0)
            {
              const std::complex<double> dVm_dflux = E * (s_flux * amp0);
              accum_from_dVm(a.idx_flux, dVm_dflux);
            }

            if (a.has_shift)
            {
              const std::complex<double> dVm_dshiftx = (minus_i * (two_pi * u)) * Vag;
              const std::complex<double> dVm_dshifty = (minus_i * (two_pi * v)) * Vag;
              accum_from_dVm(a.idx_xoff, dVm_dshiftx);
              accum_from_dVm(a.idx_yoff, dVm_dshifty);
            }

            if (do_geom && amp0 > 0.0)
            {
              const double dF_dsigma =
                s_sigma * ( -sigma * (ru * ru / (1.0 + Acl) + rv * rv / (1.0 - Acl)) );
              const std::complex<double> dVm_dsigma = Vag * dF_dsigma;
              accum_from_dVm(a.idx_sigma, dVm_dsigma);

              const double dF_dA =
                A_active
                ? 0.5 * sigma * sigma *
                    ( ru * ru / ((1.0 + Acl) * (1.0 + Acl))
                    - rv * rv / ((1.0 - Acl) * (1.0 - Acl)) )
                : 0.0;
              const std::complex<double> dVm_dA = Vag * dF_dA;
              accum_from_dVm(a.idx_A, dVm_dA);

              const double dF_dpa = ru * rv * (sa2 - sb2);
              const std::complex<double> dVm_dpa = Vag * dF_dpa;
              accum_from_dVm(a.idx_pa, dVm_dpa);
            }
          }

          for (const auto& xs : xss)
          {
            double p[11];
            p[0]  = x[xs.idx_flux];
            p[1]  = x[xs.idx_R];
            p[2]  = x[xs.idx_psi];
            p[3]  = x[xs.idx_eps];
            p[4]  = x[xs.idx_f];
            p[5]  = x[xs.idx_gax];
            p[6]  = x[xs.idx_aq];
            p[7]  = x[xs.idx_gq];
            p[8]  = x[xs.idx_pa];
            p[9]  = xs.has_shift ? x[xs.idx_xoff] : 0.0;
            p[10] = xs.has_shift ? x[xs.idx_yoff] : 0.0;

            auto fd_xs_component = [&](int slot, size_t global_idx) -> std::complex<double>
            {
              double h = step_size(std::fabs(Pr.upper_bound(global_idx) - Pr.lower_bound(global_idx)));
              if (!(h > 0.0))
                h = 1e-6 * std::max(1.0, std::fabs(x[global_idx]));

              double pp[11], pm[11];
              std::memcpy(pp, p, 11 * sizeof(double));
              std::memcpy(pm, p, 11 * sizeof(double));

              pp[slot] += h;
              pm[slot] -= h;

              const std::complex<double> Vp = xsring_component_visibility(pp, u, v);
              const std::complex<double> Vm = xsring_component_visibility(pm, u, v);
              return (Vp - Vm) / (2.0 * h);
            };

            accum_from_dVm(xs.idx_flux, fd_xs_component(0, xs.idx_flux));

            if (xs.has_shift)
            {
              accum_from_dVm(xs.idx_xoff, fd_xs_component(9, xs.idx_xoff));
              accum_from_dVm(xs.idx_yoff, fd_xs_component(10, xs.idx_yoff));
            }

            if (do_geom)
            {
              accum_from_dVm(xs.idx_R,   fd_xs_component(1, xs.idx_R));
              accum_from_dVm(xs.idx_psi, fd_xs_component(2, xs.idx_psi));
              accum_from_dVm(xs.idx_eps, fd_xs_component(3, xs.idx_eps));
              accum_from_dVm(xs.idx_f,   fd_xs_component(4, xs.idx_f));
              accum_from_dVm(xs.idx_gax, fd_xs_component(5, xs.idx_gax));
              accum_from_dVm(xs.idx_aq,  fd_xs_component(6, xs.idx_aq));
              accum_from_dVm(xs.idx_gq,  fd_xs_component(7, xs.idx_gq));
              accum_from_dVm(xs.idx_pa,  fd_xs_component(8, xs.idx_pa));
            }
          }
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, grad_local.data(), (int)Npar, MPI_DOUBLE, MPI_SUM, _Lcomm);
    }

    std::vector<double> grad = grad_local;

    {
      utils::ScopedTimer Tfd(utils::TimerID::GradientFiniteDiff, timer_ns_, timer_calls_);

      std::vector<double> y = x;
      for (size_t p = 0; p < Npar; ++p)
      {
        if (analytic_mask[p])
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
    }

    restore_basepoint();

    if (!_solve_for_gains_during_gradient && solving_for_gains_prev)
      solve_for_gains();

    return grad;
  }
  

  double likelihood_optimal_complex_gain_visibility::chi_squared(std::vector<double>& x)
  {
    distribute_gains();
    
    // Make sure that model and uncertainty are properly generated
    std::vector<double> mx(_model.size()), ux(_uncertainty.size());
    size_t i=0;
    for (size_t j=0; j<_model.size(); ++j)
      mx[j] = x[i++];
    for (size_t j=0; j<_uncertainty.size(); ++j)
      ux[j] = x[i++];

    if (auto* mr = dynamic_cast<model_image_adaptive_splined_raster*>(&_model)) {
      mr->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);
    }
    _model.generate_model(mx);
    _model.prepare_visibility_cache(_data, collect_all_datum_indices(_data.size()));
    _uncertainty.generate_uncertainty(ux);

    // Log-likelihood accumulator
    double L = 0;

    // Remove the prior?
    std::vector<double> true_sigma_g = _sigma_g;
    //_sigma_g.assign(_sigma_g.size(),2.0);


    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector< std::complex<double> > yb, y;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	//std::complex<double> err_orig = _data.datum(_datum_index_list[epoch][i]).err; // Data error
	std::complex<double> err = _uncertainty.error(_data.datum(_datum_index_list[epoch][i]));
	std::complex<double> Vd = _data.datum(_datum_index_list[epoch][i]).V;
	std::complex<double> Vm = _model.visibility(_datum_index_list[epoch][i],_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

	yb.push_back( std::complex<double>(Vm.real()/err.real(), Vm.imag()/err.imag()) );
	y.push_back( std::complex<double>(Vd.real()/err.real(), Vd.imag()/err.imag()) );
      }
      //y = _y_list[epoch];
      is1 = _is1_list[epoch];
      is2 = _is2_list[epoch];

      if (_solve_for_gains)
      {
	/*
	// Determine the initial guess for the gains based on currently stated assumptions.
	if (epoch>0)
	{
	  if (_smoothly_varying_gains)
	    _G[epoch] = _G[epoch-1];
	  else
	  {
	    for (size_t a=0; a<_sigma_g.size(); ++a)
	      _G[epoch][a] = std::complex<double>(1.0,0.0);
	  }
	}
	*/
	
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
	optimal_complex_gains(y,yb,is1,is2,_G[epoch]);
      }

      // Add in the direct likelihood
      double dL = 0.0;
      for (size_t i=0; i<y.size(); ++i)
      {
	std::complex<double> GGyb=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[i];
	dL += -0.5 * ( std::pow( y[i].real() - GGyb.real(), 2) + std::pow( y[i].imag() - GGyb.imag(), 2) );
      }

      L += dL;
    }

    // Reset the prior
    _sigma_g = true_sigma_g;

    return (-2.0*L);
  }

  double likelihood_optimal_complex_gain_visibility::chi_squared_with_gain_priors(std::vector<double>& x)
  {
    distribute_gains();

    if (auto* mr = dynamic_cast<model_image_adaptive_splined_raster*>(&_model)) {
      mr->set_cache_mode(model_image_adaptive_splined_raster::VisibilityCacheMode::Global);
    }
    
    _model.generate_model(x);
    _model.prepare_visibility_cache(_data, collect_all_datum_indices(_data.size()));
    
    // Log-likelihood accumulator
    double L = 0;

    // Remove the prior?
    std::vector<double> true_sigma_g = _sigma_g;
    //_sigma_g.assign(_sigma_g.size(),2.0);

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector< std::complex<double> > yb, y;
      std::vector<size_t> is1, is2;

      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	std::complex<double> err = _data.datum(_datum_index_list[epoch][i]).err;
	std::complex<double> Vm = _model.visibility(_datum_index_list[epoch][i],_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

	yb.push_back( std::complex<double>(Vm.real()/err.real(), Vm.imag()/err.imag()) );
      }

      y = _y_list[epoch];
      is1 = _is1_list[epoch];
      is2 = _is2_list[epoch];

      if (_solve_for_gains)
      {
	// Determine the initial guess for the gains based on currently stated assumptions.
	if (epoch>0)
	{
	  if (_smoothly_varying_gains)
	    _G[epoch] = _G[epoch-1];
	  else
	  {
	    for (size_t a=0; a<_sigma_g.size(); ++a)
	      _G[epoch][a] = std::complex<double>(1.0,0.0);
	  }
	}

	// Levenberg-Marquardt minimization of full likelihood
	optimal_complex_gains(y,yb,is1,is2,_G[epoch]);
      }

      // Add in the direct likelihood
      double dL = 0.0;
      for (size_t i=0; i<y.size(); ++i)
      {
	std::complex<double> GGyb=_G[epoch][is1[i]]*std::conj(_G[epoch][is2[i]])*yb[i];
	dL += -0.5 * ( std::pow( y[i].real() - GGyb.real(), 2) + std::pow( y[i].imag() - GGyb.imag(), 2) );	
      }

      // Add gain prior terms
      for (size_t a=0; a<_sigma_g.size(); ++a)
	dL += -0.5 * std::pow( std::real(std::log(_G[epoch][a]))/_sigma_g[a], 2);

      L += dL;
    }

    // Reset the prior
    _sigma_g = true_sigma_g;

    return (-2.0*L);
  }
  
  
  double likelihood_optimal_complex_gain_visibility::matrix_determinant(double **a)
  {
    utils::ScopedTimer T(utils::TimerID::matrix_determinant, timer_ns_, timer_calls_);

    int n = int(_sigma_g.size());
    double d;
    //double **a, d;
    //int i,*indx;
    int i;
    
    //indx = new int[n+1];
    //_indx = new int[n+1];

    ludcmp(a,n,_indx,d);

    // Find determinant of a
    for (i=1; i<=n; ++i)
      d *= a[i][i];

    // Clean up memory
    // delete[] indx;
    //delete[] _indx;

    return d;
  }
  

#define TINY 1.0e-20;
  void likelihood_optimal_complex_gain_visibility::ludcmp(double **a, int n, int *indx, double &d)
  {
    int i,imax=0,j,k;
    double big,dum,sum,temp;
    //double *vv = new double[n+1];
    //_vv = new double[n+1];
    
    
    d=1.0;
    for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
	if ((temp=std::fabs(a[i][j])) > big)
	  big=temp;
      if (big == 0.0)
	std::cerr << "Singular matrix in routine ludcmp";
      _vv[i]=1.0/big;
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
	if ( (dum=_vv[i]*std::fabs(sum)) >= big) {
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
	_vv[imax]=_vv[j];
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
    //delete[] vv;
    //delete[] _vv;
  }
#undef TINY

  std::vector<double> likelihood_optimal_complex_gain_visibility::get_gain_times()
  {
    return ( _tge );
  }

  std::vector< std::vector< std::complex<double> > > likelihood_optimal_complex_gain_visibility::get_gains()
  {
    return ( _G );
  }

  void likelihood_optimal_complex_gain_visibility::read_gain_file(std::string gain_file_name)
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
	  std::cerr << "ERROR: likelihood_optimal_complex_gain_visibility::read_gain_file too few gains in " << gain_file_name << '\n';
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

  void likelihood_optimal_complex_gain_visibility::output_gains(std::ostream& out)
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
  
  void likelihood_optimal_complex_gain_visibility::output_gains(std::string outname)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Knows that this is the bottom tempering level

    distribute_gains();

    if (rank==0)
    {
      std::ofstream out(outname.c_str());
      output_gains(out);
      out.close();
    }
  }

  void likelihood_optimal_complex_gain_visibility::output_gain_corrections(std::ostream& out)
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
  
  void likelihood_optimal_complex_gain_visibility::output_gain_corrections(std::string outname)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Knows that this is the bottom tempering level

    distribute_gains();
        
    if (rank==0)
    {
      std::ofstream out(outname.c_str());
      output_gain_corrections(out);
      out.close();
    }
  }


  double likelihood_optimal_complex_gain_visibility::optimal_complex_gains(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest)
  {
    utils::ScopedTimer T(utils::TimerID::GainsSolveTotal, timer_ns_, timer_calls_);
    
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


  /*
double likelihood_optimal_complex_gain_visibility::optimal_complex_gains_trial(
    std::vector< std::complex<double> >& y,
    std::vector< std::complex<double> >& yb,
    std::vector<size_t>& is1,
    std::vector<size_t>& is2,
    std::vector< std::complex<double> >& gest,
    double& chisq_opt,
    int* niter_out)
{
  // Get the size of y (factor of 2 from real,imag)
  int ndata = int(2*y.size());

  if (niter_out != nullptr)
    *niter_out = 0;

  if (ndata==0) {
    return 1.0;
  }

  for (size_t i=0, j=1; i<y.size(); ++i)
  {
    _ogc_y[j]   = y[i].real();
    _ogc_yb[j]  = yb[i].real();
    _ogc_is1[j] = is1[i];
    _ogc_is2[j] = is2[i];
    ++j;

    _ogc_y[j]   = y[i].imag();
    _ogc_yb[j]  = yb[i].imag();
    _ogc_is1[j] = is1[i];
    _ogc_is2[j] = is2[i];
    ++j;
  }

  int ma = 2*_sigma_g.size();

  for (int i=0, j=1; i<int(gest.size()); ++i)
  {
    _g[j++] = std::log(std::abs(gest[i]));
    _g[j++] = std::arg(gest[i]);
  }

  double alambda = -1.0;
  double chisq = 0.0, ochisq, dg2;
  double ch2limit = 1.0e-8;
  double dg2limit = 0.0;
  for (size_t i=0; i<_sigma_g.size(); ++i)
    dg2limit += _sigma_g[i]*_sigma_g[i];
  dg2limit *= 1e-12;

  bool notconverged = true;
  int iteration = 0;

  for (iteration=0; iteration<_itermax && notconverged; ++iteration)
  {
    for (int i=1; i<=ma; ++i)
      _og[i] = _g[i];

    ochisq = chisq;

    if (mrqmin(_ogc_y, ndata, _g, ma, _covar, _alpha, &chisq, &alambda))
      return -1.0;

    if (iteration>5 && chisq<ochisq)
    {
      dg2 = 0.0;
      for (int i=1; i<=ma; ++i)
        dg2 += std::pow(_g[i]-_og[i],2);

      if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
        notconverged = false;
    }
 }

  if (niter_out != nullptr)
    *niter_out = iteration;

  alambda = 0.0;
  mrqmin(_ogc_y, ndata, _g, ma, _covar, _alpha, &chisq, &alambda);

  for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
  {
    double gmag = std::exp(_g[j]);
    gest[i] = gmag * std::exp(std::complex<double>(0.0,1.0)*_g[j+1]);
  }

  double detC = matrix_determinant(_covar);

  for (size_t a=0; a<_sigma_g.size(); ++a)
    detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;

  chisq_opt = chisq;

  return std::sqrt(detC);
}
  */

  

  
  double likelihood_optimal_complex_gain_visibility::optimal_complex_gains_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    utils::ScopedTimer T(utils::TimerID::GainsSolveTrial, timer_ns_, timer_calls_);
    
    // Get the size of y (factor of 2 from real,imag)
    int ndata = int( 2*y.size() );

    if (ndata==0) {
      return 1.0;
    }

    // Make global pointers to avoid nightmares in rigging the NR stuff.
    // _ogc_y = new double[ndata+1];
    // _ogc_yb = new double[ndata+1];
    // _ogc_is1 = new size_t[ndata+1];
    // _ogc_is2 = new size_t[ndata+1];
    for (size_t i=0, j=1; i<y.size(); ++i)
    {
      _ogc_y[j] = y[i].real();
      _ogc_yb[j] = yb[i].real();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      _ogc_y[j] = y[i].imag();
      _ogc_yb[j] = yb[i].imag();
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
    }

    // Make space for mrqmin objects
    int ma = 2*_sigma_g.size(); // real,imag
    // double **covar, **alpha;
    // covar = new double*[ma+1];
    // alpha = new double*[ma+1];
    // for (int i=1; i<=ma; ++i)
    // {
    //   covar[i] = new double[ma+1];
    //   alpha[i] = new double[ma+1];
    // }
    // _covar = new double*[ma+1];
    // _alpha = new double*[ma+1];
    // for (int i=1; i<=ma; ++i)
    // {
    //   _covar[i] = new double[ma+1];
    //   _alpha[i] = new double[ma+1];
    // }

    
    // Start running mrqmin
    // double *g = new double[ma+1]; // Internal gain representation is gain correction magnitude and phase, i.e., G = exp[ g - i phi ].
    // double *og = new double[ma+1];
    // _g = new double[ma+1];
    // _og = new double[ma+1];
    for (int i=0, j=1; i<int(gest.size()); ++i)
    {
      _g[j++] = std::log(std::abs(gest[i]));
      _g[j++] = std::arg(gest[i]);
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
	_og[i] = _g[i];
      ochisq = chisq;
      
      if (mrqmin(_ogc_y,ndata,_g,ma,_covar,_alpha,&chisq,&alambda))
	return -1;

      // if (iteration>5 && chisq<ochisq) // original develop version 
      if (iteration>0 && chisq<ochisq)
      {
	dg2 = 0.0;
	for (int i=1; i<=ma; ++i)
	  dg2 += std::pow(_g[i]-_og[i],2);

	if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
	  notconverged = false;
      }
    }
    alambda=0.0;
    mrqmin(_ogc_y,ndata,_g,ma,_covar,_alpha,&chisq,&alambda);

    // Save output
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
    {
      double gmag = std::exp(_g[j]);
      //DEBUG UNCOMMENT AND CHECK AT END
      // // Limit from below
      // if (gmag<1.0/(1.0+_sigma_g[i]*_max_g[i]))
      // 	gmag = 1.0/(1.0+_sigma_g[i]*_max_g[i]);
      // // Limit from above
      // if (gmag>(1.0+_sigma_g[i]*_max_g[i]))
      // 	gmag = (1.0+_sigma_g[i]*_max_g[i]);

      gest[i] = gmag * std::exp( std::complex<double>(0.0,1.0)*_g[j+1] );
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(_covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;

    // Clean up
    // delete[] og;
    // delete[] g;
    // for (int i=1; i<=ma; ++i)
    // {
    //   delete[] covar[i];
    //   delete[] alpha[i];
    // }
    // delete[] covar;
    // delete[] alpha;
    // delete[] _og;
    // delete[] _g;
    // for (int i=1; i<=ma; ++i)
    // {
    //   delete[] _covar[i];
    //   delete[] _alpha[i];
    // }
    // delete[] _covar;
    // delete[] _alpha;
    // delete[] _ogc_is2;
    // delete[] _ogc_is1;
    // delete[] _ogc_yb;
    // delete[] _ogc_y;

    chisq_opt = chisq;
    
    return std::sqrt(detC); // Success!
  }





  double likelihood_optimal_complex_gain_visibility::optimal_complex_gains_log_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
    utils::ScopedTimer T(utils::TimerID::GainsSolveLogTrial, timer_ns_, timer_calls_);

    // Get the size of y (factor of 2 from real,imag)
    int ndata = int( 2*y.size() );

    if (ndata==0) {
      return 1.0;
    }

    // Make global pointers to avoid nightmares in rigging the NR stuff.
    // _ogc_y = new double[ndata+1];
    // _ogc_yb = new double[ndata+1];
    // _ogc_is1 = new size_t[ndata+1];
    // _ogc_is2 = new size_t[ndata+1];
    // double *sig = new double[ndata+1];
    std::complex<double> tmp;
    for (size_t i=0, j=1; i<y.size(); ++i)
    {
      tmp = y[i]/yb[i];
      if (std::abs(tmp)<1e-6)
	tmp = 1e-6;
      //_ogc_y[j] = std::log(y[i]/yb[i]).real();
      _ogc_y[j] = std::log(tmp).real();
      //sig[j] = 1.0/std::abs(y[i]);
      _sig[j] = 1.0/std::abs(y[i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
      //_ogc_y[j] = std::log(y[i]/yb[i]).imag();
      _ogc_y[j] = std::log(tmp).imag();
      _sig[j] = 1.0/std::abs(y[i]);
      _ogc_yb[j] = 0.0;
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
    }

    // Make space for mrqmin objects
    int ma = 2*_sigma_g.size(); // real,imag
    // double **covar, **alpha;
    // covar = new double*[ma+1];
    // alpha = new double*[ma+1];
    // for (int i=1; i<=ma; ++i)
    // {
    //   covar[i] = new double[ma+1];
    //   alpha[i] = new double[ma+1];
    // }
    
    // Start running mrqmin
    // double *g = new double[ma+1]; // Internal gain representation is gain correction magnitude and phase, i.e., G = exp[ g - i phi ].
    // double *og = new double[ma+1];
    for (int i=0, j=1; i<int(gest.size()); ++i)
    {
      _g[j++] = std::log(std::abs(gest[i]));
      _g[j++] = std::arg(gest[i]);
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
	_og[i] = _g[i];
      ochisq = chisq;
      
      if (mrqmin_log(_ogc_y,_sig,ndata,_g,ma,_covar,_alpha,&chisq,&alambda))
	return -1;

      // if (iteration>5 && chisq<ochisq) // original develop version 
      if (iteration>0 && chisq<ochisq)
      {
	dg2 = 0.0;
	for (int i=1; i<=ma; ++i)
	  dg2 += std::pow(_g[i]-_og[i],2);

	if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
	  notconverged = false;
      }
    }
    alambda=0.0;
    mrqmin_log(_ogc_y,_sig,ndata,_g,ma,_covar,_alpha,&chisq,&alambda);

    // Save output
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=2)
    {
      double gmag = std::exp(_g[j]);
      /* //DEBUG UNCOMMENT AND CHECK AT END
      // Limit from below
      if (gmag<1.0/(1.0+_sigma_g[i]*_max_g[i]))
	gmag = 1.0/(1.0+_sigma_g[i]*_max_g[i]);
      // Limit from above
      if (gmag>(1.0+_sigma_g[i]*_max_g[i]))
	gmag = (1.0+_sigma_g[i]*_max_g[i]);
      */
      gest[i] = gmag * std::exp( std::complex<double>(0.0,1.0)*_g[j+1] );
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(_covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]) * _opi2;

    // Clean up
    // delete[] og;
    // delete[] g;
    // for (int i=1; i<=ma; ++i)
    // {
    //   delete[] covar[i];
    //   delete[] alpha[i];
    // }
    // delete[] covar;
    // delete[] alpha;
    // delete[] _ogc_is2;
    // delete[] _ogc_is1;
    // delete[] _ogc_yb;
    // delete[] _ogc_y;
    // delete[] sig;
    
    chisq_opt = chisq;
    
    return std::sqrt(detC); // Success!
  }

  void likelihood_optimal_complex_gain_visibility::gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const
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
    std::complex<double> yc, yb; //, ytest;
    if (i%2==1) // Real
    {
      yb = std::complex<double>( _ogc_yb[i], _ogc_yb[i+1] );
      //ytest = std::complex<double>( _ogc_y[i], _ogc_y[i+1] );
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
      //ytest = std::complex<double>( _ogc_y[i-1], _ogc_y[i] );
      yc = G1*std::conj(G2)*yb;

      (*y) = yc.imag();

      // Derivatives
      dydg[2*_ogc_is1[i]+1] = ( yc ).imag(); // 1 g
      dydg[2*_ogc_is1[i]+2] = ( std::complex<double>(0.0,1.0)*yc ).imag(); // 1 phase
      dydg[2*_ogc_is2[i]+1] = ( yc ).imag(); // 2 g
      dydg[2*_ogc_is2[i]+2] = ( -std::complex<double>(0.0,1.0)*yc ).imag(); // 2 phase
    }
  }



void likelihood_optimal_complex_gain_visibility::mrqcof(double y[], int ndata, double a[], int ma,
                                                        double **alpha, double beta[], double *chisq)
{
  utils::ScopedTimer T(utils::TimerID::mrqcof, timer_ns_, timer_calls_);

  // Zero outputs
  for (int j=1; j<=ma; ++j)
  {
    beta[j] = 0.0;
    for (int k=1; k<=ma; ++k)
      alpha[j][k] = 0.0;
  }

  *chisq = 0.0;

  // Main accumulation:
  // Data are stored as (real, imag) pairs.  Each pair touches only 4 parameters:
  //   g1, phi1, g2, phi2
  // so we assemble only the corresponding 4x4 block.
  int i = 1;
  for (; i+1 <= ndata; i += 2)
  {
    const size_t s1 = _ogc_is1[i];
    const size_t s2 = _ogc_is2[i];

    const int j1g = 2*int(s1) + 1;
    const int j1p = j1g + 1;
    const int j2g = 2*int(s2) + 1;
    const int j2p = j2g + 1;

    // Gain-corrected model visibility:
    // yc = exp(g1 + i p1) * conj( exp(g2 + i p2) ) * yb
    //    = exp(g1+g2) * exp(i(p1-p2)) * yb
    const double amp   = std::exp(a[j1g] + a[j2g]);
    const double phase = a[j1p] - a[j2p];

    const std::complex<double> phase_factor(amp*std::cos(phase), amp*std::sin(phase));
    const std::complex<double> yb(_ogc_yb[i], _ogc_yb[i+1]);
    const std::complex<double> yc = phase_factor * yb;

    const double yr = yc.real();
    const double yi = yc.imag();

    // Residuals for real and imaginary parts
    const double dyr = y[i]   - yr;
    const double dyi = y[i+1] - yi;

    *chisq += dyr*dyr + dyi*dyi;

    // Nonzero derivatives for the real residual:
    // d Re(yc) / d[g1, p1, g2, p2] = [ Re(yc), Re(i yc), Re(yc), Re(-i yc) ]
    //                              = [ yr,    -yi,      yr,     yi         ]
    const int    idx[4]  = { j1g, j1p, j2g, j2p };
    const double d_re[4] = { yr,  -yi, yr,  yi  };

    // Nonzero derivatives for the imag residual:
    // d Im(yc) / d[g1, p1, g2, p2] = [ Im(yc), Im(i yc), Im(yc), Im(-i yc) ]
    //                              = [ yi,     yr,       yi,     -yr        ]
    const double d_im[4] = { yi,  yr,  yi,  -yr };

    // beta += J^T r
    for (int u=0; u<4; ++u)
      beta[idx[u]] += dyr*d_re[u] + dyi*d_im[u];

    // alpha += J^T J
    for (int u=0; u<4; ++u)
    {
      const int ju = idx[u];
      for (int v=0; v<4; ++v)
      {
        const int jv = idx[v];
        alpha[ju][jv] += d_re[u]*d_re[v] + d_im[u]*d_im[v];
      }
    }
  }

  // Safety fallback in case ndata is odd.
  // This should normally not happen, since the gain-fit data are real/imag pairs.
  if (i <= ndata)
  {
    double ymod;
    gain_optimization_likelihood(i, a, &ymod, _dyda);

    const double dy = y[i] - ymod;
    *chisq += dy*dy;

    for (int j=1; j<=ma; ++j)
    {
      const double wj = _dyda[j];
      beta[j] += dy*wj;
      for (int k=1; k<=ma; ++k)
        alpha[j][k] += wj*_dyda[k];
    }
  }

  // Add priors to alpha and beta.
  // In addition to the prior on g that is given, a weak prior on phi is provided
  // to drive the solution toward G=1 in the absence of other information.
  double oSigma2;
  for (size_t s=0, j=1; s<_sigma_g.size(); ++s, j+=2)
  {
    oSigma2 = 1.0 / (_sigma_g[s]*_sigma_g[s]);

    beta[j]   -= a[j]   * oSigma2; // g^2 / (2 sigma^2)
    beta[j+1] -= a[j+1] * _opi2;   // phi^2 / (2 varpi^2)

    alpha[j][j]     += oSigma2;
    alpha[j+1][j+1] += _opi2;

    *chisq += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
  }

  double alpha_diag_max = 0.0;
  for (int j=1; j<=ma; ++j)
    alpha_diag_max = std::max(alpha_diag_max, alpha[j][j]);

  alpha_diag_max = std::max(alpha_diag_max, 1.0);

  for (int j=1; j<=ma; ++j)
    alpha[j][j] += 1.0e-12 * alpha_diag_max;
}
  

  /* 
  void likelihood_optimal_complex_gain_visibility::mrqcof_legacy(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,dy;

    //double *dyda = new double[ma+1];

    for (j=1;j<=ma;j++) {
      beta[j]=0.0;
      for (k=1;k<=ma;k++)
	alpha[j][k]=0.0;
    }
    *chisq=0.0;
    for (i=1;i<=ndata;i++) {
      gain_optimization_likelihood(i,a,&ymod,_dyda);
      dy=y[i]-ymod;
      for (j=1;j<=ma;j++) {
	wt=_dyda[j];
	for (k=1;k<=j;k++)
	  alpha[j][k] += wt*_dyda[k];
	beta[j] += dy*wt;
      }
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

      (*chisq) += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
    }
    double alpha_diag_max = 0.0;
    for (int j=1; j<=2*int(_sigma_g.size()); j++)
      alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max,1.0);
    for (int j=1; j<=2*int(_sigma_g.size()); j++)    
      alpha[j][j] += 1.0e-12*alpha_diag_max;
    
    //alpha[j][j] = std::max(alpha[j][j],1.0e-10*alpha_diag_max);
    
    //delete[] dyda;
  }
  */
  
  void likelihood_optimal_complex_gain_visibility::gain_optimization_log_likelihood(size_t i, const double g[], double *y, double dydg[]) const
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




  void likelihood_optimal_complex_gain_visibility::mrqcof_log(double y[], double sig[], int ndata,
                                                            double a[], int ma, double **alpha,
                                                            double beta[], double *chisq)
{
  // Zero outputs
  for (int j=1; j<=ma; ++j)
  {
    beta[j] = 0.0;
    for (int k=1; k<=ma; ++k)
      alpha[j][k] = 0.0;
  }

  *chisq = 0.0;

  // Log-fit data are stored as (log-amp, phase) pairs.
  // For one baseline datum between stations s1 and s2:
  //
  //   y_amp   = g1 + g2 + const
  //   y_phase = p1 - p2 + const
  //
  // so the Jacobian is extremely sparse:
  //
  //   d y_amp   / d[g1,p1,g2,p2] = [ 1, 0, 1,  0 ]
  //   d y_phase / d[g1,p1,g2,p2] = [ 0, 1, 0, -1 ]
  //
  // Hence we update only the touched 4x4 block.

  int i = 1;
  for (; i+1 <= ndata; i += 2)
  {
    const size_t s1 = _ogc_is1[i];
    const size_t s2 = _ogc_is2[i];

    const int j1g = 2*int(s1) + 1;
    const int j1p = j1g + 1;
    const int j2g = 2*int(s2) + 1;
    const int j2p = j2g + 1;

    // Model predictions in log space.
    // Keep these exactly consistent with the original log-likelihood model:
    // amplitude row depends on g1+g2, phase row depends on p1-p2.
    const double ymod_amp   = a[j1g] + a[j2g] + _ogc_yb[i];
    const double ymod_phase = a[j1p] - a[j2p] + _ogc_yb[i+1];

    const double dy_amp   = y[i]   - ymod_amp;
    const double dy_phase = y[i+1] - ymod_phase;

    const double sig2i_amp   = 1.0 / (sig[i]   * sig[i]);
    const double sig2i_phase = 1.0 / (sig[i+1] * sig[i+1]);

    *chisq += dy_amp*dy_amp*sig2i_amp + dy_phase*dy_phase*sig2i_phase;

    // beta += J^T W r
    beta[j1g] += dy_amp   * sig2i_amp;
    beta[j2g] += dy_amp   * sig2i_amp;

    beta[j1p] += dy_phase * sig2i_phase;
    beta[j2p] -= dy_phase * sig2i_phase;

    // alpha += J^T W J
    //
    // amplitude row: [1, 0, 1, 0], weight sig2i_amp
    alpha[j1g][j1g] += sig2i_amp;
    alpha[j1g][j2g] += sig2i_amp;
    alpha[j2g][j1g] += sig2i_amp;
    alpha[j2g][j2g] += sig2i_amp;

    // phase row: [0, 1, 0, -1], weight sig2i_phase
    alpha[j1p][j1p] += sig2i_phase;
    alpha[j1p][j2p] -= sig2i_phase;
    alpha[j2p][j1p] -= sig2i_phase;
    alpha[j2p][j2p] += sig2i_phase;
  }

  // Safety fallback in case ndata is odd.
  // This should normally never happen.
  if (i <= ndata)
  {
    double ymod;
    gain_optimization_log_likelihood(i, a, &ymod, _dyda);

    const double sig2i = 1.0 / (sig[i] * sig[i]);
    const double dy    = y[i] - ymod;

    *chisq += dy*dy*sig2i;

    for (int j=1; j<=ma; ++j)
    {
      const double wj = _dyda[j] * sig2i;
      beta[j] += dy * wj;
      for (int k=1; k<=ma; ++k)
        alpha[j][k] += wj * _dyda[k];
    }
  }

  // Add priors to alpha and beta.
  // In addition to the prior on g that is given, a weak prior on phi is provided
  // to drive the solution toward G=1 in the absence of other information.
  double oSigma2;
  for (size_t s=0, j=1; s<_sigma_g.size(); ++s, j+=2)
  {
    oSigma2 = 1.0 / (_sigma_g[s] * _sigma_g[s]);

    beta[j]   -= a[j]   * oSigma2; // g^2 / (2 sigma^2)
    beta[j+1] -= a[j+1] * _opi2;   // phi^2 / (2 varpi^2)

    alpha[j][j]     += oSigma2;
    alpha[j+1][j+1] += _opi2;

    *chisq += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
  }

  double alpha_diag_max = 0.0;
  for (int j=1; j<=ma; ++j)
    alpha_diag_max = std::max(alpha_diag_max, alpha[j][j]);

  alpha_diag_max = std::max(alpha_diag_max, 1.0);

  for (int j=1; j<=ma; ++j)
    alpha[j][j] += 1.0e-12 * alpha_diag_max;
}



  /*
  void likelihood_optimal_complex_gain_visibility::mrqcof_log_legacy_dense(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,sig2i,dy;

    // double *dyda = new double[ma+1];

    for (j=1;j<=ma;j++) {
      beta[j]=0.0;
      for (k=1;k<=ma;k++)
	alpha[j][k]=0.0;
    }
    *chisq=0.0;
    for (i=1;i<=ndata;i++) {
      gain_optimization_log_likelihood(i,a,&ymod,_dyda);
      sig2i=1.0/(sig[i]*sig[i]);
      dy=y[i]-ymod;
      for (j=1;j<=ma;j++) {
	wt=_dyda[j]*sig2i;
	for (k=1;k<=j;k++)
	  alpha[j][k] += wt*_dyda[k];
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
      
      //alpha[j][j] += std::max(1e-12*std::fabs(alpha[j][j]),oSigma2);
      //alpha[j+1][j+1] += std::max(1e-12*std::fabs(alpha[j+1][j+1]),_opi2);

      (*chisq) += a[j]*a[j]*oSigma2 + a[j+1]*a[j+1]*_opi2;
    }
    double alpha_diag_max = 0.0;
    for (int j=1; j<=2*int(_sigma_g.size()); j++)
      alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
    alpha_diag_max = std::max(alpha_diag_max,1.0);
    for (int j=1; j<=2*int(_sigma_g.size()); j++)    
      alpha[j][j] += 1.0e-12*alpha_diag_max;
    
    // delete[] dyda;
  }

*/


  
  double likelihood_optimal_complex_gain_visibility::optimal_gain_amplitude_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector<std::complex<double> >& gest, double& chisq_opt)
  {
    // Get the size of y (factor of 2 from real,imag)
    int ndata = int( y.size() );

    if (ndata==0) {
      return 1.0;
    }

    //std::cout << "Started in ogat: " << ndata << std::endl;

    // Make global pointers to avoid nightmares in rigging the NR stuff.
    // _ogc_y = new double[ndata+1];
    // _ogc_yb = new double[ndata+1];
    // _ogc_is1 = new size_t[ndata+1];
    // _ogc_is2 = new size_t[ndata+1];
    for (size_t i=0, j=1; i<y.size(); ++i)
    {
      _ogc_y[j] = std::abs(y[i]);
      _ogc_yb[j] = std::abs(yb[i]);
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
    }

    // Make space for mrqmin objects
    int ma = _sigma_g.size(); // amplitude
    // double **covar, **alpha;
    // covar = new double*[ma+1];
    // alpha = new double*[ma+1];
    // for (int i=1; i<=ma; ++i)
    // {
    //   covar[i] = new double[ma+1];
    //   alpha[i] = new double[ma+1];
    // }

    // Start running mrqmin
    // double *_g = new double[ma+1]; // Internal gain representation is gain correction magnitude, i.e., G = exp[ g ].
    // double *_og = new double[ma+1];
    for (int i=0, j=1; i<int(gest.size()); ++i)
      _g[j++] = std::log(std::abs(gest[i]));

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
	_og[i] = _g[i];
      ochisq = chisq;

      if (mrqmin_amplitude(_ogc_y,ndata,_g,ma,_covar,_alpha,&chisq,&alambda))
	return -1;
      
      if (iteration>5 && chisq<ochisq)
      {
	dg2 = 0.0;
	for (int i=1; i<=ma; ++i)
	  dg2 += std::pow(_g[i]-_og[i],2);
	if (dg2<dg2limit || (ochisq-chisq)<ch2limit*chisq)
	  notconverged = false;
      }
    }
    alambda=0.0;
    mrqmin_amplitude(_ogc_y,ndata,_g,ma,_covar,_alpha,&chisq,&alambda);
    
    // Save output
    for (int i=0, j=1; i<int(gest.size()); ++i, j+=1)
    {
      double gmag = std::exp(_g[j]);
      /* //DEBUG UNCOMMENT AND CHECK AT END
      // Limit from below
      if (gmag<1.0/(1.0+_sigma_g[i]*_max_g[i]))
	gmag = 1.0/(1.0+_sigma_g[i]*_max_g[i]);
      // Limit from above
      if (gmag>(1.0+_sigma_g[i]*_max_g[i]))
	gmag = (1.0+_sigma_g[i]*_max_g[i]);
      */
      gest[i] = std::complex<double>(gmag,0.0);
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(_covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]);

    // Clean up
    // delete[] og;
    // delete[] g;
    // for (int i=1; i<=ma; ++i)
    // {
    //   delete[] covar[i];
    //   delete[] alpha[i];
    // }
    // delete[] covar;
    // delete[] alpha;
    // delete[] _ogc_is2;
    // delete[] _ogc_is1;
    // delete[] _ogc_yb;
    // delete[] _ogc_y;

    chisq_opt = chisq;
    
    return std::sqrt(detC); // Success!
  }

  void likelihood_optimal_complex_gain_visibility::gain_amplitude_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    // Gain corrected model value
    double G1 = std::exp(g[_ogc_is1[i]+1]);
    double G2 = std::exp(g[_ogc_is2[i]+1]);

    // Derivatives initialization
    for (size_t a=1; a<=_sigma_g.size(); ++a)
      dydg[a] = 0.0;

    // Value
    (*y) = G1*G2*_ogc_yb[i];
    
    // Non-vanishing derivatives
    dydg[_ogc_is1[i]+1] = (*y); // 1 g
    dydg[_ogc_is2[i]+1] = (*y); // 2 g
  }

  void likelihood_optimal_complex_gain_visibility::mrqcof_amplitude(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,dy;
    // double *dyda = new double[ma+1];
    
    for (j=1;j<=ma;j++) {
      beta[j]=0.0;
      for (k=1;k<=ma;k++)
	alpha[j][k]=0.0;
    }
    *chisq=0.0;
    for (i=1;i<=ndata;i++) {
      gain_amplitude_optimization_likelihood(i,a,&ymod,_dyda);
      dy=y[i]-ymod;
      for (j=1;j<=ma;j++) {
	wt=_dyda[j];
	for (k=1;k<=j;k++)
	  alpha[j][k] += wt*_dyda[k];
	beta[j] += dy*wt;
      }
      *chisq += dy*dy;
    }
    for (j=2;j<=ma;j++)
      for (k=1;k<j;k++)
	alpha[k][j]=alpha[j][k];

    // Add priors to alpha and beta
    // In addition to the prior on g that is given.
    double oSigma2;
    for (size_t i=0, j=1; i<_sigma_g.size(); i++, j+=1)
    {
      oSigma2 = 1.0/(_sigma_g[i]*_sigma_g[i]);

      beta[j] -= a[j]*oSigma2; // g^2/2 Sigma^2
      alpha[j][j] += std::max(1e-12*std::fabs(alpha[j][j]),oSigma2);
      (*chisq) += a[j]*a[j]*oSigma2;
    }
      
    // delete[] dyda;
  }
  

#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}
  void likelihood_optimal_complex_gain_visibility::covsrt(double **covar, int ma, int mfit)
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

  int likelihood_optimal_complex_gain_visibility::gaussj(double **a, int n, double **b, int m)
  {
    utils::ScopedTimer T(utils::TimerID::gaussj, timer_ns_, timer_calls_);

    int i,icol=0,irow=0,j,k,l,ll;
    double big,dum,pivinv,swap;

    // int *indxc = new int[n+1];
    // int *indxr = new int[n+1];
    // int *ipiv = new int[n+1];
    
    for (j=1;j<=n;j++)
      _ipiv[j]=0;
    for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
	if (_ipiv[j] != 1)
	  for (k=1;k<=n;k++) {
	    if (_ipiv[k] == 0) {
	      if (fabs(a[j][k]) >= big) {
		big=fabs(a[j][k]);
		irow=j;
		icol=k;
	      }
	    } else if (_ipiv[k] > 1) {
	      std::cerr << "gaussj: Singular Matrix-1\n";

	      // DEBUGGING
	      std::cerr << "BAR:" << std::setw(10) << _x_debug.size() << std::setw(10) << _model.size() << std::setw(10) << _uncertainty.size() << " | ";
	      for (size_t j=0; j<_x_debug.size(); ++j)
		std::cerr << std::setw(15) << _x_debug[j];
	      std::cerr << std::endl;
	      
	      // delete[] ipiv;
	      // delete[] indxr;
	      // delete[] indxc;
	      return 1;
	    }
	  }
      ++(_ipiv[icol]);
      if (irow != icol) {
	for (l=1;l<=n;l++)
	  SWAP(a[irow][l],a[icol][l]);
	for (l=1;l<=m;l++)
	  SWAP(b[irow][l],b[icol][l]);
      }
      _indxr[i]=irow;
      _indxc[i]=icol;
      if (a[icol][icol] == 0.0) {
	std::cerr << "gaussj: Singular Matrix-2\n";
	// delete[] ipiv;
	// delete[] indxr;
	// delete[] indxc;
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
      if (_indxr[l] != _indxc[l])
	for (k=1;k<=n;k++)
	  SWAP(a[k][_indxr[l]],a[k][_indxc[l]]);
    }

    // delete[] ipiv;
    // delete[] indxr;
    // delete[] indxc;
    return 0;
  }



int likelihood_optimal_complex_gain_visibility::cholesky_solve(double **a, int n, const double rhs[], double x[])
{
  // In-place Cholesky factorization of symmetric positive definite matrix a:
  // on exit, lower triangle contains L with a = L L^T.
  //
  // Uses _mrq_oneda[][1] as temporary storage for the forward-substitution vector y.
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



#undef SWAP


/*
// cholesky with diagnostics
int likelihood_optimal_complex_gain_visibility::mrqmin(double y[], int ndata, double a[], int ma,
                                                       double **covar, double **alpha,
                                                       double *chisq, double *alamda)
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
    // Final covariance/inverse: keep the old Gauss-Jordan path.
    for (j=1; j<=mfit; ++j)
      _mrq_oneda[j][1] = _mrq_beta[j];

    if (gaussj(covar, mfit, _mrq_oneda, 1))
      return 1;

    for (j=1; j<=mfit; ++j)
      _mrq_da[j] = _mrq_oneda[j][1];

    covsrt(covar, ma, mfit);
    return 0;
  }

  // Hot path: Cholesky solve of the damped LM system.
  ++g_chol_calls;

  if (cholesky_solve(covar, mfit, _mrq_beta, _mrq_da))
  {
    ++g_chol_failures;

    // Rebuild the same damped system and fall back to Gauss-Jordan.
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
  else
  {
    // Debug-only A/B comparison on the first few successful Cholesky solves.
    if (g_chol_compares < 50)
    {
      ++g_chol_compares;

      // Rebuild the same damped system in _alpha and solve with Gauss-Jordan.
      for (j=1; j<=mfit; ++j)
      {
        for (k=1; k<=mfit; ++k)
          _alpha[j][k] = alpha[j][k];
        _alpha[j][j] = alpha[j][j] * (1.0 + (*alamda));
        _mrq_oneda[j][1] = _mrq_beta[j];
      }

      if (!gaussj(_alpha, mfit, _mrq_oneda, 1))
      {
        for (j=1; j<=mfit; ++j)
        {
          const double da_chol = _mrq_da[j];
          const double da_gj   = _mrq_oneda[j][1];
          const double absdiff = std::fabs(da_chol - da_gj);
          const double reldiff = absdiff / std::max(1.0, std::fabs(da_gj));

          g_chol_max_abs_da_diff = std::max(g_chol_max_abs_da_diff, absdiff);
          g_chol_max_rel_da_diff = std::max(g_chol_max_rel_da_diff, reldiff);
        }
      }
    }
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
*/



// cholesky version, with only gaussj at the end
int likelihood_optimal_complex_gain_visibility::mrqmin(double y[], int ndata, double a[], int ma,
                                                       double **covar, double **alpha,
                                                       double *chisq, double *alamda)
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

  // Hot path: use Cholesky solve for LM step.
  ++g_chol_calls; // CHOL TMP
  
  if (cholesky_solve(covar, mfit, _mrq_beta, _mrq_da))
  {
    ++g_chol_failures; // CHOL TMP
    
    // Fallback: rebuild matrix and use old Gauss-Jordan solver.
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


/*
  int likelihood_optimal_complex_gain_visibility::mrqmin_wo_cholesky(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;

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
*/



// Cholesky version wo diagnostics:
int likelihood_optimal_complex_gain_visibility::mrqmin_log(double y[], double sig[], int ndata,
                                                           double a[], int ma,
                                                           double **covar, double **alpha,
                                                           double *chisq, double *alamda)
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
    for (j=1; j<=mfit; ++j)
      _mrq_oneda[j][1] = _mrq_beta[j];

    if (gaussj(covar, mfit, _mrq_oneda, 1))
      return 1;

    for (j=1; j<=mfit; ++j)
      _mrq_da[j] = _mrq_oneda[j][1];

    covsrt(covar, ma, mfit);
    return 0;
  }

  if (cholesky_solve(covar, mfit, _mrq_beta, _mrq_da))
  {
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



/*
  int likelihood_optimal_complex_gain_visibility::mrqmin_log_wo_cholesky(double y[], double sig[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
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
*/

  int likelihood_optimal_complex_gain_visibility::mrqmin_amplitude(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
  {
    int j,k,l;
    int mfit = ma;
    
    if (*alamda < 0.0) {
      *alamda=0.001;
      mrqcof_amplitude(y,ndata,a,ma,alpha,_mrq_beta,chisq);
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
    mrqcof_amplitude(y,ndata,_mrq_atry,ma,covar,_mrq_da,chisq);
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


  void likelihood_optimal_complex_gain_visibility::accumulate_pixel_grad_epoch(const model_image_adaptive_splined_raster& M,
									       size_t epoch,
									       std::vector<double>& grad_I) const
  {
    const auto& phase = M.phase_cache();
    const auto& sk    = M.spline_kernel_cache();
    const auto& Iflat = M.I_flat();
    
    const size_t Npix = M.Nx() * M.Ny();
    
    const auto& idx_list = _datum_index_list[epoch];
    const auto& is1 = _is1_list[epoch];
    const auto& is2 = _is2_list[epoch];
    
    for (size_t ii = 0; ii < idx_list.size(); ++ii)
      {
	const size_t d_idx = idx_list[ii];
	datum_visibility& d = _data.datum(d_idx);
	
	const std::complex<double> err = _uncertainty.error(d);
	const double inv_er = 1.0 / err.real();
	const double inv_ei = 1.0 / err.imag();
	
	const std::complex<double> y(d.V.real() * inv_er, d.V.imag() * inv_ei);
	
	const size_t off = d_idx * Npix;
	const double skd = sk[d_idx];
	
	std::complex<double> Vm(0.0, 0.0);
	for (size_t k = 0; k < Npix; ++k)
	  Vm += Iflat[k] * (skd * phase[off + k]);
	
	const std::complex<double> yb(Vm.real() * inv_er, Vm.imag() * inv_ei);
	
	const std::complex<double> g =
	  _G[epoch][is1[ii]] * std::conj(_G[epoch][is2[ii]]);
	
	const std::complex<double> pred(
					g.real() * yb.real() - g.imag() * yb.imag(),
					g.real() * yb.imag() + g.imag() * yb.real()
					);
	
	const double rr = y.real() - pred.real();
	const double ri = y.imag() - pred.imag();
	
	for (size_t k = 0; k < Npix; ++k)
	  {
	    const std::complex<double> z = skd * phase[off + k];
	    const std::complex<double> dzb(z.real() * inv_er, z.imag() * inv_ei);
	    
	    const double dp_r = g.real() * dzb.real() - g.imag() * dzb.imag();
	    const double dp_i = g.real() * dzb.imag() + g.imag() * dzb.real();
	    
	    grad_I[k] += rr * dp_r + ri * dp_i;
	  }
      }
  }
  

     // Profiling                                                                                                  
  void likelihood_optimal_complex_gain_visibility::print_timing_summary(int mpi_rank) const
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

    std::cerr << "===== Cholesky diagnostics =====\n";
    std::cerr << "chol calls          : " << g_chol_calls << "\n";
    std::cerr << "chol failures       : " << g_chol_failures << "\n";
    std::cerr << "chol compares       : " << g_chol_compares << "\n";
    std::cerr << "max |da_chol-da_gj| : " << g_chol_max_abs_da_diff << "\n";
    std::cerr << "max rel da diff     : " << g_chol_max_rel_da_diff << "\n";
  }
  
};


