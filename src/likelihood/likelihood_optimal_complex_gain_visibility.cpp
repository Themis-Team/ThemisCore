/*! 
  \file likelihood_optimal_complex_gain_visibility.cpp
  \author Avery E. Broderick
  \date  February, 2020
  \brief Implementation file for the likelihood_optimal_complex_gain_visibility likelihood class
*/


#include "random_number_generator.h"

#include "likelihood_optimal_complex_gain_visibility.h"
#include <cmath>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

namespace Themis
{  
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

  void likelihood_optimal_complex_gain_visibility::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    distribute_gains();
    
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
	//std::complex<double> V = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(_data.datum(_datum_index_list[epoch][i]).err));
	std::complex<double> V = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));
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
	std::vector< std::complex<double> > yb, y;
	std::vector<size_t> is1, is2;
	
	double lognorm = 0.0;      
	for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
	{
	  //std::complex<double> err_orig = _data.datum(_datum_index_list[epoch][i]).err; // Data error
	  std::complex<double> err = _uncertainty.error(_data.datum(_datum_index_list[epoch][i]));
	  std::complex<double> Vd = _data.datum(_datum_index_list[epoch][i]).V;
	  std::complex<double> Vm = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));
	  
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
    _model.generate_model(mx);
    _uncertainty.generate_uncertainty(ux);
    
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
	std::complex<double> Vm = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

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

    // if (wrank==2) {
    //   std::cerr << "DEBUGGING distribute gains 2 (ALMA):"
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
    // 		  << std::setw(15) << global_buff[i]
    // 		  << std::setw(15) << global_buff[i+1]
    // 		  << '\n';
    // 	i += 2*_sigma_g.size() + 1;
    //   }
    //   std::cerr << "----------------------------------\n";
    // }

    
    delete[] local_buff;
    delete[] global_buff;
  }

  

  std::vector<double> likelihood_optimal_complex_gain_visibility::gradient(std::vector<double>& x, prior& Pr)
  {
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
    _model.generate_model(mx);
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
	std::complex<double> Vm = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

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
    
    _model.generate_model(x);

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
	std::complex<double> Vm = _model.visibility(_data.datum(_datum_index_list[epoch][i]),0.25*std::abs(err));

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
  
  double likelihood_optimal_complex_gain_visibility::optimal_complex_gains_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq_opt)
  {
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
    mrqmin(_ogc_y,ndata,_g,ma,_covar,_alpha,&chisq,&alambda);

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

  void likelihood_optimal_complex_gain_visibility::mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
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

  void likelihood_optimal_complex_gain_visibility::mrqcof_log(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
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
#undef SWAP

  int likelihood_optimal_complex_gain_visibility::mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
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

  int likelihood_optimal_complex_gain_visibility::mrqmin_log(double y[], double sig[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
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

  
};


