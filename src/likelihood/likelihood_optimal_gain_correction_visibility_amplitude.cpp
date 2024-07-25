/*! 
  \file likelihood_optimal_gain_correction_visibility_amplitude.cpp
  \author Avery E. Broderick
  \date  October, 2018
  \brief Implementation file for the likelihood_optimal_gain_gorrection_visibility_amplitude likelihood class
*/


#include "random_number_generator.h"

#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include <cmath>

#include <iostream>
#include <fstream>
#include <iomanip>

namespace Themis
{  
  likelihood_optimal_gain_correction_visibility_amplitude::likelihood_optimal_gain_correction_visibility_amplitude(
  data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g)
    : _data(data), _model(model), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _smoothly_varying_gains(true), _solve_for_gains(true)
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

  likelihood_optimal_gain_correction_visibility_amplitude::likelihood_optimal_gain_correction_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge)
    : _data(data), _model(model), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(_sigma_g.size(),3.0), _tge(t_ge), _smoothly_varying_gains(true), _solve_for_gains(true)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();
  }

  likelihood_optimal_gain_correction_visibility_amplitude::likelihood_optimal_gain_correction_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g)
    : _data(data), _model(model), _station_codes(station_codes), _sigma_g(sigma_g), _max_g(max_g), _tge(t_ge), _smoothly_varying_gains(true), _solve_for_gains(true)
  {
    // Check station codes
    check_station_codes();

    // Allocate memory structures
    allocate_memory();

    // Setup organized hash tables
    organize_data_lists();
  }

  
  likelihood_optimal_gain_correction_visibility_amplitude::~likelihood_optimal_gain_correction_visibility_amplitude()
  {
    for (size_t j=0; j<=_sigma_g.size(); ++j)
      delete[] _mrq_oneda[j];
    delete[] _mrq_oneda;
    
    delete[] _mrq_da;
    delete[] _mrq_beta;
    delete[] _mrq_atry;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::check_station_codes()
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
	std::cerr << "WARNING: likelihood_optimal_gain_correction_visibility_amplitude:\n"
		  << "    Station " << _data.datum(j).Station1 << " not in station_codes list.\n"
		  << '\n';
      if ( station2_in_station_codes==false )
	std::cerr << "WARNING: likelihood_optimal_gain_correction_visibility_amplitude:\n"
		  << "    Station " << _data.datum(j).Station2 << " not in station_codes list.\n"
		  << '\n';
    }
    for (size_t k=0; k<_station_codes.size(); ++k)
      if ( station_code_in_data[k]==false )
	std::cerr << "WARNING: likelihood_optimal_gain_correction_visibility_amplitude:\n"
		  << "    station code " << _station_codes[k] << " not used in data set.\n"
		  << '\n'; 
  }
  
  void likelihood_optimal_gain_correction_visibility_amplitude::allocate_memory()
  {
    // Allocate space for marginalized gain corrections
    _g.resize(_tge.size()-1);
    for (size_t j=0; j<_tge.size()-1; ++j)
      _g[j].resize(_sigma_g.size());
    _sqrt_detC.resize(_tge.size()-1);

    int ma = _sigma_g.size();
    _mrq_atry = new double[ma+1];
    _mrq_beta = new double[ma+1];
    _mrq_da = new double[ma+1];
    _mrq_oneda = new double*[ma+1];
    for (int j=0; j<=ma; j++)
      _mrq_oneda[j] = new double[2];
  }

  
  void likelihood_optimal_gain_correction_visibility_amplitude::set_mpi_communicator(MPI_Comm comm)
  {
    _comm=comm;
    initialize_mpi();
    _model.set_mpi_communicator(comm);
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::solve_for_gains()
  {
    _solve_for_gains = true;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::fix_gains()
  {
    _solve_for_gains = false;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::assume_smoothly_varying_gains()
  {
    _smoothly_varying_gains = true;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::assume_independently_varying_gains()
  {
    _smoothly_varying_gains = false;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::output(std::ostream& out)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
      out << "# likelihood_optimal_gain_correction_visibility_amplitude output file\n#"
	  << std::setw(14) << "u (Gl)"
	  << std::setw(15) << "v (Gl)"
	  << std::setw(15) << "|V| (Jy)"
	  << std::setw(15) << "err (Jy)"
	  << std::setw(15) << "model |V| (Jy)"
	  << std::setw(15) << "residual (Jy)"
	  << '\n';

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      for (size_t i=0; i<_data.size(); ++i)
      {
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Station indexes
	  size_t a=999, b=999; // Large number to facilitate the identification of failed baseline index determination
	  for (size_t c=0; c<_sigma_g.size(); ++c)
	  {
	    if (_data.datum(i).Station1==_station_codes[c])
	      a = c;	
	    if (_data.datum(i).Station2==_station_codes[c])
	      b = c;	
	  }

	  double V = _model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err) * (1.0+_g[epoch][a])*(1.0+_g[epoch][b]);
	  if (rank==0)
	    out << std::setw(15) << _data.datum(i).u/1e9
		<< std::setw(15) << _data.datum(i).v/1e9
		<< std::setw(15) << _data.datum(i).V
		<< std::setw(15) << _data.datum(i).err
		<< std::setw(15) << V
		<< std::setw(15) << (_data.datum(i).V-V)
		<< '\n';

	  
	}
      }
    }
  }
  
  size_t likelihood_optimal_gain_correction_visibility_amplitude::number_of_independent_gains()
  {
    int number_of_gains = 0;
    
    int **baseline_list = new int*[_sigma_g.size()];
    for (size_t a=0; a<_sigma_g.size(); ++a)
      baseline_list[a] = new int[_sigma_g.size()];

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of stations contributing to each epoch
      std::vector<std::string> unique_stations(0);
      bool inlist = false;
      size_t ndata = 0;
      for (size_t a=0; a<_sigma_g.size(); ++a)
	for (size_t b=0; b<_sigma_g.size(); ++b)
	  baseline_list[a][b] = 0.0;

      for (size_t i=0; i<_data.size(); ++i)
      {
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  ndata++;

	  // Station 1
	  inlist=false;
	  for (size_t a=0; a<unique_stations.size(); ++a)
	    inlist = inlist || (_data.datum(i).Station1==unique_stations[a]);
	  if (!inlist)
	    unique_stations.push_back(_data.datum(i).Station1);

	  // Station 2
	  inlist=false;
	  for (size_t a=0; a<unique_stations.size(); ++a)
	    inlist = inlist || (_data.datum(i).Station2==unique_stations[a]);
	  if (!inlist)
	    unique_stations.push_back(_data.datum(i).Station2);

	  // Station indexes
	  size_t a=0, b=0; // Large number to facilitate the identification of failed baseline index determination
	  for (size_t c=0; c<_sigma_g.size(); ++c)
	  {
	    if (_data.datum(i).Station1==_station_codes[c])
	      a = c;	
	    if (_data.datum(i).Station2==_station_codes[c])
	      b = c;	
	  }
	  if (a>b) // Order the index
	  {
	    size_t tmp=a;
	    a=b;
	    b=tmp;
	  }
	  baseline_list[a][b] = 1;
	}
      }
 
      int number_of_baselines = 0;
      for (size_t a=0; a<_sigma_g.size(); ++a)
	for (size_t b=0; b<_sigma_g.size(); ++b)
	  number_of_baselines += baseline_list[a][b];

      int number_of_stations = int(unique_stations.size());

      number_of_gains += std::min(number_of_baselines,number_of_stations);
    }

    // Clean up
    for (size_t a=0; a<_sigma_g.size(); ++a)
      delete[] baseline_list[a];
    delete[] baseline_list;


    return size_t(number_of_gains);
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::organize_data_lists()
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
      std::vector<double> y;
      std::vector<std::string> s1,s2;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_data.size(); ++i)
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Get the index
	  id.push_back(i);

	  // Data V/sigma
	  y.push_back(_data.datum(i).V/_data.datum(i).err);

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
      _y_list[epoch]=y;
      _is1_list[epoch]=is1;
      _is2_list[epoch]=is2;

    }
  }


  double likelihood_optimal_gain_correction_visibility_amplitude::operator()(std::vector<double>& x)
  {
    _x = x;
    _model.generate_model(x);

    // Log-likelihood accumulator
    double L = 0;

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector<double> yb, y;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	yb.push_back( _model.visibility_amplitude(_data.datum(_datum_index_list[epoch][i]),0.25*_data.datum(_datum_index_list[epoch][i]).err)/_data.datum(_datum_index_list[epoch][i]).err );
      }

      y = _y_list[epoch];
      is1 = _is1_list[epoch];
      is2 = _is2_list[epoch];


      /*
      // Get vector of error-normed model and data visibilities once
      std::vector<double> yb, y;
      std::vector<std::string> s1,s2;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_data.size(); ++i)
      {
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Model V/sigma
	  yb.push_back(_model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)/_data.datum(i).err);

	  // Data V/sigma
	  y.push_back(_data.datum(i).V/_data.datum(i).err);

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
      }
      */


      double marg_term;
      if (_solve_for_gains)
      {
	// Determine the initial guess for the gains based on currently stated assumptions.
	if (epoch>0)
	{
	  if (_smoothly_varying_gains)
	  {
	    _g[epoch] = _g[epoch-1];
	  }
	  else
	  {
	    for (size_t a=0; a<_sigma_g.size(); ++a)
	      _g[epoch][a] = 0.0;
	  }
	}

	// Levenberg-Marquardt minimization of full likelihood
	_sqrt_detC[epoch] = optimal_gain_corrections(y,yb,is1,is2,_g[epoch]);
      }
      
      marg_term = _sqrt_detC[epoch];

      // Add in the direct likelihood
      double dL = 0.0;
      for (size_t i=0; i<y.size(); ++i)
	dL += -0.5 * std::pow( y[i] - (1.0+_g[epoch][is1[i]])*(1.0+_g[epoch][is2[i]])*yb[i], 2);

      // Add the Gaussian prior 
      for (size_t a=0; a<_sigma_g.size(); ++a)
	dL += -0.5*(_g[epoch][a]*_g[epoch][a])/(_sigma_g[a]*_sigma_g[a]);

      // Add a quadratic approximation of the integral over the distribution about the best-fit gain corrections
      dL += std::log(marg_term);

      L += dL;
    }

    return L;
  }

  
  double likelihood_optimal_gain_correction_visibility_amplitude::chi_squared(std::vector<double>& x)
  {
    _model.generate_model(x);

    // Log-likelihood accumulator
    double L = 0;

    // Remove the prior?
    std::vector<double> true_sigma_g = _sigma_g;
    _sigma_g.assign(_sigma_g.size(),1000.0);

    // For each gain correction epoch
    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
    {
      // Get vector of error-normed model and data visibilities once
      std::vector<double> yb, y;
      std::vector<size_t> is1, is2;
      
      for (size_t i=0; i<_datum_index_list[epoch].size(); ++i)
      {
	yb.push_back( _model.visibility_amplitude(_data.datum(_datum_index_list[epoch][i]),0.25*_data.datum(_datum_index_list[epoch][i]).err)/_data.datum(_datum_index_list[epoch][i]).err );
      }

      y = _y_list[epoch];
      is1 = _is1_list[epoch];
      is2 = _is2_list[epoch];


      /*
      // Get vector of error-normed model and data visibilities once
      std::vector<double> yb(0), y(0);
      std::vector<std::string> s1(0),s2(0);
      std::vector<size_t> is1(0), is2(0);
      
      for (size_t i=0; i<_data.size(); ++i)
      {
	if (_data.datum(i).tJ2000>=_tge[epoch] && _data.datum(i).tJ2000<_tge[epoch+1])
	{
	  // Model V/sigma
	  yb.push_back(_model.visibility_amplitude(_data.datum(i),0.25*_data.datum(i).err)/_data.datum(i).err);

	  // Data V/sigma
	  y.push_back(_data.datum(i).V/_data.datum(i).err);

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
      }
      */


      if (_solve_for_gains)
      {
	// Determine the initial guess for the gains based on currently stated assumptions.
	if (epoch>0)
        {
	  if (_smoothly_varying_gains)
	  {
	    _g[epoch] = _g[epoch-1];
	  }
	  else
	  {
	    for (size_t a=0; a<_sigma_g.size(); ++a)
	      _g[epoch][a] = 0.0;
	  }
	}

	// Levenberg-Marquardt minimization of full -likelihood
	optimal_gain_corrections(y,yb,is1,is2,_g[epoch]);
      }

      double dL = 0.0;
      for (size_t i=0; i<y.size(); ++i)
	dL += -0.5 * std::pow( y[i] - (1.0+_g[epoch][is1[i]])*(1.0+_g[epoch][is2[i]])*yb[i], 2);

      L += dL;
    }

    // Reset the prior
    _sigma_g = true_sigma_g;

    return (-2.0*L);
  }


  double likelihood_optimal_gain_correction_visibility_amplitude::matrix_determinant(double **a)
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
  void likelihood_optimal_gain_correction_visibility_amplitude::ludcmp(double **a, int n, int *indx, double &d)
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

  std::vector<double> likelihood_optimal_gain_correction_visibility_amplitude::get_gain_correction_times()
  {
    return ( _tge );
  }

  std::vector< std::vector<double> > likelihood_optimal_gain_correction_visibility_amplitude::get_gain_corrections()
  {
    return ( _g );
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::read_gain_file(std::string gain_file_name)
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

      double tmp;
      for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      {
	in >> tmp;
	in >> tmp;
	for (size_t a=0; a<_sigma_g.size(); ++a)
	  in >> _g[epoch][a];

	if (in.eof()==true)
	{
	  std::cerr << "ERROR: likelihood_optimal_gain_correction_visibility_amplitude::read_gain_file too few gains in " << gain_file_name << '\n';
	  std::exit(1);
	}
      }
    }

    size_t ngains=_sigma_g.size()*(_tge.size()-1);
    double *buff = new double[ngains];
    for (size_t epoch=0,k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
	buff[k++] = _g[epoch][a];
    MPI_Bcast(buff,ngains,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (size_t epoch=0,k=0; epoch<_tge.size()-1; ++epoch)
      for (size_t a=0; a<_sigma_g.size(); ++a)
	_g[epoch][a] = buff[k++];
    delete[] buff;

    for (size_t epoch=0; epoch<_tge.size()-1; ++epoch)
      _sqrt_detC[epoch] = 1.0;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::output_gain_corrections(std::ostream& out)
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
	out << std::setw(15) << _g[epoch][a];
      out << '\n';
    }    
  }
  
  void likelihood_optimal_gain_correction_visibility_amplitude::output_gain_corrections(std::string outname)
  {
    int rank;
    MPI_Comm_rank(_comm, &rank);

    if (rank==0)
    {
      std::ofstream out(outname.c_str());
      output_gain_corrections(out);
      out.close();
    }
  }

  double likelihood_optimal_gain_correction_visibility_amplitude::optimal_gain_corrections(std::vector<double>& y, std::vector<double>& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector<double>& gest)
  {
    // Get the size of y.
    int ndata = 0;
    for (size_t i=0; i<y.size(); ++i)
      ndata++;


    if (ndata==0) {
      return 1.0;
    }


    // Make global pointers to avoid nightmares in rigging the NR stuff.
    _ogc_y = new double[ndata+1];
    _ogc_yb = new double[ndata+1];
    _ogc_is1 = new size_t[ndata+1];
    _ogc_is2 = new size_t[ndata+1];
    for (size_t i=0, j=1; i<y.size(); ++i)
    {
      _ogc_y[j] = y[i];
      _ogc_yb[j] = yb[i];
      _ogc_is1[j] = is1[i];
      _ogc_is2[j] = is2[i];
      j++;
    }


    // Make space for mrqmin objects
    int ma = _sigma_g.size();
    double **covar, **alpha;
    covar = new double*[ma+1];
    alpha = new double*[ma+1];
    for (int i=1; i<=ma; ++i)
    {
      covar[i] = new double[ma+1];
      alpha[i] = new double[ma+1];
    }

    // Start running mrqmin
    double *g = new double[ma+1];
    double *og = new double[ma+1];
    for (int i=1; i<=ma; ++i)
      g[i] = gest[i-1];
    double alambda = -1.0;
    double chisq=0.0, ochisq, dg2;
    double dg2limit=0.0;
    for (size_t i=0; i<_sigma_g.size(); ++i)
      dg2limit += _sigma_g[i]*_sigma_g[i];
    dg2limit *= 1e-12; ///_sigma_g.size();

    bool notconverged = true;
    int iteration;
    for (iteration=0; iteration<100 && notconverged; ++iteration)
    {
      for (int i=1; i<=ma; ++i)
	og[i] = g[i];
      ochisq = chisq;
      
      mrqmin(_ogc_y,ndata,g,ma,covar,alpha,&chisq,&alambda);

      
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
    for (int i=0; i<ma; ++i)
    {
      gest[i] = g[i+1];

      // Limit from above
      gest[i] = std::max(gest[i],-_sigma_g[i]*_max_g[i]);
      // Limit from below
      gest[i] = std::min(gest[i],_sigma_g[i]*_max_g[i]);
    }

    // Determinant of the covariance matrix, which is approximately the integral of the likelihood 
    double detC = matrix_determinant(covar);

    // Renormalize by the products of 1/_sigma_g^2
    for (size_t a=0; a<_sigma_g.size(); ++a)
      detC *= 1.0/(_sigma_g[a]*_sigma_g[a]);

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

    return std::sqrt(detC); // Success!
  }
  
  void likelihood_optimal_gain_correction_visibility_amplitude::gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const
  {
    // Gain corrected model value
    (*y) = (1.0+g[_ogc_is1[i]+1])*(1.0+g[_ogc_is2[i]+1])*_ogc_yb[i];

    // Derivatives
    for (size_t a=1; a<=_sigma_g.size(); ++a)
      dydg[a] = 0.0;
    dydg[_ogc_is1[i]+1] = (1.0+g[_ogc_is2[i]+1])*_ogc_yb[i];
    dydg[_ogc_is2[i]+1] = (1.0+g[_ogc_is1[i]+1])*_ogc_yb[i];
  }
   

#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}
  void likelihood_optimal_gain_correction_visibility_amplitude::covsrt(double **covar, int ma, int mfit)
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

  void likelihood_optimal_gain_correction_visibility_amplitude::gaussj(double **a, int n, double **b, int m)
  {
    int i,icol=0,irow=0,j,k,l,ll;
    double big,dum,pivinv,swap;

    int *indxc = new int[n+1];
    int *indxr = new int[n+1];
    int *ipiv = new int[n+1];

    /*
    std::cerr << "gaussj a dump:----------------\n";
    for (j=1; j<=n; j++) {  
      std::cerr << "  ";
      for (i=1; i<=n; i++)
	std::cerr << std::setw(15) << a[i][j];
      std::cerr << '\n';
    }
    std::cerr << "------------------------------\n\n";
    */

    
    for (j=1;j<=n;j++)
      ipiv[j]=0;
    for (i=1;i<=n;i++) {
      big=0.0;
      for (j=1;j<=n;j++)
	if (ipiv[j] != 1)
	  for (k=1;k<=n;k++) {
	    if (ipiv[k] == 0) {
	      if (fabs(a[j][k]) >= big) {
		big=fabs(a[j][k]);
		irow=j;
		icol=k;
	      }
	    } else if (ipiv[k] > 1) {
	      std::cerr << "gaussj: Singular Matrix-1\n\n";
              std::cerr << "Parameter list dump:";
              for (size_t ix=0; ix<_x.size(); ++ix)
                std::cerr << std::setw(15) << _x[ix];
              std::cerr << "\n" << std::endl;
	      std::exit(1);
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
	std::cerr << "gaussj: Singular Matrix-2\n\n";
        std::cerr << "Parameter list dump:";
        for (size_t ix=0; ix<_x.size(); ++ix)
	  std::cerr << std::setw(15) << _x[ix];
        std::cerr << "\n" << std::endl;
	std::exit(1);
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
  }
#undef SWAP

  void likelihood_optimal_gain_correction_visibility_amplitude::mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq)
  {
    int i,j,k;
    double ymod,wt,dy;

    double *dyda = new double[ma+1];

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
      *chisq += dy*dy;
    }
    for (j=2;j<=ma;j++)
      for (k=1;k<j;k++)
	alpha[k][j]=alpha[j][k];

    // Add priors to alpha and beta
    for (j=1;j<=ma;j++)
    {
      beta[j] -= a[j]/(_sigma_g[j-1]*_sigma_g[j-1]);
      alpha[j][j] += 1.0/(_sigma_g[j-1]*_sigma_g[j-1]);
      (*chisq) += (a[j]*a[j])/(_sigma_g[j-1]*_sigma_g[j-1]);
    }
    double alpha_diag_max = 0.0;
    for (int j=1; j<=ma; j++)
      alpha_diag_max = std::max(alpha[j][j],alpha_diag_max);
    for (int j=1; j<=ma; j++)    
      alpha[j][j] = std::max(alpha[j][j],1.0e-10*alpha_diag_max);
      
      
    delete[] dyda;
  }

  void likelihood_optimal_gain_correction_visibility_amplitude::mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda)
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
    gaussj(covar,mfit,_mrq_oneda,1);
    for (j=1;j<=mfit;j++)
      _mrq_da[j]=_mrq_oneda[j][1];
    if (*alamda == 0.0) {
      covsrt(covar,ma,mfit);
      return;
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
  }
  
};


