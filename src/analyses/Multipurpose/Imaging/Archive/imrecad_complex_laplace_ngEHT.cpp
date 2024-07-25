/*!
  \file analyses/Imaging/imrecad_complex_nuts_diag_ngEHT.cpp
  \author Avery Broderick
  \date Sep 2020
  \brief Generic driver for image reconstruction with Themis with complex visibilities.

  \details TBD
*/


#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_closure_amplitude.h"
#include "model_image_splined_raster.h"
#include "model_image_adaptive_splined_raster.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_xsringauss.h"
#include "model_image_polynomial_variable.h"
#include "model_image.h"
#include "likelihood.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "optimizer_laplace.h"
#include "utils.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"
#include <Eigen/Core>
#include "stop_watch.h"

#include <mpi.h>
#include <memory> 
#include <string>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>

#include <cstring>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Parse the command line inputs
  std::string v_file="";
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  bool Reconstruct_gains = false;
  std::vector<std::string> gain_file_list;

  size_t Number_of_pixels_x = 4;
  size_t Number_of_pixels_y = 4;
  double Field_of_view_x = 0;
  double Field_of_view_y = 0;
  double Position_angle = -999;
  
  
  bool add_background_gaussian=false;
  bool add_ring=false;
  bool add_roving_gaussian=false;

  // Start features
  bool add_background_gaussian_start=false;
  bool add_ring_start=false;
  double initial_ring_diameter=40.0;
  bool add_roving_gaussian_start=false;
  size_t start_Number_of_pixels_x=0;
  size_t start_Number_of_pixels_y=0;
  bool scatter=false;

  
  size_t opt_ko_itermax = 0;
  double opt_ko_llrf = 10.0;
  size_t opt_instances = 50;

  double frequency = 230e9;

  std::string array = "eht2017";
  
  // Dump command line for posterity
  if (world_rank==0)
  {
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "CMD$ <mpiexec>";
    for (int k=0; k<argc; k++)
      std::cout << " " << argv[k];
    std::cout << std::endl;
    std::cout << "============================================================\n" << std::endl;
  }
  
  for (int k=1; k<argc;)
  {
    std::string opt(argv[k++]);
    if (opt=="--visibilities" || opt=="-v")
    {
      if (k<argc)
	v_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --visibilities, -v.\n";
	std::exit(1);
      }
    }
    else if (opt=="--frequency" || opt=="-rf")
    {
      if (k<argc)
      {
	frequency = atof(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A float argument must be provided after --frequency, -f: <rf in Hz>.\n";
	std::exit(1);
      }
    }
    else if (opt=="--array")
    {
      if (k<argc)
      {
	array = std::string(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --array.\n";
	std::exit(1);
      }
    }    
    else if (opt=="--parameter-file" || opt=="-p")
    {
      if (k+1<argc)
      {
	Number_start_params = atoi(argv[k++]);
	param_file = std::string(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO arguments must be provided after --parameter-file, -p: <# of params to set> and file list.\n";
	std::exit(1);
      }
    }
    else if ( opt == "--scatter" )
    {
      scatter = true;
    }
    else if (opt=="-Npx-start")
    {
      if (k<argc)
      {
	start_Number_of_pixels_x = atoi(argv[k++]);
	start_Number_of_pixels_y = start_Number_of_pixels_x;
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Npx-start.\n";
	std::exit(1);
      }
    }
    else if (opt=="-Npxy-start")
    {
      if (k+1<argc)
      {
	start_Number_of_pixels_x = atoi(argv[k++]);
	start_Number_of_pixels_y = atoi(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO int arguments must be provided after -Npxy-start.\n";
	std::exit(1);
      }
    }
    else if (opt=="-Npx")
    {
      if (k<argc)
      {
	Number_of_pixels_x = atoi(argv[k++]);
	Number_of_pixels_y = Number_of_pixels_x;
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Npx.\n";
	std::exit(1);
      }
    }
    else if (opt=="-Npxy")
    {
      if (k+1<argc)
      {
	Number_of_pixels_x = atoi(argv[k++]);
	Number_of_pixels_y = atoi(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO int arguments must be provided after -Npxy.\n";
	std::exit(1);
      }
    }
    else if (opt=="--fov")
    {
      if (k<argc)
      {
	Field_of_view_x = atof(argv[k++]) * 1e-6/3600./180.*M_PI;
	Field_of_view_y = Field_of_view_x;
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A float argument must be provided after --fov.\n";
	std::exit(1);
      }
    }
    else if (opt=="--fovxy")
    {
      if (k+1<argc)
      {
	Field_of_view_x = atof(argv[k++]) * 1e-6/3600./180.*M_PI;
	Field_of_view_y = atof(argv[k++]) * 1e-6/3600./180.*M_PI;
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO float arguments must be provided after --fovxy, corresponding to the x/y extents of the fov.\n";
	std::exit(1);
      }
    }
    else if (opt=="-pa" || opt=="--position-angle")
    {
      if (k<argc)
	Position_angle = atof(argv[k++]) * M_PI/180.0;
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A float argument must be provided after -pa or --position-angle.\n";
	std::exit(1);
      }      
    }
    else if (opt=="--seed")
    {
      if (k<argc)
	seed = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --seed.\n";
	std::exit(1);
      }
    }
    else if (opt=="-g" || opt=="--reconstruct-gains")
    {
      Reconstruct_gains=true;
    }    
    else if (opt=="-gh" || opt=="--gain-file-hi")
    {
      if (k<argc)
	gain_file_list.push_back(std::string(argv[k++]));
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --gain-file-hi, -gfh.\n";
	std::exit(1);
      }
    }
    else if (opt=="-A" || opt=="--background-gaussian")
    {
      add_background_gaussian=true;
    }    
    else if (opt=="--ring" || opt=="-X")
    {
      add_ring=true;
    }
    else if (opt=="--ring-diameter" || opt=="-XD")
    {
      if ( k <argc )
      {
	add_ring=true;
	initial_ring_diameter = atof(argv[k++]);
      }
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --ring-diameter, -XD.\n";
        std::exit(1);
      }
    }
    else if (opt=="--roving-gassian" || opt=="-rg")
    {
      add_roving_gaussian=true;
    }    
    else if (opt=="--background-gaussian-start" || opt=="-A-start")
    {
      add_background_gaussian_start=true;
    }
    else if (opt=="--ring-start" || opt=="-X-start")
    {
      add_ring_start=true;
    }
    else if (opt=="--roving-gaussian-start" || opt=="-rg-start")
    {
      add_roving_gaussian_start=true;
    }    
    else if (opt=="--opt-likelihood-factor")
    {
      if (k<argc)
	opt_ko_llrf = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --opt-likelihood-factor.\n";
	std::exit(1);
      }
    }
    else if (opt=="--opt-itermax")
    {
      if (k<argc)
	opt_ko_itermax = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --opt-itermax.\n";
	std::exit(1);
      }
    }
    else if (opt=="--opt-instances")
    {
      if (k<argc)
	opt_instances = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --opt-instances, -oi.\n";
	std::exit(1);
      }
    }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for Challenge J+\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 imrecad_complex_single -v v_file_list [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-v, --visibilities <string>\n"
		  << "\t\tSets the name of the file containing complex visibility data.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
		  << "\t-p, --parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt.  Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.).\n"
		  << "\t-rf, --frequency <float>\n"
		  << "\t\tRadio frequency of the observation in Hz. Default: 230e9.\n"
		  << "\t--array <array>\n"
		  << "\t\tArray specification.  Options are either eht2017 or an array file that contains one station code\n"
		  << "\t\tand a value giving the 1-sigma gain amplitude error per line. Default: eht2017.\n"
		  << "\t-Npx-start <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction in the parameter file.\n"
		  << "\t-Npxy-start <int> <int>\n"
		  << "\t\tSets the number of pixels along the x-axis and y-axis in the image reconstruction in the parameter file.\n"
		  << "\t--background-gaussian-start, -A-start\n"
		  << "\t\tSpecifies that the parameter file has an asymmetric background gaussian.\n"
		  << "\t--ring-start, -X-start\n"
		  << "\t\tSpecifies that the parameter file has a ring.\n"
		  << "\t--roving-gaussian-start, -rg-start\n"
		  << "\t\tSpecifies that the parameter file has an asymmetric roving gaussian.\n"
		  << "\t-Npx <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction.\n"
		  << "\t-Npxy <int> <int>\n"
		  << "\t\tSets the number of pixels along the x-axis and y-axis in the image reconstruction.\n"
		  << "\t--fov <float>\n"
		  << "\t\tSets the initial field of view (in uas).  Default 100.  Overrides the values in a parameter file\n"
		  << "\t\twith -p.\n"
		  << "\t--fovxy <float> <float>\n"
		  << "\t\tSets the initial field of view (in uas) for the x and y directions independently.  The meaning\n"
		  << "\t\tof thes directions is set by the data, ostensibly RA and DEC.  Default for both is 100.  Overrides\n"
		  << "\t\tthe values in a parameter file specified with -p.\n"
		  << "\t-pa, --position-angle <float>\n"
		  << "\t\tSets the initial position angle of the y-axis E of N in degrees.  Default is 0.  Overrides the\n"
		  << "\t\tvalues in a parameter file specified with -p.\n"
		  << "\t--scatter\n"
                  << "\t\tAdds diffractive scattering to the model to fit intrinsic source.\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	 
		  << "\t-g, --reconstruct-gains\n"
		  << "\t\tReconstructs unknown station gains.  Default off.\n"
		  << "\t-gh, --gain-file-hi <filename>\n"
		  << "\t\tSets the high-band gains to those in the specified file name.\n"
		  << "\t-A, --background-gaussian\n"
		  << "\t\tAdds a large-scale background gaussian, constrained to have an isotropized standard\n"
		  << "\t\tdeviation between 100 uas and 10 as.\n"
		  << "\t--ring, -X\n"
		  << "\t\tAdds a narrow ring feature (based on xsringauss).\n"
		  << "\t--ring-diameter, -XD <float>\n"
		  << "\t\tAdds a narrow ring feature (based on xsringauss) and initializes its diameter to the value given in microarcseconds.  Implies -X.\n"
		  << "\t--roving-gaussian, -rg\n"
		  << "\t\tAdds an asymmetric roving gaussian.\n"
		  << "\t--verbosity <int>\n"
		  << "\t\tSets the verbosity level.  Default 0.\n"
		  << "\t--opt-instances <int>\n"
		  << "\t\tSets the number of independent instances of the optimizer to run.  Defaults to the maximum allowed given the number of processors being used and number of processors per likelihood.\n"
		  << "\t--opt-likelihood-factor <float>\n"
		  << "\t\tSets the relative chi-squared that forces a kickout for the kickout Powell optimizer. Default 10.\n"
		  << "\t--opt-itermax <int>\n"
		  << "\t\tSets the number of Powell iterations to run per kickout check.  Default 20.\n"
		  << "\t--opt-rounds <int>\n"
		  << "\t\tSets the number of Powell rounds to run during kickout phase.  Default 20.\n"
		  << "\n\n";
      std::exit(0);
    }
    else
    {
      if (world_rank==0)
      {
	std::cerr << "ERROR: Unrecognized option " << opt << "\n";
	std::cerr << "Try -h or --help for options";
      }
      std::exit(1);
    }
  }  

  if (v_file=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No data file list was provided. The -v <string>,\n"
		<< "       is *required*.  See -h for more details and options.\n";
    std::exit(1);
  }
  
  // Set and fill the start parameters if provided
  // Assumes has the same format as the fit_summaries.txt file (header, index, parameters, then other items)
  std::vector<double> start_parameter_list;
  if (param_file!="")
  {
    double* buff = new double[Number_start_params];
    if (world_rank==0)
    {
      std::fstream pfin(param_file);
      if (!pfin.is_open())
      {
	std::cerr << "ERROR: Could not open " << param_file << '\n';
	std::exit(1);
      }
      double dtmp;
      std::string stmp;
      //pfin.ignore(4096,'\n');
      getline(pfin,stmp);
      // Get first index
      pfin >> dtmp;
      for (int k=0; k<Number_start_params; k++)
      {
	pfin >> dtmp;
	start_parameter_list.push_back(dtmp);
      }

      for (int k=0; k<Number_start_params; k++)
	buff[k] = start_parameter_list[k];
    }
    
    MPI_Bcast(&buff[0],Number_start_params,MPI_DOUBLE,0,MPI_COMM_WORLD);
    start_parameter_list.resize(Number_start_params);
    for (int k=0; k<Number_start_params; k++)
      start_parameter_list[k] = buff[k];
    delete[] buff;
	      
    if (start_Number_of_pixels_x==0)
    {
      start_Number_of_pixels_x = Number_of_pixels_x;
      start_Number_of_pixels_y = Number_of_pixels_y;
    }

    if (Field_of_view_x==0)
      Field_of_view_x = start_parameter_list[start_Number_of_pixels_x*start_Number_of_pixels_y];
    if (Field_of_view_y==0)
      Field_of_view_y = start_parameter_list[start_Number_of_pixels_x*start_Number_of_pixels_y+1];
  }
  else
  {
    if (Field_of_view_x==0)
      Field_of_view_x = 100.0 *1e-6/3600./180.*M_PI; // in rad
    if (Field_of_view_y==0)
      Field_of_view_y = 100.0 *1e-6/3600./180.*M_PI; // in rad
  }
  
  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "Npx_x: " << Number_of_pixels_x << "\n";
    std::cout << "Npx_y: " << Number_of_pixels_y << "\n";
    std::cout << "FOV_x: " << Field_of_view_x << "\n";
    std::cout << "FOV_y: " << Field_of_view_y << "\n";
    std::cout << "v file: " << v_file << "\n";
    if (Number_start_params>0)
    {
      std::cout << "\t";
      for (size_t j=0; j<start_parameter_list.size(); ++j)
	std::cout << std::setw(15) << start_parameter_list[j];
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
  }

  // Generate image model
  Themis::model_image_adaptive_splined_raster image_pulse(Number_of_pixels_x,Number_of_pixels_y);
  image_pulse.use_analytical_visibilities();
  Themis::model_image* image_ptr = &image_pulse;

  // Generate model image sum to create shift and possibly add components
  std::vector< Themis::model_image* > model_components;
  model_components.push_back(image_ptr);

  // Background Gaussian
  Themis::model_image_asymmetric_gaussian model_a;
  model_a.use_analytical_visibilities();
  if (add_background_gaussian)
    model_components.push_back(&model_a);

  // Ring
  Themis::model_image_xsringauss model_X;
  model_X.use_analytical_visibilities();
  if (add_ring)
    model_components.push_back(&model_X);

  // Create a roving asymmetric Gaussian
  //   Step 1: Make an asymmetric gaussian
  Themis::model_image_asymmetric_gaussian model_rg_a;
  model_rg_a.use_analytical_visibilities();
  //   Step 2: Make it a sum to create the shift in polar coords
  Themis::model_image_sum model_rg_s("polar");
  model_rg_s.add_model_image(model_rg_a);
  //   Step 3: Permit elements of rover to evolve.
  std::vector<int> order_rg(model_rg_s.size(),0);
  order_rg[5]=1;
  Themis::model_image_polynomial_variable model_rg(model_rg_s, order_rg);
  //   Step 4: Add to list of model compoents
  if (add_roving_gaussian)
    model_components.push_back(&model_rg);
    
  // Sum 
  Themis::model_image_sum image_sum(model_components);

 
  // Scattered image with Sgr A* screen
  Themis::model_visibility* model_ptr;
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image_sum);
  if (scatter)
    model_ptr = &scattimage;
  else
    model_ptr = &image_sum;

  // Generate reference for future use
  Themis::model_visibility& image=(*model_ptr);
  
  // Read in data files
  std::vector<Themis::data_visibility*> V_data;
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_optimal_complex_gain_visibility*> lvg;
  std::vector<Themis::likelihood_visibility*> lv;

  // Set gain priors
  std::vector<std::string> station_codes;
  std::vector<double> station_gain_priors;
  if (array=="eht2017")
  {
    station_codes = Themis::utils::station_codes("uvfits 2017");
    station_gain_priors.resize(station_codes.size(),0.1);
    station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  }
  else if (array=="ngEHT")
  {
    station_codes.push_back("AA");
    station_codes.push_back("AP");
    station_codes.push_back("AZ");
    station_codes.push_back("BA");
    station_codes.push_back("CN");
    station_codes.push_back("CT");
    station_codes.push_back("DC");
    station_codes.push_back("GA");
    station_codes.push_back("GS");
    station_codes.push_back("GT");
    station_codes.push_back("HY");
    station_codes.push_back("JC");
    station_codes.push_back("KP");
    station_codes.push_back("LL");
    station_codes.push_back("LM");
    station_codes.push_back("NZ");
    station_codes.push_back("OV");
    station_codes.push_back("PB");
    station_codes.push_back("PV");
    station_codes.push_back("SM");
    station_codes.push_back("SP");
    station_gain_priors.resize(station_codes.size(),0.1);
  }
  else
  {
    std::ifstream afin(array);
    if (!afin.is_open())
    {
      std::cerr << "ERROR: Could not open " << array << '\n';
      std::exit(1);
    }
    station_codes.resize(0);
    station_gain_priors.resize(0);
    std::string stmp;
    double dtmp;
    for ( afin>>stmp ; ! afin.eof(); )
    {
      afin >> dtmp;
      station_codes.push_back(stmp);
      station_gain_priors.push_back(dtmp);
      afin >> stmp;
      
      if (world_rank==0)
	std::cerr << "Adding station " << stmp << " with gain amplitude prior " << dtmp << '\n';
    }
  }

  
  // Get the list of data files
  std::vector<std::string> v_file_name_list;
  std::string v_file_name;
  int strlng;
  if (world_rank==0)
  {
    std::ifstream vin(v_file);
    for (vin>>v_file_name; !vin.eof(); vin>>v_file_name)
      v_file_name_list.push_back(v_file_name);
    strlng = v_file_name.length()+1;
  }
  int ibuff = v_file_name_list.size();
  MPI_Bcast(&ibuff,1,MPI_INT,0,MPI_COMM_WORLD);
  for (int i=0; i<ibuff; ++i)
  {
    if (world_rank==0)
      strlng = v_file_name_list[i].length()+1;
    MPI_Bcast(&strlng,1,MPI_INT,0,MPI_COMM_WORLD);

    char* cbuff = new char[strlng];
    if (world_rank==0)
      strcpy(cbuff,v_file_name_list[i].c_str());
    MPI_Bcast(&cbuff[0],strlng,MPI_CHAR,0,MPI_COMM_WORLD);
    if (world_rank>0)
      v_file_name_list.push_back(std::string(cbuff));
    delete[] cbuff;
  }

  double variance_weighted_time_average=0.0, vwta_var_norm=0.0, minimum_time=0, maximum_time=1;
  for (size_t j=0; j<v_file_name_list.size(); ++j)
  {
    V_data.push_back( new Themis::data_visibility(v_file_name_list[j],"HH") );
    V_data[j]->set_default_frequency(frequency);
    
    // Get time particulars for roving Gaussian
    for (size_t k=0; k<V_data[j]->size(); ++k)
    {
      variance_weighted_time_average += V_data[j]->datum(k).tJ2000 / std::pow(std::abs(V_data[j]->datum(k).err),2);
      vwta_var_norm += 1.0 / std::pow(std::abs(V_data[j]->datum(k).err),2);
	  
      if (j==0 && k==0)
      {
	minimum_time = maximum_time = V_data[0]->datum(0).tJ2000;
      }

      if (V_data[j]->datum(k).tJ2000<minimum_time)
	minimum_time = V_data[j]->datum(k).tJ2000;
      if (V_data[j]->datum(k).tJ2000>maximum_time)
	maximum_time = V_data[j]->datum(k).tJ2000;
    }
    
    if (Reconstruct_gains)
    {
      lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(*V_data[j],image,station_codes,station_gain_priors) );
      L.push_back( lvg[j] );

      // Set gains if gain files are provided
      if (j<gain_file_list.size() && gain_file_list[j]!="")
      {
	lvg[j]->read_gain_file(gain_file_list[j]);
	lvg[j]->fix_gains();
      }
    }
    else
    {
      lv.push_back( new Themis::likelihood_visibility(*V_data[j],image) );
      L.push_back( lv[j] );
    }
  }
  // Finish variance weighted time average and set the reference time for the rover
  variance_weighted_time_average /= vwta_var_norm;
  int coutprecorig = std::cout.precision();
  if (world_rank==0)
    std::cout << std::setprecision(20) 
	      << "Reference time for rover (if included) = " << variance_weighted_time_average << '\n'
	      << "Observation start time = " << minimum_time << '\n'
	      << "Observation end time = " << maximum_time << '\n'
	      << "Observation duration = " << maximum_time-minimum_time << '\n' 
	      << std::setprecision(coutprecorig) << std::endl;
  model_rg.set_reference_time(variance_weighted_time_average);

  // Output model tag (before scattering)  THIS IS WRONG, SHOULD HAVE MODEL_TAG PASS THROUGH FOR SCATTERED IMAGE
  image_sum.write_model_tag_file();
  
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(30,50)); // Itotal
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_x)); // fovx
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_y)); // fovy
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // PA
  
  // Generate a set of means for the initial conditions
  std::vector<double> means(Number_of_pixels_x*Number_of_pixels_y);
  std::vector<std::string> var_names;

  std::cout << "Are gains on: " << Reconstruct_gains << std::endl; 

  // Set priors if fov's are given
  double x,y;
  double sig = 0.5*Field_of_view_x; //0.17/0.5*(0.5*Field_of_view_x);
  double norm = std::log(2.5/(2.0*M_PI*sig*sig));
  for (size_t j=0,k=0; j<Number_of_pixels_x; ++j)
    for (size_t i=0; i<Number_of_pixels_y; ++i)
    {
      x = Field_of_view_x*double(i)/double(Number_of_pixels_x-1) - 0.5*Field_of_view_x;
      y = Field_of_view_y*double(j)/double(Number_of_pixels_y-1) - 0.5*Field_of_view_y;
      means[k++] = std::min( std::max(norm  - (x*x+y*y)/(2.0*sig*sig),21.0) , 59.0 );
    }
  means.push_back(Field_of_view_x);
  means.push_back(Field_of_view_y);
  if (Position_angle==-999)
  {
    means.push_back(0.0);
  }
  else
  {
    means.push_back(Position_angle);
  }

  double uas2rad = 1e-6/3600. * M_PI/180.;

  // x offset of image (center raster!)
  P.push_back(new Themis::prior_linear(-400*uas2rad,400*uas2rad));
  means.push_back(0.0);
  
  // y offset of image (center raster)
  P.push_back(new Themis::prior_linear(-400*uas2rad,400*uas2rad));
  means.push_back(0.0);
  

  // Add priors for background Gaussian if desired
  if (add_background_gaussian)
  {  
    // Itot
    P.push_back(new Themis::prior_linear(0.0,10));
    means.push_back(4.0);
    
    // Size
    P.push_back(new Themis::prior_linear(1e2*uas2rad,1e7*uas2rad));
    means.push_back(1e3*uas2rad);
    
    // Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.1);

    // Position angle
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    
    // x offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    means.push_back(0.0);

    // y offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    means.push_back(0.0);
  }


  // Add priors for ring if desired
  if (add_ring)
  {
    //   0 Itot
    P.push_back(new Themis::prior_linear(0.0,2));
    means.push_back(0.3);
    //   1 Outer size R
    P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
    means.push_back(0.5*initial_ring_diameter*uas2rad);
    //   2 psi
    P.push_back(new Themis::prior_linear(0.0001,0.05));
    means.push_back(0.01);
    //   3 1-tau
    P.push_back(new Themis::prior_linear(0.0001,0.00011));
    means.push_back(0.000105);
    //   4 f
    P.push_back(new Themis::prior_linear(0.00,1.0));
    means.push_back(0.5);
    //   5 g
    P.push_back(new Themis::prior_linear(0,1e-6));
    means.push_back(1e-8);
    //   6 a
    P.push_back(new Themis::prior_linear(0.0,1.0e-6));
    means.push_back(1e-8);
    //   7 Ig
    P.push_back(new Themis::prior_linear(0.0,1.0e-6));
    means.push_back(1e-8);
    //   8 Position angle
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0.5*M_PI);
    //   9 x offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    //  10 y offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
  }

  // Add priors for roving Gaussian if desired
  if (add_roving_gaussian)
  {  
    double dtj;

    // Itot
    P.push_back(new Themis::prior_linear(0.0,10));
    means.push_back(0.5);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[0]); ++j)
    {
      dtj=1.0/std::pow(maximum_time-minimum_time,j);
      P.push_back(new Themis::prior_linear(-10*dtj,10*dtj));
      means.push_back(0.0);
    }
    
    // Size
    P.push_back(new Themis::prior_linear(0*uas2rad,1e2*uas2rad));
    means.push_back(10*uas2rad);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[1]); ++j)
    {
      dtj=1.0/std::pow(maximum_time-minimum_time,j);
      P.push_back(new Themis::prior_linear(-1e2*uas2rad*dtj,1e2*uas2rad*dtj));
      means.push_back(0.0);
    }
    
    // Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.001);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[2]); ++j)
    {
      dtj=1.0/std::pow(maximum_time-minimum_time,j);
      P.push_back(new Themis::prior_linear(-0.99*dtj,0.99*dtj));
      means.push_back(0.0);
    }

    // Position angle
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[3]); ++j)
    {
      dtj=1.0/std::pow(maximum_time-minimum_time,j);
      P.push_back(new Themis::prior_linear(-10*M_PI*dtj,10*M_PI*dtj));
      means.push_back(0.0);
    }

    // r offset
    P.push_back(new Themis::prior_linear(0*uas2rad,50*uas2rad));
    means.push_back(25.0*uas2rad);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[4]); ++j)
    {
      dtj=1.0/std::pow(maximum_time-minimum_time,j);
      P.push_back(new Themis::prior_linear(-200*uas2rad*dtj,200*uas2rad*dtj));
      means.push_back(0.0);
    }

    // theta offset
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0.0);
    // Derivatives
    for (size_t j=1; j<=size_t(order_rg[5]); ++j)
    {
      P.push_back(new Themis::prior_linear(-2.0*M_PI/std::pow(60.0,j),2.0*M_PI/std::pow(60.0,j)));
      means.push_back(0.0);
    }
    
    // x offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);

    // y offset
    P.push_back(new Themis::prior_linear(-50*uas2rad,50*uas2rad));
    means.push_back(0.0);
  }


  
  ////////////////////////  Before checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check ==============================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "\n";
    std::cerr << "image_ptr size: " << image_ptr->size()
	      << "  model_ptr size: " << model_ptr->size()
	      << "  image_sum size: " << image_sum.size()
	      << '\n';
    for (size_t k=0; k<image.size(); ++k)
      std::cerr << std::setw(15) << means[k]
		<< '\n';
    std::cerr << "===========================================================\n";
  }
  //////////////////////////////////


  // Fill from start_parameter_list
  if ( start_parameter_list.size()>0 )
  {
    size_t k=0;
    size_t kspl=0;

    double start_Field_of_view_x = start_parameter_list[start_Number_of_pixels_x*start_Number_of_pixels_y+0];
    double start_Field_of_view_y = start_parameter_list[start_Number_of_pixels_x*start_Number_of_pixels_y+1];
    double start_Position_angle = start_parameter_list[start_Number_of_pixels_x*start_Number_of_pixels_y+2];

    std::valarray<double> xs(start_Number_of_pixels_x),ys(start_Number_of_pixels_y),fs(start_Number_of_pixels_x*start_Number_of_pixels_y);

    for (size_t i=0; i<start_Number_of_pixels_x; ++i)
      xs[i] = start_Field_of_view_x*double(i)/double(start_Number_of_pixels_x-1)-0.5*start_Field_of_view_x;
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      ys[j] = start_Field_of_view_y*double(j)/double(start_Number_of_pixels_y-1)-0.5*start_Field_of_view_y;
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	fs[j+start_Number_of_pixels_y*i] = std::exp(start_parameter_list[kspl++]);
    Themis::Interpolator2D start_image_interp(xs,ys,fs);
    start_image_interp.use_forward_difference();

    // If not specified, set the fov to the start values
    if (Field_of_view_x==0)
      Field_of_view_x = start_Field_of_view_x;
    if (Field_of_view_y==0)
      Field_of_view_y = start_Field_of_view_y;
			   
    
    for (size_t j=0; j<Number_of_pixels_y; ++j)
      for (size_t i=0; i<Number_of_pixels_x; ++i)
      {
	// Interpolate up
	double x=Field_of_view_x*double(i)/double(Number_of_pixels_x-1)-0.5*Field_of_view_x;
	double y=Field_of_view_y*double(j)/double(Number_of_pixels_y-1)-0.5*Field_of_view_y;
	double val;
	start_image_interp.bicubic(x,y,val);
	if (val<0)
	  start_image_interp.linear(x,y,val);
	
	// Moderate flux outside of boundary
	double mask=1.0;
	double xstps = xs[1]-xs[0];
	double ystps = ys[1]-ys[0];
	if (std::fabs(x)>std::fabs(xs[0]))
	{
	  if (x<0)
	    mask *= std::exp(-(x-xs[0])*(x-xs[0])/(2.0*xstps*xstps));
	  else
	    mask *= std::exp(-(x-xs[Number_of_pixels_x-1])*(x-xs[Number_of_pixels_x-1])/(2.0*xstps*xstps));
	}
	if (std::fabs(y)>std::fabs(ys[0]))
	{
	  if (y<0)
	    mask *= std::exp(-(y-ys[0])*(y-ys[0])/(2.0*ystps*ystps));
	  else
	    mask *= std::exp(-(y-ys[Number_of_pixels_y-1])*(y-ys[Number_of_pixels_y-1])/(2.0*ystps*ystps));
	}

	// Set value
	means[k++] = std::min( std::max( std::log(val*mask), 21.0 ), 59.0 );
      }

    means[k++] = Field_of_view_x;
    means[k++] = Field_of_view_y;
    if (Position_angle==-999)
      means[k++] = start_Position_angle;
    else
      means[k++] = Position_angle;

    kspl += 3;
    
    if (world_rank==0)
    {
      std::cerr << "FoVs" << std::setw(15) << Field_of_view_x << std::setw(15) << Field_of_view_y << std::endl;
      std::cerr << "Added raster image to start values k=" << k  << "  kspl=" << kspl << std::endl;
    }
    
    // x0
    means[k] = start_parameter_list[kspl]; 
    k++; kspl++;
      
    // y0
    means[k] = start_parameter_list[kspl]; 
    k++; kspl++;

    if (add_background_gaussian && add_background_gaussian_start)
    {      
      if (world_rank==0)
	std::cerr << "Adding gaussian to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_a.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }

    if (add_ring && add_ring_start)
    {
      if (world_rank==0)
	std::cerr << "Adding ring to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_X.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }

    if (add_roving_gaussian && add_roving_gaussian_start)
    {
      if (world_rank==0)
	std::cerr << "Adding roving Gaussian to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_rg.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }
  }

  ////////////////////// After checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check after =========================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "\n";
    std::cerr << "image_ptr size: " << image_ptr->size()
	      << "  model_ptr size: " << model_ptr->size()
	      << "  image_sum size: " << image_sum.size()
	      << '\n';
    for (size_t k=0; k<image.size(); ++k)
    {
      std::cerr << "PriorCheck: "
		<< std::setw(15) << means[k]
		<< std::setw(15) << P[k]->lower_bound()
		<< std::setw(15) << P[k]->upper_bound();
      if (means[k]<P[k]->lower_bound())
	std::cerr << " PRIOR RANGE ERROR <<<<<";
      else if (means[k]>P[k]->upper_bound())
	std::cerr << " PRIOR RANGE ERROR >>>>>";
      else
	std::cerr << " OK";
      std::cerr << '\n';
    }
    std::cerr << "===========================================================\n";
  }

  
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  //Themis::likelihood_power_tempered L_temp(L_obj);
  
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Ndata = 0;
  for (size_t j=0; j<V_data.size(); ++j)
    Ndata += 2*V_data[j]->size();
  int Nparam = image.size();
  if (add_ring) // Forced to be thin, no Gaussian
    Nparam -= 4;
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);
  if (Reconstruct_gains)
    for (size_t j=0; j<lvg.size(); ++j)
    {
      Ngains_list[j] = lvg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  int NDoF = Ndata - Nparam - Ngains;
  

  // Get the best point
  std::vector<double> pbest =  means;
  double lMAP;
  Themis::optimizer_laplace optim(L_obj, var_names, pbest.size());
  optim.set_parameters(1e-3, opt_ko_itermax, 200);
  optim.set_scale(Ndata*opt_ko_llrf);
  int nitr = 0;
  nitr = optim.parallel_optimizer(pbest, lMAP, opt_instances, seed+world_rank);

  if (world_rank == 0){  
    std::cout << "LBFGSB ran for " << nitr << " iterations\n";

    std::cout << "Maximum at \n";
    for ( size_t i = 0; i < pbest.size(); ++i )
     std::cout << std::setw(15) <<  pbest[i];
    std::cout << std::endl;
    std::cout << "map: " << L_obj(pbest) << std::endl;

    std::cout << "Finding the Precision matix" << std::endl;
  }
  Eigen::MatrixXd precision;
  Eigen::VectorXd pmax = Eigen::VectorXd::Constant(image.size(), 0.0);
  for ( size_t i = 0; i < pbest.size(); ++i )
    pmax(i) = pbest[i];
  //optim.find_precision(pmax, precision, "laplace.txt");
  
  std::stringstream sumoutname;
  sumoutname << "laplace" << "_fit_summaries" << ".txt";
  std::ofstream sumout;
  if (world_rank==0)
  {
    sumout.open(sumoutname.str().c_str());
    sumout << std::setw(10) << "# Index";
    for (size_t k=0; k<image_pulse.size(); ++k)
    {
      std::stringstream var;
      var << "p" << k;
      sumout << std::setw(15) << var.str();
    }
    sumout << std::setw(15) << "img-x"
	   << std::setw(15) << "img-y";
    if (add_background_gaussian)
    {
      sumout << std::setw(15) << "A-I"
             << std::setw(15) << "A-sig"
             << std::setw(15) << "A-A"
             << std::setw(15) << "A-phi"
             << std::setw(15) << "A-x"
             << std::setw(15) << "A-y";
    }
    if (add_ring)
    {
      sumout << std::setw(15) << "Isx (Jy)"
	     << std::setw(15) << "Rp (uas)"
	     << std::setw(15) << "psi"
	     << std::setw(15) << "ecc"
	     << std::setw(15) << "f"
	     << std::setw(15) << "gax"
	     << std::setw(15) << "a"
	     << std::setw(15) << "ig"
	     << std::setw(15) << "PA"
	     << std::setw(15) << "xsX"
	     << std::setw(15) << "ysX";
    }
    if (add_roving_gaussian)
    {
      for (size_t j=0; j<=size_t(order_rg[0]); ++j)
	sumout << std::setw(15) << "RG-I"+std::to_string(j);
      for (size_t j=0; j<=size_t(order_rg[1]); ++j)
	sumout << std::setw(15) << "RG-sig"+std::to_string(j);
      for (size_t j=0; j<=size_t(order_rg[2]); ++j)
	sumout << std::setw(15) << "RG-A"+std::to_string(j);
      for (size_t j=0; j<=size_t(order_rg[3]); ++j)
	sumout << std::setw(15) << "RG-phi"+std::to_string(j);
      for (size_t j=0; j<=size_t(order_rg[4]); ++j)
	sumout << std::setw(15) << "RG-r"+std::to_string(j);
      for (size_t j=0; j<=size_t(order_rg[5]); ++j)
	sumout << std::setw(15) << "RG-theta"+std::to_string(j);
      sumout << std::setw(15) << "RG-x";
      sumout << std::setw(15) << "RG-y";
    }
    for (size_t j=0; j<L.size(); ++j) 
      sumout << std::setw(15) << "V" + std::to_string(j) + " rc2";
    sumout << std::setw(15) << "Total rc2"
	   << std::setw(15) << "log-liklhd"
	   << "     FileNames"
	   << std::endl;
  }


  if (Reconstruct_gains)
    for (size_t j=0; j<lvg.size(); ++j)
    {
      std::stringstream gc_name, cgc_name;
      gc_name << "laplace"  << "_gain_corrections_" << std::setfill('0') << j << ".d";
      cgc_name << "laplace" << "_complex_gains_" << std::setfill('0') << j << ".d";
      lvg[j]->output_gain_corrections(gc_name.str());
      lvg[j]->output_gains(cgc_name.str());
    }


    // Generate summary and ancillary output

  //   Summary file parameter location:
  if (world_rank==0)
  {
    sumout << std::setw(10) << 0;
    for (size_t k=0; k<image.size(); ++k)
      sumout << std::setw(15) << pbest[k];
  }

  //   Data-set specific output
  double V_rchi2=0.0, rchi2=0.0;
  for (size_t j=0; j<L.size(); ++j)
  {
    // output residuals
    std::stringstream v_res_name;
    v_res_name << "laplace" << "_residuals_" << std::setfill('0') << j << ".d";
      L[j]->output_model_data_comparison(v_res_name.str());

    // Get the data-set specific chi-squared
    V_rchi2 = (L[j]->chi_squared(pbest));
    if (world_rank==0)
    {
      std::cout << "V" << j << " rchi2: " << V_rchi2  << " / " << 2*V_data[j]->size() << " - " << Nparam << " - " << Ngains_list[j] << std::endl;
      // Summary file values
      sumout << std::setw(15) << V_rchi2 / ( 2*int(V_data[j]->size()) - Nparam - Ngains_list[j]);
    }
  }
  rchi2 = L_obj.chi_squared(pbest);
  if (world_rank==0)
    std::cout << "Tot rchi2: " << rchi2  << " / " << NDoF << std::endl;
  rchi2 = rchi2 / NDoF;

  // Generate summary file      
  if (world_rank==0)
  {
    sumout << std::setw(15) << rchi2
	   << std::setw(15) << lMAP;
    for (size_t j=0; j<v_file_name_list.size(); ++j)
      sumout << "   " << v_file_name_list[j];
    sumout << std::endl;
  }
    
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
