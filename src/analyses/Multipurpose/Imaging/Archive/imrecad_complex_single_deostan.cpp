/*!
  \file analyses/Imaging/imrecad_complex_single_deostan.cpp
  \author Avery Broderick & Paul Tiede
  \date Feb 2020
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
#include "model_image.h"
#include "likelihood.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "sampler_differential_evolution_deo_tempered_MCMC.h"
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_stan_adapt_dense_e_nuts_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"
#include "optimizer_kickout_powell.h"
#include "utils.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"

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
  int Number_of_steps = 10; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  bool Reconstruct_gains = false;
  std::string gain_high_file="";

  size_t Number_of_pixels_x = 4;
  size_t Number_of_pixels_y = 4;
  double Field_of_view_x = 0;
  double Field_of_view_y = 0;
  double Position_angle = -999;
  
  //int Number_of_tempering_levels=world_size;
  //int round_geometric_factor = 2;
  double initial_ladder_spacing = 1.15;
  int thin_factor = 1;
  int Temperature_stride = 50;
  std::string annealing_ladder_file = "";
  int refresh_rate = 1;
  int verbosity = 0;
  
  size_t Number_of_reps = 7;

  bool add_background_gaussian=false;
  bool add_ring=false;

  // Start features
  bool add_background_gaussian_start=false;
  bool add_ring_start=false;
  size_t start_Number_of_pixels_x=0;
  size_t start_Number_of_pixels_y=0;
  bool scatter=false;

  bool restart_flag = false;
  
  bool preoptimize_flag = false;
  //bool postoptimize_flag = false;
  double opt_ko_llrf = 10.0;
  size_t opt_ko_itermax = 5;
  size_t opt_ko_rounds = 5;
  size_t opt_instances = 0;

  // Dump command line for posterity
  if (world_rank==0)
  {
    std::cout << "\n=============================================================" << std::endl;
    std::cout << "CMD$ <mpiexec>";
    for (int k=0; k<argc; k++)
      std::cout << " " << argv[k];
    std::cout << std::endl;
    std::cout << "=============================================================\n" << std::endl;
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
    else if (opt=="-Ns")
    {
      if (k<argc)
	Number_of_steps = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Ns.\n";
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
    else if (opt=="-Nr")
    {
      if (k<argc)
	Number_of_reps = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Nr.\n";
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
    /*
    else if (opt=="--tempering-levels")
    {
      if (k<argc)
	Number_of_tempering_levels = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --tempering-levels.\n";
	std::exit(1);
      }
    }
    else if (opt=="--tempering-ladder")
    {
      if (k<argc)
	Tempering_ladder = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An float argument must be provided after --tempering-ladder.\n";
	std::exit(1);
      }
    }
    else if (opt=="--tempering-time")
    {
      if (k<argc)
	Tempering_time = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An float argument must be provided after --tempering-time.\n";
	std::exit(1);
      }
    }
    */
    else if (opt=="--number-of-rounds" || opt=="-nor")
    {
      if (k<argc)
	Number_of_reps = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --number-of-rounds, -nor.\n";
	std::exit(1);
      }
    }
    /*
    else if (opt=="--round-geometric-factor" || opt=="-rgf")
    {
      if (k<argc)
	round_geometric_factor = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --round-geometric-factor, -rgf.\n";
	std::exit(1);
      }
    }
    */
    else if (opt=="--initial-ladder-spacing" || opt=="-ils")
    {
      if (k<argc)
	initial_ladder_spacing = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --initial-ladder-spacing, -ils.\n";
	std::exit(1);
      }
    }
    else if (opt=="--temperature-stride" || opt=="-ts")
    {
      if (k<argc)
	Temperature_stride = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --temperature-stride, -ts.\n";
	std::exit(1);
      }
    }
    else if (opt=="--annealing-ladder-file" || opt=="-alf")
    {
      if (k<argc)
	annealing_ladder_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --annealing-ladder-file, -alf.\n";
	std::exit(1);
      }
    }
    else if (opt=="--thin-factor" || opt=="-tf")
    {
      if (k<argc)
	thin_factor = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --thin-factor, -tf.\n";
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
	gain_high_file = std::string(argv[k++]);
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
    else if (opt=="--background-gaussian-start" || opt=="-A-start")
    {
      add_background_gaussian_start=true;
    }
    else if (opt=="--ring-start" || opt=="-X-start")
    {
      add_ring_start=true;
    }
    else if (opt=="--continue" || opt=="--restart")
    {
      restart_flag = true;
    }
    else if (opt=="--pre-optimize" || opt=="-po")
    {
      preoptimize_flag = true;
    }
    /*
    else if (opt=="--post-optimize" || opt=="-Po")
    {
      postoptimize_flag = true;
    }
    */
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
    else if (opt=="--opt-rounds")
    {
      if (k<argc)
	opt_ko_rounds = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --opt-rounds.\n";
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
    else if ( opt == "--refresh" )
    {
      if ( k <argc )
        refresh_rate = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --refresh.\n";
        std::exit(1);
      }
    }
    else if (opt=="-v" || opt=="--verbosity")
    {
      if (k<argc)
	verbosity=atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -v, --verbosity\n";
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
		  << "\t\tto set parameter 1, etc.). This also shrinks the affected ranges.\n"
		  << "\t-Npx-start <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction in the parameter file.\n"
		  << "\t-Npxy-start <int> <int>\n"
		  << "\t\tSets the number of pixels along the x-axis and y-axis in the image reconstruction in the parameter file.\n"
		  << "\t--background-gaussian-start, -A-start\n"
		  << "\t\tSpecifies that the parameter file has an asymmetric background gaussian.\n"
		  << "\t--ring-start, -X-start\n"
		  << "\t\tSpecifies that the parameter file has a ring.\n"
		  << "\t-Ns <int>\n"
		  << "\t\tSets the number of MCMC steps to take for each repetition chain.  Defaults to 1000.\n"
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
	  //<< "\t--tempering-levels <int>\n"
	  //<< "\t\tSets the number of tempering levels.  Defaults to the number of processes.\n"
		  << "\t--number-of-rounds, -nor, -Nr <int>\n"
		  << "\t\tSets the number of rounds for each DEO repitition.  Defaults to 7.\n"
	  //<< "\t--round-geometric-factor, -rgf <int>\n"
	  //<< "\t\tSets the factor by which subsequent rounds increase the number of steps.  Defaults to 2.\n"
		  << "\t--initial-ladder-spacing, -ils <float>\n"
		  << "\t\tSets the geometric factor by which subsequent ladder beta increases.  Defaults to 1.15.\n"
		  << "\t--annealing-ladder-file, -alf <string>\n"
		  << "\t\tStarts the temperature ladder from a prior run as specified in the filename provided (e.g., Annealing.dat).  Note that this obviates --initial-ladder-spacing.\n"
		  << "\t--temperature-stride, -ts <int>\n"
		  << "\t\tSets the number of steps between tempering level swaps.  Defaults to 50.\n"
		  << "\t--thin-factor, -tf <int>\n"
		  << "\t\tSets the factor by which chain outputs are thined.  Defaults to 10.\n"
		  << "\t--continue/--restart\n"
		  << "\t\tRestarts a run using the local MCMC.ckpt file.  Note that this restarts at step 0, i.e., does not continue to the next step.\n"
		  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
                  << "\t\tDefault is 1 step.\n"
		  << "\t-v, --verbosity <int>\n"
		  << "\t\tSets the verbosity level.  Default 0.\n"
		  << "\t-po, --pre-optimize\n"
		  << "\t\tMakes an attempt to identify an optimal fit prior to running the sampler.  Optimal models may be found in PreOptimizationSummary.dat\n"
	  //<< "\t-Po, --post-optimize\n"
	  //<< "\t\tOptimizes after sampling but prior to generating the fit_summaries file.  Optimal models may be found in PostOptimizationSummary.dat\n"
		  << "\t--opt-instances <int>\n"
		  << "\t\tSets the number of independent instances of the optimizer to run.  Defaults to the maximum allowed given the number of processors being used and number of processors per likelihood.\n"
		  << "\t--opt-likelihood-factor <float>\n"
		  << "\t\tSets the relative chi-squared that forces a kickout for the kickout Powell optimizer. Default 10.\n"
		  << "\t--opt-itermax <int>\n"
		  << "\t\tSets the number of Powell iterations to run per kickout check.  Default 5.\n"
		  << "\t--opt-rounds <int>\n"
		  << "\t\tSets the number of Powell rounds to run during kickout phase.  Default 5.\n"
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

  //std::cerr << "Debug A " << world_rank << "\n";

  
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

  //std::cerr << "Debug B " << world_rank << "\n";
  
  
  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "Npx_x: " << Number_of_pixels_x << "\n";
    std::cout << "Npx_y: " << Number_of_pixels_y << "\n";
    std::cout << "FOV_x: " << Field_of_view_x << "\n";
    std::cout << "FOV_y: " << Field_of_view_y << "\n";
    std::cout << "Nr: " << Number_of_reps << "\n";
    std::cout << "Ns: " << Number_of_steps << "\n";
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

  //std::cerr << "Debug C " << world_rank << "\n";

  // Generate image model
  Themis::model_image_adaptive_splined_raster image_pulse(Number_of_pixels_x,Number_of_pixels_y);
  image_pulse.use_analytical_visibilities();
  Themis::model_image* image_ptr = &image_pulse;

  // Sum with background gaussian if desired
  std::vector< Themis::model_image* > model_components;
  Themis::model_image_asymmetric_gaussian model_a;
  Themis::model_image_xsringauss model_X;
  model_components.push_back(image_ptr);
  if (add_background_gaussian)
    model_components.push_back(&model_a);
  if (add_ring)
    model_components.push_back(&model_X);
  model_a.use_analytical_visibilities();
  model_X.use_analytical_visibilities();
  Themis::model_image_sum image_sum(model_components);
  Themis::model_image* model_ptr;
  //if (add_background_gaussian || add_ring)
  model_ptr = &image_sum;
  //else
  //  model_ptr = image_ptr;  
  Themis::model_image& image=(*model_ptr);
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image);

  //std::cerr << "Debug D " << world_rank << "\n";

  // Output model tag
  image.write_model_tag_file();
  
  
  // Read in data files
  Themis::data_visibility V_data_hi;
  std::vector<Themis::likelihood_base*> L;
  Themis::likelihood_optimal_complex_gain_visibility *lv_hi=0;

  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  

  std::string v_file_name;
  int strlng;
  if (world_rank==0)
  {
    std::ifstream vin(v_file);    
    vin >> v_file_name;
    strlng = v_file_name.length()+1;
  }
  MPI_Bcast(&strlng,1,MPI_INT,0,MPI_COMM_WORLD);
  char* cbuff = new char[strlng];
  strcpy(cbuff,v_file_name.c_str());
  MPI_Bcast(&cbuff[0],strlng,MPI_CHAR,0,MPI_COMM_WORLD);
  v_file_name = std::string(cbuff);
  delete[] cbuff;
  
  //std::cerr << "Debug D.2 " << world_rank << "  " << v_file_name << '\n';

  
  
  V_data_hi.add_data(v_file_name,"HH");
  if (Reconstruct_gains) 
  {
    if (!scatter)
      lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,image,station_codes,station_gain_priors);
    else
      lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,scattimage,station_codes,station_gain_priors);
    L.push_back(lv_hi);
    if (gain_high_file!="")
    {
      lv_hi->read_gain_file(gain_high_file);
      lv_hi->fix_gains();
    }
  }
  else
  {
    if (!scatter)
      lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,image,station_codes,station_gain_priors);
    else
      lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,scattimage,station_codes,station_gain_priors);
    L.push_back(lv_hi);
  }

  //std::cerr << "Debug E " << world_rank << "\n";


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(20,60)); // Itotal
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_x)); // fovx
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_y)); // fovy
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // PA

  
  // Generate a set of means and ranges for the initial conditions
  std::vector<double> means(Number_of_pixels_x*Number_of_pixels_y), ranges(Number_of_pixels_x*Number_of_pixels_y,2.0);
  std::vector<std::string> var_names;

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
  ranges.push_back(0.05*Field_of_view_x);
  means.push_back(Field_of_view_y);
  ranges.push_back(0.05*Field_of_view_y);
  if (Position_angle==-999)
  {
    means.push_back(0.0);
    ranges.push_back(0.5*M_PI);
  }
  else
  {
    means.push_back(Position_angle);
    ranges.push_back(0.001*M_PI);
  }

  //std::cerr << "Debug F " << world_rank << "\n";

  double uas2rad = 1e-6/3600. * M_PI/180.;

  // x offset of image (center raster!)
  P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);
  
  // y offset of image (center raster)
  P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
  means.push_back(0.0);
  ranges.push_back(1e-7*uas2rad);
  

  // Add priors for background Gaussian if desired
  if (add_background_gaussian)
  {  
    // Itot
    P.push_back(new Themis::prior_linear(0.0,10));
    means.push_back(4.0);
    ranges.push_back(2.0);
    
    // Size
    P.push_back(new Themis::prior_linear(1e2*uas2rad,1e7*uas2rad));
    means.push_back(1e3*uas2rad);
    ranges.push_back(0.5e3*uas2rad);
    
    // Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.1);
    ranges.push_back(0.1);

    // Position angle
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    
    // x offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-2000*uas2rad,2000*uas2rad));
    means.push_back(0.0);
    ranges.push_back(50*uas2rad);

    // y offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-2000*uas2rad,2000*uas2rad));
    means.push_back(0.0);
    ranges.push_back(50*uas2rad);
  }


  // Add priors for ring if desired
  if (add_ring)
  {
    //   0 Itot
    P.push_back(new Themis::prior_linear(0.0,2));
    means.push_back(0.3);
    ranges.push_back(0.3e-2);
    //   1 Outer size R
    P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
    means.push_back(20*uas2rad);
    ranges.push_back(20e-3*uas2rad);
    //   2 psi
    P.push_back(new Themis::prior_linear(0.0001,0.05));
    means.push_back(0.01);
    ranges.push_back(0.01e-2);
    //   3 1-tau
    P.push_back(new Themis::prior_linear(0.0001,0.00011));
    means.push_back(0.000105);
    ranges.push_back(0.0000001);
    //   4 f
    P.push_back(new Themis::prior_linear(0.00,1.0));
    means.push_back(0.5);
    ranges.push_back(0.5e-2);
    //   5 g
    P.push_back(new Themis::prior_linear(0,1e-6));
    means.push_back(1e-8);
    ranges.push_back(1e-10);
    //   6 a
    P.push_back(new Themis::prior_linear(0.0,1.0e-6));
    means.push_back(1e-8);
    ranges.push_back(1e-10);
    //   7 Ig
    P.push_back(new Themis::prior_linear(0.0,1.0e-6));
    means.push_back(1e-8);
    ranges.push_back(1e-10);
    //   8 Position angle
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(M_PI);
    //   9 x offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    ranges.push_back(5*uas2rad);
    //  10 y offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    ranges.push_back(5*uas2rad);

  }

  //std::cerr << "Debug G " << world_rank << "\n";
  
  ////////////////////////  Before checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check ==============================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "  ranges size: " << ranges.size()
	      << "\n";
    std::cerr << "image_ptr size: " << image_ptr->size()
	      << "  model_ptr size: " << model_ptr->size()
	      << "  image_sum size: " << image_sum.size()
	      << '\n';
    for (size_t k=0; k<image.size(); ++k)
      std::cerr << std::setw(15) << means[k]
		<< std::setw(15) << ranges[k]
		<< '\n';
    std::cerr << "===========================================================\n";
  }
  //////////////////////////////////


  // Fill from start_parameter_list
  if ( start_parameter_list.size()>0 )
  {
    size_t k=0;
    for (k=0; k<image_pulse.size(); ++k)
      ranges[k] = 1e-3;
    ranges[image_pulse.size()-3] = 1e-3*uas2rad; // Fov_x
    ranges[image_pulse.size()-2] = 1e-3*uas2rad; // Fov_y
    
    k=0;
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
    
    //if ( (add_background_gaussian && add_background_gaussian_start) || (add_ring && add_ring_start) )
    {
      if (world_rank==0)
	std::cerr << "Adding additional components to start values k=" << k  << "  kspl=" << kspl << std::endl;
      // x0
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-8*uas2rad;
      k++; kspl++;
      
      // y0
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-8*uas2rad;
      k++; kspl++;
    }

    if (add_background_gaussian && add_background_gaussian_start)
    {      
      if (world_rank==0)
	std::cerr << "Adding gaussian to start values k=" << k  << "  kspl=" << kspl << std::endl;
      // I
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3;
      k++; kspl++;

      // sig
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3*means[k];
      k++; kspl++;
      
      // A
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3*means[k];
      k++; kspl++;
      
      // phi
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3;
      k++; kspl++;
      
      // xA
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3*uas2rad;
      k++; kspl++;
      
      // yA
      means[k] = start_parameter_list[kspl]; 
      ranges[k] = 1e-3*uas2rad;
      k++; kspl++;
    }

    if (add_ring && add_ring_start)
    {
      if (world_rank==0)
	std::cerr << "Adding ring to start values k=" << k  << "  kspl=" << kspl << std::endl;
      //   0 Itot
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;
      
      //   1 Outer size R
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   2 psi
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   3 1-tau
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   4 f
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   5 g
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   6 a
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   7 Ig
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //   8 Position angle
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //  12 x offset
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;

      //  13 y offset
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*means[k];
      k++; kspl++;
    }
  }

  
  //std::cerr << "Debug H " << world_rank << "\n";

  ////////////////////// After checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check after =========================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "  ranges size: " << ranges.size()
	      << "\n";
    std::cerr << "image_ptr size: " << image_ptr->size()
	      << "  model_ptr size: " << model_ptr->size()
	      << "  image_sum size: " << image_sum.size()
	      << '\n';
    for (size_t k=0; k<image.size(); ++k)
    {
      std::cerr << "PriorCheck: "
		<< std::setw(15) << means[k]
		<< std::setw(15) << ranges[k]
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


  //std::cerr << "Debug I " << world_rank << "\n";

  
  //MPI_Barrier(MPI_COMM_WORLD);


  //std::cerr << "Debug J " << world_rank << "\n";
  
  
  //////////////////////

  /*
  // Interpolation tests
  // Output hack chain file
  std::ofstream chout("Chain-hack.dat");
  for (size_t j=0; j<3; ++j)
  {
    for (size_t k=0; k<means.size(); ++k)
      chout << std::setw(15) << means[k];
    chout << std::endl;
  }
  std::ofstream chout2("Chain-in.dat");
  for (size_t j=0; j<3; ++j)
  {
    for (size_t k=0; k<start_parameter_list.size(); ++k)
      chout2 << std::setw(15) << start_parameter_list[k];
    chout2 << std::endl;
  }
  MPI_Finalize();
  return 0;
  */

  
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  Themis::likelihood_power_tempered L_temp(L_obj);

  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "Likelihood is " << Lstart << '\n';

  
  // Setup MCMC sampler
  int Number_of_procs_per_lklhd = 1;
  //int Chi2_stride = 10;
  int Ckpt_frequency = 10;
  //int out_precision = 8;

  std::vector<double> pbest =  means;
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);

  if (preoptimize_flag)
  {
    lv_hi->set_iteration_limit(20);
      
    opt_obj.set_cpu_distribution(Number_of_procs_per_lklhd);
    opt_obj.set_kickout_parameters(opt_ko_llrf,opt_ko_itermax,opt_ko_rounds);
    means = opt_obj.run_optimizer(L_obj, 2*V_data_hi.size(), means, "PreOptimizeSummary.dat",opt_instances);

    lv_hi->set_iteration_limit(50);
    
    pbest = means;
    
    // Reset priors
    int k=0;
    //  Image
    for (size_t j=0; j<image_pulse.size(); ++j)
    {
      means[k] = pbest[k];
      ranges[k] = 1e-3;
      k++;
    }
    ranges[k-3] = 1e-3*uas2rad; // Fov_x
    ranges[k-2] = 1e-3*uas2rad; // Fov_y
    
    if (add_background_gaussian || add_ring)
    {
      // x offset of image (center raster!)
      means[k]=pbest[k];
      ranges[k] = 1e-7*uas2rad;
      k++;
      
      // y offset of image (center raster)
      means[k]=pbest[k];
      ranges[k] = 1e-7*uas2rad;
      k++;
    }
      
    if (add_background_gaussian)
    {
      // I0
      means[k]=pbest[k];
      ranges[k] = 1e-3;
      k++;
      
      // Size
      means[k]=pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;
    
      // Asymmetry
      means[k]=pbest[k];
      ranges[k] = 1e-4;
      k++;

      // Position angle
      means[k]=pbest[k];
      ranges[k] = 1e-3;
      k++;
    
      // x offset
      means[k]=pbest[k];
      ranges[k] = 1e-3*uas2rad;
      k++;

      // y offset
      means[k]=pbest[k];
      ranges[k] = 1e-3*uas2rad;
      k++;
    }
    
    if (add_ring)
    {
      //   0 Itot
      means[k]=pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;
      
      //   1 Outer size R
      means[k]=pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;

      //   2 psi
      means[k]=pbest[k];
      ranges[k] = 1e-5*pbest[k];
      k++;

      //   3 1-tau
      means[k]=pbest[k];
      ranges[k] = 1e-5*pbest[k];
      k++;

      //   4 f
      means[k]=pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;

      //   5 g
      means[k]=pbest[k];
      ranges[k] = 1e-5*pbest[k];
      k++;

      //   6 a
      means[k]=pbest[k];
      ranges[k] = 1e-5*pbest[k];
      k++;

      //   7 Ig
      means[k]=pbest[k];
      ranges[k] = 1e-5*pbest[k];
      k++;

      //   8 Position angle
      means[k]=pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;
    
      // x offset
      means[k]=pbest[k];
      ranges[k] = 1e-3*uas2rad;
      k++;

      // y offset
      means[k]=pbest[k];
      ranges[k] = 1e-3*uas2rad;
      k++;
    }
  }

  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_dense_e_nuts_MCMC> DEO(seed, L_temp, var_names, means.size());

  // If continuing
  if (restart_flag)
    DEO.read_checkpoint("MCMC.ckpt");

  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  //If you want to access the exploration sampler to change some setting you can!
  Themis::sampler_stan_adapt_dense_e_nuts_MCMC* hmc = DEO.get_sampler();

  hmc->set_max_depth(6);
  
  /*
  //Lets say you want to change the inverse matrix for the sampler. You can!
  std::vector<double> inverse_metric;
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.5);
  inverse_metric.push_back(1.9);
  inverse_metric.push_back(1.1);
  hmc->set_initial_inverse_metric(inverse_metric);
  DEO.set_sampler_summary("stan_summary_deov5.txt"); 
  */
  
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(means);

  //We can also change the annealing schedule.
  double initial_spacing = initial_ladder_spacing; //initial geometric spacre_stride;
  DEO.set_annealing_schedule(initial_spacing);  
  DEO.set_deo_round_params(Number_of_steps,Temperature_stride);
  
  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  int num_thin = thin_factor;
  int num_swamps_adapt = 2000;
  bool save_warmup = true; //false;

  // Generate some output information
  std::stringstream chain_file_name, anneal_file_name, lklhd_file_name, stansum_file_name, chi2_file_name;
  chain_file_name << "chain.dat";
  anneal_file_name << "annealing.dat";
  lklhd_file_name << "state.dat";
  stansum_file_name << "stan_summary_deov5.dat";
  
  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream(chain_file_name.str(),lklhd_file_name.str(),stansum_file_name.str());
  
  //Sets the output for the annealing summary information
  DEO.set_annealing_output(anneal_file_name.str());

  // Chose adaption details
  hmc->set_adaptation_parameters( num_swamps_adapt, save_warmup);
  
  // Start looping over repetitions
  // Extra loop is to make final files
  pbest = means;
  for (size_t rep=0; rep<=Number_of_reps; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started round " << rep << std::endl;

    // Generate fit summaries file
    std::stringstream sumoutname;
    sumoutname << "fit_summaries_" << std::setfill('0') << std::setw(3) << rep << ".txt";
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
      sumout << std::setw(15) << "V rc2"
	     << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileName"
	     << std::endl;
    }

    std::stringstream v_res_name_hi, gc_name_hi, cgc_name_hi;
    v_res_name_hi << "V_residuals_" << std::setfill('0') << std::setw(3) << rep << ".d";
    gc_name_hi << "gain_corrections_" << std::setfill('0') << std::setw(3) << rep << ".d";
    cgc_name_hi << "complex_gains_" << std::setfill('0') << std::setw(3) << rep << ".d";
    
    int Ndata = 2*V_data_hi.size();
    int Nparam = image_pulse.size();
    if (add_ring) // Forced to be thin, no Gaussian
      Nparam -= 4;

    int Ngains = 0;
    if (world_rank==0)
      for (size_t j=0; j<means.size(); ++j)
	std::cout << pbest[j] << std::endl;


    double Lval = L_obj(pbest);
    if (Reconstruct_gains)
    {
      Ngains += lv_hi->number_of_independent_gains();
      lv_hi->output_gain_corrections(gc_name_hi.str());
      lv_hi->output_gains(cgc_name_hi.str());
    }
    
    int NDoF = Ndata - Nparam - Ngains;
    double V_rchi2=0.0, rchi2=0.0;
    L[0]->output_model_data_comparison(v_res_name_hi.str());
    V_rchi2 = (L[0]->chi_squared(pbest));
    if (world_rank==0)
      std::cout << "V rchi2: " << V_rchi2  << " / " << 2*V_data_hi.size() << " - " << Nparam << " - " << Ngains << std::endl;
    V_rchi2 = V_rchi2 / NDoF;      
    rchi2 = L_obj.chi_squared(pbest);
    if (world_rank==0)
      std::cout << "Tot rchi2: " << rchi2  << " / " << NDoF << std::endl;
    rchi2 = rchi2 / NDoF;

    // Generate summary file      
    if (world_rank==0)
    {
      sumout << std::setw(10) << 0;
      for (size_t k=0; k<image.size(); ++k)
	sumout << std::setw(15) << pbest[k];
      sumout << std::setw(15) << V_rchi2 
	     << std::setw(15) << rchi2
	     << std::setw(15) << Lval
	     << "   " << v_file
	     << std::endl;
    }

    // Run the sampler with the given settings
    if (rep<Number_of_reps) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	std::cerr << "Starting MCMC on round " << rep << std::endl;
      DEO.run_sampler( 1, num_thin, refresh_rate, verbosity);
      clock_t end = clock();
      if (world_rank == 0)
	std::cerr << "Done MCMC on round " << rep << std::endl
		  << "it took " << (end-start)/CLOCKS_PER_SEC/3600.0 << " hours" << std::endl;
    }
    
    // Reset pbest
    pbest = hmc->find_best_fit();
  }
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
