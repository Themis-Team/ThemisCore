/*!
  \file analyses/Imaging/imrecad_crosshands_deostan.cpp
  \author Avery Broderick
  \date May 2020
  \brief Generic driver for polarized image reconstruction with Themis with full crosshand visibilities that uses DEO and the Stan NUTS drivers.

  \details TBD
*/

#include "data_crosshand_visibilities.h"
#include "model_polarized_image.h"
#include "model_polarized_image_sum.h"
#include "model_polarized_image_adaptive_splined_raster.h"
#include "model_polarized_image_constant_polarization.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_xsringauss.h"
#include "likelihood.h"
#include "likelihood_crosshand_visibilities.h"
#include "likelihood_optimal_complex_gain_constrained_crosshand_visibilities.h"
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_stan_adapt_dense_e_nuts_MCMC.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"
#include "optimizer_kickout_powell.h"
#include "utils.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"
#include "stop_watch.h"

#include <mpi.h>
#include <vector>
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
  std::string x_file="";
  int Number_of_steps = 10; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;

  bool model_Dterms = false;

  bool Reconstruct_gains = false;
  std::vector<std::string> gain_file_list;

  size_t Number_of_pixels_x = 4;
  size_t Number_of_pixels_y = 4;
  double Field_of_view_x = 0;
  double Field_of_view_y = 0;
  double Position_angle = -999;

  double initial_ladder_spacing = 1.15;
  int thin_factor = 1;
  int Temperature_stride = 50;
  std::string annealing_ladder_file = "";
  int refresh_rate = 1;
  int tree_depth = 6;
  int number_of_adaption_steps = 2000;
  int Ckpt_frequency = 10; // per swaps
  int verbosity = 0;
  
  size_t Number_of_reps = 7;
  
  bool add_background_gaussian=false;
  bool add_ring=false;
  bool add_roving_gaussian=false;

  // Start features
  bool add_background_gaussian_start=false;
  bool add_ring_start=false;
  bool add_roving_gaussian_start=false;
  size_t start_Number_of_pixels_x=0;
  size_t start_Number_of_pixels_y=0;
  bool scatter=false;

  bool restart_flag = false;
  
  bool preoptimize_flag = false;
  double opt_ko_llrf = 10.0;
  size_t opt_ko_itermax = 5;
  size_t opt_ko_rounds = 5;
  size_t opt_instances = 0;


  std::string sampler_type = 'nuts_dense'

    
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
    if (opt=="--crosshands" || opt=="-x")
    {
      if (k<argc)
	x_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --crosshands, -x.\n";
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
    else if (opt=="-d" || opt=="--model-Dterms")
    {
      model_Dterms=true;
    }    
    else if (opt=="-g" || opt=="--reconstruct-gains")
    {
      Reconstruct_gains=true;
    }    
    else if (opt=="-gf" || opt=="--gain-file")
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
    else if (opt=="--continue" || opt=="--restart")
    {
      restart_flag = true;
    }
    else if (opt=="--pre-optimize" || opt=="-po")
    {
      preoptimize_flag = true;
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
    else if ( opt == "--sampler" )
    {
      if ( k <argc )
        number_of_adaption_steps = std::string(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An string argument must be provided after --sampler.\n";
        std::exit(1);
      }
    }
    else if ( opt == "--adaption-steps" )
    {
      if ( k <argc )
        number_of_adaption_steps = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --adaption-steps.\n";
        std::exit(1);
      }
    }
    else if ( opt == "--tree-depth" || opt == "-td" )
    {
      if ( k <argc )
        tree_depth = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --tree-depth, -td.\n";
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
    else if ( opt == "--checkpoint-stride" )
    {
      if ( k <argc )
	Ckpt_frequency = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --checkpoint-stride.\n";
        std::exit(1);
      }
    }
    else if (opt=="--verbosity")
    {
      if (k<argc)
	verbosity=atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --verbosity\n";
	std::exit(1);
      }	
    }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for performing polarized image reconstruction with the adaptive splined-raster\n"
		  << "\tmodel using the crosshand visibilities.\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 200 imrecad_crosshands_deostan -x x_file_list [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-x, --crosshands <string>\n"
		  << "\t\tSets the name of the file containing complex visibility data.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
		  << "\t-p, --parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt.  Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.).\n"
		  << "\t-Npx-start <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction in the parameter file.\n"
		  << "\t-Npxy-start <int> <int>\n"
		  << "\t\tSets the number of pixels along the x-axis and y-axis in the image reconstruction in the parameter file.\n"
	  //		  << "\t--background-gaussian-start, -A-start\n"
	  //		  << "\t\tSpecifies that the parameter file has an asymmetric background gaussian.\n"
	  //		  << "\t--ring-start, -X-start\n"
	  //		  << "\t\tSpecifies that the parameter file has a ring.\n"
	  //		  << "\t--roving-gaussian-start, -rg-start\n"
	  //		  << "\t\tSpecifies that the parameter file has an asymmetric roving gaussian.\n"
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
	  //		  << "\t--scatter\n"
	  //                  << "\t\tAdds diffractive scattering to the model to fit intrinsic source.\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"
		  << "\t-d, --model-Dterms\n"
		  << "\t\tReconstructs the station D terms.  Default off.\n"
		  << "\t-g, --reconstruct-gains\n"
		  << "\t\tReconstructs unknown station gains.  Default off.\n"
		  << "\t-g, --gain-file <filename>\n"
		  << "\t\tSets the gains to those in the specified file name.  These should be listed in the same order as data files in the file list past via -x.\n"
	  //		  << "\t-A, --background-gaussian\n"
	  //		  << "\t\tAdds a large-scale background gaussian, constrained to have an isotropized standard\n"
	  //		  << "\t\tdeviation between 100 uas and 10 as.\n"
	  //		  << "\t--ring, -X\n"
	  //		  << "\t\tAdds a narrow ring feature (based on xsringauss).\n"
	  //		  << "\t--roving-gaussian, -rg\n"
	  //		  << "\t\tAdds an asymmetric roving gaussian.\n"
	  //		  << "\t--tempering-levels <int>\n"
	  //		  << "\t\tSets the number of tempering levels.  Defaults to the number of processes.\n"
		  << "\t--number-of-rounds, -nor, -Nr <int>\n"
		  << "\t\tSets the number of rounds for each DEO repitition.  Defaults to 7.\n"
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
		  << "\t--sampler <string>\n"
		  << "\t\tSelects the exploration sampler.  Options include: 'nuts_dense', 'nuts_diag', and 'slice'.  Default: 'nuts_dense'.\n"
		  << "\t--adaption-steps <int>\n"
		  << "\t\tSets the number of adaption steps to perform.  Defaults to 2000.\n"
		  << "\t--tree-depth, -td <int>\n"
		  << "\t\tSets the maximum tree depth.  Defaults to 6.\n"
		  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
                  << "\t\tDefault is 1 step.\n"
		  << "\t--checkpoint-stride <int>\n"
		  << "\t\tSets the number of tempering level *swaps* after which to checkpoint the state of the run.  Default: 10.\n"
		  << "\t--verbosity <int>\n"
		  << "\t\tSets the verbosity level.  Default 0.\n"
		  << "\t-po, --pre-optimize\n"
		  << "\t\tMakes an attempt to identify an optimal fit prior to running the sampler.  Optimal models may be found in PreOptimizationSummary.dat\n"
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


  if (x_file=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No data file list was provided. The -x <string>,\n"
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
      Field_of_view_x = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y];
    if (Field_of_view_y==0)
      Field_of_view_y = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y+1];

    if (Position_angle==-999)
      Position_angle = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y+2];
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
    std::cout << "Nr: " << Number_of_reps << "\n";
    std::cout << "Ns: " << Number_of_steps << "\n";
    std::cout << "X file: " << x_file << "\n";
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
  Themis::model_polarized_image_adaptive_splined_raster image_pulse(Number_of_pixels_x,Number_of_pixels_y);
  image_pulse.use_analytical_visibilities();
  Themis::model_polarized_image* image_ptr = &image_pulse;
  
  // Generate sum to get a shift and add possibly components
  Themis::model_polarized_image_sum image;
  image.add_model_polarized_image(image_pulse);

  std::cerr << "A: " <<  image.size() << '\n';
  
  // Background Gaussian
  Themis::model_image_asymmetric_gaussian model_img_a;
  model_img_a.use_analytical_visibilities();
  Themis::model_polarized_image_constant_polarization model_a(model_img_a);
  if (add_background_gaussian)
    image.add_model_polarized_image(model_a);

  std::cerr << "B: " <<  image.size() << '\n';

  // Ring
  Themis::model_image_xsringauss model_img_X;
  model_img_X.use_analytical_visibilities();
  Themis::model_polarized_image_constant_polarization model_X(model_img_X);
  if (add_ring)
    image.add_model_polarized_image(model_X);

  std::cerr << "C: " <<  image.size() << '\n';

  /* TBD
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
  */

  // Scattered image with Sgr A* screen
  Themis::model_crosshand_visibilities* model_ptr = &image;
  /*
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image_sum);
  if (scatter)
    model_ptr = &scattimage;
  else
    model_ptr = &image_sum;
  */
  
  
  // Read in data files
  std::vector<Themis::data_crosshand_visibilities*> X_data;
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_optimal_complex_gain_constrained_crosshand_visibilities*> lxg;
  std::vector<Themis::likelihood_crosshand_visibilities*> lx;

  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

  // Get the list of data files
  std::vector<std::string> x_file_name_list;
  std::string x_file_name;
  int strlng;
  if (world_rank==0)
  {
    std::ifstream xin(x_file);
    for (xin>>x_file_name; !xin.eof(); xin>>x_file_name)
      x_file_name_list.push_back(x_file_name);
    strlng = x_file_name.length()+1;
  }
  int ibuff = x_file_name_list.size();
  MPI_Bcast(&ibuff,1,MPI_INT,0,MPI_COMM_WORLD);
  for (int i=0; i<ibuff; ++i)
  {
    if (world_rank==0)
      strlng = x_file_name.length()+1;
    MPI_Bcast(&strlng,1,MPI_INT,0,MPI_COMM_WORLD);

    char* cbuff = new char[strlng];
    if (world_rank==0)
      strcpy(cbuff,x_file_name.c_str());
    MPI_Bcast(&cbuff[0],strlng,MPI_CHAR,0,MPI_COMM_WORLD);
    if (world_rank>0)
      x_file_name_list.push_back(std::string(cbuff));
    delete[] cbuff;
  }

  //double variance_weighted_time_average=0.0, vwta_var_norm=0.0, minimum_time=0, maximum_time=1;
  for (size_t j=0; j<x_file_name_list.size(); ++j)
  {
    X_data.push_back( new Themis::data_crosshand_visibilities(x_file_name_list[j],"HH") );

    /*
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
    */

    if (Reconstruct_gains)
    {
      lxg.push_back( new Themis::likelihood_optimal_complex_gain_constrained_crosshand_visibilities(*X_data[j],image,station_codes,station_gain_priors) );
      L.push_back( lxg[j] );

      // Set gains if gain files are provided
      if (j<gain_file_list.size() && gain_file_list[j]!="")
      {
	lxg[j]->read_gain_file(gain_file_list[j]);
	lxg[j]->fix_gains();
      }
    }
    else
    {
      lx.push_back( new Themis::likelihood_crosshand_visibilities(*X_data[j],image) );
      L.push_back( lx[j] );
    }
  }
  /*
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
  */

  // Model D terms
  if (model_Dterms)
    image.model_Dterms(station_codes);

  image.write_model_tag_file();
  
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(20,60)); // log(Itotal)
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(-10,-0.1)); // log(m) max at 90%
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // EVPA
  for (size_t j=0; j<Number_of_pixels_x*Number_of_pixels_y; ++j)
    P.push_back(new Themis::prior_linear(-1.0,1.0)); // muV
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_x)); // fovx
  P.push_back(new Themis::prior_linear(0,5*Field_of_view_y)); // fovy
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // PA
  
  // Generate a set of initial conditions
  std::vector<double> means(4*Number_of_pixels_x*Number_of_pixels_y);
  std::vector<std::string> var_names;
  double x,y;
  double sig = 0.5*Field_of_view_x;
  double norm = std::log(2.5/(2.0*M_PI*sig*sig));
  for (size_t j=0,k=0; j<Number_of_pixels_x; ++j)
    for (size_t i=0; i<Number_of_pixels_y; ++i)
    {
      x = Field_of_view_x*double(i)/double(Number_of_pixels_x-1) - 0.5*Field_of_view_x;
      y = Field_of_view_y*double(j)/double(Number_of_pixels_y-1) - 0.5*Field_of_view_y;
      means[k++] = std::min( std::max(norm  - (x*x+y*y)/(2.0*sig*sig),21.0) , 59.0 );
    }
  for (size_t j=Number_of_pixels_x*Number_of_pixels_y; j<2*Number_of_pixels_x*Number_of_pixels_y; ++j)
    means[j] = std::log(1e-2); // log(m)
  for (size_t j=2*Number_of_pixels_x*Number_of_pixels_y; j<3*Number_of_pixels_x*Number_of_pixels_y; ++j)
    means[j] = 0.0; // EVPA
  for (size_t j=3*Number_of_pixels_x*Number_of_pixels_y; j<4*Number_of_pixels_x*Number_of_pixels_y; ++j)
    means[j] = 0.0; // muV
  means.push_back(Field_of_view_x);
  means.push_back(Field_of_view_y);
  if (Position_angle==-999)
    means.push_back(0.0);
  else
    means.push_back(Position_angle);

  
  double uas2rad = 1e-6/3600. * M_PI/180.;

  // x offset of image (center raster!)
  P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
  means.push_back(0.0);
  
  // y offset of image (center raster)
  P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
  means.push_back(0.0);

  
  // Add priors for background Gaussian if desired
  if (add_background_gaussian)
  {  
    // Itot
    P.push_back(new Themis::prior_linear(0.0,10));
    means.push_back(1.0);
    // Size
    P.push_back(new Themis::prior_linear(1e2*uas2rad,1e7*uas2rad));
    means.push_back(1e3*uas2rad);
    // Asymmetry
    P.push_back(new Themis::prior_linear(0.0,0.99));
    means.push_back(0.1);
    // Position angle
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    // p
    P.push_back(new Themis::prior_linear(0,0.99)); // max at 99%
    means.push_back(0.01);
    // EVPA
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0.0);
    // muV
    P.push_back(new Themis::prior_linear(-0.99,0.99));
    means.push_back(0.0);
    // x offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-2000*uas2rad,2000*uas2rad));
    means.push_back(0.0);
    // y offset
    //P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
    P.push_back(new Themis::prior_linear(-2000*uas2rad,2000*uas2rad));
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
    means.push_back(20*uas2rad);
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
    //   9 p
    P.push_back(new Themis::prior_linear(0,0.99)); // max at 99%
    means.push_back(0.01);
    //  10 EVPA
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0.0);
    //  11 muV
    P.push_back(new Themis::prior_linear(-0.99,0.99));
    means.push_back(0.0);
    //  12 x offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    //  13 y offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
  }

  /* TBD
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
  */

  if (model_Dterms)
  {
    for (size_t k=0; k<station_codes.size(); ++k)
      for (size_t j=0; j<4; ++j)
      {
	P.push_back(new Themis::prior_linear(-1.0,1.0));
	means.push_back(0.0);
      }
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
	      << "  image_sum size: " << image.size()
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

    double start_Field_of_view_x = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y+0];
    double start_Field_of_view_y = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y+1];
    double start_Position_angle = start_parameter_list[4*start_Number_of_pixels_x*start_Number_of_pixels_y+2];

    if (world_rank==0)
    {
      std::cerr << "Start fov/PA: " << std::setw(15) << start_Field_of_view_x << std::setw(15) << start_Field_of_view_y << std::setw(15) << start_Position_angle << '\n';
      std::cerr << "New fov/PA:   " << std::setw(15) << Field_of_view_x << std::setw(15) << Field_of_view_y << std::setw(15) << Position_angle << '\n';
    }
    
    
    std::valarray<double> xs(start_Number_of_pixels_x),ys(start_Number_of_pixels_y),Is(start_Number_of_pixels_x*start_Number_of_pixels_y),ms(start_Number_of_pixels_x*start_Number_of_pixels_y),EVPAs(start_Number_of_pixels_x*start_Number_of_pixels_y),muVs(start_Number_of_pixels_x*start_Number_of_pixels_y);

    for (size_t i=0; i<start_Number_of_pixels_x; ++i)
      xs[i] = start_Field_of_view_x*double(i)/double(start_Number_of_pixels_x-1)-0.5*start_Field_of_view_x;
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      ys[j] = start_Field_of_view_y*double(j)/double(start_Number_of_pixels_y-1)-0.5*start_Field_of_view_y;
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	Is[j+start_Number_of_pixels_y*i] = std::exp(start_parameter_list[kspl++]);
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	ms[j+start_Number_of_pixels_y*i] = std::exp(start_parameter_list[kspl++]);
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	EVPAs[j+start_Number_of_pixels_y*i] = start_parameter_list[kspl++];
    for (size_t j=0; j<start_Number_of_pixels_y; ++j)
      for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	muVs[j+start_Number_of_pixels_y*i] = start_parameter_list[kspl++];


    // DEBUG
    if (world_rank==0)
    {
      for (size_t j=0; j<start_Number_of_pixels_y; ++j)
	for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	  std::cerr << std::setw(5) << i << std::setw(5) << j << std::setw(15) << Is[j+start_Number_of_pixels_y*i] << std::setw(15) << ms[j+start_Number_of_pixels_y*i] << std::setw(15) << EVPAs[j+start_Number_of_pixels_y*i] << std::setw(15) << muVs[j+start_Number_of_pixels_y*i] << '\n';
      std::cerr << '\n';
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::valarray<double> Qs(start_Number_of_pixels_x*start_Number_of_pixels_y),Us(start_Number_of_pixels_x*start_Number_of_pixels_y),Vs(start_Number_of_pixels_x*start_Number_of_pixels_y);
    for (size_t j=0; j<start_Number_of_pixels_x*start_Number_of_pixels_y; ++j)
    {
      Qs[j] = Is[j] * ms[j] * std::cos(2.0*EVPAs[j]) * std::sqrt(1.0 - muVs[j]*muVs[j]);
      Us[j] = Is[j] * ms[j] * std::sin(2.0*EVPAs[j]) * std::sqrt(1.0 - muVs[j]*muVs[j]);
      Vs[j] = Is[j] * ms[j] * muVs[j];
    }
    Themis::Interpolator2D start_image_interp_I(xs,ys,Is);
    Themis::Interpolator2D start_image_interp_Q(xs,ys,Qs);
    Themis::Interpolator2D start_image_interp_U(xs,ys,Us);
    Themis::Interpolator2D start_image_interp_V(xs,ys,Vs);
    start_image_interp_I.use_forward_difference();
    start_image_interp_Q.use_forward_difference();
    start_image_interp_U.use_forward_difference();
    start_image_interp_V.use_forward_difference();

    
    // DEBUG
    if (world_rank==0)
    {
      for (size_t j=0; j<start_Number_of_pixels_y; ++j)
	for (size_t i=0; i<start_Number_of_pixels_x; ++i)
	  std::cerr << std::setw(5) << i << std::setw(5) << j << std::setw(15) << Is[j+start_Number_of_pixels_y*i] << std::setw(15) << Qs[j+start_Number_of_pixels_y*i] << std::setw(15) << Us[j+start_Number_of_pixels_y*i] << std::setw(15) << Vs[j+start_Number_of_pixels_y*i] << '\n';
      std::cerr << '\n';
    }
    MPI_Barrier(MPI_COMM_WORLD);
        
    
    // If not specified, set the fov to the start values
    if (Field_of_view_x==0)
      Field_of_view_x = start_Field_of_view_x;
    if (Field_of_view_y==0)
      Field_of_view_y = start_Field_of_view_y;
			   
    size_t km = 0;
    for (size_t j=0; j<Number_of_pixels_y; ++j)
      for (size_t i=0; i<Number_of_pixels_x; ++i)
      {
	// Interpolate up
	double x=Field_of_view_x*double(i)/double(Number_of_pixels_x-1)-0.5*Field_of_view_x;
	double y=Field_of_view_y*double(j)/double(Number_of_pixels_y-1)-0.5*Field_of_view_y;
	double Ival,Qval,Uval,Vval;
	start_image_interp_I.bicubic(x,y,Ival);
	start_image_interp_Q.bicubic(x,y,Qval);
	start_image_interp_U.bicubic(x,y,Uval);
	start_image_interp_V.bicubic(x,y,Vval);
	if (Ival<0)
	{
	  start_image_interp_I.linear(x,y,Ival);
	  start_image_interp_Q.linear(x,y,Qval);
	  start_image_interp_U.linear(x,y,Uval);
	  start_image_interp_V.linear(x,y,Vval);
	}

	// DEBUG
	if (world_rank==0)
	  std::cerr << std::setw(5) << i << std::setw(5) << j << std::setw(15) << Ival << std::setw(15) << Qval << std::setw(15) << Uval << std::setw(15) << Vval << '\n';
	
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

	double Inew,Qnew,Unew,Vnew;
	Inew = Ival*mask;
	Qnew = Qval*mask;
	Unew = Uval*mask;
	Vnew = Vval*mask;

	double mnew,EVPAnew,muVnew;
	mnew = std::sqrt( Qnew*Qnew + Unew*Unew + Vnew*Vnew )/Inew;
	if (mnew>1.0)
	  mnew = 0.9;
	else if (mnew<0.0)
	  mnew = 1e-4;

	EVPAnew = 0.5*std::atan2(Unew,Qnew);
	muVnew = Vnew/std::sqrt(Qnew*Qnew+Unew*Unew+Vnew*Vnew);
	if (muVnew>1.0)
	  muVnew = 0.9999;
	else if (muVnew<-1.0)
	  muVnew = -0.9999;

	// DEBUG
	if (world_rank==0)
	  std::cerr << std::setw(5) << i << std::setw(5) << j << std::setw(15) << Inew << std::setw(15) << mnew << std::setw(15) << EVPAnew << std::setw(15) << muVnew << '\n';
	
	// Set values
	means[km] = std::min( std::max( std::log(Inew), 21.0 ), 59.0 );
	means[km+Number_of_pixels_x*Number_of_pixels_y] = std::min( std::max( std::log(mnew), -10.0), -0.1);
	means[km+2*Number_of_pixels_x*Number_of_pixels_y] = std::min( std::max( EVPAnew, -M_PI), M_PI);
	means[km+3*Number_of_pixels_x*Number_of_pixels_y] = std::min( std::max( muVnew, -1.0), 1.0);
	km+=1;
	k+=4;
      }

    // DEBUG
    if (world_rank==0)
      std::cerr << '\n';
    MPI_Barrier(MPI_COMM_WORLD);
    
    means[k++] = Field_of_view_x;
    means[k++] = Field_of_view_y;
    if (Position_angle==-999)
      means[k++] = start_Position_angle;
    else
      means[k++] = Position_angle;
    kspl += 3;

    // x0
    means[k] = start_parameter_list[kspl]; 
    k++; kspl++;
    
    // y0
    means[k] = start_parameter_list[kspl]; 
    k++; kspl++;

    
    if (world_rank==0)
    {
      std::cerr << "FoVs" << std::setw(15) << Field_of_view_x << std::setw(15) << Field_of_view_y << std::endl;
      std::cerr << "Added raster image to start values k=" << k  << "  kspl=" << kspl << std::endl;
    }
    
    if ( (add_background_gaussian && add_background_gaussian_start) || (add_ring && add_ring_start) )
    {
      if (world_rank==0)
	std::cerr << "Adding additional components to start values k=" << k  << "  kspl=" << kspl << std::endl;
    }

    if (add_background_gaussian && add_background_gaussian_start)
    {      
      if (world_rank==0)
	std::cerr << "Adding gaussian to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_a.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }
    else if (add_background_gaussian) // To push D-term parameter locations
      k += model_a.size()+2;
    else if (add_background_gaussian_start) // To push D-term start-parameter locations
      kspl += model_a.size()+2;
    
    if (add_ring && add_ring_start)
    {
      if (world_rank==0)
	std::cerr << "Adding ring to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_X.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }
    else if (add_ring) // To push D-term parameter locations
      k += model_X.size()+2;
    else if (add_ring_start) // To push D-term start-parameter locations
      kspl += model_X.size()+2;

    /* TBD
    if (add_roving_gaussian && add_roving_gaussian_start)
    {
      if (world_rank==0)
	std::cerr << "Adding roving Gaussian to start values k=" << k  << "  kspl=" << kspl << std::endl;
      for (size_t j=0; j<model_rg.size()+2; ++j)
	means[k++] = start_parameter_list[kspl++];
    }
    */

    // D term initialization, but we will need to do this after the rest of the components
    if (model_Dterms)
    {
      if (int(start_parameter_list.size())-kspl < 4*station_codes.size())
      {
	std::cerr << "ERROR: Insufficient number of start parameters provided to initialized D terms.\n";
	std::exit(1);
      }
      
      for (size_t j=0; j<4*station_codes.size(); ++j)
	means[k++] = start_parameter_list[kspl++];
      
      if (world_rank==0)
	std::cerr << "Added D terms to start values k=" << k  << "  kspl=" << kspl << std::endl;
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
	      << "  image_sum size: " << image.size()
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
  Themis::likelihood_power_tempered L_temp(L_obj);
  
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Ndata = 0;
  for (size_t j=0; j<X_data.size(); ++j)
    Ndata += 8*X_data[j]->size();
  int Nparam = image.size();
  if (add_ring) // Forced to be thin, no Gaussian
    Nparam -= 4;
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);
  if (Reconstruct_gains)
    for (size_t j=0; j<lxg.size(); ++j)
    {
      Ngains_list[j] = lxg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  int NDoF = Ndata - Nparam - Ngains;
  
  // Setup MCMC sampler
  int Number_of_procs_per_lklhd = 1;

  // Get the best point
  std::vector<double> pbest =  means;
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);
  if (preoptimize_flag)
  {
    if (Reconstruct_gains)
      for (size_t j=0; j<lxg.size(); ++j)
	lxg[j]->set_iteration_limit(20);
      
    opt_obj.set_cpu_distribution(Number_of_procs_per_lklhd);
    opt_obj.set_kickout_parameters(opt_ko_llrf,opt_ko_itermax,opt_ko_rounds);
    means = opt_obj.run_optimizer(L_obj, Ndata, means, "PreOptimizeSummary.dat",opt_instances);

    if (Reconstruct_gains)
      for (size_t j=0; j<lxg.size(); ++j)
	lxg[j]->set_iteration_limit(50);
    
    pbest = means;
  }

  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_dense_e_nuts_MCMC> DEO(seed, L_temp, var_names, means.size());
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO_nuts_diag(seed, L_temp, var_names, means.size());
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO_slice(seed, L_temp, var_names, means.size());
  
  
  // slice Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO_nuts_diag(seed, L_temp, var_names, means.size());
  
  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "stan_summary_deov5.dat");
    
  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat");

  // If passed an existing ladderr
  if (annealing_ladder_file!="")
    DEO.read_initial_ladder(annealing_ladder_file);
  
  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  //If you want to access the exploration sampler to change some setting you can!
  DEO.get_sampler()->set_max_depth(tree_depth);

  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  bool save_warmup = true;
  DEO.get_sampler()->set_adaptation_parameters( number_of_adaption_steps, save_warmup);
    
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(means);

  //We can also change the annealing schedule.
  double initial_spacing = initial_ladder_spacing; //initial geometric spacre_stride;
  DEO.set_annealing_schedule(initial_spacing);  
  DEO.set_deo_round_params(Number_of_steps,Temperature_stride);
  
  // If continuing
  int round_start = 0;
  if (restart_flag)
  {
    DEO.read_checkpoint("MCMC.ckpt");
    round_start = DEO.get_round();
  }
    
  // Start looping over repetitions
  // Extra loop is to make final files
  pbest = means;
  for (size_t rep=round_start; rep<=Number_of_reps; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started round " << rep << std::endl;

    // Generate fit summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3) << rep << "_fit_summaries" << ".txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index";

      for (size_t j=0; j<Number_of_pixels_y; ++j)
	for (size_t i=0; i<Number_of_pixels_x; ++i)
	  sumout << std::setw(15) << "I("+std::to_string(i)+","+std::to_string(j)+")";

      for (size_t j=0; j<Number_of_pixels_y; ++j)
	for (size_t i=0; i<Number_of_pixels_x; ++i)
	  sumout << std::setw(15) << "p("+std::to_string(i)+","+std::to_string(j)+")";

      for (size_t j=0; j<Number_of_pixels_y; ++j)
	for (size_t i=0; i<Number_of_pixels_x; ++i)
	  sumout << std::setw(15) << "EVPA("+std::to_string(i)+","+std::to_string(j)+")";

      for (size_t j=0; j<Number_of_pixels_y; ++j)
	for (size_t i=0; i<Number_of_pixels_x; ++i)
	  sumout << std::setw(15) << "muV("+std::to_string(i)+","+std::to_string(j)+")";

      sumout << std::setw(15) << "fovx"
	     << std::setw(15) << "fovy"
	     << std::setw(15) << "PA"
	     << std::setw(15) << "img-x"
	     << std::setw(15) << "img-y";
      
      if (add_background_gaussian)
      {
	sumout << std::setw(15) << "A-I"
	       << std::setw(15) << "A-sig"
	       << std::setw(15) << "A-A"
	       << std::setw(15) << "A-phi"
	       << std::setw(15) << "A-p"
	       << std::setw(15) << "A-EVPA"
	       << std::setw(15) << "A-muV"
	       << std::setw(15) << "A-x"
	       << std::setw(15) << "A-y";
      }
      if (add_ring)
      {
	sumout << std::setw(15) << "X-Isx (Jy)"
	       << std::setw(15) << "X-Rp (uas)"
	       << std::setw(15) << "X-psi"
	       << std::setw(15) << "X-ecc"
	       << std::setw(15) << "X-f"
	       << std::setw(15) << "X-gax"
	       << std::setw(15) << "X-a"
	       << std::setw(15) << "X-ig"
	       << std::setw(15) << "X-PA"
	       << std::setw(15) << "X-p"
	       << std::setw(15) << "X-EVPA"
	       << std::setw(15) << "X-muV"
	       << std::setw(15) << "xsX"
	       << std::setw(15) << "ysX";
      }
      /* TBD
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
      */
      if (model_Dterms)
	for (size_t j=0; j<station_codes.size(); ++j)
	  sumout << std::setw(15) << "D-"+station_codes[j]+"-R.r"
		 << std::setw(15) << "D-"+station_codes[j]+"-R.i"
		 << std::setw(15) << "D-"+station_codes[j]+"-L.r"
		 << std::setw(15) << "D-"+station_codes[j]+"-L.i";
      
      for (size_t j=0; j<L.size(); ++j) 
	sumout << std::setw(15) << "X" + std::to_string(j) + " rc2";
      sumout << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileNames"
	     << std::endl;
    }


    // Output the current location
    if (world_rank==0)
      for (size_t j=0; j<pbest.size(); ++j)
	std::cout << pbest[j] << std::endl;

    double Lval = L_obj(pbest);
    if (Reconstruct_gains)
    {
      for (size_t j=0; j<lxg.size(); ++j)
      {
	std::stringstream gc_name, cgc_name;
	gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" << std::setfill('0') << j << ".d";
	cgc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_complex_gains_" << std::setfill('0') << j << ".d";
	lxg[j]->output_gain_corrections(gc_name.str());
	lxg[j]->output_gains(cgc_name.str());
      }
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
    double X_rchi2=0.0, rchi2=0.0;
    for (size_t j=0; j<L.size(); ++j)
    {
      // output residuals
      std::stringstream x_res_name;
      x_res_name << "round" << std::setfill('0') << std::setw(3) << rep << "_residuals_" << std::setfill('0') << j << ".d";
      L[j]->output_model_data_comparison(x_res_name.str());

      // Get the data-set specific chi-squared
      X_rchi2 = (L[j]->chi_squared(pbest));
      if (world_rank==0)
      {
	std::cout << "X" << j << " rchi2: " << X_rchi2  << " / " << 8*X_data[j]->size() << " - " << Nparam << " - " << Ngains_list[j] << std::endl;
	// Summary file values
	sumout << std::setw(15) << X_rchi2 / ( 8*int(X_data[j]->size()) - Nparam - Ngains_list[j]);
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
	     << std::setw(15) << Lval;
      for (size_t j=0; j<x_file_name_list.size(); ++j)
	sumout << "   " << x_file_name_list[j];
      sumout << std::endl;
    }
    
    // Run the sampler with the given settings
    if (rep<Number_of_reps) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	std::cerr << "Starting MCMC on round " << rep << std::endl;
      DEO.run_sampler( 1, thin_factor, refresh_rate, verbosity);
      clock_t end = clock();
      if (world_rank == 0)
	std::cerr << "Done MCMC on round " << rep << std::endl
		  << "it took " << (end-start)/CLOCKS_PER_SEC/3600.0 << " hours" << std::endl;
    }
    
    // Reset pbest
    pbest = DEO.get_sampler()->find_best_fit();
  }
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
