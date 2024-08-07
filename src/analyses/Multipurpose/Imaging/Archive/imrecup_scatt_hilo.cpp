/*!
  \file analyses/Imaging/imrecup_scatt_hilo.cpp
  \author
  \date May 2019
  \brief Generic driver for image reconstruction with Themis that upsamples image.

  \details TBD
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "data_closure_amplitude.h"
#include "model_image_raster.h"
#include "model_image_upsample.h"
#include "model_image_smooth.h"
#include "model_image_splined_raster.h"
#include "model_image_sum.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_xsringauss.h"
#include "model_image.h"
#include "model_galactic_center_diffractive_scattering_screen.h"
#include "likelihood.h"
#include "likelihood_closure_amplitude.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory> 
#include <string>

#include <iostream>
#include <iomanip>
#include <fstream>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

    // Parse the command line inputs
  std::string vm_file="", cp_file="", ca_file="";
  int Number_of_steps = 1000; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  bool Reconstruct_gains = false;
  size_t Number_of_pixels = 4;
  size_t Number_of_reps = 2;
  double Field_of_view_x = 100.0 *1e-6/3600./180.*M_PI; // in rad
  double Field_of_view_y = 100.0 *1e-6/3600./180.*M_PI; // in rad
  bool smooth_image=false;
  bool smooth_image_start=false;
  double smoothing_fwhm=0.0;

  int Number_of_tempering_levels=40;
  double Tempering_time=1000.0;
  double Tempering_ladder=1.4;
  int Number_of_walkers=0;

  bool add_background_gaussian=false;
  bool add_background_gaussian_start=false;
  bool add_ring=false;
  bool add_ring_start=false;
  size_t start_image_size=0;
  bool upsample_image=false;
  size_t upsample_factor=1;
  bool do_fft_spline=true;
  double start_fov=0.0;
  bool restart_flag = false;
  
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
    if (opt=="--visibility-amplitudes" || opt=="-vm")
    {
      if (k<argc)
	vm_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --visibility-amplitudes, -vm.\n";
	std::exit(1);
      }
    }
    else if (opt=="--closure-phases" || opt=="-cp")
    {
      if (k<argc)
	cp_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --closure-phases, -cp.\n";
	std::exit(1);
      }
    }
    else if (opt=="--closure-amplitudes" || opt=="-ca")
    {
      if (k<argc)
	ca_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --closure-amplitudes, -ca.\n";
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
    else if (opt=="-Npx-start")
    {
      if (k<argc)
      {
	start_image_size = atoi(argv[k++]);
	start_image_size *= start_image_size;
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Npx-start.\n";
	std::exit(1);
      }
    }
    else if (opt=="--fov-start")
    {
      if (k<argc)
	start_fov = atof(argv[k++]) * 1e-6/3600./180.*M_PI;
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A float argument must be provided after --fov-start.\n";
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
	Number_of_pixels = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Npx.\n";
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
      if (k<argc)
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
    else if (opt=="-g" || opt=="--reconstruct-gains")
    {
      Reconstruct_gains=true;
    }    
    else if (opt=="-A" || opt=="--background-gaussian")
    {
      add_background_gaussian=true;
    }    
    else if (opt=="--ring" || opt=="-X")
    {
      add_ring=true;
    }    
    else if (opt=="--smooth" || opt=="-s")
    {
      if (k+1<argc)
      {
	smooth_image=true;
	smoothing_fwhm=atof(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A float argument must be provided after --smooth/-s.\n";
	std::exit(1);
      }
    }
    else if (opt=="--smooth-start" || opt=="-s-start")
    {
      smooth_image_start=true;
    }
    else if (opt=="--background-gaussian-start" || opt=="-A-start")
    {
      add_background_gaussian_start=true;
    }
    else if (opt=="--ring-start" || opt=="-X-start")
    {
      add_ring_start=true;
    }
    else if (opt=="--upsample" || opt=="-u")
    {
      if (k+1<argc)
      {
	upsample_image=true;
	upsample_factor=atoi(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An integer argument must be provided after --upsample/-u.\n";
	std::exit(1);
      }
    }
    else if (opt=="--pulse") 
    {
      do_fft_spline=false;
    }
    else if (opt=="--splined")
    { 
      do_fft_spline=true;
    }
    else if (opt=="--walkers")
    {
      if (k<argc)
	Number_of_walkers = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --walkers.\n";
	std::exit(1);
      }
    }
    else if (opt=="--continue" || opt=="--restart")
    {
      restart_flag = true;
    }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for Challenge J+\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 imrecup_hilo -vm vm_file_list -cp cp_file_list [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-vm, --visibility-amplitudes <string>\n"
		  << "\t\tSets the name of the file containing visibility amplitude data.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-cp, --closure-phases <string>\n"
		  << "\t\tSets the name of the file containing closure phase data.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-ca, --closure-amplitudes <string>\n"
		  << "\t\tSets the name of the file containing closure amplitude data.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
		  << "\t-p, --parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt, with the same number of lines as the vm_file_list and\n"
		  << "\t\tcp_file_list.  Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.). This also shrinks the affected ranges.\n"
		  << "\t-Npx-start <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction in the parameter file.\n"
		  << "\t--fov-start <float>\n"
		  << "\t\tSets the field of view of image reconstruction in the parameter file in uas (defaults to the current fov).\n"
		  << "\t--smooth-start, -s-start\n"
		  << "\t\tSpecifies that the parameter file has a smoothing kernel.\n"
		  << "\t--background-gaussian-start, -A-start\n"
		  << "\t\tSpecifies that the parameter file has an asymmetric background gaussian.\n"
		  << "\t--ring-start, -X-start\n"
		  << "\t\tSpecifies that the parameter file has a ring.\n"
		  << "\t-Ns <int>\n"
		  << "\t\tSets the number of MCMC steps to take for each repetition chain.  Defaults to 1000.\n"
		  << "\t-Npx <int>\n"
		  << "\t\tSets the number of pixels along the x-axis in the image reconstruction.\n"
		  << "\t--fov <float>\n"
		  << "\t\tSets the field of view (in uas).  Default 100.\n"
		  << "\t--fovxy <float>\n"
		  << "\t\tSets the field of view (in uas) for the x and y directions independently.  The meaning\n"
		  << "\t\tof thes directions is set by the data, ostensibly RA and DEC.  Default for both is 100.\n"
		  << "\t-s, --smooth <float>\n"
		  << "\t\tTurns on smoothing by a symmetric Gaussian, sets the FWHM (in uas) of the kernel.  Default is off.\n"
		  << "\t-Nr <int>\n"
		  << "\t\tSets the number of repetitions to perform.  Defaults to 2.\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	 
		  << "\t-g, --reconstruct-gains\n"
		  << "\t\tReconstructs unknown station gains.  Default off.\n"
		  << "\t-A, --background-gaussian\n"
		  << "\t\tAdds a large-scale background gaussian, constrained to have an isotropized standard\n"
		  << "\t\tdeviation between 100 uas and 10 as.\n"
		  << "\t--ring, -X\n"
		  << "\t\tAdds a narrow ring feature (based on xsringauss).\n"
		  << "\t-u, --upsample <int>\n"
		  << "\t\tTurns on and sets the factor by which to upsample the image, sets --pulse.  Default is off.\n"
		  << "\t--tempering-levels <int>\n"
		  << "\t\tSets the number of tempering levels.  Defaults to 8.\n"
		  << "\t--tempering-ladder <float>\n"
		  << "\t\tSets the ladder factor for the tempering levels.  Defaults to 2.\n"
		  << "\t--tempering-time <float>\n"
		  << "\t\tSets the number of MCMC steps over which to reduce the tempering ladder\n"
		  << "\t\toptimization evolution to reduce by half.  Defaults to 1000.\n"
		  << "\t--walkers <int>\n"
		  << "\t\tSets the number of chains per tempering level.  Defaults to 360.\n"
		  << "\t--pulse/--splined\n"
		  << "\t\tUses the pulse raster model (delta functions) or performs a Fourier-domain spline (cubic spline control points; default).\n"
		  << "\t--continue/--restart\n"
		  << "\t\tRestarts a run using the local MCMC.ckpt file.  Note that this restarts at step 0, i.e., does not continue to the next step.\n"
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

  if (vm_file=="" && cp_file=="" && ca_file=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No data file list was provided. The -vm <string>,\n"
		<< "       -cp <string> or -ca <string> options are *required*.  See -h for more\n"
		<< "       details and options.\n";
    std::exit(1);
  }



  
  // Set and fill the start parameters if provided
  // Assumes has the same format as the fit_summaries.txt file (header, index, parameters, then other items)
  std::vector<double> start_parameter_list;
  if (param_file!="")
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
    for (; !pfin.eof();)
    {
      for (int k=0; k<Number_start_params; k++)
      {
	pfin >> dtmp;
	start_parameter_list.push_back(dtmp);
      }
      // Kill remainder of line
      //pfin.ignore(4096,'\n');
      getline(pfin,stmp);
      // Get next index
      pfin >> dtmp;      
    }
  }  

  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "Npx: " << Number_of_pixels << "\n";
    std::cout << "Nr: " << Number_of_reps << "\n";
    std::cout << "Ns: " << Number_of_steps << "\n";
    std::cout << "VM file: " << vm_file << "\n";
    std::cout << "CP file: " << cp_file << "\n";
    std::cout << "CA file: " << ca_file << "\n";
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
  size_t Number_of_pixels_x = Number_of_pixels;
  size_t Number_of_pixels_y = int(Field_of_view_y/Field_of_view_x+0.5) * Number_of_pixels;  
  Themis::model_image_raster image_pulse_raw(-0.5*Field_of_view_x,0.5*Field_of_view_x,Number_of_pixels_x,-0.5*Field_of_view_y,0.5*Field_of_view_y,Number_of_pixels_y);
  Themis::model_image_splined_raster image_splined_raw(-0.5*Field_of_view_x,0.5*Field_of_view_x,Number_of_pixels_x,-0.5*Field_of_view_y,0.5*Field_of_view_y,Number_of_pixels_y);
  image_splined_raw.use_analytical_visibilities();
  Themis::model_image_upsample image_upsamp(image_pulse_raw,upsample_factor);
  Themis::model_image* image_pulse_ptr;
  if (upsample_image==false)
    if (do_fft_spline)
      image_pulse_ptr = &image_splined_raw;
    else
      image_pulse_ptr = &image_pulse_raw;
  else
    image_pulse_ptr = &image_upsamp;
  Themis::model_image& image_pulse=(*image_pulse_ptr);

  
  // Smooth if desired
  Themis::model_image_smooth image_smooth(image_pulse);
  Themis::model_image* image_ptr;
  if (smooth_image==false)
    image_ptr = &image_pulse;
  else
    image_ptr = &image_smooth;

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
  if (add_background_gaussian || add_ring)
    model_ptr = &image_sum;
  else
    model_ptr = image_ptr;  
  Themis::model_image& image=(*model_ptr);

  // Add scattering
  Themis::model_galactic_center_diffractive_scattering_screen image_scatt(image);


  
  // Read in data files
  Themis::data_visibility_amplitude VM_data_hi, VM_data_lo;
  Themis::data_closure_phase CP_data;
  Themis::data_closure_amplitude CA_data;
  std::vector<Themis::likelihood_base*> L;
  Themis::likelihood_optimal_gain_correction_visibility_amplitude *lva_hi, *lva_lo;

  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  

  if (vm_file!="")
  {
    std::ifstream vmin(vm_file);    
    std::string vm_file_name;

    vmin >> vm_file_name;
    VM_data_hi.add_data(vm_file_name,"HH");
    if (Reconstruct_gains) 
    {
      lva_hi = new Themis::likelihood_optimal_gain_correction_visibility_amplitude(VM_data_hi,image_scatt,station_codes,station_gain_priors);
      L.push_back(lva_hi);
    }
    else
      L.push_back(new Themis::likelihood_visibility_amplitude(VM_data_hi,image_scatt));

    vmin >> vm_file_name;
    if (vmin.eof()==true)
    {
      std::cerr << "ERROR: Expects two VM file names in " << vm_file << std::endl;
      return 1;
    }
    VM_data_lo.add_data(vm_file_name,"HH");
    if (Reconstruct_gains) 
    {
      lva_lo = new Themis::likelihood_optimal_gain_correction_visibility_amplitude(VM_data_lo,image_scatt,station_codes,station_gain_priors);
      L.push_back(lva_lo);
    }
    else
      L.push_back(new Themis::likelihood_visibility_amplitude(VM_data_lo,image_scatt));
  }

  if (cp_file!="")
  {
    std::ifstream cpin(cp_file);    
    std::string cp_file_name;

    cpin >> cp_file_name;
    CP_data.add_data(cp_file_name,"HH");
    if (cpin.eof()==true)
    {
      std::cerr << "ERROR: Expects two CP file names in " << cp_file << std::endl;
      return 1;
    }
    cpin >> cp_file_name;
    CP_data.add_data(cp_file_name,"HH");
    L.push_back(new Themis::likelihood_closure_phase(CP_data,image));
  }

  if (ca_file!="")
  {
    CA_data.add_data(ca_file);//,"HH");
    L.push_back(new Themis::likelihood_closure_amplitude(CA_data,image));
  }

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  for (size_t j=0; j<image_pulse.size(); ++j)
    P.push_back(new Themis::prior_linear(20,60)); // Itotal

  // Generate a set of means and ranges for the initial conditions
  std::vector<double> means(image_pulse.size()), ranges(image_pulse.size(),2.0);
  std::vector<std::string> var_names;

  double x,y;
  double sig = 0.17/0.5*(0.5*Field_of_view_x);
  double norm = std::log(2.5/(2.0*M_PI*sig*sig));
  for (size_t j=0,k=0; j<Number_of_pixels; ++j)
    for (size_t i=0; i<Number_of_pixels; ++i)
    {
      x = Field_of_view_x*double(i)/double(Number_of_pixels_x-1) - 0.5*Field_of_view_x;
      y = Field_of_view_y*double(j)/double(Number_of_pixels_y-1) - 0.5*Field_of_view_y;
      means[k++] = norm  - (x*x+y*y)/(2.0*sig*sig);
    }



  // If smoothing the image, add some parameters.  Assume a fixed smoothing kernel (circular, with given FWMH)
  double uas2rad = 1e-6/3600. * M_PI/180.;
  if (smooth_image)
  {
    // Sigma
    double smoothing_sigma = smoothing_fwhm / std::sqrt(8.0*std::log(2.0));
    P.push_back(new Themis::prior_linear((smoothing_sigma-1e-5)*uas2rad,(smoothing_sigma+1e-5)*uas2rad));
    means.push_back(smoothing_sigma*uas2rad);
    ranges.push_back(1e-6*uas2rad);
    
    // Asymmetry
    P.push_back(new Themis::prior_linear(0.0,1e-6));
    means.push_back(1e-8);
    ranges.push_back(1e-7);
    
    // Position angle
    P.push_back(new Themis::prior_linear(0,1e-6));
    means.push_back(1e-8);
    ranges.push_back(1e-7);
  }

  // Adding additional feature, kill shift
  if (add_background_gaussian || add_ring)
  {
    // x offset of image (center raster!)
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
    
    // y offset of image (center raster)
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
  }

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
    ////  12 x offset REVISIT
    //P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    //means.push_back(0.0);
    //ranges.push_back(1e-7*uas2rad);
    ////  13 y offset REVISIT
    //P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    //means.push_back(0.0);
    //ranges.push_back(1e-7*uas2rad);
    //  12 x offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    ranges.push_back(5*uas2rad);
    //  13 y offset REVISIT
    P.push_back(new Themis::prior_linear(-40*uas2rad,40*uas2rad));
    means.push_back(0.0);
    ranges.push_back(5*uas2rad);

  }

  
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


  
  if ( start_image_size==0 )
    start_image_size = start_parameter_list.size();

  if ( start_image_size>0)
  {
    size_t k=0;
    for (k=0; k<image_pulse.size(); ++k)
      ranges[k] = 1e-3;

    k=0;
    size_t kspl=0;

    // Up-/Down-sample image using bicubic interpolation
    if (start_fov==0)
      start_fov=Field_of_view_x;

    size_t Ns = size_t(std::sqrt(start_image_size));
    std::valarray<double> xs(Ns),ys(Ns),fs(Ns*Ns);
    for (size_t j=0; j<Ns; ++j)
    {
      xs[j] = start_fov*double(j)/double(Ns-1)-0.5*start_fov;
      ys[j] = start_fov*double(j)/double(Ns-1)-0.5*start_fov;
      for (size_t i=0; i<Ns; ++i)
	fs[j+Ns*i] = std::exp(start_parameter_list[kspl++]);
    }
    Themis::Interpolator2D start_image_interp(xs,ys,fs);
    start_image_interp.use_forward_difference();
    

    for (size_t j=0; j<Number_of_pixels; ++j)
      for (size_t i=0; i<Number_of_pixels; ++i)
      {
	// Interpolate up
	double x=Field_of_view_x*double(i)/double(Number_of_pixels-1)-0.5*Field_of_view_x;
	double y=Field_of_view_x*double(j)/double(Number_of_pixels-1)-0.5*Field_of_view_x;
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
	    mask *= std::exp(-(x-xs[Ns-1])*(x-xs[Ns-1])/(2.0*xstps*xstps));
	}
	if (std::fabs(y)>std::fabs(ys[0]))
	{
	  if (y<0)
	    mask *= std::exp(-(y-ys[0])*(y-ys[0])/(2.0*ystps*ystps));
	  else
	    mask *= std::exp(-(y-ys[Ns-1])*(y-ys[Ns-1])/(2.0*ystps*ystps));
	}

	// Set value
	means[k++] = std::max(20.0,std::min(60.0,std::log(val*mask)));
      }
   
    if (world_rank==0) 
      std::cerr << "FoVs" << std::setw(15) << start_fov << std::setw(15) << Field_of_view_x << std::endl;
    if (world_rank==0)
      std::cerr << "Added raster image to start values k=" << k  << "  kspl=" << kspl << std::endl;

    
    // Add remaining parameters
    if (smooth_image) // Smoothing kernel already fixed, skip means and ranges to next steps
      k += 3;

    if (smooth_image_start) // Smoothing kernel already fixed, skip means and ranges to next steps
      kspl += 3;

    
    if ( (add_background_gaussian && add_background_gaussian_start) || (add_ring && add_ring_start) )
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
      ranges[k] = 1e-3*means[k];
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
      ranges[k] = 1e-3*uas2rad;
      k++; kspl++;

      //  13 y offset
      means[k] = start_parameter_list[kspl];
      ranges[k] = 1e-3*uas2rad;
      k++; kspl++;
    }
  }

  ////////////////////// After checking for initial data
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
      std::cerr << "PriorCheck: "
		<< std::setw(10) << k
		<< std::setw(15) << means[k]
		<< std::setw(15) << ranges[k]
		<< '\n';
    std::cerr << "===========================================================\n";
  }
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

  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "Likelihood is " << Lstart << '\n';

  
  // Setup MCMC sampler
  //int Number_of_steps = 20000;
  int Number_of_chains; // Number of walkers
  if (Number_of_walkers==0)
    Number_of_chains = Number_of_pixels*Number_of_pixels*4;
  else
    Number_of_chains = Number_of_walkers;
  if (Number_of_chains<Number_of_pixels*Number_of_pixels*4)
  {
    std::cerr << "WARNING: Insufficient number of walkers for desired number of pixels.\n"
	      << "         Minimum number is 4*Npx^2 = " << Number_of_pixels*Number_of_pixels*4 << ".\n";
  }
  int Number_of_temperatures = Number_of_tempering_levels;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  //bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;

  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);


  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set tempering schedule
  MCMC_obj.set_tempering_schedule(Tempering_time,1.,Tempering_ladder);
  //MCMC_obj.set_tempering_schedule(1000.,1.,1.1);

  // Parallelization settings
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);


  for (size_t rep=0; rep<Number_of_reps; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started rep " << rep << std::endl;
    
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
      if (smooth_image)
      {
	sumout << std::setw(15) << "s-sig"
	       << std::setw(15) << "s-A"
	       << std::setw(15) << "s-phi";
      }
      if (add_background_gaussian)
      {
	sumout << std::setw(15) << "img-x"
	       << std::setw(15) << "img-y"
	       << std::setw(15) << "A-I"
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
      sumout << std::setw(15) << "VA rc2"
	     << std::setw(15) << "CP rc2"
	     << std::setw(15) << "CA rc2"
	     << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileName"
	     << std::endl;
    }
      

    std::stringstream chain_file_name, lklhd_file_name, chi2_file_name;
    chain_file_name << "Chain_" << std::setfill('0') << std::setw(3) << rep << ".dat";
    lklhd_file_name << "Lklhd_" << std::setfill('0') << std::setw(3) << rep << ".dat";
    chi2_file_name << "Chi2_" << std::setfill('0') << std::setw(3) << rep << ".dat";
    
    MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
			 chain_file_name.str(), lklhd_file_name.str(), chi2_file_name.str(),
			 means, ranges, var_names, restart_flag, out_precision, verbosity);
    
    
    
    std::vector<double> pbest = MCMC_obj.find_best_fit(chain_file_name.str(),lklhd_file_name.str());
    double Lval = L_obj(pbest);
    
    if (world_rank==0)
    {
      std::cout << "Lbest = " << Lval << std::endl;
      for (size_t k=0; k<image.size(); ++k)
	std::cout << std::setw(15) << pbest[k];
      std::cout << std::endl;
    }

    std::stringstream vm_res_name_hi, vm_res_name_lo, cp_res_name, ca_res_name, gc_name_hi, gc_name_lo;
    vm_res_name_hi << "VM_residuals_hi_" << std::setfill('0') << std::setw(3) << rep << ".d";
    vm_res_name_lo << "VM_residuals_lo_" << std::setfill('0') << std::setw(3) << rep << ".d";
    cp_res_name << "CP_residuals_" << std::setfill('0') << std::setw(3) << rep << ".d";
    ca_res_name << "CA_residuals_" << std::setfill('0') << std::setw(3) << rep << ".d";
    gc_name_hi << "gain_corrections_hi_" << std::setfill('0') << std::setw(3) << rep << ".d";
    gc_name_lo << "gain_corrections_lo_" << std::setfill('0') << std::setw(3) << rep << ".d";

      
    int Ndata = VM_data_hi.size() + VM_data_lo.size() + CP_data.size() + CA_data.size();

    // Adjust number of parameters if we are smoothing to actual independent pixels
    int Nparam = image.size();
    if (smooth_image)
    {
      double smoothing_sigma = smoothing_fwhm / std::sqrt(8.0*std::log(2.0));
      int Nindpx = int(std::ceil(Field_of_view_x*Field_of_view_y / (M_PI*smoothing_sigma*smoothing_sigma)));
      Nparam = std::min(Nparam,Nindpx);
    }

    if (add_background_gaussian || add_ring)
      Nparam -= 2; // Raster shift

    if (add_ring) // Fixed ring elements (REVISIT)
    {
      //Nparam -= 6;
      Nparam -= 4;
    }
    
    int Ngains = 0;
    if (Reconstruct_gains)
    {
      Ngains += lva_hi->number_of_independent_gains();
      lva_hi->output_gain_corrections(gc_name_hi.str());
      Ngains += lva_lo->number_of_independent_gains();
      lva_lo->output_gain_corrections(gc_name_lo.str());
    }
    int NDoF = Ndata - Nparam - Ngains;

    double VM_rchi2=0.0, CP_rchi2=0.0, CA_rchi2=0.0, rchi2=0.0;
    size_t k=0;
    if (vm_file!="")
    {
      L[k]->output_model_data_comparison(vm_res_name_hi.str());
      L[k+1]->output_model_data_comparison(vm_res_name_lo.str());
      VM_rchi2 = (L[k]->chi_squared(pbest)+L[k+1]->chi_squared(pbest)); // / (VM_data_lo.size()+VM_data_hi.size()-Nparam-Ngains);
      if (world_rank==0)
	std::cout << "VM rchi2: " << VM_rchi2  << " / " << VM_data_hi.size() << " + " << VM_data_lo.size() << " - " << Nparam << " - " << Ngains << std::endl;
      VM_rchi2 = VM_rchi2 / (VM_data_hi.size()+VM_data_lo.size()-Nparam-Ngains);      
      k+=2;
    }
    if (cp_file!="")
    {
      L[k]->output_model_data_comparison(cp_res_name.str());
      CP_rchi2 = L[k]->chi_squared(pbest);
      if (world_rank==0)
	std::cout << "CP rchi2: " << CP_rchi2  << " / " << CP_data.size() << " - " << Nparam << std::endl;
      CP_rchi2 = CP_rchi2 / (CP_data.size()-Nparam);
      k++;
    }
    if (ca_file!="")
    {
      L[k]->output_model_data_comparison(ca_res_name.str());
      CA_rchi2 = L[k]->chi_squared(pbest);
      if (world_rank==0)
	std::cout << "CA rchi2: " << CA_rchi2  << " / " << CA_data.size() << " - " << Nparam << std::endl;
      CA_rchi2 = CA_rchi2 / (CA_data.size()-Nparam);
      k++;
    }
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
      sumout << std::setw(15) << VM_rchi2 
	     << std::setw(15) << CP_rchi2 
	     << std::setw(15) << CA_rchi2 
	     << std::setw(15) << rchi2
	     << std::setw(15) << Lval
	     << "   " << vm_file
	     << "   " << cp_file
	     << "   " << ca_file
	     << std::endl;
    }


    // Reset priors
    k=0;
    //  Image
    for (size_t j=0; j<image_pulse.size(); ++j)
    {
      means[k] = pbest[k];
      ranges[k] = 1e-3;
      k++;
    }
    //  Smoothing
    if (smooth_image)
    {
      // Size
      means[k] = pbest[k];
      ranges[k] = 1e-3*pbest[k];
      k++;

      // Asymmetry
      means[k] = pbest[k];
      ranges[k] = 1e-7;
      k++;

      // PA
      means[k] = pbest[k];
      ranges[k] = 1e-7;
      k++;
    }

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
      
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
