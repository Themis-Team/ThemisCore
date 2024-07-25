/*!
    \file multigauss_complex_single_deo_diagstan.cpp
    \author Paul Tiede
    \date April, 2021
    \brief Driver file for M87 synthetic data analysis with geometric models:
              Large assymmetric Gaussian background
	      one or more symmetric Gaussians
	      one or more asymmetric Gaussians
    \details Takes complex visibility files for fitting, created using themispy:
	     These must be passed as -v option.
    \todo 
*/


#include "model_image_fixed_lightcurve.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"
#include "model_image_score_sgra.h"
// unceheaderfil
#include "uncertainty_visibility.h"
#include "uncertainty_visibility_loose_change.h"
#include "uncertainty_visibility_power_change.h"
#include "uncertainty_visibility_broken_power_change.h"

#include "optimizer_kickout_powell.h"


#include "sampler_deo_tempering_MCMC.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_power_tempered.h"
#include "utils.h"
#include "read_data.h"
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <complex>

/*
 * Reads in a config file and returns a vector with the parameters
 * like the config file
 * #                      Percentile
#         param           0.5%           2.5%            25%            50%            75%          97.5%          99.5%
              a      0.0187211      0.0228614      0.0618002        0.18952       0.472129       0.934031       0.988063
             u0      0.0313386      0.0797632       0.333633       0.625367        1.26685         3.0245        3.86391
              b       0.954535        1.42388        2.35213        2.86059        3.37849        4.55225        6.14047
              c      0.0600666       0.328347        3.42426        7.14147        10.9416        14.6142        14.9329
          a@4Gl      0.0104484      0.0109439      0.0123039      0.0130988      0.0139344      0.0157996       0.017002

 */
std::vector<std::vector<double>> read_config(std::string file);

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  std::string v_file="", nc_file="", imfile="", headerfile="";
  std::string anneal_start_fname = "";
  int Number_temperatures = 0;
  int Number_of_swaps = 10; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  std::string model_glob = "";
  bool restart_flag = false;
  bool scatter=false;
  bool Reconstruct_gains=false;


  std::string lc_file="";
  bool model_noise = false;
  bool model_noise_priors_sgra = true; // false;
  std::vector<std::string> gain_file_list;


  // Tempering stuff default options
  size_t number_of_rounds = 10;
  double initial_ladder_spacing = 1.1;
  int Thin_factor = 1;
  int Temperature_stride = 50;

  //AFSS adatation parameters
  int init_buffer = 25;
  int window = 150;
  int number_of_adaptation = 10000;
  bool save_adapt = true;
  int refresh = 10;


  bool preoptimize_flag = false;
  double opt_ko_llrf = 10.0;
  size_t opt_ko_itermax = 20;
  size_t opt_ko_rounds = 20;
  size_t opt_instances = 0;

  double frequency=230e9;

  bool constrain_4lambda = false;

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
    else if (opt == "--imfile")
    {
      if (k < argc)
        imfile = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --imfile \n";
	std::exit(1);

      }
    }
    else if (opt == "--header")
    {
      if (k < argc)
        headerfile = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --header \n";
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
    else if (opt == "--noise-config" || opt=="-nc")
    {
      if (k<argc)
	nc_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --noise-config, -nc.\n";
	std::exit(1);
      }
    }

    else if (opt=="--anneal-ladder-file" || opt=="-alf")
    {
      if (k<argc)
      {
        anneal_start_fname = std::string(argv[k++]);
      }
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: One argument must be provided after --annealing-ladder-file, -alf: annealing.dat file.\n";
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
    else if (opt=="-c4" || opt=="--constrain-4Glambda")
    {
      constrain_4lambda=true;
    }
    else if (opt=="-Ns")
    {
      if ( k< argc )
        Number_of_swaps = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after -Ns\n";
        std::exit(1);
        
      }
    }
    else if (opt=="--number-of-rounds" || opt=="-nor")
    {
      if (k<argc)
        number_of_rounds = atoi(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An int argument must be provided after --number-of-rounds, -nor.\n";
        std::exit(1);
      }
    }
    else if (opt=="--number-of-temperatures" || opt=="-not")
    {
      if (k<argc)
	Number_temperatures = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --number-of-temperatures, -not.\n";
	std::exit(1);
      }
    }
    else if ( opt == "--refresh" )
    {
      if ( k <argc )
        refresh = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --refresh.\n";
        std::exit(1);
      }
    }
    else if (opt=="--tempering-stride" || opt=="-ts")
    {
      if (k<argc)
        Temperature_stride = atoi(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An int argument must be provided after --tempering-stride, -ts.\n";
        std::exit(1);
      }
    }
    else if (opt=="--initial-ladder-spacing" || opt =="-ils")
    {
      if (k<argc)
        initial_ladder_spacing = atof(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: A float argument must be provided after --initial-ladder-spacing, -ils.\n";
        std::exit(1);
      }
    }
    else if (opt=="--scatter")
    {
        scatter=true;
    }
    else if (opt=="-lc" || opt=="--light-curve")
    {
      if (k<argc)
	lc_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after --light-curve, -lc.\n";
	std::exit(1);
      }
    }
    else if (opt=="-n" || opt=="--model-noise")
    {
      model_noise=true;
    }    
    else if (opt=="-nmprsgra" || opt=="--noise-prior-sgra")
    {
      model_noise_priors_sgra=true;
    }    
    else if (opt=="--thin-factor" || opt=="-tf")
    {
      if (k<argc)
        Thin_factor = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --thin-factor, -tf.\n";
	std::exit(1);
      }
    }
    else if (opt == "--number-of-adaptation" || opt == "-Na")
    {
      if ( k < argc )
        number_of_adaptation = atoi(argv[k++]);
      else
      {
        if (world_rank == 0)
          std::cerr << "ERROR: An int argument must be provided after --number-of-adaptation, -Na.\n";
        std::exit(1);
      }
    }
    else if (opt == "--initial-adaptation-steps" || opt == "-ias")
    {
      if ( k < argc )
        init_buffer = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after -ias.\n";
        std::exit(1);
      }
    }
    else if (opt == "--initial-window-steps" || opt == "-iws")
    {
      if ( k < argc )
        window = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after -iws.\n";
        std::exit(1);
      }
    }
    else if (opt == "--clear-adapt")
    {
      save_adapt = false;
    }
    else if (opt=="--continue")
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
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for multigauss fit using deo if differential evolution\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 multigauss_complex_single_deo_diff -v v_file [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-v, --visibilities <string>\n"
		  << "\t\tSets the name of the file containing file path to the complex visibilities data file.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
		  << "\t-p, -parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt.Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.).\n"
		  << "\t-Ns <int>\n"
		  << "\t\tSets the number of tempering swaps to take for the first round.  Defaults to 50.\n"
		  << "\t-Nr <int>\n"
		  << "\t\tSets the number of repetitions to perform.  Defaults to 2.\n"
		  << "\t-m, --model <string>\n"
		  << "\t\tSets the model definition string.  This may any combination of G, A, a, g\n"
		  << "\t\tin any order, with only one of each (so G, A, aagggG, etc.).  The first sets the origin.\n"
		  << "\t--morder <int>\n"
                  << "\t\tOrder of the mring expansion DEFAULT is 2\n"
                  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
		  << "\t-g, --reconstruct-gains\n"
		  << "\t\tReconstructs unknown station gains.  Default off.\n"
		  << "\t-c4, --constrain-4Glambda\n"
		  << "\t\tConstrains the noise model at 4 Glambda instead of usual\n"
                  << "\t\tDefault is 10 steps.\n"
                  << "\t--temperature-stride, -ts <int>\n"
		  << "\t\tSets the number of steps between tempering level swaps.  Defaults to 25.\n"
		  << "\t--number-of-rounds, -nor <int>\n"
		  << "\t\tSets the number of rounds for each DEO repetition.  Defaults to 7.\n"
		  << "\t--initial-ladder-spacing, -ils <float>\n"
		  << "\t\tSets the geometric factor by which subsequent ladder beta increases.  Defaults to 1.15.\n"
		  << "\t--thin-factor, -tf <int>\n"
		  << "\t\tSets the factor by which chain outputs are thinned.  Defaults to 1.\n"
                  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	  
                  << "\t--initial-adaptation-steps, -ias <int>\n"
		  << "\t\tSets the number of steps to take before the initial widths are updated.\n"
		  << "\t\tDefaults to 20.\n"
                  << "\t--intial-window-steps, -iws <int>\n"
		  << "\t\tSets the number of steps to take before the covariance is adapted.\n"
		  << "\t\tDefaults to 100.\n"
                  << "\t--max-depth <int>\n"
                  << "\t\tSets the max tree depth for NUTS.\n Default is 10."
                  << "\t--number-of-adaptation, -Na <int>\n"
		  << "\t\tSets the total number of STEPS the exploration samplers take in the adaptation phase.\n"
		  << "\t\tDefaults to 10,000.\n"	  
                  << "\t--clear-adapt\n"
		  << "\t\tBy default the adaptation phase of each round will be saved. This will turn that off.\n"
                  << "\t--continue\n"
		  << "\t\t Restarts the previous run from the ckpt file.\n"
                  << "\t--scatter\n"
		  << "\t\t Scatters the model using the Johnson 2018 scattering screen\n"
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
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
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
  if (imfile=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No image file was provided. The --imfile <string>,\n"
		<< "       is *required*.  See -h for more details and options.\n";
    std::exit(1);
  }
  if (headerfile=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No header file was provided. The --imfile <string>,\n"
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
  }
  //  Output these for check
  if (world_rank==0)
  {
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
  
  

  // Choose the model to compare (other models present for rapid build out)
  Themis::model_image_score_sgra image_sum(imfile, headerfile, false, 0);

  image_sum.write_model_tag_file();
  // Normalize by light curve if desired
  Themis::model_image* model_image_ptr = &image_sum;
  Themis::model_image_fixed_lightcurve* model_flc_ptr;
  if (lc_file!="")
  {
    model_flc_ptr = new Themis::model_image_fixed_lightcurve(image_sum,lc_file);
    model_image_ptr = model_flc_ptr;
  }
  
  // Scattered image with Sgr A* screen
  Themis::model_visibility* model_ptr;
  //Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image_sum);
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage( *model_image_ptr );
  if (scatter)
    model_ptr = &scattimage;
  else
    model_ptr = &image_sum;

  // Generate reference for future use
  Themis::model_visibility& image=(*model_ptr);

  
  Themis::uncertainty_visibility_broken_power_change uncertainty; 
  uncertainty.constrain_noise_at_4Glambda();
  uncertainty.logarithmic_ranges(); 
  
  //Read in the data
  std::vector<Themis::data_visibility*> V_data;
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_optimal_complex_gain_visibility*> lvg;
  std::vector<Themis::likelihood_visibility*> lv;
  //Themis::likelihood_visibility *lv_hi=0;

  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  //Use the proper calibration priors
  station_gain_priors[0] = 0.01;//AA
  station_gain_priors[1] = 0.01;//AP
  station_gain_priors[2] = 0.1;//AZ
  station_gain_priors[3] = 0.01;//JC
  station_gain_priors[4] = 0.2;//LM
  station_gain_priors[5] = 0.1;//PV
  station_gain_priors[6] = 0.01;//SM
  station_gain_priors[7] = 0.1;//SP
  

  // Read in the data into the buffer
  std::vector<std::string> v_file_name_list;
  read_vfile_mpi(v_file_name_list, v_file, MPI_COMM_WORLD);

  for (size_t j=0; j<v_file_name_list.size(); ++j)
  {
    V_data.push_back( new Themis::data_visibility(v_file_name_list[j],"HH") );
    V_data[j]->set_default_frequency(frequency); 
    
    if (Reconstruct_gains && (!model_noise) )
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
    else if ( (Reconstruct_gains) && (model_noise) )
    {
      if (world_rank==0) 
	{
	  std::cout<<"Including uncertainty model in likelihood_optimal_complex_gain_visibility object"<<std::endl;
	}
      lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(*V_data[j],image,uncertainty,station_codes,station_gain_priors) );
      L.push_back( lvg[j] );

    }
    else
    {
      if (model_noise) {
	std::cout<<"Including uncertainty model in likelihood_visibility object"<<std::endl;
	lv.push_back( new Themis::likelihood_visibility(*V_data[j],image,uncertainty) );
      }
      else {
	lv.push_back( new Themis::likelihood_visibility(*V_data[j],image) );
      }
      L.push_back( lv[j] );
    }
  }
  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means;
  std::vector<std::string> var_names;
    
    
  double uas2rad = 1e-6/3600. * M_PI/180.;

  //Add crescent priors and transforms 
  //   0 M/D
  P.push_back(new Themis::prior_linear(5.02, 5.04));
  means.push_back(5.03);
  //   1 PA
  P.push_back(new Themis::prior_linear(-M_PI, M_PI));
  means.push_back(0.0);


  if (model_noise) { 

    if (nc_file == "") // Set default permissive priors
    {
      if (world_rank==0)
	std::cout<<"ADJUSTING NOISE MODELING PRIOR RANGES to SGRA* PRE-MODELING CONSTRAINTS"<<std::endl;
      P.push_back(new Themis::prior_gaussian(std::log(0.004),5.0));
      P.push_back(new Themis::prior_gaussian(std::log(0.010),5.0));
      // P.push_back(new Themis::prior_gaussian(std::log(0.010),5.0));
      // P.push_back(new Themis::prior_gaussian(std::log(1.5),2.5));
      P.push_back(new Themis::prior_linear(std::log(0.0001),std::log(0.1000)));
      P.push_back(new Themis::prior_linear(std::log(0.01),std::log(10.0)));
      P.push_back(new Themis::prior_linear(1.0,5.0));
      P.push_back(new Themis::prior_linear(1.5,2.5));

      // don't reset if restart. Check whether start_parameter_list was used prior to here in order to set these
      // if (Number_start_params <= start_Number_of_pixels_x*start_Number_of_pixels_y+5) {
      //   means.push_back(std::log(0.004)); // noise threshold
      // 	means.push_back(std::log(0.010)); // fractional (mimicking non-closing errors)
      // 	means.push_back(std::log(0.018)); // bpc noise amplitude at 4Glambda
      // 	means.push_back(std::log(1.5)); // uv distance where two powerlaws break
      // 	means.push_back(2.5); // long baseline index
      // 	means.push_back(2.0); // short baseline index        
      // }
      // else
      // 	for (size_t k=start_Number_of_pixels_x*start_Number_of_pixels_y+5;k<start_Number_of_pixels_x*start_Number_of_pixels_y+5+6; k++)
      // 	{
      // 	  if (world_rank==0)
      // 	    std::cout<<"k="<<k<<" "<<start_parameter_list[k]<<std::endl;
      // 	  means.push_back(start_parameter_list[k]);
      // 	}
    }
    else // Read in a bpl_stats.txt file and generate priors from them.
    { 
      double noise_sigma[4];
      double noise_med[4];
      double noise_lo[4], noise_hi[4];
      //Set theshold and fractional components
      P.push_back(new Themis::prior_gaussian(std::log(0.004),1.0));
      P.push_back(new Themis::prior_gaussian(std::log(0.010),1.0));
      //Now read in the config file for the others
      if (world_rank == 0)
      {
	std::vector<std::vector<double> > params = read_config(nc_file);
	for ( int nn=1; nn<=4; ++nn )
	{
	  // Interquartile range is about 1.35 * std dev.
	  noise_sigma[nn%4] = (params[nn][4]-params[nn][2])/1.35;
	  noise_med[nn%4] = params[nn][3];

	  noise_lo[nn%4] = params[nn][2];
	  noise_hi[nn%4] = params[nn][4];

	  std::cerr << "Noise BPL intialized at [" << (nn%4) << "] = " << noise_med[nn%4] << " +- " << noise_sigma[nn%4]
		    << "  log: " << std::log(noise_med[nn%4]) << " +- " << noise_sigma[nn%4]/noise_med[nn%4]
		    << '\n';
	}
      }
      //Now broadcast to everyone
      MPI_Bcast(&noise_sigma[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_med[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_lo[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_hi[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      //Now set the priors
      // for ( size_t nt=0; nt<2; ++nt )
      // 	P.push_back(new Themis::prior_gaussian(std::log(noise_med[nt]), noise_sigma[nt]/noise_med[nt]));

      for (size_t nt=0; nt<2; ++nt)
	P.push_back(new Themis::prior_linear(std::log(noise_lo[nt]),std::log(noise_hi[nt])));
      
      //for ( size_t nt=2; nt<4; ++nt )
      //P.push_back(new Themis::prior_gaussian(noise_med[nt], noise_sigma[nt]));
      //P.push_back(new Themis::prior_linear(1.0,5.0));
      P.push_back(new Themis::prior_linear(noise_lo[2],noise_hi[2]));
      P.push_back(new Themis::prior_linear(1.5,2.5));

      // if (world_rank==0)
      // 	std::cerr << "NOISE PRIOR INITIALIZATION: "
      // 		  << start_Number_of_pixels_x*start_Number_of_pixels_y+5 << " "
      // 		  << Number_start_params << '\n';
											
      if (Number_start_params <= image.size())
      {
      	means.push_back(std::log(0.004)); // noise threshold
      	means.push_back(std::log(0.01)); // fractional (mimicking non-closing errors)
      	means.push_back(std::log(noise_med[0])); // bpc noise amplitude at 4Glambda
      	means.push_back(std::log(noise_med[1])); // uv distance where two powerlaws break
      	means.push_back(noise_med[2]); // long baseline index
      	//means.push_back(noise_med[3]); // short baseline index        
      	//means.push_back(2.8); // long baseline index
      	means.push_back(2.0); // short baseline index        
      }
      else
      	for (size_t k=image.size();k<image.size()+6; k++)
      	{
      	  if (world_rank==0)
      	    std::cout<<"k="<<k<<" "<<start_parameter_list[k]<<std::endl;
      	  means.push_back(start_parameter_list[k]);
      	}


    }



  }



  if ( (means.size() != P.size())){
    std::cerr << "Error number of means does equal number of priors!\n";
    std::cerr << " means.size: " << means.size()
              << "\n P.size(): " << P.size() << std::endl;
    std::exit(1);
  }
  int nparams = image.size();
  if (model_noise)
      nparams += uncertainty.size();
  if ( (nparams != means.size())){
    std::cerr << "Error number of image params does equal number of means!\n";
    std::cerr << " nparams: " << nparams
              << "\n means.size(): " << means.size() << std::endl;
    std::exit(1);
  }

  for (size_t k=0; k<nparams; ++k)
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
  if ( (P.size() != means.size())){
    std::cerr << "Error number of transforms does equal number of means!\n";
    std::exit(1);
  }
  if (world_rank==0)
    std::cout << "Finished fusing prior lists, now at " << means.size() << std::endl;


  if (Number_start_params>int(means.size()))
  {
    std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
    std::exit(1);
  }
  for ( int i = 0; i < Number_start_params; ++i)
    means[i] = start_parameter_list[i];

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  Themis::likelihood_power_tempered L_beta(L_obj);
  
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Ndata = 0;
  for (size_t j=0; j<V_data.size(); ++j)
    Ndata += 2*V_data[j]->size();
  int Nparam = image_sum.size()-2.0;
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);
  if (Reconstruct_gains)
    for (size_t j=0; j<lvg.size(); ++j)
    {
      Ngains_list[j] = lvg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  int NDoF = Ndata - Nparam - Ngains;
  
  // Setup MCMC sampler
  int Number_of_procs_per_lklhd = 1;

  // Get the best point 
  std::vector<double> pbest =  means;
  // double lMAP;
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);
  // Themis::optimizer_laplace optim(L_obj, var_names, pbest.size());
  // optim.set_parameters(1e-2, 0, 200);
  // optim.set_scale(Ndata);
  // int nitr = 0;
  // nitr = optim.parallel_optimizer(pbest, lMAP, opt_instances, seed);

  if (preoptimize_flag)
  {
    if (Reconstruct_gains)
      for (size_t j=0; j<lvg.size(); ++j)
	lvg[j]->set_iteration_limit(20);
      
    opt_obj.set_cpu_distribution(Number_of_procs_per_lklhd);
    opt_obj.set_kickout_parameters(opt_ko_llrf,opt_ko_itermax,opt_ko_rounds);
    means = opt_obj.run_optimizer(L_obj, Ndata, means, "PreOptimizeSummary.dat",opt_instances);

    std::cout<<"means.size()="<<means.size()<<std::endl;
    for (size_t element=0; element<means.size(); ++element)
      {
	std::cout<<"means="<<means[element]<<",";
	std::cout<<std::endl;
      }
    if (Reconstruct_gains)
      for (size_t j=0; j<lvg.size(); ++j)
	lvg[j]->set_iteration_limit(50);
    
    pbest = means;
  }
  if (Reconstruct_gains)
    for (size_t j=0; j<lvg.size(); ++j)
    {
      std::stringstream gc_name, cgc_name;
      gc_name << "kickout_powell"  << "_gain_corrections_" << std::setfill('0') << j << ".d";
      cgc_name << "kickout_powell" << "_complex_gains_" << std::setfill('0') << j << ".d";
      lvg[j]->output_gain_corrections(gc_name.str());
      lvg[j]->output_gains(cgc_name.str());
    }

  // Generate a chain
  int Ckpt_frequency = 100;
  int out_precision = 8;
  int verbosity = 0;
  
  // Create a sampler object
  //Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO(seed, L_beta, var_names, means.size());
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed, L_beta, var_names, means.size());

  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "summary.dat");
    
  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat");

  if (anneal_start_fname != "")
     DEO.read_initial_ladder(anneal_start_fname);
  
  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set annealing schedule
  DEO.set_annealing_schedule(initial_ladder_spacing); 
  DEO.set_deo_round_params(Number_of_swaps, Temperature_stride);
  
  //Set the initial location of the sampler
  DEO.set_initial_location(means);
  
  //Get the exploration sampler and set some options
  //Themis::sampler_stan_adapt_diag_e_nuts_MCMC* kexplore = DEO.get_sampler();
  //kexplore->set_max_depth(10);
  Themis::sampler_automated_factor_slice_sampler_MCMC* kexplore = DEO.get_sampler();
  kexplore->set_adaptation_parameters(number_of_adaptation, save_adapt);
  kexplore->set_window_parameters(init_buffer, window);
  // Set a checkpoint




 
  int round_start = 0;
  if (restart_flag){
    DEO.read_checkpoint("MCMC.ckpt");
    round_start = DEO.get_round();
    restart_flag=false;
  }


  //////////////////// DEVEL
  std::cerr << "N temps: " << Number_temperatures << '\n';
  std::cerr << "world size: " << world_size << '\n';
  std::cerr << "world rank: " << world_rank << '\n';
  std::cerr << "N per L: " << world_size/Number_temperatures << '\n';
  if (Number_temperatures==0)
    Number_temperatures = world_size;
  int Number_per_likelihood = world_size/Number_temperatures;
  if (world_size%Number_per_likelihood != 0){
    if (world_rank == 0){
      std::cerr << "The total number of MPI processes must be divisible by the number of cores per likelihood evaluation!\n";
      std::cerr << "The distribution is currently: " << std::endl
                << "\tNumber of procs: " << world_size << std::endl
                << "\tNumber per lklhd: " << Number_per_likelihood << std::endl
                << "\tRemainder: " << world_size%Number_per_likelihood << std::endl;
    }
    std::exit(1);
  }
  DEO.set_cpu_distribution(Number_temperatures, Number_per_likelihood);
  
  
  //Start looping over the rounds
  for ( size_t rep = round_start; rep < number_of_rounds; ++rep)
  {
    if ( world_rank == 0 )
      std::cerr << "Started round " << rep << std::endl;

    //Generate fit_summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3)<< rep << "_fit_summaries.txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index";
      sumout << std::setw(15) << "M/D"
             << std::setw(15) << "PA";
      if (model_noise)
	sumout << std::setw(15) << "n-t"
	       << std::setw(15) << "n-f"
	       << std::setw(15) << "n-sigp"
	       << std::setw(15) << "n-up"
	       << std::setw(15) << "n-ahi"
	       << std::setw(15) << "n-alo";
	  
      for (size_t j=0; j<L.size(); ++j) 
	sumout << std::setw(15) << "V" + std::to_string(j) + " rc2";
      sumout << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileNames"
	     << std::endl;
    }

    if ( rep != 0 ){
      pbest = kexplore->find_best_fit();
    }
    
    // Output the current location
    if (world_rank==0)
      for (size_t j=0; j<pbest.size(); ++j)
	std::cout <<"pbest[j]="<< pbest[j] << std::endl;

    double Lval = L_obj(pbest);
    if (Reconstruct_gains)
      for (size_t j=0; j<lvg.size(); ++j)
      {
	std::stringstream gc_name, cgc_name;
	gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" << std::setfill('0') << j << ".d";
	cgc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_complex_gains_" << std::setfill('0') << j << ".d";
	lvg[j]->output_gain_corrections(gc_name.str());
	lvg[j]->output_gains(cgc_name.str());
      }
    
    // Generate summary and ancillary output

    //   Summary file parameter location:
    if (world_rank==0)
    {
      sumout << std::setw(10) << 0;
      for (size_t k=0; k<nparams; ++k)
	sumout << std::setw(15) << pbest[k];
    }

    //   Data-set specific output
    double V_rchi2=0.0, rchi2=0.0;
    for (size_t j=0; j<L.size(); ++j)
    {
      // output residuals
      std::stringstream v_res_name;
      v_res_name << "round" << std::setfill('0') << std::setw(3) << rep << "_residuals_" << std::setfill('0') << j << ".d";
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
	     << std::setw(15) << Lval;
      for (size_t j=0; j<v_file_name_list.size(); ++j)
	sumout << "   " << v_file_name_list[j];
      sumout << std::endl;
    }
    
    // Run the sampler with the given settings
    if (rep<number_of_rounds) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	std::cerr << "Starting MCMC on round " << rep << std::endl;
      DEO.run_sampler( 1, Thin_factor, refresh, verbosity);
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

std::vector<std::vector<double> > read_config(std::string cfile)
{
    std::ifstream in(cfile); 
    if (!in.is_open())
    {
        std::cerr << "read_config: Configuration file not found " << cfile << std::endl;
        std::exit(1);
    }
    std::string line;
    //skip first two lines
    std::getline(in, line);
    std::getline(in, line);
    //get first parameter quantiles
    std::string word;
    std::vector<std::vector<double> > params;
    for (int i = 0; i < 5; ++i)
    {
        in >> word; //skip first line since it is a character
        std::vector<double> tmp(7, 0.0);
        for ( int j = 0; j < 7; ++j )
        {
            in >> word;
            tmp[j] = std::stod(word.c_str());
        }
        params.push_back(tmp);
    }
    in.close();
    return params;
}
