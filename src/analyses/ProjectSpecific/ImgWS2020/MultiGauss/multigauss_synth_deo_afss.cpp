/*!
    \file multigauss_complex_single_deo_afss.cpp
    \author Avery E Broderick, Paul Tiede
    \date June, 2019
    \brief Driver file for Sgr A* synthetic data analysis with geometric models:
              Large assymmetric Gaussian background
	      one or more symmetric Gaussians
	      one or more asymmetric Gaussians
    \details Takes complex visibility files for fitting, created using themispy:
	     These must be passed as -v option.
    \todo 
*/


#include "model_image_xsringauss.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_smooth.h"
#include "optimizer_kickout_powell.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_power_tempered.h"
#include "utils.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"

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
#include <unistd.h>
int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  std::string v_file="";
  std::string anneal_start_fname = "";
  int Number_of_swaps = 10; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  std::string model_glob = "";
  bool restart_flag = false;
  bool scatter = false;

  // Tempering stuff default options
  size_t number_of_rounds = 7;
  double initial_ladder_spacing = 1.1;
  int Thin_factor = 25;
  int Temperature_stride = 50;

  //AFSS adatation parameters
  int init_buffer = 20;
  int window = 500;
  int number_of_adaptation = 5000;
  bool save_adapt = true;
  int refresh = 10;


  bool preoptimize_flag = false;
  double opt_ko_llrf = 10.0;
  size_t opt_ko_itermax = 20;
  size_t opt_ko_rounds = 20;
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
    else if (opt=="-m" || opt=="--model")
    {
      if (k<argc)
	model_glob = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A string argument must be provided after -m, --model.\n";
	std::exit(1);
      }
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
    else if ( opt == "--scatter" )
    {
      scatter = true;
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
		  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
                  << "\t\tDefault is 10 steps.\n"
                  << "\t--temperature-stride, -ts <int>\n"
		  << "\t\tSets the number of steps between tempering level swaps.  Defaults to 50.\n"
		  << "\t--number-of-rounds, -nor <int>\n"
		  << "\t\tSets the number of rounds for each DEO repetition.  Defaults to 7.\n"
		  << "\t--initial-ladder-spacing, -ils <float>\n"
		  << "\t\tSets the geometric factor by which subsequent ladder beta increases.  Defaults to 1.15.\n"
		  << "\t--thin-factor, -tf <int>\n"
		  << "\t\tSets the factor by which chain outputs are thinned.  Defaults to 10.\n"
                  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	  
                  << "\t--initial-adaptation-steps, -ias <int>\n"
		  << "\t\tSets the number of steps to take before the initial widths are updated.\n"
		  << "\t\tDefaults to 20.\n"
                  << "\t--intial-window-steps, -iws <int>\n"
		  << "\t\tSets the number of steps to take before the covariance is adapted.\n"
		  << "\t\tDefaults to 100.\n"
                  << "\t--number-of-adaptation, -Na <int>\n"
		  << "\t\tSets the total number of STEPS the exploration samplers take in the adaptation phase.\n"
		  << "\t\tDefaults to 10,000.\n"	  
                  << "\t--clear-adapt\n"
		  << "\t\tBy default the adaptation phase of each round will be saved. This will turn that off.\n"
                  << "\t--continue\n"
		  << "\t\t Restarts the previous run from the ckpt file.\n"
		  << "\t--scatter\n"
                  << "\t\tAdds diffractive scattering to the model to fit intrinsic source.\n"
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
  
  //Model information and parsing
  // Parse model_glob
  std::vector< std::string > model_list;
  int fix_model = 1;
  for (size_t j=0; j<model_glob.length(); ++j) 
  {
    model_list.push_back(model_glob.substr(j,1));
    if (model_list[model_list.size()-1]=="G")
    {
      std::cerr << "Added large-scale symmetric Gaussian\n";
      fix_model++;
    }
    else if (model_list[model_list.size()-1]=="A")
    {
      std::cerr << "Added large-scale asymmetric Gaussian\n";
      fix_model++;
    }
    else if (model_list[model_list.size()-1]=="g")
      std::cerr << "Added small-scale symmetric Gaussian\n";
    else if (model_list[model_list.size()-1]=="a")
      std::cerr << "Added small-scale asymmetric Gaussian\n";
    else
    {
      std::cerr << "ERROR: Unrecognized model option " << model_list[model_list.size()-1] << '\n';
      std::exit(1);
    }
  }

  /* 
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
        sleep(5);
  */


  

  // Choose the model to compare (other models present for rapid build out)
  std::vector< Themis::model_image* > model_components;
  std::vector< Themis::model_image_symmetric_gaussian* > model_g;
  std::vector< Themis::model_image_asymmetric_gaussian* > model_a;
  for (size_t mc=0; mc<model_list.size(); ++mc)
  {
    if (model_list[mc]=="G" || model_list[mc]=="g")
    {
      model_g.push_back( new Themis::model_image_symmetric_gaussian );
      model_components.push_back(model_g[model_g.size()-1]);
    }
    else if (model_list[mc]=="A" || model_list[mc]=="a")
    {
      model_a.push_back( new Themis::model_image_asymmetric_gaussian );
      model_components.push_back(model_a[model_a.size()-1]);
    }
  }

  // Create a sum model
  Themis::model_image_sum image(model_components);
  image.write_model_tag_file();
  //Blur the image because it is Sgr A*
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image);
  
  // Use analytical Visibilities
  for (size_t mc=0; mc<model_g.size(); ++mc)
    model_g[mc]->use_analytical_visibilities();
  for (size_t mc=0; mc<model_a.size(); ++mc)
    model_a[mc]->use_analytical_visibilities();
  
  
  //Read in the data
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
  if (!scatter)
    lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,image,station_codes,station_gain_priors);
  else
    lv_hi = new Themis::likelihood_optimal_complex_gain_visibility(V_data_hi,scattimage,station_codes,station_gain_priors);
  L.push_back(lv_hi);
  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<Themis::transform_base*> T;
  std::vector<double> means;
  std::vector<std::string> var_names;
    
    
  double uas2rad = 1e-6/3600. * M_PI/180.;
    

  for (size_t mc=0; mc<model_list.size(); ++mc)
  {
    //   0 Itot
    P.push_back(new Themis::prior_linear(0.0,20));
    T.push_back(new Themis::transform_none());
    means.push_back(4.0);

    //   1 Size
    if (model_list[mc]=="G" || model_list[mc]=="A")
    {
      P.push_back(new Themis::prior_logarithmic(1e4*uas2rad,1e8*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(1e5*uas2rad);
    }
    else
    {
      P.push_back(new Themis::prior_linear(0.0,1e2*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(20*uas2rad);
    }

    // If asymmetric Gaussian, add asymmetry parameters
    if (model_list[mc]=="A" || model_list[mc]=="a")
    {
      //   2 Asymmetry
      P.push_back(new Themis::prior_linear(0.0,0.99));
      T.push_back(new Themis::transform_none());
      means.push_back(0.1);
      //   3 Position angle
      P.push_back(new Themis::prior_linear(0,M_PI));
      T.push_back(new Themis::transform_none());
      means.push_back(0.5*M_PI);
    }

    // If first component, fix at origin, otherwise, let move
    if (mc==0 || model_list[mc]=="G" || model_list[mc]=="A") 
    {
      //   4 x offset if first component then fix
      P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(0.0);
      //   5 y offset if first component then fix
      P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(0.0);
    }
    else
    {
      //   4 x offset
      P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(0.0);
      //   5 y offset
      P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
      T.push_back(new Themis::transform_none());
      means.push_back(0.0);
    }
  }

  if ( (T.size() != P.size())){
    std::cerr << "Error number of transforms does equal number of priors!\n";
    std::exit(1);
  }
  if ( (T.size() != means.size())){
    std::cerr << "Error number of transforms does equal number of means!\n";
    std::exit(1);
  }

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
  Themis::likelihood L_obj(P,T, L, W);
  Themis::likelihood_power_tempered L_beta(L_obj);

  double llll = L_beta(means);
  if (world_rank == 0)
    std::cerr << "Starting likelihood: " << llll << std::endl;
  std::vector<double> pbest =  means;
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);

  //Now pre-optimize
  if (preoptimize_flag)
  {
    lv_hi->set_iteration_limit(20);
      
    opt_obj.set_cpu_distribution(1);
    opt_obj.set_kickout_parameters(opt_ko_llrf,opt_ko_itermax,opt_ko_rounds);
    means = opt_obj.run_optimizer(L_obj, 2*V_data_hi.size(), means, "PreOptimizeSummary.dat",opt_instances);

    lv_hi->set_iteration_limit(50);
    
    pbest = means;

    //Reset the priors
    // Set up the revised means
    for (size_t mc=0,j=0; mc<model_list.size(); ++mc)
    {
      // 0 Itot
      means[j] = pbest[j];
      
      // 1 Size
      means[j] = pbest[j];
      
      // If asymmetric Gaussian, add asymmetry parameters
      if (model_list[mc]=="A" || model_list[mc]=="a")
      {
	//   2 Asymmetry
 	means[j] = pbest[j];
	//   3 Position angle
	means[j] = pbest[j];
      }
      if ( model_list[mc]=="G" || model_list[mc]=="A" || mc==0 )
      {
        //   4 x offset if first component then fix
        means[j] = pbest[j];
        //   5 y offset if first component then fix
        means[j] = pbest[j];
      }
      else
      {
        //   4 x offset if first component then fix
        means[j] = pbest[j];
        //   5 y offset if first component then fix
        means[j] = pbest[j];
      }  
    }
  }

  // Generate a chain
  int Ckpt_frequency = 10;
  int out_precision = 8;
  int verbosity = 0;
  
  // Create a sampler object
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed, L_beta, var_names, means.size());

  if (restart_flag)
    DEO.read_checkpoint("MCMC.ckpt");

  //Get the exploration sampler and set some options
  Themis::sampler_automated_factor_slice_sampler_MCMC* sexplore = DEO.get_sampler();
  sexplore->set_window_parameters(init_buffer, window);
  sexplore->set_adaptation_parameters(number_of_adaptation, save_adapt);
 
  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");

  // Set annealing schedule
  DEO.set_annealing_schedule(initial_ladder_spacing); 
  DEO.set_deo_round_params(Number_of_swaps, Temperature_stride);
  
  //Set the initial location of the sampler
  DEO.set_initial_location(means);

  //Set the annealing output
  std::string anneal_name = "annealing.dat";
  DEO.set_annealing_output(anneal_name);

  std::stringstream chain_file_name, state_file_name, sampler_file_name;
  chain_file_name << "chain_deo_afss.dat";
  state_file_name << "state_deo_afss.dat";
  sampler_file_name << "sampler_deo_afss.dat";
  //Set the output
  DEO.set_output_stream(chain_file_name.str(), state_file_name.str(), sampler_file_name.str(), out_precision);

  if (anneal_start_fname != ""){
     DEO.read_initial_ladder(anneal_start_fname);
  }
  
  //Start looping over the rounds
  for ( size_t rep = 0; rep < number_of_rounds; ++rep)
  {
    if ( world_rank == 0 )
      std::cerr << "Started round " << rep << std::endl;

    //Generate fit_summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3)
               << rep << "fit_summaries.txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index"; 
      for (size_t mc=0; mc<model_list.size(); ++mc) 
      {
        if (model_list[mc]=="G" || model_list[mc]=="g")
        {
          sumout << std::setw(15) << "Ig (Jy)"
                 << std::setw(15) << "sigg (uas)"
                 << std::setw(15) << "xg"
                 << std::setw(15) << "yg";  
        }

        if (model_list[mc]=="A" || model_list[mc]=="a")
        {
          sumout << std::setw(15) << "Ia (Jy)"
                 << std::setw(15) << "siga (uas)"
                 << std::setw(15) << "A"
                 << std::setw(15) << "phi"
                 << std::setw(15) << "xa"
                 << std::setw(15) << "ya";
        }
      }
      sumout << std::setw(15) << "V rc2"
             << std::setw(15) << "Total rc2"
             << std::setw(15) << "log-liklhd"
             << "     FileName"
             << std::endl;
    }

    if ( rep != 0 )
      pbest = sexplore->find_best_fit();
    
 
    std::stringstream v_res_name_hi, gc_name_hi, cgc_name_hi;
    v_res_name_hi << "round" << std::setfill('0') << std::setw(3) 
                  << rep << "_V_residuals.d";
    gc_name_hi << "round" << std::setfill('0') << std::setw(3) 
               << rep << "_gain_corrections.d";
    cgc_name_hi << "round" << std::setfill('0') << std::setw(3) 
                << rep << "_complex_gains.d";
    
    int Ndata = 2*V_data_hi.size();
    int Nparam = image.size();
    Nparam -= 2*fix_model;

    int Ngains = 0;

    // Set up the revised means 
    for (size_t mc=0,j=0; mc<model_list.size(); ++mc)
    {
      // 0 Itot
      means[j] = pbest[j];
      
      // 1 Size
      means[j] = pbest[j];
      
      // If asymmetric Gaussian, add asymmetry parameters
      if (model_list[mc]=="A" || model_list[mc]=="a")
      {
	//   2 Asymmetry
 	means[j] = pbest[j];
	//   3 Position angle
	means[j] = pbest[j];
      }
      if ( model_list[mc]=="G" || model_list[mc]=="A" || mc==0 )
      {
        //   4 x offset if first component then fix
        means[j] = pbest[j];
        //   5 y offset if first component then fix
        means[j] = pbest[j];
      }
      else
      {
        //   4 x offset if first component then fix
        means[j] = pbest[j];
        //   5 y offset if first component then fix
        means[j] = pbest[j];
      }  
    }

    double Lval = L_obj(pbest);

    if (world_rank==0)
    {
      std::cerr << "After " << rep << " repetitions the best fit is at \n";
      for (size_t j=0; j<means.size(); ++j)
        std::cerr << pbest[j] << std::endl;
      std::cerr << "With likelihood " << Lval << std::endl;
    }
    
    Ngains += lv_hi->number_of_independent_gains();
    lv_hi->output_gain_corrections(gc_name_hi.str());
    lv_hi->output_gains(cgc_name_hi.str());
    
    int NDoF = Ndata - Nparam - Ngains;
    double V_rchi2=0.0, rchi2=0.0;
    L[0]->output_model_data_comparison(v_res_name_hi.str());
    V_rchi2 = (L[0]->chi_squared(pbest));
    if (world_rank==0)
      std::cerr << "V rchi2: " << V_rchi2  << " / " << 2*V_data_hi.size() << " - " << Nparam << " - " << Ngains << std::endl;
    V_rchi2 = V_rchi2 / NDoF;      
    rchi2 = L_obj.chi_squared(pbest);
    if (world_rank==0)
      std::cerr << "Tot rchi2: " << rchi2  << " / " << NDoF << std::endl;
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
    clock_t start = clock();
    if ( world_rank == 0 ){
      std::cerr << "Starting MCMC on round " << rep << std::endl;
    }
    DEO.run_sampler(1, Thin_factor, refresh, verbosity);
    clock_t end = clock();
    if (world_rank == 0){
      std::cerr << "Done MCMC on round " << rep << std::endl
              << "it took " << (end-start)/CLOCKS_PER_SEC/3600.0 << " hours" << std::endl;
    }
    //Reset the priors
    sumout.close();
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
