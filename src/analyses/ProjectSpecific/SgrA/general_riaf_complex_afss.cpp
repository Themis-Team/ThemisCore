/*!
    \file sgrA_riaf_fitting.cpp
    \author Paul Tiede
    \date May, 2020
    \brief Driver file for SgrA analysis with Broderick et al. 2016 RIAF model which is static
    \details Takes file lists generated via something like:
	     These must be passed a -v <file> which contains a list of files to be read in.
*/

#include "optimizer_kickout_powell.h"

#include "utils.h"

#include "sampler_deo_tempering_MCMC.h"
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

#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_power_tempered.h"

#include "model_visibility_galactic_center_diffractive_scattering_screen.h"
#include "model_image_general_riaf.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"

int main(int argc, char* argv[])
{
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  int verbosity = 0;
  std::string v_file="";
  std::string anneal_start_fname = "";
  int Number_of_swaps = 10; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  std::string model_glob = "";
  bool restart_flag = false;

  // Tempering stuff default options
  size_t number_of_rounds = 7;
  double initial_ladder_spacing = 1.1;
  int thin_factor = 1;
  int Temperature_stride = 25;

  //AFSS adatation parameters
  int init_buffer = 75;
  int window = 100;
  int number_of_adaptation = 5000;
  bool save_adapt = true;
  int refresh = 10;
  bool add_G = false;

  //RIAF options
  size_t npix = 64;
  size_t nref = 0;
  double fov = 60;
  size_t Number_per_likelihood = 1; //Number of cores per likelihood
  bool Reconstruct_gains = false;


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
    else if (opt=="--Gains" || opt=="-g")
    {
      Reconstruct_gains = true;
    }
    else if (opt=="--Gaussian" || opt=="-G")
    {
      add_G = true;
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
        thin_factor = atoi(argv[k++]);
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
    else if (opt=="--npix")
    {
      if (k+1<argc)
      {
	npix = atoi(argv[k++]);
        nref = atoi(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: Two int arguments must be provided after --npix.\n";
	std::exit(1);
      }
    }
    else if (opt=="--fov")
    {
      if (k<argc)
	fov = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A double argument must be provided after --fov.\n";
	std::exit(1);
      }
    }
    else if (opt=="-Nl" || opt =="--number-likelihood")
    {
      if (k<argc)
        Number_per_likelihood = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -Nl.\n";
	std::exit(1);
      }
    }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for the general riaf fit using deo with afss\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 multigauss_complex_single_deo_afss -v v_file [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  
                  << "\t-v, --visibilities <string>\n"
		  << "\t\tSets the name of the file containing file path to the complex visibilities data file.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
                  << "\t--npix <int,int>\n"
                  << "\t\tSets the number of pixels to be used in the image and number of refines to use. DEFAULT 64, 0, i.e. creates as 64x64 image with no refines.\n"
                  << "\t--fov <double>\n"
                  << "\t\tSets the field of view of the image in units of M. DEFAULT is 60M, i.e. creates a 60Mx60M image\n"
                  << "\t-Nl, --number-likelihood <int>\n"
                  << "\t\tSets the number of processors to use per likelihood evaluation. Default is 1 which is WAY to few.\n"
                  << "\t-p, -parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt.Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.).\n"
		  << "\t-Ns <int>\n"
		  << "\t\tSets the number of tempering swaps to take for the first round.  Defaults to 50.\n"
		  << "\t-Nr <int>\n"
		  << "\t\tSets the number of repetitions to perform.  Defaults to 2.\n"
		  << "\t-G, --Gaussian\n"
		  << "\t\tAdds a large scale gaussian to the image.\n"
		  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
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
                  << "\t--number-of-adaptation, -Na <int>\n"
		  << "\t\tSets the total number of STEPS the exploration samplers take in the adaptation phase.\n"
		  << "\t\tDefaults to 10,000.\n"	  
                  << "\t--clear-adapt\n"
		  << "\t\tBy default the adaptation phase of each round will be saved. This will turn that off.\n"
                  << "\t--continue\n"
		  << "\t\t Restarts the previous run from the ckpt file.\n"
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
    std::cout << "Npix: " << npix << std::endl
              << "Nref: " << nref << std::endl;
    std::cout << "FOV: " << fov << std::endl;
    std::cout << "nor: " << number_of_rounds << "\n";
    std::cout << "nswaps: " << Number_of_swaps << "\n";
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
 
  



  //RIAF model plus a large scale gaussian
  Themis::model_image_general_riaf riaf;
  riaf.set_image_resolution(npix, nref);
  riaf.set_screen_size(fov/2.0);

  //Large scale gaussian
  Themis::model_image_symmetric_gaussian gauss;
  gauss.use_analytical_visibilities();

  Themis::model_image_sum image;
  image.add_model_image(riaf);
  if (add_G)
    image.add_model_image(gauss);

  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattimage(image);

  
  // Read in data files
  std::vector<Themis::data_visibility*> V_data;
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_optimal_complex_gain_visibility*> lvg;
  std::vector<Themis::likelihood_visibility*> lv;
  
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

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
      strlng = v_file_name.length()+1;
    MPI_Bcast(&strlng,1,MPI_INT,0,MPI_COMM_WORLD);

    char* cbuff = new char[strlng];
    if (world_rank==0)
      strcpy(cbuff,v_file_name.c_str());
    MPI_Bcast(&cbuff[0],strlng,MPI_CHAR,0,MPI_COMM_WORLD);
    if (world_rank>0)
      v_file_name_list.push_back(std::string(cbuff));
    delete[] cbuff;
  }

  for (size_t j=0; j<v_file_name_list.size(); ++j)
  {
    V_data.push_back( new Themis::data_visibility(v_file_name_list[j],"HH") );
 
    if (Reconstruct_gains)
    {
      lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(*V_data[j],scattimage,station_codes,station_gain_priors) );
      L.push_back( lvg[j] );
    }
    else
    {
      lv.push_back( new Themis::likelihood_visibility(*V_data[j],scattimage) );
      L.push_back( lv[j] );
    }
  }


  image.write_model_tag_file();

  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means;
  std::vector<std::string> var_names;

  double uas2rad = M_PI/180.0/3600.0/1e6;
  // RIAF params
  //1. Black hole mass in Msun
  P.push_back( new Themis::prior_linear(3e6,5e6) );
  means.push_back(4.25e6);
  var_names.push_back("Mass");
  
  //2. Black hole spin parameter
  P.push_back( new Themis::prior_linear(0,0.998) );
  means.push_back(0.1);
  var_names.push_back("spin");
  
  //3. Cosine of inclination
  P.push_back( new Themis::prior_linear(-1,1) );
  means.push_back(-0.5);
  var_names.push_back("cos(Theta)");
  
  //4. electron density normalization
  P.push_back( new Themis::prior_logarithmic(1e6,1e8) );
  means.push_back(6e6);
  var_names.push_back("nth_e");
  
  //5.electron radial power law index for accretion
  P.push_back( new Themis::prior_linear(-1.5,0.0) );
  means.push_back(-0.92);
  var_names.push_back("s_th");
  
  //6. electron height ratio h/r
  P.push_back( new Themis::prior_linear(1e-5,2.0) );
  means.push_back(0.5);
  var_names.push_back("H/R");
  
  //7. electron temperature normalization
  P.push_back( new Themis::prior_logarithmic(8e9,1.5e11) );
  means.push_back(9.3e10);
  var_names.push_back("T_e");
  
  //8. electron temperature radial index
  P.push_back( new Themis::prior_linear(-1.0,0.0) );
  means.push_back(-0.46);
  var_names.push_back("s_T");
  
  //9. non-thermal electron normalization
  P.push_back( new Themis::prior_logarithmic(1e4,1e6) );
  means.push_back(1e5);
  var_names.push_back("n_nth");
  
  //10. non-thermal electron radial power-law index
  P.push_back( new Themis::prior_linear(-3.5, -1.0) );
  means.push_back(-2.02);
  var_names.push_back("s_nth");
  
  //11. non-thermal electron h/r
  P.push_back( new Themis::prior_linear(0.0,2.00) );
  means.push_back(1.0);
  var_names.push_back("H/R_nth");
  
  //12. Radial infall parameter
  P.push_back( new Themis::prior_linear(0.0,1.0) );
  means.push_back(0.05);
  var_names.push_back("infall");
  
  //13. sukep factor
  P.push_back(new Themis::prior_linear(1e-3,1.0));
  means.push_back(0.90);
  var_names.push_back("subkep");
  
  //14. position angle
  P.push_back(new Themis::prior_linear(-2*M_PI,2*M_PI));
  means.push_back(M_PI/4.0);
  var_names.push_back("pos");
    
  //15. x position
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  var_names.push_back("x0");
  
  //15. y position
  P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
  means.push_back(0.0);
  var_names.push_back("y0");
  
  if ( add_G )
  {
    //16. Igauss
    P.push_back(new Themis::prior_linear(0,10));
    means.push_back(0.3);
    var_names.push_back("Ig");
  
  
    //17. gaussian size
    P.push_back(new Themis::prior_linear(100*uas2rad,1e6*uas2rad));
    means.push_back(1e-6);
    var_names.push_back("sigma_g");
  
    //18. x position
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    var_names.push_back("x0_g");
  
    //19. y position
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    var_names.push_back("y0_g");
  }
  ////////////////////////  Before checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check ==============================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "\n";
    for (size_t k=0; k<image.size(); ++k)
      std::cerr << std::setw(15) << means[k]
		<< '\n';
    std::cerr << "===========================================================\n";
  }
  //////////////////////////////////

  if (Number_start_params>int(means.size()))
  {
    std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
    std::exit(1);
  }

  if (start_parameter_list.size()>0)
  {
    for (int j=0; j<Number_start_params; ++j)
    {
      // Set desired means
      means[j] = start_parameter_list[j];
    }
  }
  if (world_rank==0)
  {
    
    std::cout << "mass   :" << means[0] << std::endl
              << "spin   :" << means[1] << std::endl
              << "cosInc :" << means[2] << std::endl
              << "ne_th  :" << means[3] << std::endl
              << "s_th   :" << means[4] << std::endl
              << "h/r    :" << means[5] << std::endl
              << "Te     :" << means[6] << std::endl
              << "s_Te   :" << means[7] << std::endl
              << "ne_nnth:" << means[8] << std::endl
              << "s_nnth :" << means[9] << std::endl
              << "h/rnnth:" << means[10] << std::endl
              << "infall :" << means[11] << std::endl
              << "subkep :" << means[12] << std::endl
              << "posang :" << means[13] << std::endl
              << "xpos   :" << means[14] << std::endl
              << "ypos   :" << means[15] << std::endl;
    if ( add_G )
      std::cout << "IG     :" << means[16] << std::endl
                << "sigma  :" << means[17] << std::endl
                << "xpos   :" << means[18] << std::endl
                << "ypos   :" << means[19] << std::endl
                << std::endl;
  }
  ////////////////////// After checking for initial data
  if (world_rank==0)
  {
    std::cerr << "Priors check after =========================================\n";
    std::cerr << "Image size: " << image.size()
	      << "  Priors size: " << P.size()
	      << "  means size: " << means.size()
	      << "\n";
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
  for (size_t j=0; j<V_data.size(); ++j)
    Ndata += 2*V_data[j]->size();
  int Nparam = image.size() - 2;
  if ( add_G )
    Nparam -= 2;
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);
  if (Reconstruct_gains)
  {
    for (size_t j=0; j<lvg.size(); ++j)
    {
      Ngains_list[j] = lvg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  }
  int NDoF = Ndata - Nparam - Ngains;
  

  // Get the best point
  std::vector<double> pbest =  means;
  Themis::optimizer_kickout_powell opt_obj(seed+world_rank+10*world_size);
  if (preoptimize_flag)
  {
    if (Reconstruct_gains)
      for (size_t j=0; j<lvg.size(); ++j)
	lvg[j]->set_iteration_limit(20);
      
    opt_obj.set_cpu_distribution(Number_per_likelihood);
    opt_obj.set_kickout_parameters(opt_ko_llrf,opt_ko_itermax,opt_ko_rounds);
    means = opt_obj.run_optimizer(L_obj, Ndata, means, "PreOptimizeSummary.dat",opt_instances);

    if (Reconstruct_gains)
      for (size_t j=0; j<lvg.size(); ++j)
	lvg[j]->set_iteration_limit(50);
    
    pbest = means;
  }
  
  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed, L_temp, var_names, means.size());
  
  //Get the exploration sampler and set some options
  Themis::sampler_automated_factor_slice_sampler_MCMC* skernel = DEO.get_sampler();
  skernel->set_window_parameters(init_buffer, window);
  skernel->set_adaptation_parameters(number_of_adaptation, save_adapt);
  // Set a checkpoint
  DEO.set_checkpoint(5,"MCMC.ckpt");

  // Set annealing schedule
  DEO.set_annealing_schedule(initial_ladder_spacing); 
  DEO.set_deo_round_params(Number_of_swaps, Temperature_stride);
  
  //Set the initial location of the sampler
  DEO.set_initial_location(means);

  //Set the annealing output
  std::string anneal_name = "annealing_afss.dat";
  DEO.set_annealing_output(anneal_name);

  std::stringstream chain_file_name, state_file_name, sampler_file_name;
  chain_file_name << "chain_deo_afss.dat";
  state_file_name << "state_deo_afss.dat";
  sampler_file_name << "sampler_deo_afss.dat";
  //Set the output
  int out_precision = 8;
  DEO.set_output_stream(chain_file_name.str(), state_file_name.str(), sampler_file_name.str(), out_precision);
 
  if (restart_flag)
    DEO.read_checkpoint("MCMC.ckpt");


  int Number_temperatures = world_size/Number_per_likelihood;
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
  
  
  // Start looping over repetitions
  // Extra loop is to make final files
  pbest = means;
  for (size_t rep=0; rep<=number_of_rounds; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started round " << rep << std::endl;

    // Generate fit summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3) 
               << rep << "_fit_summaries" << ".txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index";
      for (size_t k=0; k<image.size(); ++k)
      {
	sumout << std::setw(15) << var_names[k];
      }
      for (size_t j=0; j<L.size(); ++j) 
	sumout << std::setw(15) << "V" + std::to_string(j) + " rc2";
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
      for (size_t k=0; k<image.size(); ++k)
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
  
    std::stringstream riaf_name; 
    riaf_name << "round" << std::setfill('0') << std::setw(3) << rep << "_riaf_image" << ".d";
    riaf.output_image(riaf_name.str(), true);
    
    // Run the sampler with the given settings
    if (rep<number_of_rounds) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	std::cerr << "Starting MCMC on round " << rep << std::endl;
      DEO.run_sampler( 1, thin_factor, refresh, verbosity);
      clock_t end = clock();
      if (world_rank == 0)
	std::cerr << "Done MCMC on round " << rep << std::endl
		  << "it took " << (end-start)/CLOCKS_PER_SEC/3600.0 << " hours" << std::endl;
    }
    
    // Reset pbest
    pbest = skernel->find_best_fit();
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
  
}
