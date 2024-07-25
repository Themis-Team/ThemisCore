/*!
    \file m87_ffj_fitting.cpp
    \author Avery E Broderick & Paul Tiede
    \date Jan, 2019
    \brief Driver file for M87 analysis with Broderick et al. 2009 FFJ model which is static
    \details Takes file lists generated via something like:
	     These must be passed as -vm <file> and -cp <file> options.
	     Reads a single vm and cp file for a day, since SgrA changes on timescale of minutes
*/


#include "model_image_force_free_jet.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "utils.h"
#include "constants.h"
#include "vrt2.h"

#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  std::string vm_file="", cp_file="";
  int istart = 0;
  int iend = -1;
  int Number_of_steps_A = 20000; 
  int Number_of_steps_B = 10000; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  bool restart_flag = false;
  size_t npix = 64;
  size_t nref = 0;
  double fov = 60;
  size_t nwalkers = 64; //number of walkers
  size_t ntemps = 8; //number of tempering levels must be greater than 1
  size_t nlklhd = 8; //Number of cores per likelihood
  size_t tempering_time_scale = 200; //Freeze out time of tempering
  size_t tempering_spacing = 2.0; //Geometric spacing between tempering levels

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
	  std::cerr << "ERROR: TWO arguments must be provided after --parameter-file, -p: < # of params to set> and file list.\n";
	std::exit(1);
      }
    }
    else if (opt=="-NA")
    {
      if (k<argc)
	Number_of_steps_A = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -NA.\n";
	std::exit(1);
      }
    }
    else if (opt=="-NB")
    {
      if (k<argc)
	Number_of_steps_B = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -NB.\n";
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
    else if ( opt == "--restart")
    {
      restart_flag = true;
      if (world_rank == 0)
        std::cout << "Turning on restart flag will restart the Chain from the last checkpoint\n";
    
    }
    else if (opt=="-npix")
    {
      if (k+1<argc)
      {
	npix = atoi(argv[k++]);
        nref = atoi(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: Two int arguments must be provided after -npix.\n";
	std::exit(1);
      }
    }
    else if (opt=="-fov")
    {
      if (k<argc)
	fov = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: A double argument must be provided after -fov.\n";
	std::exit(1);
      }
    }
    else if (opt=="-nwalkers")
    {
      if (k<argc)
        nwalkers = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -nwalkers.\n";
	std::exit(1);
      }
    }
    else if (opt=="-ntemps")
    {
      if (k<argc)
      {

        ntemps = atoi(argv[k++]);
      
        if (ntemps <= 1)
        {
          std::cerr <<"Number of tempering levels must be greater than unity\n";
          std::exit(1);
        }
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -ntemps.\n";
	std::exit(1);
      }
    }
    else if (opt=="-sTemp")
    {
      if (k<argc)
        tempering_spacing = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An double argument must be provided after -sTemp.\n";
	std::exit(1);
      }
    }
    else if (opt=="-fTemp")
    {
      if (k<argc)
        tempering_time_scale = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -fTemp.\n";
	std::exit(1);
      }
    } 
    else if (opt=="-nlklhd")
    {
      if (k<argc)
        nlklhd = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -nlklhd.\n";
	std::exit(1);
      }
    }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for FFJ model\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 ./m87_ffj_fitting -vm vm_file -cp cp_file [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-vm, --visibility-amplitudes <string>\n"
		  << "\t\tSets the name of the file containing the of visibility amplitude files.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-cp, --closure-phases <string>\n"
		  << "\t\tSets the name of the file containing the closure phases.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "DESCRIPTION\n"
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
		  << "\t-e, --end <int>\n"
		  << "\t\tSets the end index (plus 1), beginning with 1, in the vm_file, cp_file,\n"
		  << "\t\tand param_file list (if provided) to begin running.  So \"-s 3 -f 5\" will run\n"
		  << "\t\tindexes 3 and 4 and then stop.\n"
		  << "\t-p, -parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt. \n" 
		  << "\t-NA <int>\n"
		  << "\t\tSets the number of MCMC steps to take for chain A.  Defaults to 20000.\n"
		  << "\t-NB <int>\n"
		  << "\t\tSets the number of MCMC steps to take for chain B.  Defaults to 10000.\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"
                  << "\t--restart\n"
                  << "\t\tTurns on the restart flag for the sampler. Will restart from the last checkpoint file saved.\n"
                  << "\t-npix <int,int>\n"
                  << "\t\tSets the number of pixels to be used in the image and number of refines to use. DEFAULT 64, 0, i.e. creates as 64x64 image with no refines.\n"
                  << "\t-fov <double>\n"
                  << "\t\tSets the field of view of the image in units of M. DEFAULT is 60M, i.e. creates a 60Mx60M image\n"
                  << "\t-nwalkers <int>\n"
                  << "\t\tSets the number of walkers to use. DEFAULT is 64\n"
                  << "\t-ntemps <int>\n"
                  << "\t\tSets the number of temperatures to use, must be > 1. DEFAULT is 8\n"
                  << "\t-sTemp <double>\n"
                  << "\t\tSets the geometric spacing factor for the temperatures. DEFAULT is 2\n"
                  << "\t-fTemp <int>\n"
                  << "\t\tSets the tempering freeze out timescale. DEFAULT is 200 steps\n."
                  << "\t-nlklhd <int>\n"
                  << "\t\tSets the number of processors to use per likelihood evaluation. DEFAULT is 8.\n";

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

  if (vm_file=="" || cp_file=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No data file was provided. The -vm <string> and\n"
		<< "       -cp <string> options are *required*.  See -h for more\n"
		<< "       details and options.\n";
    std::exit(1);
  }

  if (world_rank==0)
  {
    std::cout << "Using cpu distribution:\n"
              << "\tnumber of walkers        : " << nwalkers << std::endl
              << "\tnumber of temps          : " << ntemps << std::endl
              << "\tnumber of procs per lklhd: " << nlklhd << std::endl << std::endl;
    std::cout << "Using tempering settings:\n"
              << "\ttempering time scale: " << tempering_time_scale << std::endl
              << "\ttempering spacing   : " << tempering_spacing << std::endl;
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
    pfin.ignore(4096,'\n');
    // Get first index
    pfin >> dtmp;
    for (int k=0; k<Number_start_params; k++)
    {
	pfin >> dtmp;
	start_parameter_list.push_back(dtmp);
    }
    // Kill remainder of line
    pfin.ignore(4096,'\n');
     
  }  

  //  Output these for check
  if (world_rank==0)
  {
    if (Number_start_params>0)
      std::cout << "start parameters: (" << start_parameter_list.size() << ")\n";
    std::cout << "---------------------------------------------------\n" << std::endl;
  }
 
  
  // Prepare the output summary file
  std::stringstream sumoutnameA, sumoutnameB;
  sumoutnameA << "fit_summaries_A_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  sumoutnameB << "fit_summaries_B_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  std::ofstream sumoutA, sumoutB;
  if (world_rank==0)
  {
    sumoutA.open(sumoutnameA.str().c_str());
    sumoutA << std::setw(10) << "# index";
    sumoutA << std::setw(15) << "m (Ms)"
	    << std::setw(15) << "spin"
	    << std::setw(15) << "cosinc"
            << std::setw(15) << "ne_th"
            << std::setw(15) << "s_th"
            << std::setw(15) << "h/r"
            << std::setw(15) << "Te"
            << std::setw(15) << "s_Te"
            << std::setw(15) << "ne_nnth"
            << std::setw(15) << "s_nnth"
            << std::setw(15) << "h/r_nnth"
	    << std::setw(15) << "infall"
	    << std::setw(15) << "subkep"
	    << std::setw(15) << "pos_ang (rad)"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos"
            << std::setw(15) << "Ia"
            << std::setw(15) << "sigma"
            << std::setw(15) << "asym"
            << std::setw(15) << "phi"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos"
	    << std::setw(15) << "Ig"
	    << std::setw(15) << "sig (rad)"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos";
    sumoutA << std::setw(15) << "va red. chisq"
            << std::setw(15) << "cp red. chisq"
	    << std::setw(15) << "red. chisq"
	    << std::setw(15) << "log-liklhd"
	    << "     filename"
	    << std::endl;

    
    sumoutB.open(sumoutnameB.str().c_str());
    sumoutB << std::setw(10) << "# index";
    sumoutB << std::setw(15) << "m (Ms)"
	    << std::setw(15) << "spin"
	    << std::setw(15) << "cosinc"
            << std::setw(15) << "ne_th"
            << std::setw(15) << "s_th"
            << std::setw(15) << "h/r"
            << std::setw(15) << "Te"
            << std::setw(15) << "s_Te"
            << std::setw(15) << "ne_nnth"
            << std::setw(15) << "s_nnth"
            << std::setw(15) << "h/r_nnth"
	    << std::setw(15) << "infall"
	    << std::setw(15) << "subkep"
	    << std::setw(15) << "pos_ang (rad)"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos"
            << std::setw(15) << "Ia"
            << std::setw(15) << "sigma"
            << std::setw(15) << "asym"
            << std::setw(15) << "phi"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos"
	    << std::setw(15) << "Ig"
	    << std::setw(15) << "sig (rad)"
            << std::setw(15) << "xpos"
            << std::setw(15) << "ypos";
    sumoutB << std::setw(15) << "va red. chisq"
            << std::setw(15) << "cp red. chisq"
	    << std::setw(15) << "red. chisq"
	    << std::setw(15) << "log-liklhd"
	    << "     filename"
	    << std::endl;
    
  }


  //FFJ model at M87 distance
  Themis::model_image_force_free_jet ffj(230e9);
  ffj.set_image_resolution(npix, nref);
  ffj.set_screen_size(fov/2.0);

  //Large scale gaussian
  Themis::model_image_symmetric_gaussian gauss;
  gauss.use_analytical_visibilities();

  Themis::model_image_sum image;
  image.add_model_image(ffj);
  image.add_model_image(gauss);

  // Read in data
  Themis::data_visibility_amplitude VM(vm_file,"HH");
  Themis::data_closure_phase CP(cp_file,"HH");

  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  std::vector<Themis::transform_base*> T;


  // FFJ params
  //1. Black hole mass in Msun
  P.push_back( new Themis::prior_logarithmic(1e9,9e9) );
  means.push_back(6.5e9);
  ranges.push_back(1.0e9);
  T.push_back(new Themis::transform_none());
  
  //2. Distance in kpc
  P.push_back( new Themis::prior_logarithmic(2e3,25e3) );
  means.push_back(16.9e3);
  ranges.push_back(2e9);
  T.push_back(new Themis::transform_none());
  
  //3. Black hole spin parameter
  P.push_back( new Themis::prior_linear(0,0.998) );
  means.push_back(0.8);
  ranges.push_back(0.2);
  T.push_back(new Themis::transform_none());
  
  //4. Cosine of inclination
  P.push_back( new Themis::prior_linear(-1,1) );
  means.push_back(std::cos(M_PI/180.0*163.0));
  ranges.push_back(0.2);
  T.push_back(new Themis::transform_none());
  
  //5. radial power law
  P.push_back( new Themis::prior_linear(0.0,2.0) );
  means.push_back(2.0/3);
  ranges.push_back(0.5);
  T.push_back(new Themis::transform_none());

  //6. opening angle
  P.push_back( new Themis::prior_linear(5.0,30.0));
  means.push_back(10.0);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());

  //7. rload
  P.push_back( new Themis::prior_linear(2.5,20.0) );
  means.push_back(4.0);
  ranges.push_back(1.0);
  T.push_back(new Themis::transform_none());

  //8. Gamma inf
  P.push_back( new Themis::prior_linear(1,1e5) );
  means.push_back(5.0);
  ranges.push_back(1.0);
  T.push_back(new Themis::transform_fixed(5.0));

  //9. electron density normalization
  P.push_back( new Themis::prior_logarithmic(1e-1,1e2) );
  means.push_back(10.0);
  ranges.push_back(5.0);
  T.push_back(new Themis::transform_none());
  
  //10.electron energy spectral index
  P.push_back( new Themis::prior_linear(0.0,2.0) );
  means.push_back(1.19);
  ranges.push_back(0.5);
  T.push_back(new Themis::transform_fixed(1.19));
   
  //11. magnetic field normalization
  P.push_back( new Themis::prior_linear(10,500) );
  means.push_back(100);
  ranges.push_back(20);
  T.push_back(new Themis::transform_none());
  
  //12. electron temperature radial index
  P.push_back( new Themis::prior_linear(10,500) );
  means.push_back(100.0);
  ranges.push_back(10);
  T.push_back(new Themis::transform_fixed(100.0));

  //13. inner accretion disk edge
  P.push_back( new Themis::prior_linear(1.0,5.0) );
  means.push_back(1.01);
  ranges.push_back(1e-2);
  T.push_back(new Themis::transform_none());
  
  //14. position angle
  P.push_back(new Themis::prior_linear(-2*M_PI,2*M_PI));
  means.push_back(-1.78);
  ranges.push_back(0.5);
  T.push_back(new Themis::transform_none());
    
  //15. x position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  
  //16. y position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  
  //17. Igauss
  P.push_back(new Themis::prior_linear(0,10));
  means.push_back(0.6);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());
  //18. gaussian size
  P.push_back(new Themis::prior_linear(1e-8,1e-2));
  means.push_back(1e-3);
  ranges.push_back(1e-10);
  T.push_back(new Themis::transform_fixed(1e-3));
  //19. x position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  //20. y position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));


  if (Number_start_params>int(means.size()))
  {
    std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
    std::exit(1);
  }

  if (param_file!="")
  {
    for (int j=0; j<Number_start_params; ++j)
    {
      // Set desired means
      means[j] = start_parameter_list[j];
      // Restrict initial ranges
      ranges[j] *= 1e-2;
    }
  }
  if (world_rank==0)
  {
    
    std::cout << "mass    :" << means[0] << std::endl
              << "distance:" << means[1] << std::endl
              << "spin    :" << means[2] << std::endl
              << "cosInc  :" << means[3] << std::endl
              << "rad_pl  :" << means[4] << std::endl
              << "op_ang  :" << means[5] << std::endl
              << "rLoad   :" << means[6] << std::endl
              << "G_inf   :" << means[7] << std::endl
              << "n_e     :" << means[8] << std::endl
              << "s_e     :" << means[9] << std::endl
              << "B0      :" << means[10] << std::endl
              << "gam_min :" << means[11] << std::endl
              << "inner_e :" << means[12] << std::endl
              << "posang  :" << means[13] << std::endl
              << std::endl;
  }
  
  // Set the likelihood functions
  // Visibility Amplitudes
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  Themis::likelihood_optimal_gain_correction_visibility_amplitude lva(VM,image,station_codes,station_gain_priors);
  // Closure Phases
  Themis::likelihood_closure_phase lcp(CP,image);
    
  std::vector<Themis::likelihood_base*> L;
  L.push_back(&lva);
  L.push_back(&lcp);
    
  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);
  
  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);

  // Output residual data
  std::stringstream VA_res_name;
  std::stringstream gc_name;
  std::stringstream CP_res_name;
  int index=0;
  VA_res_name << "VA_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
  CP_res_name << "CP_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
  gc_name << "gain_corrections_" << std::setfill('0') << std::setw(5) << index << ".d";
  image.generate_model(means);
  L_obj(means);
  std::cout << "Starting Likelihood value " << L_obj.chi_squared(means) << std::endl;

  lva.output_gain_corrections(gc_name.str());
  L[0]->output_model_data_comparison(VA_res_name.str());
  L[1]->output_model_data_comparison(CP_res_name.str());
    
  // Create a sampler object
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);

  // Generate a chain
  int Number_of_chains = nwalkers;
  int Number_of_temperatures =  ntemps;
  int Number_of_procs_per_lklhd = nlklhd;
  int Temperature_stride = 25;
  int Chi2_stride = 60;
  int Ckpt_frequency = 75;
  int out_precision = 8;
  int verbosity = 0;
  
  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");

  // Set tempering schedule
  MCMC_obj.set_tempering_schedule(tempering_time_scale,1.,tempering_spacing);
    
  
  /////////////
  // First run from random positions
  // Run the Sampler
  std::stringstream ChainA_name, LklhdA_name, Chi2A_name;
  ChainA_name << "Chain_A_" << std::setfill('0') << std::setw(5) << index << ".dat";
  LklhdA_name << "Lklhd_A_" << std::setfill('0') << std::setw(5) << index << ".dat";
  Chi2A_name << "Chi2_A_" << std::setfill('0') << std::setw(5) << index << ".dat";
  MCMC_obj.run_sampler(L_obj, Number_of_steps_A, Temperature_stride, Chi2_stride, ChainA_name.str(), LklhdA_name.str(), Chi2A_name.str(), means, ranges, var_names, restart_flag, out_precision, verbosity);

  ////////////
  // Prepare for second run:
  // Get the best fit and produce residual/gain files
  std::vector<double> pmax = MCMC_obj.find_best_fit(ChainA_name.str(),LklhdA_name.str());
  ffj.generate_model(pmax);
  L_obj(pmax);

  std::cerr << "Read pmax run A\n";

  int Ndof_VM = int(VM.size()) - int(lva.number_of_independent_gains()); 
  int Ndof_CP = int(CP.size());
  int Ndof;
  Ndof = Ndof_VM + Ndof_CP - (int(image.size())-5-3); //just kill FFJ stuff and G stuff

  double chi2_VM = L[0]->chi_squared(pmax);
  double chi2_CP = L[1]->chi_squared(pmax);
  double chi2 = L_obj.chi_squared(pmax);
  double Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutA << std::setw(10) << index;
    for (size_t j=0; j<image.size(); ++j)
      sumoutA << std::setw(15) << pmax[j];
    sumoutA << std::setw(15) << chi2_VM/Ndof_VM
	    << std::setw(15) << chi2_CP/Ndof_CP
	    << std::setw(15) << chi2/Ndof
	    << std::setw(15) << Lmax
	    << "     " << vm_file
	    << "     " << cp_file
	    << std::endl;
  }
    
  lva.output_gain_corrections(gc_name.str());
  L[0]->output_model_data_comparison(VA_res_name.str());
  L[1]->output_model_data_comparison(CP_res_name.str());
    

  /////////
  // Set up the revised means and ranges
  for (size_t j=0; j<image.size(); ++j)
  {
    means[j] = pmax[j];
    if (Number_start_params==0)
      ranges[j] *= 1e-2;
  }
    
  // First run from random positions
  // Run the Sampler
  std::stringstream ChainB_name, LklhdB_name, Chi2B_name;
  ChainB_name << "Chain_B_" << std::setfill('0') << std::setw(5) << index << ".dat";
  LklhdB_name << "Lklhd_B_" << std::setfill('0') << std::setw(5) << index << ".dat";
  Chi2B_name << "Chi2_B_" << std::setfill('0') << std::setw(5) << index << ".dat";
  MCMC_obj.run_sampler(L_obj, Number_of_steps_B, Temperature_stride, Chi2_stride, ChainB_name.str(), LklhdB_name.str(), Chi2B_name.str(), means, ranges, var_names, restart_flag, out_precision, verbosity);



  ////////////
  // Record final results
  // Get the best fit and produce residual/gain files
  pmax = MCMC_obj.find_best_fit(ChainB_name.str(),LklhdB_name.str());
  ffj.generate_model(pmax);

  chi2_VM = L[0]->chi_squared(pmax);
  chi2_CP = L[1]->chi_squared(pmax);
  chi2 = L_obj.chi_squared(pmax);
  Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutB << std::setw(10) << index;
    for (size_t j=0; j<image.size(); ++j)
      sumoutB << std::setw(15) << pmax[j];
    sumoutB << std::setw(15) << chi2_VM/Ndof_VM
	    << std::setw(15) << chi2_CP/Ndof_CP
	    << std::setw(15) << chi2/Ndof
	    << std::setw(15) << Lmax
	    << "     " << vm_file
	    << "     " << cp_file
	    << std::endl;
  }
    
  lva.output_gain_corrections(gc_name.str());
  L[0]->output_model_data_comparison(VA_res_name.str());
  L[1]->output_model_data_comparison(CP_res_name.str());

  
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
  
}
