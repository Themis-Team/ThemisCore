/*!
    \file sgrA_general_sspotriaf_fitting_diff.cpp
    \author Avery E Broderick & Paul Tiede
    \date Jan, 2019
    \brief Driver file for SgrA analysis with Broderick et al. 2016 RIAF model + shearing hotspot
    \details Takes file lists generated via something like:
	     These must be passed as -vm <file> and -cp <file> options.
	     Reads a single vm and cp file for a day, since SgrA changes on timescale of minutes
*/


#include "model_movie_general_riaf_shearing_spot.h"
#include "model_galactic_center_diffractive_scattering_screen.h"
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
  int Number_of_steps_A = 20000; 
  int Number_of_steps_B = 10000; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  //Underlying riaf params
  std::string riaf_param_file="";
  bool riaf_fix = false;
  double tbegin_hr = 0;
  size_t nframes = 12;
  size_t npix = 64;
  double fov = 60;

  
  //MCMC params
  bool restart_flag = false;
  int Number_of_tempering_levels=20;
  double Tempering_halving_time = 1000.0;
  double Tempering_ladder = 1.4;
  int Number_of_walkers=360;
  
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
    else if (opt == "-rp" || opt=="-riaf-parameter-file")
    {
      if (k+1 < argc)
      {
        int tmp = atoi(argv[k++]);
        if (tmp==1)
        {
          std::cout << "Fixing riaf fit params\n";
          riaf_fix=true;
        }
        else
        {
          std::cout << "Not fixing riaf fit params\n";
          riaf_fix=false;
        }
        riaf_param_file = std::string(argv[k++]);
      }
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: TWO arguments must be provided after -rp: < (1,0) i.e. whether to fix riaf or not > and riaf fit file" << std::endl;
        std::exit(1);
      }
    }
    else if (opt=="--parameter-file" || opt=="-p")
    {
      Number_start_params=19;
      if (k<argc)
      {
	param_file = std::string(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO arguments must be provided after --parameter-file, -p: < # of params to set> and file list.\n";
	std::exit(1);
      }
    }
    else if ( opt == "-t0")
    {
      if ( k < argc)
        tbegin_hr = atof(argv[k++]);
      else
      {
        if ( world_rank == 0)
          std::cerr << "ERROR: An float argument must be provided after -t0";
        std::exit(1);
      }
    }
    else if ( opt == "-nF"  )
    {
      if ( k < argc)
        nframes = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0)
          std::cerr << "ERROR: An int argument must be provided after -nF";
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
    else if (opt=="--tempering-levels")
    {
      if ( k<argc)
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
      if ( k<argc)
        Tempering_ladder = atof(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An int argument must be provided after --tempering-ladder.\n";
        std::exit(1);
      }
    }
    else if (opt=="--tempering-time")
    {
      if ( k<argc)
        Tempering_halving_time = atof(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An float argument must be provided after --tempering-time.\n";
        std::exit(1);
      }
    }
    else if (opt=="--walkers")
    {
      if ( k<argc)
        Number_of_walkers = atoi(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An int argument must be provided after --tempering-levels.\n";
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
      if (k<argc)
	npix = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after -npix.\n";
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
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for RIAF model\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 ./sgrA_riaf_fitting -vm vm_file -cp cp_file [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-vm, --visibility-amplitudes <string>\n"
		  << "\t\tSets the name of the file containing the of visibility amplitude files.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-cp, --closure-phases <string>\n"
		  << "\t\tSets the name of the file containing the closure phases.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-rp -riaf-parameter-file <int><string>\n"
                  << "\t\tReads in a general riaf model fit to modify some of the parameters of the RIAF.\n"
                  << "\t\tExpects an integer first, which specifies whether to fix the parameters.\n"
                  << "\t\t 1 means fix, and 0 means fit all the parameters.\n"
                  << "\t\t next requires the absolute path of the general riaf fits summaries file"
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
                  << "\t-t0 <float,hours>\n"
                  << "\t\tSets the starting time to make the first frame of the movie relative to\n"
                  << "\t\tthe starting time of the observation in hours. Defaults to the beginning of the observation\n"
                  << "\t-nF <int>\n"
                  << "\t\tSets the number of frames to make in a movie. Defaults to 6\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"
                  << "\t--walkers <int>\n"
                  << "\t\tSets the number of walkers to use. Default is 64.\n"
                  << "\t--tempering-levels <int>\n"
                  << "\t\tNumber of tempering levels to use. Default is 20.\n"
                  << "\t--tempering-ladder <float>\n"
                  << "\t\tGeometric temperature spacing level. Default is 1.5.\n"
                  << "\t--tempering-time <float>\n"
                  << "\t\tTempering halving time for freeze out of adaption. Default is 1000 steps.\n"
                  << "\t--restart\n"
                  << "\t\tTurns on the restart flag for the sampler. Will restart from the last checkpoint file saved.\n"
                  << "\t-npix <int>\n"
                  << "\t\tSets the number of pixels to be used in the image. DEFAULT 64, i.e. creates as 64x64 image.\n"
                  << "\t-fov <double>\n"
                  << "\t\tSets the field of view of the image in units of M. DEFAULT is 60M, i.e. creates a 60Mx60M image\n";
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

  std::vector<double> riaf_start_parameter_list;
  if (riaf_param_file!="")
  {
    std::fstream pfin(riaf_param_file);
    if (!pfin.is_open())
    {
      std::cerr << "ERROR: Could not open " << riaf_param_file << '\n';
      std::exit(1);
    }

    double dtmp;
    pfin.ignore(4096,'\n');
    // Get first index
    pfin >> dtmp;
    for (int k=0; k<14; k++)
    {
      pfin >> dtmp;
      riaf_start_parameter_list.push_back(dtmp);
    }
    // Kill remainder of line
    pfin.ignore(4096,'\n');
    pfin.close();
  }
  else
  {
    std::cerr << "Must pass a riaf fit parameter list!";
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


  if (world_rank==0)
  {
    if (riaf_start_parameter_list.size()>0)
    {
      std::cout << "riaf starting parameters list: (" << riaf_start_parameter_list.size() << ")\n";
      std::cout << "-------------------------------------------------\n" << std::endl;
      std::cout << "mass   :" << riaf_start_parameter_list[0] << std::endl
                << "spin   :" << riaf_start_parameter_list[1] << std::endl
                << "cosInc :" << riaf_start_parameter_list[2] << std::endl
                << "ne_th  :" << riaf_start_parameter_list[3] << std::endl
                << "s_th   :" << riaf_start_parameter_list[4] << std::endl
                << "h/r    :" << riaf_start_parameter_list[5] << std::endl
                << "Te     :" << riaf_start_parameter_list[6] << std::endl
                << "s_Te   :" << riaf_start_parameter_list[7] << std::endl
                << "ne_nnth:" << riaf_start_parameter_list[8] << std::endl
                << "s_nnth :" << riaf_start_parameter_list[9] << std::endl
                << "h/rnnth:" << riaf_start_parameter_list[10] << std::endl
                << "infall :" << riaf_start_parameter_list[11] << std::endl
                << "subkep :" << riaf_start_parameter_list[12] << std::endl
                << "posang :" << riaf_start_parameter_list[13] << std::endl
                << std::endl;
    }

  }


 
  
  // Prepare the output summary file
  std::stringstream sumoutnameA, sumoutnameB;
  sumoutnameA << "fit_summaries_A_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << 0 << ".txt";
  sumoutnameB << "fit_summaries_B_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << 0 << ".txt";
  std::ofstream sumoutA, sumoutB;
  if (world_rank==0)
  {
    sumoutA.open(sumoutnameA.str().c_str());
    sumoutA << std::setw(10) << "# index";
    sumoutA << std::setw(15) << "m (Ms)"
	    << std::setw(15) << "spin"
	    << std::setw(15) << "cosinc"
            << std::setw(15) << "ne_s"
            << std::setw(15) << "R_s"
            << std::setw(15) << "t0_s"
            << std::setw(15) << "r0_s"
            << std::setw(15) << "phi0_s"
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
            << std::setw(15) << "IG (Jy)"
            << std::setw(15) << "sG (Rad)"
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
            << std::setw(15) << "ne_s"
            << std::setw(15) << "R_s"
            << std::setw(15) << "t0_s"
            << std::setw(15) << "r0_s"
            << std::setw(15) << "phi0_s"
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
            << std::setw(15) << "IG (Jy)"
            << std::setw(15) << "sG (Rad)"
    sumoutB << std::setw(15) << "va red. chisq"
            << std::setw(15) << "cp red. chisq"
	    << std::setw(15) << "red. chisq"
	    << std::setw(15) << "log-liklhd"
	    << "     filename"
	    << std::endl;
    
  }

  
  // Read in data
  Themis::data_visibility_amplitude VM(vm_file,"HH");
  Themis::data_closure_phase CP(cp_file,"HH", false);
  
  //Set up observing times for the spot, not we will pick the begining and end as fixed points
  double t0 = VM.datum(0).tJ2000;
  double tstart = t0 + tbegin_hr*3600.0;
  double tend = VM.datum(VM.size()-1).tJ2000;
  if (world_rank == 0)
    std::cout << "First frame made at " << tbegin_hr 
              << " hrs after the start time " << t0/3600.0 
              << " end time " << tend/3600.0
              << std::endl;

  //Now select the time we want to create a frame.
  //Selecting this correct is tricky because we need to capture the flare.
  //
  double dt = (tend-tstart)/(nframes-1);
  std::vector<double> observing_times;
  for (int i = 0; i < nframes; i++)
    observing_times.push_back(tstart+dt*i);
  
  Themis::model_movie_general_riaf_shearing_spot movie(tstart, observing_times);
  movie.set_image_resolution(npix);  
  movie.set_screen_size(fov/2.0); 

  //Blur the movie
  Themis::model_galactic_center_diffractive_scattering_screen diff_movie(movie);



  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  std::vector<Themis::transform_base*> T;
 

  // RIAF params
  //Black hole mass in Msun
  P.push_back( new Themis::prior_linear(2e6,6e6) );
  means.push_back(riaf_start_parameter_list[0]);
  ranges.push_back(1e5);
  var_names.push_back("M(Ms)");
  T.push_back(new Themis::transform_none());
  // Black hole spin parameter
  P.push_back( new Themis::prior_linear(0,0.998) );
  means.push_back(riaf_start_parameter_list[1]);
  ranges.push_back(0.3);
  T.push_back(new Themis::transform_none());
  var_names.push_back("spin");
  // Cosine of inclination
  P.push_back( new Themis::prior_linear(-1,1) );
  means.push_back(riaf_start_parameter_list[2]);
  ranges.push_back(0.3);
  T.push_back(new Themis::transform_none());
  var_names.push_back("cosI");
  //spot electron density normalization
  P.push_back( new Themis::prior_logarithmic(1e4,1e8));
  means.push_back(2.5e6);
  ranges.push_back(5e5);
  T.push_back(new Themis::transform_none());
  var_names.push_back("ne_s");
  //spot size in M
  P.push_back( new Themis::prior_linear(1e-2,5));
  means.push_back(1.0);
  ranges.push_back(0.5);
  T.push_back(new Themis::transform_none());
  var_names.push_back("Rs(M)");
  //Initial spot time (s) relative to start of observation
  P.push_back( new Themis::prior_linear(-3600*2,(tend-tstart)+3600*2.0));
  means.push_back(0.0);
  ranges.push_back(3600);
  T.push_back(new Themis::transform_none());
  var_names.push_back("t0rel(sec)");
  //initial spot radius in M
  P.push_back( new Themis::prior_linear(2,20));
  means.push_back(10.0);
  ranges.push_back(2.0);
  T.push_back(new Themis::transform_none());
  var_names.push_back("r0(M)");
  //initial spot azimuthal location in radians
  P.push_back( new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0);
  ranges.push_back(1.0);
  T.push_back(new Themis::transform_none());
  var_names.push_back("phi0(rad)");
  //electron density normalization
  P.push_back( new Themis::prior_logarithmic(1e4,1e9) );
  means.push_back(riaf_start_parameter_list[3]); //Set to half of what Sgr A 2016 fitting says
  ranges.push_back(1e6);
  T.push_back(new Themis::transform_none());
  var_names.push_back("ne_th");
  //electron radial power law index for accretion
  P.push_back( new Themis::prior_linear(-2.5,-0.5) );
  means.push_back(riaf_start_parameter_list[4]);
  ranges.push_back(0.1);
  var_names.push_back("sr_th");
  //electron height ratio h/r
  P.push_back( new Themis::prior_linear(1e-5,1.0) );
  means.push_back(riaf_start_parameter_list[5]);
  ranges.push_back(0.1);
  var_names.push_back("h/r");
  //electron temperature normalization
  P.push_back( new Themis::prior_logarithmic(1e9,1e12) );
  means.push_back(riaf_start_parameter_list[6]);
  ranges.push_back(1e8);
  var_names.push_back("Te");
  //electron temperature radial index
  P.push_back( new Themis::prior_linear(-1.5,0) );
  means.push_back(riaf_start_parameter_list[7]);
  ranges.push_back(0.1);
  var_names.push_back("sr_Te");
  //non-thermal electron normalization
  P.push_back( new Themis::prior_logarithmic(1e-5,5e6) );
  means.push_back(riaf_start_parameter_list[8]);
  ranges.push_back(1e-5);
  var_names.push_back("ne_nnth");
  //non-thermal electron radial power-law index
  P.push_back( new Themis::prior_linear(-2.2, -1) );
  means.push_back(riaf_start_parameter_list[9]);
  ranges.push_back(1e-5);
  var_names.push_back("sr_nnth");
  //non-thermal electron h/r
  P.push_back( new Themis::prior_linear(0.0,2.001) );
  means.push_back(riaf_start_parameter_list[10]);
  ranges.push_back(1e-4);
  var_names.push_back("h/r_nnth");
  //Radial infall parameter
  P.push_back( new Themis::prior_linear(0,1) );
  means.push_back(riaf_start_parameter_list[11]);
  ranges.push_back(1e-2);
  var_names.push_back("alphaR");
  //sukep factor
  P.push_back(new Themis::prior_linear(0,1));
  means.push_back(riaf_start_parameter_list[12]);
  ranges.push_back(0.1);
  var_names.push_back("subkep");
  //position angle
  P.push_back(new Themis::prior_linear(-2*M_PI,2*M_PI));
  means.push_back(riaf_start_parameter_list[13]);
  ranges.push_back(1.0);
  var_names.push_back("posang(rad)");

   
  if (Number_start_params>0 && Number_start_params!=int(means.size()))
  {
    std::cerr << "ERROR: Not equal start parameters provided for chosen model.\n";
    std::exit(1);
  }
  for (size_t j=0; j<start_parameter_list.size(); ++j)
  {
    // Set desired means
    means[j] = start_parameter_list[j];
    // Restrict initial ranges
    ranges[j] *= 1e-3;
  }
  
  //Now choose whether to fix the riaf parameters
  if (riaf_fix)
  {
    size_t preSize = T.size();
    for ( size_t i = preSize; i < means.size()-3; i++)
      T.push_back(new Themis::transform_fixed(means[i]));
    //radial infall
    T.push_back(new Themis::transform_none());
    //subkep 
    T.push_back(new Themis::transform_none());
    //pos_ang 
    T.push_back(new Themis::transform_fixed(means.size()-1));
  }
  else 
  {
    //Themal electron params
    for (size_t i = 0; i < 4; i++)
      T.push_back(new Themis::transform_none());
    //Non-thermal electrons
    size_t preSize = T.size();
    for ( size_t i = preSize; i < means.size()-3; i++)
      T.push_back(new Themis::transform_fixed(means[i])); 
    //radial infall
    T.push_back(new Themis::transform_none());
    //subkep 
    T.push_back(new Themis::transform_none());
    //pos_ang 
    T.push_back(new Themis::transform_none());
  }
  //Now let's make sure I didn't screw anything up
  if (world_rank==0)
    std::cout << "means  size  " << means.size() << std::endl
              << "ranges size " << ranges.size() << std::endl
              << "prior size " << P.size() << std::endl
              << "tranf  size " << T.size() << std::endl;
  //Print read in means and starting location
  if ( world_rank==0)
  {
    std::cout << "Starting location of chains\n";
    for ( size_t i = 0; i < means.size(); i++)
      std::cout << means[i] << std::endl; 
  }
  // Set the likelihood functions
  // Visibility Amplitudes
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  Themis::likelihood_optimal_gain_correction_visibility_amplitude lva(VM,diff_movie,station_codes,station_gain_priors);
  // Closure Phases
  Themis::likelihood_closure_phase lcp(CP,movie);
    
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
  movie.generate_model(means);
  L_obj(means);

  lva.output_gain_corrections(gc_name.str());
  L[0]->output_model_data_comparison(VA_res_name.str());
  L[1]->output_model_data_comparison(CP_res_name.str());
    
  // Create a sampler object
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);

  // Generate a chain
  int Number_of_chains = Number_of_walkers;
  int Number_of_temperatures =  Number_of_tempering_levels;
  int Number_of_procs_per_lklhd = 8;
  int Temperature_stride = 50;
  int Chi2_stride = 40;
  int Ckpt_frequency = 75;
  int out_precision = 8;
  int verbosity = 0;
  
  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");

  // Set tempering schedule
  MCMC_obj.set_tempering_schedule(Tempering_halving_time,1.,Tempering_ladder);
    

  if (world_rank==0)
    std::cout << "Setting MCMC parameters----------------\n"
              << "Number of walker  :  " << Number_of_chains << std::endl
              << "Number of temps   : " << Number_of_temperatures << std::endl
              << "Temp halving time : " << Tempering_halving_time << std::endl
              << "Temp ladder space : " << Tempering_ladder << std::endl;
  
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
  movie.generate_model(pmax);
  L_obj(pmax);

  std::cerr << "Read pmax run A\n";

  int Ndof_VM = int(VM.size()) - int(lva.number_of_independent_gains()); 
  int Ndof_CP = int(CP.size());
  int Ndof = Ndof_VM + Ndof_CP - int(movie.size());

  double chi2_VM = L[0]->chi_squared(pmax);
  double chi2_CP = L[1]->chi_squared(pmax);
  double chi2 = L_obj.chi_squared(pmax);
  double Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutA << std::setw(10) << index;
    for (size_t j=0; j<movie.size(); ++j)
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
  for (size_t j=0; j<movie.size(); ++j)
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
  movie.generate_model(pmax);

  chi2_VM = L[0]->chi_squared(pmax);
  chi2_CP = L[1]->chi_squared(pmax);
  chi2 = L_obj.chi_squared(pmax);
  Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutB << std::setw(10) << index;
    for (size_t j=0; j<movie.size(); ++j)
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
