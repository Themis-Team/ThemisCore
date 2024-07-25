/*!
    \file sgrA_general_riaf_fitting_diff.cpp
    \author Avery E Broderick & Paul Tiede
    \date Jan, 2019
    \brief Driver file for SgrA analysis with Broderick et al. 2016 RIAF model which is static
    \details Takes file lists generated via something like:
	     These must be passed as -vm <file> and -cp <file> options.
	     Reads a single vm and cp file for a day, since SgrA changes on timescale of minutes
*/


#include "model_image_general_riaf.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"
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


  //RIAF model plus a large scale gaussian
  Themis::model_image_general_riaf riaf;
  riaf.set_image_resolution(npix, nref);
  riaf.set_screen_size(fov/2.0);

  //Large scale gaussian
  Themis::model_image_symmetric_gaussian gauss;
  gauss.use_analytical_visibilities();

  Themis::model_image_sum image;
  image.add_model_image(riaf);
  image.add_model_image(gauss);

  Themis::model_galactic_center_diffractive_scattering_screen diff_image(image);

  // Read in data
  Themis::data_visibility_amplitude VM(vm_file,"HH");
  Themis::data_closure_phase CP(cp_file,"HH", false);

  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  std::vector<Themis::transform_base*> T;


  // RIAF params
  //1. Black hole mass in Msun
  P.push_back( new Themis::prior_linear(3e6,5e6) );
  means.push_back(4.1e6);
  ranges.push_back(5e5);
  T.push_back(new Themis::transform_none());
  var_names.push_back("Mass");
  
  //2. Black hole spin parameter
  P.push_back( new Themis::prior_linear(0,0.998) );
  means.push_back(0.5);
  ranges.push_back(0.25);
  T.push_back(new Themis::transform_none());
  var_names.push_back("spin");
  
  //3. Cosine of inclination
  P.push_back( new Themis::prior_linear(-1,1) );
  means.push_back(-0.5);
  ranges.push_back(0.2);
  T.push_back(new Themis::transform_none());
  var_names.push_back("cos(Theta)");
  
  //4. electron density normalization
  P.push_back( new Themis::prior_logarithmic(1e6,1e9) );
  means.push_back(6e6);
  ranges.push_back(2e6);
  T.push_back(new Themis::transform_none());
  var_names.push_back("n^th_e");
  
  //5.electron radial power law index for accretion
  P.push_back( new Themis::prior_linear(-3.0,-0.5) );
  means.push_back(-0.92);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());
  var_names.push_back("s_th");
  
  //6. electron height ratio h/r
  P.push_back( new Themis::prior_linear(1e-5,1.0) );
  means.push_back(0.5);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());
  var_names.push_back("H/R");
  
  //7. electron temperature normalization
  P.push_back( new Themis::prior_logarithmic(1e9,1e12) );
  means.push_back(9.3e10);
  ranges.push_back(1e10);
  T.push_back(new Themis::transform_none());
  var_names.push_back("T_e");
  
  //8. electron temperature radial index
  P.push_back( new Themis::prior_linear(-1.0,0.0) );
  means.push_back(-0.46);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());
  var_names.push_back("s_T");
  
  //9. non-thermal electron normalization
  P.push_back( new Themis::prior_logarithmic(1e-5,5e6) );
  means.push_back(1e-2);
  ranges.push_back(1e-5);
  T.push_back(new Themis::transform_fixed(1e-2));
  var_names.push_back("n_nth");
  
  //10. non-thermal electron radial power-law index
  P.push_back( new Themis::prior_linear(-3, -1) );
  means.push_back(-2.02);
  ranges.push_back(1e-5);
  T.push_back(new Themis::transform_fixed(-2.02));
  var_names.push_back("s_nth");
  
  //11. non-thermal electron h/r
  P.push_back( new Themis::prior_linear(0.0,1.001) );
  means.push_back(1.0);
  ranges.push_back(1e-4);
  T.push_back(new Themis::transform_fixed(1.0));
  var_names.push_back("H/R_nth");
  
  //12. Radial infall parameter
  P.push_back( new Themis::prior_linear(0.0,1.0) );
  means.push_back(0.1);
  ranges.push_back(0.025);
  T.push_back(new Themis::transform_none());
  var_names.push_back("infall");
  
  //13. sukep factor
  P.push_back(new Themis::prior_linear(1e-3,1.0));
  means.push_back(0.90);
  ranges.push_back(0.05);
  T.push_back(new Themis::transform_none());
  var_names.push_back("subkep");
  
  //14. position angle
  P.push_back(new Themis::prior_linear(-2*M_PI,2*M_PI));
  means.push_back(M_PI/4.0);
  ranges.push_back(0.5);
  T.push_back(new Themis::transform_none());
  var_names.push_back("pos");
    
  //15. x position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  var_names.push_back("x0");
  
  //15. y position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  var_names.push_back("y0");
  
  //16. Igauss
  P.push_back(new Themis::prior_linear(0,10));
  means.push_back(0.3);
  ranges.push_back(0.1);
  T.push_back(new Themis::transform_none());
  var_names.push_back("Ig");
  
  
  //17. gaussian size
  P.push_back(new Themis::prior_linear(1e-8,1e-2));
  means.push_back(1e-6);
  ranges.push_back(1e-10);
  T.push_back(new Themis::transform_fixed(1e-6));
  var_names.push_back("sigma_g");
  
    //18. x position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));
  var_names.push_back("x0_g");
  
  //19. y position
  P.push_back(new Themis::prior_linear(-1e-12,1e-12));
  means.push_back(0.0);
  ranges.push_back(1e-14);
  T.push_back(new Themis::transform_fixed(0.0));;
  var_names.push_back("y0_g");


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
              << "ypos   :" << means[15] << std::endl
              << "IG     :" << means[16] << std::endl
              << "sigma  :" << means[17] << std::endl
              << "xpos   :" << means[18] << std::endl
              << "ypos   :" << means[19] << std::endl
              << std::endl;
  }
  
  // Set the likelihood functions
  // Visibility Amplitudes
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  // Specify the priors we will be assuming (to 20% by default)
  std::vector<double> station_gain_priors(station_codes.size(),0.2);
  station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  Themis::likelihood_optimal_gain_correction_visibility_amplitude lva(VM,diff_image,station_codes,station_gain_priors);
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
  diff_image.generate_model(means);
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
  int Temperature_stride = 50;
  int Chi2_stride = 100;
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
  riaf.generate_model(pmax);
  L_obj(pmax);

  std::cerr << "Read pmax run A\n";

  int Ndof_VM = int(VM.size()) - int(lva.number_of_independent_gains()); 
  int Ndof_CP = int(CP.size());
  int Ndof = Ndof_VM + Ndof_CP - int(riaf.size())-2;

  double chi2_VM = L[0]->chi_squared(pmax);
  double chi2_CP = L[1]->chi_squared(pmax);
  double chi2 = L_obj.chi_squared(pmax);
  double Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutA << std::setw(10) << index;
    for (size_t j=0; j<riaf.size(); ++j)
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
  for (size_t j=0; j<riaf.size(); ++j)
  {
    means[j] = pmax[j];
    if (Number_start_params==0)
      ranges[j] *= 1e-5;
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
  riaf.generate_model(pmax);

  chi2_VM = L[0]->chi_squared(pmax);
  chi2_CP = L[1]->chi_squared(pmax);
  chi2 = L_obj.chi_squared(pmax);
  Lmax = L_obj(pmax);

    
  if (world_rank==0)
  {
    sumoutB << std::setw(10) << index;
    for (size_t j=0; j<riaf.size(); ++j)
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
