/*!
    \file m87_ccm_mexico_afss.cpp
    \author Avery E Broderick, Paul Tiede
    \date Jan, 2019
    \brief Driver file for M87 analysis with the concordance crescent model to multiple data files:
              Large symmetric Gaussian background
              smoothed xsringauss
              one or two asymmetric Gaussians
              central symmetric Gaussians
    \details Takes file lists generated via something like:
             readlink -f ~/Themis/Themis/sim_data/ChallengeJ+/wsB/V_*.d > v_file_list
	     These must be passed as -v <file> option.
	     Uses the automated factored slice sampler.
    \todo 
*/

#include "data_visibility.h"

#include "model_image_xsringauss.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_smooth.h"
#include "utils.h"
#include "read_data.h"


#include "likelihood.h"
#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"


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
  std::string v_file="", nc_file="";
  int Number_of_steps = 10; 
  size_t Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  bool Reconstruct_gains = false;
  std::string lc_file="";
  std::vector<std::string> gain_file_list;

  int Number_temperatures = 0;
  double initial_ladder_spacing = 1.15;
  int thin_factor = 1;
  int Temperature_stride = 50;
  std::string annealing_ladder_file = "";
  int refresh_rate = 1;
  int init_buffer = 10;
  int window = 0;
  int number_of_adaption_steps = 0; // 10000;
  int Ckpt_frequency = 10; // per swaps
  int verbosity = 0;
  size_t Number_of_reps = 7;

  bool restart_flag = false;
  
  std::string model_glob = "sXaagG";


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
    	gain_file_list.push_back(std::string(argv[k++]));
      else
      {
    	if (world_rank==0)
    	  std::cerr << "ERROR: A string argument must be provided after --gain-file-hi, -gfh.\n";
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
    else if (opt=="--continue" || opt=="--restart")
    {
      restart_flag = true;
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
    else if ( opt == "--init-buffer" || opt == "-ib" )
    {
      if ( k <argc )
        init_buffer = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --init-buffer, -ib.\n";
        std::exit(1);
      }
    }
    else if ( opt == "--window" || opt == "-w" )
    {
      if ( k <argc )
        window = atoi(argv[k++]);
      else
      {
        if ( world_rank == 0 )
          std::cerr << "ERROR: An int argument must be provided after --window, -w.\n";
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
		  << "\tDriver executable for M87 CCM analyses\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 m87_ccm_afss -v v_file_list [OPTIONS]\n\n"
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
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	 
		  << "\t-g, --reconstruct-gains\n"
		  << "\t\tReconstructs unknown station gains.  Default off.\n"
		  << "\t-gh, --gain-file-hi <filename>\n"
		  << "\t\tSets the gains to those in the specified file name.  Must have one per data file.\n"
		  << "\t--array <array>\n"
		  << "\t\tArray specification.  Options are either eht2017 or an array file that contains one station code\n"
		  << "\t\tand a value giving the 1-sigma gain amplitude error per line. Default: eht2017.\n"
		  << "\t--tempering-levels <int>\n"
		  << "\t\tSets the number of tempering levels.  Defaults to the number of processes.\n"
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
		  << "\t--adaption-steps <int>\n"
		  << "\t\tSets the number of adaption steps to perform.  Defaults to (dimension^2)*4.\n"
		  << "\t--init-buffer, -ib <int>\n"
		  << "\t\tSets the initial buffer size for the slice sampler.  Defaults to 10.\n"
		  << "\t--window, -w <int>\n"
		  << "\t\tSets the window size for the slice sampler.  Defaults to (dimension^2)/2.\n"
		  << "\t--refresh <int>\n"
                  << "\t\tHow often the cout stream is refreshed to show the progress of the sampler.\n"
                  << "\t\tDefault is 1 step.\n"
		  << "\t--checkpoint-stride <int>\n"
		  << "\t\tSets the number of tempering level *swaps* after which to checkpoint the state of the run.  Default: 10.\n"
		  << "\t--verbosity <int>\n"
		  << "\t\tSets the verbosity level.  Default 0.\n"
		  << "\t-m, --model <string>\n"
		  << "\t\tSets the model definition string.  This may any combination of sX, a, g, G,\n"
		  << "\t\tin that order, with only one of each (so sXag, sX, sXg, ag, sXaag, etc.).\n"
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
      for (size_t k=0; k<Number_start_params; k++)
      {
	pfin >> dtmp;
	start_parameter_list.push_back(dtmp);
      }

      for (size_t k=0; k<Number_start_params; k++)
	buff[k] = start_parameter_list[k];
    }
    
    MPI_Bcast(&buff[0],Number_start_params,MPI_DOUBLE,0,MPI_COMM_WORLD);
    start_parameter_list.resize(Number_start_params);
    for (size_t k=0; k<Number_start_params; k++)
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

  // Parse model_glob
  bool add_sX=false, add_a=false, add_g=false, add_G=false;
  size_t number_a=0;
  size_t mgi=0;
  if (model_glob.substr(mgi,2)=="sX")
  {
    std::cerr << "Adding smoothed xsringauss\n";
    add_sX=true;
    mgi+=2;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=1;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=2;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="g")
  {
    std::cerr << "Adding fixed central gaussian\n";
    add_g=true;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="G")
  {
    std::cerr << "Adding fixed large gaussian\n";
    add_G=true;
    mgi+=1;
  }
  if (model_glob.substr(mgi)!="")
    std::cerr << "ERROR: Unrecognized model option " << model_glob.substr(mgi) << '\n';

  // Choose the model to compare (other models present for rapid build out)
  //  1 Crescent
  Themis::model_image_xsringauss model_X;
  Themis::model_image_smooth model_sX(model_X);
  //  1 Asymmetric Gaussian
  Themis::model_image_asymmetric_gaussian model_a1, model_a2;
  //  1 Symmetric Gaussian for the center
  Themis::model_image_symmetric_gaussian model_g, model_G;

  Themis::model_image_sum image;
  if (add_sX)
    image.add_model_image(model_sX);
  if (add_a)
  {
    image.add_model_image(model_a1);
    if (number_a==2)
      image.add_model_image(model_a2);
  }
  if (add_g)
    image.add_model_image(model_g);
  if (add_G)
    image.add_model_image(model_G);

  // Use analytical Visibilities
  model_X.use_analytical_visibilities();
  model_a1.use_analytical_visibilities();
  model_a2.use_analytical_visibilities();
  model_g.use_analytical_visibilities();
  model_G.use_analytical_visibilities();
  

  // Get ready to read in data files
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
  Themis::utils::read_vfile_mpi(v_file_name_list, v_file, MPI_COMM_WORLD);

  for (size_t j=0; j<v_file_name_list.size(); ++j)
  {
    V_data.push_back( new Themis::data_visibility(v_file_name_list[j],"HH") );
    V_data[j]->set_default_frequency(frequency);
    
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

  // Output model tag
  image.write_model_tag_file();


  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  double uas2rad = 1e-6/3600. * M_PI/180.;

  // sX params
  if (add_sX)
  {
    if (world_rank==0)
      std::cout << "Pushed back sX prior list, now at " << means.size() << std::endl;

    //   0 Itot
    P.push_back(new Themis::prior_linear(0.0,2));
    means.push_back(0.6);
    ranges.push_back(0.3);
    //   1 Outer size R
    P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
    means.push_back(50*uas2rad);
    ranges.push_back(20*uas2rad);
    //   2 psi
    P.push_back(new Themis::prior_linear(0.0001,0.9999));
    means.push_back(0.2);
    ranges.push_back(0.2);
    //   3 1-tau
    P.push_back(new Themis::prior_linear(0.0001,0.9999));
    means.push_back(0.2);
    ranges.push_back(0.2);
    //   4 f
    P.push_back(new Themis::prior_linear(0.00,1.0));
    means.push_back(0.5);
    ranges.push_back(0.5);
    //   5 g
    P.push_back(new Themis::prior_linear(0,3.0));
    means.push_back(0.1);
    ranges.push_back(0.1);
    //   6 a
    P.push_back(new Themis::prior_linear(0.0,100.0));
    means.push_back(5.0);
    ranges.push_back(2.5);
    //   7 Ig
    P.push_back(new Themis::prior_linear(0.0,1.0));
    means.push_back(0.3);
    ranges.push_back(0.3);
    //   8 Position angle
    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
    means.push_back(0);
    ranges.push_back(M_PI);
    //   9 Smoothing kernel size
    P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
    means.push_back(1*uas2rad);
    ranges.push_back(1*uas2rad);
    //  10 Smoothing kernel Asymmetry
    P.push_back(new Themis::prior_linear(0.0,1e-6));
    means.push_back(0.5e-6);
    ranges.push_back(0.1e-6);
    //  11 Smoothing kernel phi
    P.push_back(new Themis::prior_linear(0,M_PI));
    means.push_back(0.5*M_PI);
    ranges.push_back(0.5*M_PI);
    //  12 x offset
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
    //  13 y offset
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
  }
  if (add_a)
  {
    for (size_t ia=0; ia<number_a; ++ia)
    {
      if (world_rank==0)
	std::cout << "Pushed back a prior list, now at " << means.size() << std::endl;
	
      //   0 Itot 
      P.push_back(new Themis::prior_linear(0,2));
      means.push_back(0.1);
      ranges.push_back(0.1);
      //   1 Size
      P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
      means.push_back(20*uas2rad);
      ranges.push_back(20*uas2rad);
      //   2 Asymmetry
      P.push_back(new Themis::prior_linear(0.0,0.99));
      means.push_back(0.1);
      ranges.push_back(0.1);
      //   3 Position angle
      P.push_back(new Themis::prior_linear(0,M_PI));
      means.push_back(0.5*M_PI);
      ranges.push_back(0.5*M_PI);
      if (means.size()==4 && ia==0)
      {
	//   4 x offset if first component then fix
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-7*uas2rad);
	//   5 y offset if first component then fix
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-7*uas2rad);
      }
      else
      {
	//   4 x offset
	P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
	means.push_back(0.0);
	ranges.push_back(50*uas2rad);
	//   5 y offset
	P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
	means.push_back(0.0);
	ranges.push_back(50*uas2rad);
      }
    }
  }
  if (add_g)
  {
    if (world_rank==0)
      std::cout << "Pushed back g prior list, now at " << means.size() << std::endl;

    //   0 Itot 
    P.push_back(new Themis::prior_linear(0,2));
    means.push_back(0.1);
    ranges.push_back(0.1);
    //   1 Size
    P.push_back(new Themis::prior_linear(0.0,25*uas2rad));
    means.push_back(12.5*uas2rad);
    ranges.push_back(12.5*uas2rad);
    //   2 x offset -- FIX AT CENTER NO MATTER WHAT
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
    //   3 y offset -- FIX AT CENTER NO MATTER WHAT
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
  }
  if (add_G)
  {
    if (world_rank==0)
      std::cout << "Pushed back G prior list, now at " << means.size() << std::endl;
    
    //   0 Itot 
    P.push_back(new Themis::prior_linear(0,10.));
    means.push_back(0.1);
    ranges.push_back(0.1);
    //   1 Size
    P.push_back(new Themis::prior_logarithmic(1e4*uas2rad,1e7*uas2rad));
    means.push_back(1e5*uas2rad);
    ranges.push_back(0.5e5*uas2rad);
    //   2 x offset -- FIX AT CENTER NO MATTER WHAT
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
    //   3 y offset -- FIX AT CENTER NO MATTER WHAT
    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
    means.push_back(0.0);
    ranges.push_back(1e-7*uas2rad);
  }

  if (world_rank==0)
    std::cout << "Finished pusing prior lists, now at " << means.size() << std::endl;


  if (Number_start_params>int(means.size()))
  {
    std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
    std::exit(1);
  }
  for (size_t j=0; j<start_parameter_list.size(); ++j)
  {
    // Set desired means
    means[j] = start_parameter_list[j];
    // Restrict initial ranges
    ranges[j] *= 1e-5;
  }
  int pind=0;
  if (add_sX)
  {
    pind += 14;
    ranges[pind-1] = ranges[pind-2] = 1e-6*uas2rad;
  }
  if (add_a)
  {
    pind += 6;
    if (pind==6)
    {
      ranges[pind-1] = ranges[pind-2] = 1e-6*uas2rad;
    }
  }
  if (add_a)
  {
    pind += 6;
    if (pind==6)
    {
      ranges[pind-1] = ranges[pind-2] = 1e-6*uas2rad;
    }
  }
  if (add_g)
  {
    pind += 4;
    ranges[pind-1] = ranges[pind-2] = 1e-6*uas2rad;
  }
  if (add_G)
  {
    pind += 4;
    ranges[pind-1] = ranges[pind-2] = 1e-6*uas2rad;
  }

  // Output priors with bounds check
  if (world_rank==0)
  {
    std::cerr << "Priors check after =========================================\n";
    std::cerr << "model size: " << image.size()
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
  
  // double Lstart = L_obj(means);
  // if (world_rank==0)
  //   std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Ndata = 0;
  for (size_t j=0; j<V_data.size(); ++j)
    Ndata += 2*V_data[j]->size();
  int Nparam = image.size();
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);
  if (Reconstruct_gains)
    for (size_t j=0; j<lvg.size(); ++j)
    {
      Ngains_list[j] = lvg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  int NDoF = Ndata - Nparam - Ngains;
  
  // Get the best point at start
  std::vector<double> pbest =  means;

  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed, L_temp, var_names, means.size());

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
  if (window==0)
    window = means.size()*means.size()/2;
  DEO.get_sampler()->set_window_parameters(init_buffer,window);

  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  bool save_warmup = true;
  if (number_of_adaption_steps==0)
    number_of_adaption_steps = means.size()*means.size()*4;
  DEO.get_sampler()->set_adaptation_parameters( number_of_adaption_steps, save_warmup);
    
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(means);

  //We can also change the annealing schedule.
  double initial_spacing = initial_ladder_spacing; //initial geometric spacre_stride;
  DEO.set_annealing_schedule(initial_spacing);  
  DEO.set_deo_round_params(Number_of_steps,Temperature_stride);
  
  // If continuing
  int round_start = 0;
  if (restart_flag && Themis::utils::isfile("MCMC.ckpt"))
  {
    DEO.read_checkpoint("MCMC.ckpt");
    round_start = DEO.get_round();
    restart_flag=false;
  }else if (restart_flag && !Themis::utils::isfile("MCMC.ckpt")) {
    std::cerr << "!!!Warning no ckpt found starting run from beginning!!!\n";
  }

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

  //// BECAUSE WE SHORT-CIRCUIT RECOMPUTATION, WE MUST DO THIS AFTER
  //// SETTING THE PROCESS TOPOLOGY
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';
  
  // Start looping over repetitions
  // Extra loop is to make final files
  pbest = means;
  for (size_t rep=round_start; rep<=Number_of_reps; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started round " << rep << std::endl;
  
    // Prepare the output summary file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3) << rep << "_fit_summaries" << ".txt";
    std::ofstream sumout;

    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index";
      if (add_sX)
	sumout << std::setw(15) << "Isx (Jy)"
	       << std::setw(15) << "Rp (uas)"
	       << std::setw(15) << "psi"
	       << std::setw(15) << "ecc"
	       << std::setw(15) << "f"
	       << std::setw(15) << "gax"
	       << std::setw(15) << "a"
	       << std::setw(15) << "ig"
	       << std::setw(15) << "PA"
	       << std::setw(15) << "s-sig"
	       << std::setw(15) << "s-A"
	       << std::setw(15) << "s-phi"
	       << std::setw(15) << "xsX"
	       << std::setw(15) << "ysX";
      if (add_a)
	for (size_t ia=0; ia<number_a; ++ia)
	  sumout << std::setw(15) << "Ia (Jy)"
		 << std::setw(15) << "siga (uas)"
		 << std::setw(15) << "A"
		 << std::setw(15) << "phi"
		 << std::setw(15) << "xa"
		 << std::setw(15) << "ya";
      if (add_g)
	sumout << std::setw(15) << "Ig (Jy)"
	       << std::setw(15) << "sigg (uas)"
	       << std::setw(15) << "xg"
	       << std::setw(15) << "yg";
      if (add_G)
	sumout << std::setw(15) << "Ig (Jy)"
	       << std::setw(15) << "sigg (uas)"
	       << std::setw(15) << "xg"
	       << std::setw(15) << "yg";
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


