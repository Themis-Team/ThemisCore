/*!
    \file 3c279_mexico.cpp
    \author Avery E Broderick
    \date Mar, 2019
    \brief Driver file for 3c279 analysis with geometric models:
              Large assymmetric Gaussian background
	      one or more symmetric Gaussians
	      one or more asymmetric Gaussians
    \details Takes file lists generated via something like:
             readlink -f ~/Themis/Themis/sim_data/ChallengeJ+/wsB/VM_*.d > vm_file_list
             readlink -f ~/Themis/Themis/sim_data/ChallengeJ+/wsB/CP_*.d > cp_file_list
	     These must be passed as -vm <file> and -cp <file> options.
    \todo 
*/


#include "model_image_xsringauss.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_smooth.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "utils.h"

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
  std::string vm_file_list="", cp_file_list="";
  int istart = 0;
  int iend = -1;
  int Number_of_steps_A = 20000; 
  int Number_of_steps_B = 10000; 
  int Number_start_params = 0;
  std::string param_file="";
  unsigned int seed = 42;
  std::string model_glob = "";

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

    if (opt=="--start" || opt=="-s")
    {
      if (k<argc)
	istart = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --start, -s.\n";
	std::exit(1);
      }
    }
    else if (opt=="--end" || opt=="-e")
    {
      if (k<argc)
	iend = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An int argument must be provided after --end, -e.\n";
	std::exit(1);
      }
    }
    else if (opt=="--visibility-amplitudes" || opt=="-vm")
    {
      if (k<argc)
	vm_file_list = std::string(argv[k++]);
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
	cp_file_list = std::string(argv[k++]);
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
	  std::cerr << "ERROR: TWO arguments must be provided after --parameter-file, -p: <# of params to set> and file list.\n";
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
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
	std::cerr << "NAME\n"
		  << "\tDriver executable for Challenge J+\n\n"
		  << "SYNOPSIS"
		  << "\tmpirun -np 40 chjp_sXag -vm vm_file_list -cp cp_file_list [OPTIONS]\n\n"
		  << "REQUIRED OPTIONS\n"
		  << "\t-vm, --visibility-amplitudes <string>\n"
		  << "\t\tSets the name of the file containing the list of visibility amplitude data file names.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\t-cp, --closure-phases <string>\n"
		  << "\t\tSets the name of the file containing the list of closure phase data file names.\n"
		  << "\t\tFile names must include the full absolute paths.\n"
		  << "\tNOTE: The vm_file_list and cp_file_list must have the same number of lines.\n"
		  << "DESCRIPTION\n"
		  << "\t-h,--help\n"
		  << "\t\tPrint this message.\n"
		  << "\t-s, --start <int>\n"
		  << "\t\tSets the start index, beginning with 0, in the vm_file_list, cp_file_list,\n"
		  << "\t\tand param_file list (if provided) to begin running.\n"
		  << "\t-e, --end <int>\n"
		  << "\t\tSets the end index (plus 1), beginning with 1, in the vm_file_list, cp_file_list,\n"
		  << "\t\tand param_file list (if provided) to begin running.  So \"-s 3 -f 5\" will run\n"
		  << "\t\tindexes 3 and 4 and then stop.\n"
		  << "\t-p, -parameter-file <int> <string>\n"
		  << "\t\tNumber of parameters to set and name of parameter list file, formatted as\n"
		  << "\t\tfit_summaries_*.txt, with the same number of lines as the vm_file_list and\n"
		  << "\t\tcp_file_list.  Parameters are set in order (i.e., you must fit parameter 0\n"
		  << "\t\tto set parameter 1, etc.). This also shrinks the affected ranges.\n"
		  << "\t-NA <int>\n"
		  << "\t\tSets the number of MCMC steps to take for chain A.  Defaults to 20000.\n"
		  << "\t-NB <int>\n"
		  << "\t\tSets the number of MCMC steps to take for chain B.  Defaults to 10000.\n"
		  << "\t-m, --model <string>\n"
		  << "\t\tSets the model definition string.  This may any combination of G, A, a, g\n"
		  << "\t\tin any order, with only one of each (so G, A, Gaaggg, etc.).  The first sets the origin.\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	  
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

  if (vm_file_list=="" || cp_file_list=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No data file list was provided. The -vm <string> and\n"
		<< "       -cp <string> options are *required*.  See -h for more\n"
		<< "       details and options.\n";
    std::exit(1);
  }

  // Read in the data file lists
  std::vector<std::string> vm_file_names, cp_file_names;
  std::string stmp;
  std::fstream vmfnin(vm_file_list);
  if (!vmfnin.is_open())
  {
    std::cerr << "ERROR: Could not open " << vm_file_list << '\n';
    std::exit(1);
  }
  for (vmfnin >> stmp;  !vmfnin.eof(); vmfnin >> stmp)
    vm_file_names.push_back(stmp);
  std::fstream cpfnin(cp_file_list);
  if (!cpfnin.is_open())
  {
    std::cerr << "ERROR: Could not open " << cp_file_list << '\n';
    std::exit(1);
  }
  for (cpfnin >> stmp;  !cpfnin.eof(); cpfnin >> stmp)
    cp_file_names.push_back(stmp);
  if (vm_file_names.size()!=cp_file_names.size())
  {
    std::cerr << "ERROR: Visibility magnitude and closure phase\n";
    std::cerr << "       file lists provided must be of equal length.\n";
    std::exit(1);
  }

  // Set and fill the start parameters if provided
  // Assumes has the same format as the fit_summaries.txt file (header, index, parameters, then other items)
  std::vector< std::vector<double> > start_parameter_list(vm_file_names.size());
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
    for (size_t j=0; j<start_parameter_list.size() && !pfin.eof(); ++j)
    {
      for (int k=0; k<Number_start_params; k++)
      {
	pfin >> dtmp;
	start_parameter_list[j].push_back(dtmp);
      }
      // Kill remainder of line
      pfin.ignore(4096,'\n');
      // Get next index
      pfin >> dtmp;      
    } 
  }  

  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "VM files: (" << vm_file_names.size() << ")\n";
    std::cout << "CP files: (" << cp_file_names.size() << ")\n";
    if (Number_start_params>0)
      std::cout << "start parameters: (" << start_parameter_list.size() << ", " << start_parameter_list[0].size() << ")\n"; 
    for (size_t i=0; i<vm_file_names.size(); ++i)
    {
	std::cout << "\t" << vm_file_names[i] << std::endl;
	std::cout << "\t" << cp_file_names[i] << std::endl;
	if (Number_start_params>0)
	{
	  std::cout << "\t";
	  for (size_t j=0; j<start_parameter_list[i].size(); ++j)
	    std::cout << std::setw(15) << start_parameter_list[i][j];
	  std::cout << std::endl;
	}
	std::cout << std::endl;
    }
    std::cout << "---------------------------------------------------\n" << std::endl;
  }
  // Sort out the end point
  if (iend<0 || iend>int(vm_file_names.size()) )
  {
    if (world_rank==0)
      std::cerr << "WARNING: iend not set or set past end of file list.\n";
    iend = vm_file_names.size();
  }
  // Sort out the start point (after iend is sorted)
  if (istart>iend)
  {
    if (world_rank==0)
      std::cerr << "ERROR: istart is set to beyond iend or past end of file list.\n";
    std::exit(1);
  }
  // Parse model_glob
  std::vector< std::string > model_list;
  for (size_t j=0; j<model_glob.length(); ++j) 
  {
    model_list.push_back(model_glob.substr(j,1));
    if (model_list[model_list.size()-1]=="G")
      std::cerr << "Added large-scale symmetric Gaussian\n";
    else if (model_list[model_list.size()-1]=="A")
      std::cerr << "Added large-scale asymmetric Gaussian\n";
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


  // Prepare the output summary file
  std::stringstream sumoutnameA, sumoutnameB;
  sumoutnameA << "fit_summaries_A_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  sumoutnameB << "fit_summaries_B_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  std::ofstream sumoutA, sumoutB;
  if (world_rank==0)
  {
    sumoutA.open(sumoutnameA.str().c_str());
    sumoutA << std::setw(10) << "# Index";
    sumoutB.open(sumoutnameB.str().c_str());
    sumoutB << std::setw(10) << "# Index";
    
    for (size_t mc=0; mc<model_list.size(); ++mc) 
    {
      if (model_list[mc]=="G" || model_list[mc]=="g")
      {
	sumoutA << std::setw(15) << "Ig (Jy)"
		<< std::setw(15) << "sigg (uas)"
		<< std::setw(15) << "xg"
		<< std::setw(15) << "yg";  
	sumoutB << std::setw(15) << "Ig (Jy)"
		<< std::setw(15) << "sigg (uas)"
		<< std::setw(15) << "xg"
		<< std::setw(15) << "yg";  
      }
      if (model_list[mc]=="A" || model_list[mc]=="a")
      {
	sumoutA << std::setw(15) << "Ia (Jy)"
		<< std::setw(15) << "siga (uas)"
		<< std::setw(15) << "A"
		<< std::setw(15) << "phi"
		<< std::setw(15) << "xa"
		<< std::setw(15) << "ya";
	sumoutB << std::setw(15) << "Ia (Jy)"
		<< std::setw(15) << "siga (uas)"
		<< std::setw(15) << "A"
		<< std::setw(15) << "phi"
		<< std::setw(15) << "xa"
		<< std::setw(15) << "ya";
      }
    }
    sumoutA << std::setw(15) << "VA red. chisq"
	    << std::setw(15) << "CP red. chisq"
	    << std::setw(15) << "red. chisq"
	    << std::setw(15) << "log-liklhd"
	    << "     FileName"
	    << std::endl;
    sumoutB << std::setw(15) << "VA red. chisq"
	    << std::setw(15) << "CP red. chisq"
	    << std::setw(15) << "red. chisq"
	    << std::setw(15) << "log-liklhd"
	    << "     FileName"
	    << std::endl;
  }
  
  
  ////////////////////////////
  // Begin loop over the relevant portion of file_name_list and running chains.
  // ALL items are in the loop to scope potential problems, though this is probably unncessary.
  // This happens in two steps.
  //   1. First a long run from totally random position
  //   2. A refined run from the best fit

  for (size_t index=size_t(istart); index<size_t(iend); ++index)
  {
    // Read in data
    //Themis::data_visibility_amplitude VM(Themis::utils::global_path(vm_file_names[index]),"HH");
    //Themis::data_closure_phase CP(Themis::utils::global_path(cp_file_names[index]));
    Themis::data_visibility_amplitude VM(vm_file_names[index],"HH");
    Themis::data_closure_phase CP(cp_file_names[index]);

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
    
    // Use analytical Visibilities
    for (size_t mc=0; mc<model_g.size(); ++mc)
      model_g[mc]->use_analytical_visibilities();
    for (size_t mc=0; mc<model_a.size(); ++mc)
      model_a[mc]->use_analytical_visibilities();
    
    
    /////////////////
    // Set up priors and initial walker ensemble starting positions
    //
    // Container of base prior class pointers
    std::vector<Themis::prior_base*> P;
    std::vector<double> means, ranges;
    std::vector<std::string> var_names;
    
    
    double uas2rad = 1e-6/3600. * M_PI/180.;
    

    for (size_t mc=0; mc<model_list.size(); ++mc)
    {
      //   0 Itot
      P.push_back(new Themis::prior_linear(0.0,10));
      means.push_back(4.0);
      ranges.push_back(2.0);

      //   1 Size
      if (model_list[mc]=="G" || model_list[mc]=="A")
      {
	P.push_back(new Themis::prior_linear(1e2*uas2rad,1e7*uas2rad));
	means.push_back(1e3*uas2rad);
	ranges.push_back(0.5e3*uas2rad);
      }
      else
      {
	P.push_back(new Themis::prior_linear(0.0,1e2*uas2rad));
	means.push_back(20*uas2rad);
	ranges.push_back(20*uas2rad);	
      }

      // If asymmetric Gaussian, add asymmetry parameters
      if (model_list[mc]=="A" || model_list[mc]=="a")
      {
	//   2 Asymmetry
	P.push_back(new Themis::prior_linear(0.0,0.99));
	means.push_back(0.1);
	ranges.push_back(0.1);
	//   3 Position angle
	P.push_back(new Themis::prior_linear(0,M_PI));
	means.push_back(0.5*M_PI);
	ranges.push_back(0.5*M_PI);
      }

      // If first component, fix at origin, otherwise, let move
      if (mc==0) 
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

    if (world_rank==0)
      std::cout << "Finished pusing prior lists, now at " << means.size() << std::endl;


    if (Number_start_params>int(means.size()))
    {
      std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
      std::exit(1);
    }
    for (size_t j=0; j<start_parameter_list[index].size(); ++j)
    {
      // Set desired means
      means[j] = start_parameter_list[index][j];
      // Restrict initial ranges
      ranges[j] *= 1e-5;
    }
    if (model_list[0]=="G" || model_list[0]=="g")
    {
      ranges[2] = ranges[3] = 1e-7*uas2rad;
    }
    else
    {
      ranges[4] = ranges[5] = 1e-7*uas2rad;
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
    Themis::likelihood L_obj(P, L, W);

    // Output residual data
    std::stringstream VA_res_name, CP_res_name, gc_name;
    VA_res_name << "VA_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    CP_res_name << "CP_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    gc_name << "gain_corrections_" << std::setfill('0') << std::setw(5) << index << ".d";
    L_obj(means);

    lva.output_gain_corrections(gc_name.str());
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());
  
    // Create a sampler object
    Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);

    // Generate a chain
    int Number_of_chains = 240;
    int Number_of_temperatures = 40;// 16; // 8;
    int Number_of_procs_per_lklhd = 1;
    int Temperature_stride = 50;
    int Chi2_stride = 10;
    int Ckpt_frequency = 500;
    bool restart_flag = false;
    int out_precision = 8;
    int verbosity = 0;
  
    // Set the CPU distribution
    MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
    // Set a checkpoint
    MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");

    // Set tempering schedule
    //MCMC_obj.set_tempering_schedule(1000.,1.,2.0);
    MCMC_obj.set_tempering_schedule(2000.,1.,1.25);
    
  
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
    L_obj(pmax);

    std::cerr << "Read pmax run A\n";
    if (world_rank==0)
    {
      std::cout << "-------------------------\n";
      for (size_t j=0; j<pmax.size(); ++j)
	std::cout << "pmax[" << j << "] = " << pmax[j] << '\n';
      std::cout << "-------------------------\n";
      std::cout << std::endl;
    }

    int Ndof_VM = int(VM.size()) - int(lva.number_of_independent_gains()) - int(image.size()) + 2;
    int Ndof_CP = int(CP.size()) - int(image.size()) + 2;
    int Ndof = int(VM.size()+CP.size()) - int(lva.number_of_independent_gains()) - int(image.size()) + 2;

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
	      << "     " << vm_file_names[index]
	      << "     " << cp_file_names[index]
	      << std::endl;
    }
    
    lva.output_gain_corrections(gc_name.str());
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());
    

    /////////
    // Set up the revised means and ranges
    for (size_t mc=0,j=0; mc<model_list.size(); ++mc)
    {
      // 0 Itot
      means[j] = pmax[j];
      ranges[j++] = 1e-7;
      
      // 1 Size
      means[j] = pmax[j];
      ranges[j++] = 1e-7 * uas2rad;
      
      // If asymmetric Gaussian, add asymmetry parameters
      if (model_list[mc]=="A" || model_list[mc]=="a")
      {
	//   2 Asymmetry
 	means[j] = pmax[j];
 	ranges[j++] = 1e-7;	
	//   3 Position angle
	means[j] = pmax[j];
	ranges[j++] = 1e-7;
      }

      //   4 x offset if first component then fix
      means[j] = pmax[j];
      ranges[j++] = 1e-7*uas2rad;
      //   5 y offset if first component then fix
      means[j] = pmax[j];
      ranges[j++] = 1e-7*uas2rad;
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
    L_obj(pmax);

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
	      << "     " << vm_file_names[index]
	      << "     " << cp_file_names[index]
	      << std::endl;
    }
    
    lva.output_gain_corrections(gc_name.str());
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());

  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
