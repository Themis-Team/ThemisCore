/*!
    \file sgra_eccm_linear.cpp
    \author Avery E Broderick & Rufus Ni
    \date Aug, 2019
    \brief Driver file for M87 analysis with the concordance crescent model that uses all days and bands:
              Large symmetric Gaussian background
              smoothed xsringauss
              one or two asymmetric Gaussians
              central symmetric Gaussians
    \details Takes file lists generated via something like:
             readlink -f ~/Themis/Themis/sim_data/ChallengeJ+/wsB/VM_*.d > vm_file_list
             readlink -f ~/Themis/Themis/sim_data/ChallengeJ+/wsB/CP_*.d > cp_file_list
	     These must be passed as -vm <file> and -cp <file> options.
	     Reads these two at a time (i.e., four files at a time) and assumes that these
	     are identical underlying structures (e.g., hi and lo bands).
    \todo 
*/


#include "model_image_xsringauss.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"
#include "model_image_smooth.h"
#include "model_image_polynomial_variable.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "model_galactic_center_diffractive_scattering_screen.h"
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
  std::string model_glob = "sXaagG";

  std::string offset_coordinates="Cartesian";
  
  int Number_of_tempering_levels=40;
  double Tempering_time=1000.0;
  double Tempering_ladder=1.4;
  int Number_of_walkers=360;

  //bool use_diffractive_scattering=true;

  bool static_image=false;

  
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
	  std::cerr << "ERROR: An float argument must be provided after --tempering-ladder.\n";
	std::exit(1);
      }
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
    else if (opt=="--static")
    {
      static_image=true;
      std::cerr << "WARNING: Assuming image is static.\n";
    }
    else if (opt=="--polar")
    {
      offset_coordinates="polar";
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
		  << "\t\tSets the model definition string.  This may any combination of sX, a, g, G,\n"
		  << "\t\tin that order, with only one of each (so sXag, sX, sXg, ag, sXaag, etc.).\n"
		  << "\t--seed <int>\n"
		  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
		  << "\t\tDefaults to 42.\n"	  
		  << "\t--tempering-levels <int>\n"
		  << "\t\tSets the number of tempering levels.  Defaults to 40\n"
		  << "\t--tempering-ladder <float>\n"
		  << "\t\tSets the initial geometric tempering ladder spacing.  Defaults to 1.4.\n"
		  << "\t--tempering-time <float>\n"
		  << "\t\tSets the number of MCMC steps over which to reduce the tempering ladder\n"
		  << "\t\toptimization evolution to reduce by half.  Defaults to 1000.\n"
		  << "\t--walkers <int>\n"
		  << "\t\tSets the number of chains per tempering level.  Defaults to 360.\n"
                  << "\t--static\n"
                  << "\t\tSets the image model to be static, i.e., not linearly evolving.\n"
		  << "\t--polar\n"
		  << "\t\tSets the offset coordinates for model components to be polar instead of Cartesian.\n"
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
  //std::vector< std::vector<double> > start_parameter_list(vm_file_names.size());
  std::vector< std::vector<double> > start_parameter_list(1);
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
    for (size_t j=0; j<start_parameter_list.size() && !pfin.eof(); j+=8) // Jumps by eight to keep structure with index stepping below
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
    }
    if (Number_start_params>0)
    {
      std::cout << "\t";
      for (size_t j=0; j<start_parameter_list[0].size(); ++j)
	std::cout << std::setw(15) << start_parameter_list[0][j];
      std::cout << std::endl;
    }
    //std::cout << std::endl;
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
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=3;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=4;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=5;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=6;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=7;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=8;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=9;
    mgi+=1;
  }
  if (model_glob.substr(mgi,1)=="a")
  {
    std::cerr << "Adding asymmetric gaussian\n";
    add_a=true;
    number_a=10;
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
  
  // Prepare the output summary file
  std::stringstream sumoutnameA, sumoutnameB;
  sumoutnameA << "fit_summaries_A_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  sumoutnameB << "fit_summaries_B_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  std::ofstream sumoutA, sumoutB;
  if (world_rank==0)
  {
   sumoutA.open(sumoutnameA.str().c_str());
   sumoutA << std::setw(10) << "# Index";
   if (add_sX)
   {
     sumoutA << std::setw(15) << "Isx (Jy)";
     if (!static_image)
       sumoutA << std::setw(15) << "Isx_1 (Jy/s)";
     sumoutA << std::setw(15) << "Rp (uas)";
     if (!static_image)
       sumoutA << std::setw(15) << "Rp_1 (uas/s)";
     sumoutA << std::setw(15) << "psi";
     if (!static_image)
       sumoutA << std::setw(15) << "psi_1";
     sumoutA << std::setw(15) << "ecc";
     if (!static_image)
       sumoutA << std::setw(15) << "ecc_1";
     sumoutA << std::setw(15) << "f";
     if (!static_image)
       sumoutA << std::setw(15) << "f_1";
     sumoutA << std::setw(15) << "gax";
     if (!static_image)
       sumoutA << std::setw(15) << "gax_1";
     sumoutA << std::setw(15) << "a";
     if (!static_image)
       sumoutA << std::setw(15) << "a_1";
     sumoutA << std::setw(15) << "ig";
     if (!static_image)
       sumoutA << std::setw(15) << "ig_1";
     sumoutA << std::setw(15) << "PA";
     if (!static_image)
       sumoutA << std::setw(15) << "PA_1";
     sumoutA << std::setw(15) << "s-sig";
     if (!static_image)
       sumoutA << std::setw(15) << "s-sig_1";
     sumoutA << std::setw(15) << "s-A";
     if (!static_image)
       sumoutA << std::setw(15) << "s-A_1";
     sumoutA << std::setw(15) << "s-phi";
     if (!static_image)
       sumoutA << std::setw(15) << "s-phi_1";
     if (offset_coordinates=="Cartesian")
     {
       sumoutA << std::setw(15) << "xsX";
       if (!static_image)
	 sumoutA << std::setw(15) << "xsX_1";
       sumoutA << std::setw(15) << "ysX";
       if (!static_image)
	 sumoutA << std::setw(15) << "ysX_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutA << std::setw(15) << "rsX";
       if (!static_image)
	 sumoutA << std::setw(15) << "rsX_1";
       sumoutA << std::setw(15) << "thsX";
       if (!static_image)
	 sumoutA << std::setw(15) << "thsX_1";
     }
   }
   if (add_a)
     for (size_t ia=0; ia<number_a; ++ia)
     {
       sumoutA << std::setw(15) << "Ia (Jy)";
       if (!static_image)
         sumoutA << std::setw(15) << "Ia_1 (Jy/s)";
       sumoutA << std::setw(15) << "siga (rad)";
       if (!static_image)
         sumoutA << std::setw(15) << "siga_1 (rad/s)";
       sumoutA << std::setw(15) << "A";
       if (!static_image)
         sumoutA << std::setw(15) << "A_1";
       sumoutA << std::setw(15) << "phi";
       if (!static_image)
         sumoutA << std::setw(15) << "phi_1";
       if (offset_coordinates=="Cartesian")
       {
	 sumoutA << std::setw(15) << "xa";
	 if (!static_image)
	   sumoutA << std::setw(15) << "xa_1";
	 sumoutA << std::setw(15) << "ya";
	 if (!static_image)
	   sumoutA << std::setw(15) << "ya_1";
       }
       else if (offset_coordinates=="polar")
       {
	 sumoutA << std::setw(15) << "ra";
	 if (!static_image)
	   sumoutA << std::setw(15) << "ra_1";
	 sumoutA << std::setw(15) << "tha";
	 if (!static_image)
	   sumoutA << std::setw(15) << "tha_1";
       }
     }
   
   if (add_g)
   {
     sumoutA << std::setw(15) << "Ig (Jy)";
     if (!static_image)
       sumoutA << std::setw(15) << "Ig_1 (Jy/s)";
     sumoutA << std::setw(15) << "sigg (rad)";
     if (!static_image)
       sumoutA << std::setw(15) << "sigg_1 (rad/s)";
     if (offset_coordinates=="Cartesian")
     {
       sumoutA << std::setw(15) << "xg";
       if (!static_image)
	 sumoutA << std::setw(15) << "xg_1";
       sumoutA << std::setw(15) << "yg";
       if (!static_image)
	 sumoutA << std::setw(15) << "yg_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutA << std::setw(15) << "rg";
       if (!static_image)
	 sumoutA << std::setw(15) << "rg_1";
       sumoutA << std::setw(15) << "thg";
       if (!static_image)
	 sumoutA << std::setw(15) << "thg_1";
     }
   }
   
   if (add_G)
   {
     sumoutA << std::setw(15) << "Ig (Jy)";
     if (!static_image)
        sumoutA << std::setw(15) << "Ig_1 (Jy/s)";
     sumoutA << std::setw(15) << "sigg (rad)";
     if (!static_image)
        sumoutA << std::setw(15) << "sigg_1 (rad/s)";
     if (offset_coordinates=="Cartesian")
     {
       sumoutA << std::setw(15) << "xG";
       if (!static_image)
	 sumoutA << std::setw(15) << "xG_1";
       sumoutA << std::setw(15) << "yG";
       if (!static_image)
	 sumoutA << std::setw(15) << "yG_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutA << std::setw(15) << "rG";
       if (!static_image)
	 sumoutA << std::setw(15) << "rG_1";
       sumoutA << std::setw(15) << "thG";
       if (!static_image)
	 sumoutA << std::setw(15) << "thG_1";
     }
   }
   
   sumoutA << std::setw(15) << "VA red. chisq"
	   << std::setw(15) << "CP red. chisq"
	   << std::setw(15) << "red. chisq"
	   << std::setw(15) << "log-liklhd"
	   << "     FileName"
	   << std::endl;

   sumoutB.open(sumoutnameB.str().c_str());
   sumoutB << std::setw(10) << "# Index";
   if (add_sX)
   {
     sumoutB << std::setw(15) << "Isx (Jy)";
     if (!static_image)
       sumoutB << std::setw(15) << "Isx_1 (Jy/s)";
     sumoutB << std::setw(15) << "Rp (uas)";
     if (!static_image)
       sumoutB << std::setw(15) << "Rp_1 (uas/s)";
     sumoutB << std::setw(15) << "psi";
     if (!static_image)
       sumoutB << std::setw(15) << "psi_1";
     sumoutB << std::setw(15) << "ecc";
     if (!static_image)
       sumoutB << std::setw(15) << "ecc_1";
     sumoutB << std::setw(15) << "f";
     if (!static_image)
       sumoutB << std::setw(15) << "f_1";
     sumoutB << std::setw(15) << "gax";
     if (!static_image)
       sumoutB << std::setw(15) << "gax_1";
     sumoutB << std::setw(15) << "a";
     if (!static_image)
       sumoutB << std::setw(15) << "a_1";
     sumoutB << std::setw(15) << "ig";
     if (!static_image)
       sumoutB << std::setw(15) << "ig_1";
     sumoutB << std::setw(15) << "PA";
     if (!static_image)
       sumoutB << std::setw(15) << "PA_1";
     sumoutB << std::setw(15) << "s-sig";
     if (!static_image)
       sumoutB << std::setw(15) << "s-sig_1";
     sumoutB << std::setw(15) << "s-A";
     if (!static_image)
       sumoutB << std::setw(15) << "s-A_1";
     sumoutB << std::setw(15) << "s-phi";
     if (!static_image)
       sumoutB << std::setw(15) << "s-phi_1";
     if (offset_coordinates=="Cartesian")
     {
       sumoutB << std::setw(15) << "xsX";
       if (!static_image)
	 sumoutB << std::setw(15) << "xsX_1";
       sumoutB << std::setw(15) << "ysX";
       if (!static_image)
	 sumoutB << std::setw(15) << "ysX_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutB << std::setw(15) << "rsX";
       if (!static_image)
	 sumoutB << std::setw(15) << "rsX_1";
       sumoutB << std::setw(15) << "thsX";
       if (!static_image)
	 sumoutB << std::setw(15) << "thsX_1";
     }
   }
   if (add_a)
     for (size_t ia=0; ia<number_a; ++ia)
     {
       sumoutB << std::setw(15) << "Ia (Jy)";
       if (!static_image)
         sumoutB << std::setw(15) << "Ia_1 (Jy/s)";
       sumoutB << std::setw(15) << "siga (rad)";
       if (!static_image)
         sumoutB << std::setw(15) << "siga_1 (rad/s)";
       sumoutB << std::setw(15) << "A";
       if (!static_image)
         sumoutB << std::setw(15) << "A_1";
       sumoutB << std::setw(15) << "phi";
       if (!static_image)
         sumoutB << std::setw(15) << "phi_1";
       if (offset_coordinates=="Cartesian")
       {
	 sumoutB << std::setw(15) << "xa";
	 if (!static_image)
	   sumoutB << std::setw(15) << "xa_1";
	 sumoutB << std::setw(15) << "ya";
	 if (!static_image)
	   sumoutB << std::setw(15) << "ya_1";
       }
       else if (offset_coordinates=="polar")
       {
	 sumoutB << std::setw(15) << "ra";
	 if (!static_image)
	   sumoutB << std::setw(15) << "ra_1";
	 sumoutB << std::setw(15) << "tha";
	 if (!static_image)
	   sumoutB << std::setw(15) << "tha_1";
       }
     }
   
   if (add_g)
   {
     sumoutB << std::setw(15) << "Ig (Jy)";
     if (!static_image)
       sumoutB << std::setw(15) << "Ig_1 (Jy/s)";
     sumoutB << std::setw(15) << "sigg (rad)";
     if (!static_image)
       sumoutB << std::setw(15) << "sigg_1 (rad/s)";
     if (offset_coordinates=="Cartesian")
     {
       sumoutB << std::setw(15) << "xg";
       if (!static_image)
	 sumoutB << std::setw(15) << "xg_1";
       sumoutB << std::setw(15) << "yg";
       if (!static_image)
	 sumoutB << std::setw(15) << "yg_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutB << std::setw(15) << "rg";
       if (!static_image)
	 sumoutB << std::setw(15) << "rg_1";
       sumoutB << std::setw(15) << "thg";
       if (!static_image)
	 sumoutB << std::setw(15) << "thg_1";
     }
   }
   
   if (add_G)
   {
     sumoutB << std::setw(15) << "Ig (Jy)";
     if (!static_image)
        sumoutB << std::setw(15) << "Ig_1 (Jy/s)";
     sumoutB << std::setw(15) << "sigg (rad)";
     if (!static_image)
        sumoutB << std::setw(15) << "sigg_1 (rad/s)";
     if (offset_coordinates=="Cartesian")
     {
       sumoutB << std::setw(15) << "xG";
       if (!static_image)
	 sumoutB << std::setw(15) << "xG_1";
       sumoutB << std::setw(15) << "yG";
       if (!static_image)
	 sumoutB << std::setw(15) << "yG_1";
     }
     else if (offset_coordinates=="polar")
     {
       sumoutB << std::setw(15) << "rG";
       if (!static_image)
	 sumoutB << std::setw(15) << "rG_1";
       sumoutB << std::setw(15) << "thG";
       if (!static_image)
	 sumoutB << std::setw(15) << "thG_1";
     }
   }
  
 
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

  //for (size_t index=size_t(istart); index<size_t(iend); index+=4)  // Jump by two
  {
    // Read in data
    std::vector< Themis::data_visibility_amplitude* > VM;
    Themis::data_closure_phase CP;

    for (size_t index=size_t(istart); index<size_t(iend); index++)
    {
      VM.push_back( new Themis::data_visibility_amplitude(vm_file_names[index],"HH") );
      CP.add_data(cp_file_names[index],"HH");
    }
    
    // Choose the model to compare (other models present for rapid build out)
    //  1 Crescent
    Themis::model_image_xsringauss model_X;
    Themis::model_image_smooth model_sX(model_X);
    //  1 Asymmetric Gaussian
    Themis::model_image_asymmetric_gaussian model_a1, model_a2, model_a3, model_a4, model_a5, model_a6, model_a7, model_a8, model_a9, model_a10;
    //  1 Symmetric Gaussian for the center
    Themis::model_image_symmetric_gaussian model_g, model_G;

    Themis::model_image_sum image_sum(offset_coordinates);
    if (add_sX)
      image_sum.add_model_image(model_sX);
    if (add_a)
    {
      image_sum.add_model_image(model_a1);
      if (number_a>=2)
	image_sum.add_model_image(model_a2);
      if (number_a>=3)
	image_sum.add_model_image(model_a3);
      if (number_a>=4)
	image_sum.add_model_image(model_a4);
      if (number_a>=5)
	image_sum.add_model_image(model_a5);
      if (number_a>=6)
	image_sum.add_model_image(model_a6);
      if (number_a>=7)
	image_sum.add_model_image(model_a7);
      if (number_a>=8)
	image_sum.add_model_image(model_a8);
      if (number_a>=9)
	image_sum.add_model_image(model_a9);
      if (number_a>=10)
	image_sum.add_model_image(model_a10);
    }
    if (add_g)
      image_sum.add_model_image(model_g);
    if (add_G)
      image_sum.add_model_image(model_G);

    // Use analytical Visibilities
    model_X.use_analytical_visibilities();
    model_a1.use_analytical_visibilities();
    model_a2.use_analytical_visibilities();
    model_a3.use_analytical_visibilities();
    model_a4.use_analytical_visibilities();
    model_a5.use_analytical_visibilities();
    model_a6.use_analytical_visibilities();
    model_a7.use_analytical_visibilities();
    model_a8.use_analytical_visibilities();
    model_a9.use_analytical_visibilities();
    model_a10.use_analytical_visibilities();
    model_g.use_analytical_visibilities();
    model_G.use_analytical_visibilities();

    int poly_order = 1;
    if (static_image)
      poly_order = 0;
    Themis::model_image_polynomial_variable image_var(image_sum,poly_order,VM[0]->datum(0).tJ2000);
    Themis::model_galactic_center_diffractive_scattering_screen image(image_var);

    std::cerr << "I think that the image size is : " << image.size() << std::endl;
    
    /////////////////
    // Set up priors and initial walker ensemble starting positions
    //
    // Container of base prior class pointers
    std::vector<Themis::prior_base*> P;
    std::vector<double> means, ranges;
    std::vector<std::string> var_names;
    bool first_component=true;

    double uas2rad = 1e-6/3600. * M_PI/180.;

    // sX params
    if (add_sX)
    {
      if (world_rank==0)
	std::cout << "Pushed back sX prior list, now at " << means.size() << std::endl;

      //   0.1 Itot
      P.push_back(new Themis::prior_linear(0.0,3));
      means.push_back(1.0);
      ranges.push_back(0.01);
      if (!static_image)
      {
        //   0.2 Itot
        P.push_back(new Themis::prior_linear(-3e-3,3e-3));
        means.push_back(0.0);
        ranges.push_back(3e-7);
      }

      //   1.1 Outer size R
      P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
      means.push_back(20*uas2rad);
      ranges.push_back(0.2*uas2rad);
      if (!static_image)
      {
	//   1.2 Outer size R
	P.push_back(new Themis::prior_linear(-1e-2*uas2rad,1e-2*uas2rad));
	means.push_back(0.0*uas2rad);
	ranges.push_back(1e-5*uas2rad);
      }

      //   2.1 psi
      P.push_back(new Themis::prior_linear(0.0001,0.9999));
      means.push_back(0.1);
      ranges.push_back(0.001);
      if (!static_image)
      {
	//   2.2 psi
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(1e-8);
      }

      //   3.1 1-tau
      P.push_back(new Themis::prior_linear(0.0001,0.9999));
      means.push_back(0.1);
      ranges.push_back(0.001);
      if (!static_image)
      {
	//   3.2 1-tau
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(1e-8);
      }

      //   4.1 f
      P.push_back(new Themis::prior_linear(0.00,1.0));
      means.push_back(0.1);
      ranges.push_back(0.001);
      if (!static_image)
      {
	//   4.2 f
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(1e-8);
      }

      //   5.1 g
      P.push_back(new Themis::prior_linear(0,3.0));
      means.push_back(2.0);
      ranges.push_back(0.1);
      if (!static_image)
      {
      //   5.2 g
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(3e-8);
      }

      //   6.1 a
      P.push_back(new Themis::prior_linear(0.0,100.0));
      means.push_back(8.0);
      ranges.push_back(0.08);
      if (!static_image)
      {
	//   6.2 a
	P.push_back(new Themis::prior_linear(-1e-2,1e-2));
	means.push_back(0.0);
	ranges.push_back(1e-6);
      }

      //   7.1 Ig
      P.push_back(new Themis::prior_linear(0.0,1.0));
      means.push_back(0.5);
      ranges.push_back(0.005);
      if (!static_image)
      {
	//   7.2 Ig
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(1e-8);
      }

      //   8.1 Position angle
      P.push_back(new Themis::prior_linear(-M_PI,M_PI));
      means.push_back(2.0);
      ranges.push_back(0.02);
      if (!static_image)
      {
	//   8.2 Position angle
	P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	means.push_back(0.0);
	ranges.push_back(1e-8*M_PI);
      }

      //   9.1 Smoothing kernel size
      P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
      means.push_back(1*uas2rad);
      ranges.push_back(0.01*uas2rad);
      if (!static_image)
      {
	//   9.2 Smoothing kernel size
	P.push_back(new Themis::prior_linear(-1e-2*uas2rad,1e-2*uas2rad));
	means.push_back(0.0*uas2rad);
	ranges.push_back(1e-6*uas2rad);
      }

      //  10.1 Smoothing kernel Asymmetry
      P.push_back(new Themis::prior_linear(0.0,1e-6));
      means.push_back(1e-8);
      ranges.push_back(1e-10);
      if (!static_image)
      {
	//  10.2 Smoothing kernel Asymmetry
	P.push_back(new Themis::prior_linear(-1e-12,1e-12));
	means.push_back(0.0);
	ranges.push_back(0.1e-16);
      }

      //  11.1 Smoothing kernel phi
      P.push_back(new Themis::prior_linear(0,M_PI));
      means.push_back(1.64223);
      ranges.push_back(0.01*M_PI);
      if (!static_image)
      {
	//  11.2 Smoothing kernel phi
	P.push_back(new Themis::prior_linear(-1e-10*M_PI,1e-10*M_PI));
	means.push_back(0.0);
	ranges.push_back(1e-12*M_PI);
      }


      if (offset_coordinates=="Cartesian")
      {
	//  12.1 x offset
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
        {
	  //  12.2 x offset
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}

	//  13.1 y offset
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
        {
	  //  13.2 y offset
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}
      }
      else if (offset_coordinates=="polar")
      {
	//  12.1 r offset
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(5e-7*uas2rad);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
        {
	  //  12.2 r offset
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}

	//  13.1 theta offset
	P.push_back(new Themis::prior_linear(-M_PI,M_PI));
	means.push_back(0.0);
	ranges.push_back(1e-3*M_PI);
	if (!static_image)
        {
	  //  13.2 theta offset
	  P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	  means.push_back(0.0);
	  ranges.push_back(1e-6*M_PI);
	}
      }
	
      first_component = false;
    }
    if (add_a)
    {
      for (size_t ia=0; ia<number_a; ++ia)
      {
	if (world_rank==0)
	  std::cout << "Pushed back a prior list, now at " << means.size() << std::endl;
	
	//   0 Itot 
	P.push_back(new Themis::prior_linear(0,3));
	means.push_back(0.1);
	ranges.push_back(0.001);
	if (!static_image)
        {
	  //   0 Itot 
	  P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	  means.push_back(0.0);
	  ranges.push_back(2e-6);
	}

	//   1.1 Size
	P.push_back(new Themis::prior_linear(0.0,100*uas2rad));
	means.push_back(10*uas2rad);
	ranges.push_back(0.01*uas2rad);
	if (!static_image)
        {
	  //   1.2 Size
	  P.push_back(new Themis::prior_linear(-1e-2*uas2rad,1e-2*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-4*uas2rad);
	}

	//   2.1 Asymmetry
	P.push_back(new Themis::prior_linear(0.0,0.99));
	means.push_back(0.1);
	ranges.push_back(0.001);
	if (!static_image)
        {
	  //   2.2 Asymmetry
	  P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	  means.push_back(0.0);
	  ranges.push_back(1e-8);
	}

	//   3.1 Position angle
	P.push_back(new Themis::prior_linear(0,M_PI));
	means.push_back(2.5);
	ranges.push_back(2.5e-2);
	if (!static_image)
        {
	  //   3.2 Position angle
	  P.push_back(new Themis::prior_linear(-4e-3*M_PI,4e-3*M_PI));
	  means.push_back(0.0);
	  ranges.push_back(1e-5*M_PI);
	}

	if (first_component) 
	{
	  if (offset_coordinates=="Cartesian")
	  {
	    //  4.1 x offset
	    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	    means.push_back(0.0);
	    ranges.push_back(1e-8*uas2rad);
	    if (!static_image)
	    {
	      //  4.2 x offset
	      P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-14*uas2rad);
	    }

	    //  5.1 y offset
	    P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	    means.push_back(0.0);
	    ranges.push_back(1e-8*uas2rad);
	    if (!static_image)
	    {
	      //  5.2 y offset
	      P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-14*uas2rad);
	    }
	  }
	  else if (offset_coordinates=="polar")
	  {
	    //  4.1 r offset
	    P.push_back(new Themis::prior_linear(0.0*uas2rad,1e-6*uas2rad));
	    means.push_back(5e-7*uas2rad);
	    ranges.push_back(1e-8*uas2rad);
	    if (!static_image)
	    {
	      //  4.2 r offset
	      P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-14*uas2rad);
	    }

	    //  5.1 theta offset
	    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
	    means.push_back(0.0);
	    ranges.push_back(1e-3*M_PI);
	    if (!static_image)
	    {
	      //  5.2 theta offset
	      P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	      means.push_back(0.0);
	      ranges.push_back(1e-6*M_PI);
	    }
	  }	  
	}
	else
	{
	  if (offset_coordinates=="Cartesian")
	  {
	    //   4.1 x offset
	    P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
	    means.push_back(2*uas2rad);
	    ranges.push_back(0.02*uas2rad);
	    if (!static_image)
	    {
	      //   4.2 x offset
	      P.push_back(new Themis::prior_linear(-1e-2*uas2rad,1e-2*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-4*uas2rad);
	    }

	    //   5.1 y offset
	    P.push_back(new Themis::prior_linear(-200*uas2rad,200*uas2rad));
	    means.push_back(2*uas2rad);
	    ranges.push_back(0.02*uas2rad);
	    if (!static_image)
            {
	      //   5.2 y offset
	      P.push_back(new Themis::prior_linear(-1e-2*uas2rad,1e-2*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-4*uas2rad);
	    }
	  }
	  else if (offset_coordinates=="polar")
	  {
	    //  4.1 r offset
	    P.push_back(new Themis::prior_linear(0*uas2rad,250*uas2rad));
	    means.push_back(10.0*uas2rad);
	    ranges.push_back(1.0*uas2rad);
	    if (!static_image)
	    {
	      //  4.2 r offset
	      P.push_back(new Themis::prior_linear(-1e-3*250*uas2rad,1e-3*250*uas2rad));
	      means.push_back(0.0);
	      ranges.push_back(1e-5*uas2rad);
	    }

	    //  5.1 theta offset
	    P.push_back(new Themis::prior_linear(-M_PI,M_PI));
	    means.push_back(0.0);
	    ranges.push_back(1e-3*M_PI);
	    if (!static_image)
	    {
	      //  5.2 theta offset
	      P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	      means.push_back(0.0);
	      ranges.push_back(1e-6*M_PI);
	    }
	  }	  
	}
	first_component = false;
      }
    }
    if (add_g)
    {
      if (world_rank==0)
	std::cout << "Pushed back g prior list, now at " << means.size() << std::endl;

      //   0.1 Itot 
      P.push_back(new Themis::prior_linear(0,3));
      means.push_back(0.1);
      ranges.push_back(0.001);
      if (!static_image)
      {
	//   0.2 Itot 
	P.push_back(new Themis::prior_linear(-1e-3,1e-3));
	means.push_back(0.0);
	ranges.push_back(1e-6);
      }

      //   1.1 Size
      P.push_back(new Themis::prior_linear(0.0,25*uas2rad));
      means.push_back(5*uas2rad);
      ranges.push_back(0.02*uas2rad);
      if (!static_image)
      {
	//   1.2 Size
	P.push_back(new Themis::prior_linear(-25e-3,25e-3*uas2rad));
	means.push_back(0.0);
	ranges.push_back(25e-5*uas2rad);
      }

      if (offset_coordinates=="Cartesian")
      {
	//   2.1 x offset -- FIX AT CENTER NO MATTER WHAT
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //   2.2 x offset -- FIX AT CENTER NO MATTER WHAT
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}

	//   3.1 y offset -- FIX AT CENTER NO MATTER WHAT
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //   3.2 y offset -- FIX AT CENTER NO MATTER WHAT
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}
      }
      else if (offset_coordinates=="polar")
      {
	//  2.1 r offset
	P.push_back(new Themis::prior_linear(0.0*uas2rad,1e-6*uas2rad));
	means.push_back(5e-7*uas2rad);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //  2.2 r offset
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}
	
	//  3.1 theta offset
	P.push_back(new Themis::prior_linear(-M_PI,M_PI));
	means.push_back(0.0);
	ranges.push_back(1e-3*M_PI);
	if (!static_image)
	{
	  //  3.2 theta offset
	  P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	  means.push_back(0.0);
	  ranges.push_back(1e-6*M_PI);
	}
      } 

      first_component = false;
    }
    if (add_G)
    {
      if (world_rank==0)
	std::cout << "Pushed back G prior list, now at " << means.size() << std::endl;

      //   0.1 Itot 
      P.push_back(new Themis::prior_linear(0,10.));
      means.push_back(0.05);
      ranges.push_back(0.0005);
      if (!static_image)
      {
	//   0.2 Itot 
	P.push_back(new Themis::prior_linear(-1e-5,1e-5));
	means.push_back(0.0);
	ranges.push_back(1e-7);
      }

      //   1.1 Size
      P.push_back(new Themis::prior_logarithmic(1e4*uas2rad,1e7*uas2rad));
      means.push_back(1e5*uas2rad);
      ranges.push_back(1e3*uas2rad);
      if (!static_image)
      {
	//   1.2 Size
	P.push_back(new Themis::prior_linear(-10*uas2rad,10*uas2rad));
	means.push_back(0.0);
	ranges.push_back(0.1*uas2rad);
      }

      if (offset_coordinates=="Cartesian")
      {
	//   2.1 x offset -- FIX AT CENTER NO MATTER WHAT
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //   2.2 x offset -- FIX AT CENTER NO MATTER WHAT
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}

	//   3.1 y offset -- FIX AT CENTER NO MATTER WHAT
	P.push_back(new Themis::prior_linear(-1e-6*uas2rad,1e-6*uas2rad));
	means.push_back(0.0);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //   3.2 y offset -- FIX AT CENTER NO MATTER WHAT
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}
      }
      else if (offset_coordinates=="polar")
      {
	//  2.1 r offset
	P.push_back(new Themis::prior_linear(0.0*uas2rad,1e-6*uas2rad));
	means.push_back(5e-7*uas2rad);
	ranges.push_back(1e-8*uas2rad);
	if (!static_image)
	{
	  //  2.2 r offset
	  P.push_back(new Themis::prior_linear(-1e-12*uas2rad,1e-12*uas2rad));
	  means.push_back(0.0);
	  ranges.push_back(1e-14*uas2rad);
	}
	
	//  3.1 theta offset
	P.push_back(new Themis::prior_linear(-M_PI,M_PI));
	means.push_back(0.0);
	ranges.push_back(1e-3*M_PI);
	if (!static_image)
	{
	  //  3.2 theta offset
	  P.push_back(new Themis::prior_linear(-1e-3*M_PI,1e-3*M_PI));
	  means.push_back(0.0);
	  ranges.push_back(1e-6*M_PI);
	}
      } 

      first_component = false;
    }

    if (world_rank==0)
      std::cout << "Finished pusing prior lists, now at " << means.size() << std::endl;


    if (Number_start_params>int(means.size()))
    {
      std::cerr << "ERROR: Too many start parameters provided for chosen model.\n";
      std::exit(1);
    }
    for (size_t j=0; j<start_parameter_list[0].size(); ++j)
    {
      std::cerr << std::setw(4) << world_rank
		<< std::setw(5) << j
		<< std::setw(15) << start_parameter_list[0][j]
		<< std::setw(15) << means[j]
		<< std::setw(15) << ranges[j]
		<< std::setw(15) << ranges[j]*1e-5
		<< std::endl;
      // Set desired means
      means[j] = start_parameter_list[0][j];
      // Restrict initial ranges
      ranges[j] *= 1e-5;
    }
    
    // Set the likelihood functions
    // Visibility Amplitudes
    std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
    // Specify the priors we will be assuming (to 20% by default)
    std::vector<double> station_gain_priors(station_codes.size(),0.2);
    station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%

    std::vector< Themis::likelihood_optimal_gain_correction_visibility_amplitude* > lva;
    for (size_t index=0; index<VM.size(); ++index)
      lva.push_back( new Themis::likelihood_optimal_gain_correction_visibility_amplitude((*VM[index]),image,station_codes,station_gain_priors) );
    
    // Closure Phases
    Themis::likelihood_closure_phase lcp(CP,image_var);
    //Themis::likelihood_closure_phase lcp(CP,image);
    
    std::vector<Themis::likelihood_base*> L;
    for (size_t index=0; index<lva.size(); ++index)
      L.push_back(lva[index]);
    L.push_back(&lcp);
    
    // Set the weights for likelihood functions
    std::vector<double> W(L.size(), 1.0);
  
    // Make a likelihood object
    Themis::likelihood L_obj(P, L, W);

    // Output residual data
    std::vector< std::string > VA_res_name, gc_name;
    for (size_t index=0; index<VM.size(); ++index)
    {
      std::stringstream VA_tmp, gc_tmp;
      VA_tmp << "VA_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
      VA_res_name.push_back(VA_tmp.str());
      gc_tmp << "gain_corrections_" << std::setfill('0') << std::setw(5) << index << ".d";
      gc_name.push_back(gc_tmp.str());
    }
    std::string CP_res_name="CP_residuals.d";
    //std::stringstream CP_res_name;
    //CP_res_name << "CP_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    L_obj(means);


    for (size_t index=0; index<VM.size(); ++index)
    {
      lva[index]->output_gain_corrections(gc_name[index]);
      L[index]->output_model_data_comparison(VA_res_name[index]);
    }
    L[L.size()-1]->output_model_data_comparison(CP_res_name);
    
    // Create a sampler object
    Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(seed+world_rank);

    // Generate a chain
    int Number_of_chains = Number_of_walkers; //360;
    int Number_of_temperatures = Number_of_tempering_levels; //40; //20; //16; // 8;
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
    //MCMC_obj.set_tempering_schedule(1000.,1.,1.4);
    MCMC_obj.set_tempering_schedule(Tempering_time,1.,Tempering_ladder);
    
  
    /////////////
    // First run from random positions
    // Run the Sampler
    std::stringstream ChainA_name, LklhdA_name, Chi2A_name;
    ChainA_name << "Chain_A_" << std::setfill('0') << std::setw(5) << 0 << ".dat";
    LklhdA_name << "Lklhd_A_" << std::setfill('0') << std::setw(5) << 0 << ".dat";
    Chi2A_name << "Chi2_A_" << std::setfill('0') << std::setw(5) << 0 << ".dat";

    std::cerr << "About to start running sampler, should always get here!" << std::endl;
    
    MCMC_obj.run_sampler(L_obj, Number_of_steps_A, Temperature_stride, Chi2_stride, ChainA_name.str(), LklhdA_name.str(), Chi2A_name.str(), means, ranges, var_names, restart_flag, out_precision, verbosity);

    ////////////
    // Prepare for second run:
    // Get the best fit and produce residual/gain files
    std::vector<double> pmax = MCMC_obj.find_best_fit(ChainA_name.str(),LklhdA_name.str());
    L_obj(pmax);

    std::cerr << "Read pmax run A\n";

    int Ndof_VM = 0;
    for (size_t index=0; index<VM.size(); ++index)
      Ndof_VM += int(VM[index]->size()) - int(lva[index]->number_of_independent_gains());
    Ndof_VM -= int(image.size());
    int Ndof_CP = int(CP.size()) - int(image.size());
    int Ndof = Ndof_VM + Ndof_CP + int(image.size());

    double chi2_VM = 0;
    for (size_t index=0; index<VM.size(); ++index)
      chi2_VM += L[index]->chi_squared(pmax);
    double chi2_CP = L[L.size()-1]->chi_squared(pmax);
    double chi2 = L_obj.chi_squared(pmax);
    double Lmax = L_obj(pmax);

    
    if (world_rank==0)
    {
      sumoutA << std::setw(10) << 0;
      for (size_t j=0; j<image.size(); ++j)
	sumoutA << std::setw(15) << pmax[j];
      sumoutA << std::setw(15) << chi2_VM/Ndof_VM
	      << std::setw(15) << chi2_CP/Ndof_CP
	      << std::setw(15) << chi2/Ndof
	      << std::setw(15) << Lmax;
      for (size_t index=size_t(istart); index<size_t(iend); index++)
	sumoutA << "     " << vm_file_names[index];
      for (size_t index=size_t(istart); index<size_t(iend); index++)
	sumoutA << "     " << cp_file_names[index];
      sumoutA << std::endl;
    }
    for (size_t index=0; index<VM.size(); ++index)
    {
      lva[index]->output_gain_corrections(gc_name[index]);
      L[index]->output_model_data_comparison(VA_res_name[index]);
    }
    L[L.size()-1]->output_model_data_comparison(CP_res_name);

    

    /////////
    // Set up the revised means and ranges
    for (size_t j=0; j<image.size(); ++j)
    {
      means[j] = pmax[j];
      if (Number_start_params==0)
      	ranges[j] *= 1e-5;
    }
    
    // First run from random positions
    // Run the Sampler
    std::stringstream ChainB_name, LklhdB_name, Chi2B_name;
    ChainB_name << "Chain_B_" << std::setfill('0') << std::setw(5) << 0 << ".dat";
    LklhdB_name << "Lklhd_B_" << std::setfill('0') << std::setw(5) << 0 << ".dat";
    Chi2B_name << "Chi2_B_" << std::setfill('0') << std::setw(5) << 0 << ".dat";
    MCMC_obj.run_sampler(L_obj, Number_of_steps_B, Temperature_stride, Chi2_stride, ChainB_name.str(), LklhdB_name.str(), Chi2B_name.str(), means, ranges, var_names, restart_flag, out_precision, verbosity);



    ////////////
    // Record final results
    // Get the best fit and produce residual/gain files
    pmax = MCMC_obj.find_best_fit(ChainB_name.str(),LklhdB_name.str());
    L_obj(pmax);

    chi2_VM = 0;
    for (size_t index=0; index<VM.size(); ++index)
      chi2_VM += L[index]->chi_squared(pmax);
    chi2_CP = L[L.size()-1]->chi_squared(pmax);
    chi2 = L_obj.chi_squared(pmax);
    Lmax = L_obj(pmax);

    
    if (world_rank==0)
    {
      sumoutB << std::setw(10) << 0;
      for (size_t j=0; j<image.size(); ++j)
	sumoutB << std::setw(15) << pmax[j];
      sumoutB << std::setw(15) << chi2_VM/Ndof_VM
	      << std::setw(15) << chi2_CP/Ndof_CP
	      << std::setw(15) << chi2/Ndof
	      << std::setw(15) << Lmax;
      for (size_t index=size_t(istart); index<size_t(iend); index++)
	sumoutB << "     " << vm_file_names[index];
      for (size_t index=size_t(istart); index<size_t(iend); index++)
	sumoutB << "     " << cp_file_names[index];
      sumoutB << std::endl;
    }
    for (size_t index=0; index<VM.size(); ++index)
    {
      lva[index]->output_gain_corrections(gc_name[index]);
      L[index]->output_model_data_comparison(VA_res_name[index]);
    }
    L[L.size()-1]->output_model_data_comparison(CP_res_name);



  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
