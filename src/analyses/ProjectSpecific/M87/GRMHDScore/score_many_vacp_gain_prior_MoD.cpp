/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Hung-Yi Pu
    \date  Nov, 2018
    \brief test model_image_score clas by mcmc runs *with* gain calibration
    \details Takes a file list generated via something like:
             readlink -f ~/Themis/Themis/sim_data/Score/README.txt ~/Themis/Themis/sim_data/Score/example_image.dat > file_list
*/

#include "data_visibility_amplitude.h"
#include "model_image_score.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "image_family_error.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  std::string image_file_list="";
  int istart = 0;
  int iend = -1;
  bool reflect_image=false;
  bool include_theory_errors=false;
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
	  std::cerr << "ERROR: An argument must be provided after --start, -s.\n";
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
	  std::cerr << "ERROR: An argument must be provided after --end, -e.\n";
	std::exit(1);
      }
    }
    else if (opt=="--imagefilelist" || opt=="-f")
    {
      if (k<argc)
	image_file_list = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --imagefilelist, -f.\n";
	std::exit(1);
      }
    }
    else if (opt=="--reflect" || opt=="-r")
    {
      reflect_image=true;
      std::cout << "Reflecting images." << std::endl;
    }
    else if (opt=="--theory-error" || opt=="-te")
    {
      include_theory_errors=true;
      std::cout << "Including theoretical variability as independent errors." << std::endl;
    }
    else
    {
      if (world_rank==0)
	std::cerr << "ERROR: Unrecognized option " << opt << "\n";
      std::exit(1);
    }
  }  
  if (image_file_list=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No image file list was provided.\n"
		<< "       The image file list is simply a text file\n"
		<< "       with one file name per line containing the\n"
		<< "       *absolute* path to the image files.  The\n"
		<< "       first line must be the README.txt file that\n"
		<< "       describes the image file parameters.\n";
    std::exit(1);
  }

  // Read in the file list
  std::string README_file_name, stmp;
  std::vector<std::string> image_file_names;
  std::fstream ifnin(image_file_list);
  ifnin >> README_file_name;
  for (ifnin >> stmp;  !ifnin.eof(); ifnin >> stmp)
    image_file_names.push_back(stmp);
  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "README file:\n\t" << README_file_name << std::endl;
    std::cout << "Image files: (" << image_file_names.size() << ")\n";
    for (size_t i=0; i<image_file_names.size(); ++i)
      std::cout << "\t" << image_file_names[i] << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
  }
  // Sort out the end point
  if (iend<0 || iend>int(image_file_names.size()) )
  {
    if (world_rank==0)
      std::cerr << "WARNING: iend not set or set past end of file list.\n";
    iend = image_file_names.size();
  }
  // Sort out the start point (after iend is sorted)
  if (istart>iend)
  {
    if (world_rank==0)
      std::cerr << "ERROR: istart is set to beyond iend or past end of file list.\n";
    std::exit(1);
  }

  // Prepare the output summary file
  std::stringstream sumoutname;
  sumoutname << "fit_summaries_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  std::ofstream sumout;
  if (world_rank==0)
  {
   sumout.open(sumoutname.str().c_str());
   sumout << std::setw(10) << "# Index"
	  << std::setw(15) << "I (Jy)"
	  << std::setw(15) << "M/D (uas)"
	  << std::setw(15) << "PA (deg)"
	  << std::setw(15) << "VA red. chisq"
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
  //   1. First element of list is run in longer chain to get good starting location (cutting PA to +-0.5 rad or +-30 deg)
  //   2. All remaining elements are run in shorter chains that exploit more rapid convergence

  double PAguess = M_PI;


  // Read in data
  Themis::data_visibility_amplitude VM_obs(Themis::utils::global_path("eht_data/ER5/Mexico/Processed_1/hi/VM_hops_3598_M87+netcal.d"),"HH");
  Themis::data_closure_phase CP_obs(Themis::utils::global_path("eht_data/ER5/Mexico/Processed_1/hi/CP_hops_3598_M87+netcal.d"));

  // Prepare theory error computation
  Themis::image_family_static_error theory_error;
  Themis::data_visibility_amplitude VM;
  Themis::data_closure_phase CP;
  if (include_theory_errors)
  {
    std::vector<double> p;
    p.push_back(0.6);
    p.push_back(3.66);
    p.push_back(-72*M_PI/180.);
    theory_error.generate_error_estimates(image_file_names,README_file_name,p);
    VM = theory_error.data_visibility_amplitude(VM_obs);
    CP = theory_error.data_closure_phase(CP_obs);
  }
  else
  {
    VM = VM_obs;
    CP = CP_obs;
  }
  
  for (size_t index=size_t(istart); index<size_t(iend); ++index)
  {

    // Choose the model to compare
    //Themis::model_image_score image(Themis::utils::global_path("sim_data/Score/example_image.dat"),Themis::utils::global_path("sim_data/Score/README.txt"));
    Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image);
				  
    // Container of base prior class pointers
    std::vector<Themis::prior_base*> P;
    std::vector<double> means, ranges;
    
    P.push_back(new Themis::prior_linear(0.1,10.)); // total I
    means.push_back(0.5);  
    ranges.push_back(0.1);
    
    
    //====gaussian priors for M/D
    //P.push_back(new Themis::prior_gaussian(3.75,0.5)); // (M/D) in uas
    //means.push_back(3.5);  
    //ranges.push_back(0.5);
	  
    //====linear priors for M/D
    P.push_back(new Themis::prior_linear(2.01,5.34)); // (M/D) in uas
    means.push_back(3.75);  
    ranges.push_back(0.5);
    
    P.push_back(new Themis::prior_linear(-2.0*M_PI,2.0*M_PI)); // position angle
    means.push_back(PAguess);
    if (index==size_t(istart))
      ranges.push_back(M_PI);
    else
      ranges.push_back(0.5);
   
	  

	  

	  

    // vector to hold the name of variables, if the names are provided it would be added
    // as the header to the chain file
    std::vector<std::string> var_names;
    var_names.push_back("I");
    var_names.push_back("M/D");
    var_names.push_back("PA");

    // Set the likelihood functions
    std::vector<Themis::likelihood_base*> L;

    // Visability amplitudes with gain correction
    std::vector<std::string> stations = Themis::utils::station_codes("uvfits 2017");
    std::vector<double> gain_sigmas(stations.size(),0.2);
    gain_sigmas[4] = 1.0; // LMT is poorly calibrated
    Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(VM,image,stations,gain_sigmas);
    L.push_back(&L_ogva);
    
    // Closure phases
    Themis::likelihood_closure_phase L_cp(CP,image);
    L.push_back(&L_cp);
    
    // Set the weights for likelihood functions
    std::vector<double> W(L.size(), 1.0);
    
    // Make a likelihood object
    Themis::likelihood L_obj(P, L, W);
    
    // Output residual data
    L_obj(means);
    std::stringstream VA_res_name, CP_res_name, gc_name;
    VA_res_name << "VA_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    CP_res_name << "CP_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    gc_name << "gain_corrections_" << std::setfill('0') << std::setw(5) << index << ".d";
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());
    L_ogva.output_gain_corrections(gc_name.str());

    // Create a sampler object
    //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
    Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);

    // Generate a chain
    int Number_of_chains = 20;
    int Number_of_temperatures = 4;
    int Number_of_procs_per_lklhd = 1;
    int Number_of_steps = 500;
    int Temperature_stride = 50;
    int Chi2_stride = 100;
    int Ckpt_frequency = 2000;
    bool restart_flag = false;
    int out_precision = 8;
    int verbosity = 0;

    if (index==size_t(istart))
      Number_of_steps *= 5;

    // Set the CPU distribution
    MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);

    // Set a checkpoint
    MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
    // Run the Sampler
    std::stringstream Chain_name, Lklhd_name, Chi2_name;
    Chain_name << "Chain_" << std::setfill('0') << std::setw(5) << index << ".dat";
    Lklhd_name << "Lklhd_" << std::setfill('0') << std::setw(5) << index << ".dat";
    Chi2_name << "Chi2_" << std::setfill('0') << std::setw(5) << index << ".dat";
    
    MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, Chain_name.str(), Lklhd_name.str(), Chi2_name.str(), means, ranges, var_names, restart_flag, out_precision, verbosity);

    // Get the best fit and produce residual/gain files
    std::vector<double> pmax = MCMC_obj.find_best_fit(Chain_name.str(),Lklhd_name.str());
    L_obj(pmax);

    int Ndof_VM = int(VM.size()) - int(L_ogva.number_of_independent_gains()) - int(image.size());
    int Ndof_CP = int(CP.size()) - int(image.size());
    int Ndof = int(VM.size()+CP.size()) - int(L_ogva.number_of_independent_gains()) - int(image.size());

    double chi2_VM = L[0]->chi_squared(pmax);
    double chi2_CP = L[1]->chi_squared(pmax);
    double chi2 = L_obj.chi_squared(pmax);
    double Lmax = L_obj(pmax);
    
    if (world_rank==0)
      sumout << std::setw(10) << index
	     << std::setw(15) << pmax[0]
	     << std::setw(15) << pmax[1]
	     << std::setw(15) << pmax[2]*180./M_PI
	     << std::setw(15) << chi2_VM/Ndof_VM
	     << std::setw(15) << chi2_CP/Ndof_CP
	     << std::setw(15) << chi2/Ndof
	     << std::setw(15) << Lmax
	     << "     " << image_file_names[index]
	     << std::endl;
    
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());
    L_ogva.output_gain_corrections(gc_name.str());
    
    PAguess = pmax[2];
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
