/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Hung-Yi Pu
    \date  Jan. 2023
    \brief test model_image_score clas by mcmc runs *with* gain calibration
    \details Takes a file list generated via something like:
             readlink -f ~/Themis/Themis/sim_data/Score/README.txt ~/Themis/Themis/sim_data/Score/example_image.dat > file_list
*/

#include "data_visibility_amplitude.h"
#include "model_image_score.h"

#include "sampler_deo_tempering_MCMC.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"

#include "likelihood_power_tempered.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"

#include "image_family_error.h"
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

  int verbosity = 0;


  int seed = 42;

  bool restart_flag = false;

  std::string visibility_amplitude_data_file=Themis::utils::global_path("eht_data/M87_2018/VM_hops_3644_M87_b4_wosb.netcal_10s_scanavg_f1_tygtd.dat");
  std::string closure_phase_data_file=Themis::utils::global_path("eht_data/M87_2018/CP_hops_3644_M87_b4_wosb.netcal_10s_scanavg_f1_tygtd.dat");


  // Tempering stuff default options
  size_t number_of_rounds = 6;

  //AFSS adatation parameters
  int init_buffer = 25;
  int window = 150;
  int number_of_adaptation = 500;
  bool save_adapt = true;
  int refresh = 10;
    
  
  for (int k=1; k<argc;)
  {
    std::string opt(argv[k++]);

    if (opt=="--start" || opt=="-s")
    {
      if (k<argc)
	istart = atoi(argv[k++]);
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
	iend = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --end, -e.\n";
	std::exit(1);
      }
    }
    else if (opt=="--seed" )
    {
      if (k<argc)
	seed = atoi(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --seed.\n";
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
    else if (opt=="--visibility-amplitude-data" || opt=="-vad")
    {
      if (k<argc)
	visibility_amplitude_data_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --visiiblity-amplitude-data, -vad.\n";
	std::exit(1);
      }
    }
    else if (opt=="--closure-phase-data" || opt=="-cpd")
    {
      if (k<argc)
	closure_phase_data_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --closure-phase-data, -cpd.\n";
	std::exit(1);
      }
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
    std::cout << "VA data file:\n\t" << visibility_amplitude_data_file << std::endl;
    std::cout << "CP data file:\n\t" << closure_phase_data_file << std::endl;
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
	  << "     VA Data file"
	  << "     CP Data file"
	  << std::endl;
  }


  ////////////////////////////
  // Begin loop over the relevant portion of file_name_list and running chains.
  // ALL items are in the loop to scope potential problems, though this is probably unncessary.
  // This happens in two steps.
  //   1. First element of list is run in longer chain to get good starting location (cutting PA to +-0.5 rad or +-30 deg)
  //   2. All remaining elements are run in shorter chains that exploit more rapid convergence

  //double PAguess = M_PI;


  // Read in data
  Themis::data_visibility_amplitude VM_obs(visibility_amplitude_data_file,"HH");
  Themis::data_closure_phase CP_obs(closure_phase_data_file);

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
    Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image,0); // No window
    //Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image,2); // Hann
    //Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image,3); // Blackman
    //Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image,-3); // Cyl Blackman
    //Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image,-10); // Cyl Bump
				  
    // Container of base prior class pointers
    std::vector<Themis::prior_base*> P;
    std::vector<double> means, ranges;
    
    P.push_back(new Themis::prior_linear(0.1, 5.0)); // total I
    means.push_back(0.5);  
    ranges.push_back(0.1);
    
    P.push_back(new Themis::prior_linear(1.,8.)); // (M/D) in uas
    means.push_back(4.);  
    ranges.push_back(1.);
    
    // P.push_back(new Themis::prior_linear(-10.0*M_PI,10.0*M_PI)); // position angle
    P.push_back(new Themis::prior_linear(0.0, 2*M_PI)); // position angle
    //means.push_back(PAguess);
    //if (index==size_t(istart))
    //  ranges.push_back(M_PI);
    //else
    //  ranges.push_back(0.5);
    means.push_back(M_PI);
    ranges.push_back(M_PI);

    // Define the pbest vector (used below)
    std::vector<double> pbest =  means;


    // vector to hold the name of variables, if the names are provided it would be added
    // as the header to the chain file
    std::vector<std::string> var_names;
    var_names.push_back("I");
    var_names.push_back("M/D");
    var_names.push_back("PA");

    // Set the likelihood functions
    std::vector<Themis::likelihood_base*> L;

    // Visability amplitudes with gain correction
    std::vector<std::string> stations = Themis::utils::station_codes("uvfits 2018");
    std::vector<double> gain_sigmas(stations.size(),0.2);
    // following Britt's gain budget
    gain_sigmas[1] = 0.2; // Apex
    gain_sigmas[4] = 0.3; // LMT
    gain_sigmas[5] = 0.2; // PV
    gain_sigmas[8] = 1.0; // GLT
    Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(VM,image,stations,gain_sigmas);
    L.push_back(&L_ogva);
    
    // Closure phases
    Themis::likelihood_closure_phase L_cp(CP,image);
    L.push_back(&L_cp);
    
    // Set the weights for likelihood functions
    std::vector<double> W(L.size(), 1.0);
    
    // Make a likelihood object
    Themis::likelihood L_obj(P, L, W);
    Themis::likelihood_power_tempered L_beta(L_obj);
    
    // Output residual data
    L_obj(means);
    std::stringstream VA_res_name, CP_res_name, gc_name, chain_name, state_name, summary_name, anneal_name;
    chain_name  << "chain_"     << std::setfill('0') << std::setw(5) << index << ".dat";
    state_name  << "state_"     << std::setfill('0') << std::setw(5) << index << ".dat";
    summary_name<< "summary_"   << std::setfill('0') << std::setw(5) << index << ".dat";
    anneal_name << "annealing_" << std::setfill('0') << std::setw(5) << index << ".dat";
    VA_res_name << "VA_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    CP_res_name << "CP_residuals_" << std::setfill('0') << std::setw(5) << index << ".d";
    gc_name << "gain_corrections_" << std::setfill('0') << std::setw(5) << index << ".d";
    L[0]->output_model_data_comparison(VA_res_name.str());
    L[1]->output_model_data_comparison(CP_res_name.str());
    //L[0]->output_model_data_comparison(CP_res_name.str());
    L_ogva.output_gain_corrections(gc_name.str());

    int Ngains = L_ogva.number_of_independent_gains();

    // Create a sampler object
    //Themis::sampler_affine_invariant_tempered_MCMC MCMC_obj(42+world_rank);
    //Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(142+world_rank);

    
    Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> 
	          DEO(seed, L_beta, var_names, means.size());

    DEO.set_cpu_distribution(world_size, 1);
    DEO.set_output_stream(chain_name.str(), state_name.str(), summary_name.str());
    DEO.set_annealing_output(anneal_name.str());
    DEO.set_checkpoint(1000000,"MCMC.ckpt");

    DEO.set_annealing_schedule(5.0); 
    DEO.set_deo_round_params(2, 2);


    DEO.set_initial_location(means);




    Themis::sampler_automated_factor_slice_sampler_MCMC* kexplore = DEO.get_sampler();
    kexplore->set_adaptation_parameters(number_of_adaptation, save_adapt);
    kexplore->set_window_parameters(init_buffer, window);

    int round_start = 0;
    if (restart_flag){
      DEO.read_checkpoint("MCMC.ckpt");
      round_start = DEO.get_round();
      restart_flag=false;
    }


    for ( size_t rep = 0; rep < number_of_rounds; ++rep)
    {
      if ( world_rank == 0 )
        std::cerr << "Started round " << rep << std::endl;

      //Generate fit_summaries file
      std::stringstream sumoutname;
      sumoutname << "round"  << std::setfill('0') << std::setw(3)<< rep 
                 << "_fit_summaries" << std::setfill('0') << std::setw(5) << index << ".txt";
      std::ofstream sumout;
      if (world_rank==0)
      {
        sumout.open(sumoutname.str().c_str());
        sumout << std::setw(10) << "# Index";
        sumout << std::setw(15) << "I"
               << std::setw(15) << "M/D"
               << std::setw(15) << "PA";
	  
	      sumout << std::setw(15) << "VA rc2"
	             << std::setw(15) << "CP rc2"
               << std::setw(15) << "Total rc2"
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
	    std::stringstream gc_name, cgc_name;
	    gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" 
                << std::setfill('0') << std::setw(5) << index << ".d";
	    L_ogva.output_gain_corrections(gc_name.str());
    
    // Generate summary and ancillary output

    //   Summary file parameter location:
    if (world_rank==0)
    {
      sumout << std::setw(10) << 0;
      for (size_t k=0; k<3; ++k)
	      sumout << std::setw(15) << pbest[k];
    }

    //   Data-set specific output
    double VA_rchi2=0.0, CP_rchi2=0.0, rchi2=0.0;
    // output residuals
    std::stringstream va_res_name, cp_res_name;
    va_res_name << "round" << std::setfill('0') << std::setw(3) << rep << "_VA_residuals_" 
               << std::setfill('0') << std::setw(5) << index << ".d";
    L[0]->output_model_data_comparison(va_res_name.str());
    cp_res_name << "round" << std::setfill('0') << std::setw(3) << rep << "_CP_residuals_" 
               << std::setfill('0') << std::setw(5) << index << ".d";
    L[1]->output_model_data_comparison(cp_res_name.str());
    // Get the data-set specific chi-squared
    VA_rchi2 = (L[0]->chi_squared(pbest));
    CP_rchi2 = (L[1]->chi_squared(pbest));
    if (world_rank==0)
    {
	    sumout << std::setw(15) << VA_rchi2 / ( int(VM_obs.size()) - 3 - Ngains)
	           << std::setw(15) << CP_rchi2 / ( int(CP_obs.size()) - 2);
    }
    rchi2 = L_obj.chi_squared(pbest);
    rchi2 = rchi2 / (VM_obs.size() + CP_obs.size() - 3 - Ngains);

    // Generate summary file      
    if (world_rank==0)
    {
      sumout << std::setw(15) << rchi2
	     << std::setw(15) << Lval;
      sumout << std::setw(15) << "   " << image_file_names[index]
             << "   " << visibility_amplitude_data_file
             << "   " << closure_phase_data_file;
      sumout << std::endl;
    }
    
    // Run the sampler with the given settings
    if (rep<number_of_rounds) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	      std::cerr << "Starting MCMC on round " << rep << std::endl;
      
      DEO.run_sampler( 1, 1, refresh, verbosity);
      clock_t end = clock();
      if (world_rank == 0)
	      std::cerr << "Done MCMC on round " << rep << std::endl
		              << "it took " << (end-start)/CLOCKS_PER_SEC/3600.0 << " hours" << std::endl;
    }
    
    // Reset pbest
    pbest = DEO.get_sampler()->find_best_fit();
 
  }


    //PAguess = pmax[2];
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
