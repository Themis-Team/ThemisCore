/*!
    \file binary_gaussian.cpp
    \author Ali Sarertoosi, Avery Broderick
    \date March 2023
    \brief Driver file to test the model_image_vae_interpolated_riaf.h model.
    \todo 
*/


#include "model_image_vae_interpolated_riaf.h"

#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_power_tempered.h"
#include "utils.h"
#include "read_data.h"
#include "sampler_stan_adapt_diag_e_nuts_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"

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

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;


  // Some specifications


  int Number_temperatures = 20;
  int Number_of_steps = 10; 
  double initial_ladder_spacing = 1.15;
  int thin_factor = 1;
  int Temperature_stride = 20;
  int refresh_rate = 1;
  int tree_depth = 6;
  int number_of_adaption_steps = 2000;
  int Ckpt_frequency = 10; // per swaps
  int verbosity = 0;
  size_t number_of_rounds = 11;
  unsigned int seed = 42;

  std::string annealing_ladder_file = "";
  bool restart_flag = false;

  
  // // int Number_temperatures = world_size;
  // int Number_temperatures = 20;  
  // int Number_of_swaps = 10;
  // int seed = 42;

  // // Tempering stuff default options
  // size_t number_of_rounds = 10;
  // double initial_ladder_spacing = 1.1;
  // int Thin_factor = 1;
  // int Temperature_stride = 10;

  // // AFSS adatation parameters
  // int init_buffer = 10; //25;
  // int window = 0; //150;
  // int number_of_adaptation = 0; //10000;
  // bool save_adapt = true;
  // int refresh = 1; //10;

  // Define model
  Themis::model_image_vae_interpolated_riaf image("vae_model");

  // Read in complex visibility data
  Themis::data_visibility V_data("./V_im1_seed10_netcal_scanavg_scanavg_f1_tygtd.dat","HH");
  
  // Choose gain particulars
  std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  std::vector<double> station_gain_priors(station_codes.size(),0.1);
  
  // Construct likelihood
  Themis::likelihood_optimal_complex_gain_visibility lvg(V_data,image,station_codes,station_gain_priors);
  // Themis::likelihood_visibility lvg(V_data,image);
  std::vector<Themis::likelihood_base*> L;
  L.push_back(&lvg);


  // Output model tag
  image.write_model_tag_file();
  
  
  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means;
  std::vector<std::string> var_names;
  
  // Add priors for the latent variables
  for (size_t j=0; j<image.latent_size(); ++j)
  {
    P.push_back(new Themis::prior_gaussian(0.0,1.0));
    means.push_back(0.0);
  }

  // Add prior for total flux
  P.push_back(new Themis::prior_linear(-5,5));
  means.push_back(0.0);

  // Add prior for M/D rescale
  P.push_back(new Themis::prior_linear(-2,2));
  means.push_back(0.0);

  // Add prior for PA
  P.push_back(new Themis::prior_linear(-M_PI,M_PI));
  means.push_back(0.0);



  // Start values
  std::ifstream fsin("fsstart.txt");
  fsin.ignore(4096,'\n');
  double dtmp;
  fsin >> dtmp;
  for (size_t j=0; j<image.size(); ++j)
    fsin >> means[j];
  fsin.close();

  
  // Check
  if ( (means.size() != P.size())){
    std::cerr << "Error number of means does equal number of priors!\n";
    std::cerr << " means.size: " << means.size()
              << "\n P.size(): " << P.size() << std::endl;
    std::exit(1);
  }
  int nparams = image.size();

  if (world_rank==0)
  {
    for (int k=0; k<nparams; ++k)
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
  }


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);
  Themis::likelihood_power_tempered L_beta(L_obj);
  
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Ndata = 2*V_data.size();
  int Nparam = image.size()-2.0;
  int Ngains = lvg.number_of_independent_gains();
  // int Ngains = 0;
  int NDoF = Ndata - Nparam - Ngains;
  
  // Get the best point 
  std::vector<double> pbest =  means;

  // Generate a chain
  //int Ckpt_frequency = 100;
  // int out_precision = 8;
  // int verbosity = 0;

  //Create the tempering sampler which is templated off of the exploration sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO(seed, L_beta, var_names, means.size());

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
  DEO.get_sampler()->set_max_depth(tree_depth);
  
  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  bool save_warmup = true;
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
  
  
  // Start looping over the rounds
  for ( size_t rep = round_start; rep < number_of_rounds; ++rep)
  {
    if ( world_rank == 0 )
      std::cerr << "Started round " << rep << std::endl;

    // Generate fit_summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3)<< rep << "_fit_summaries.txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "index";
      for (size_t j=0; j<image.latent_size(); ++j)
	sumout << std::setw(15) << "z"+std::to_string(j);
      sumout << std::setw(15) << "flux ratio"
	     << std::setw(15) << "mass ratio"
	     << std::setw(15) << "PA (rad)";
      for (size_t j=0; j<L.size(); ++j) 
	sumout << std::setw(15) << "V" + std::to_string(j) + " rc2";
      sumout << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileNames"
	     << std::endl;
    }

    if ( rep != 0 ){
      pbest = DEO.get_sampler()->find_best_fit();
    }
    
    // Output the current location
    if (world_rank==0)
      for (size_t j=0; j<pbest.size(); ++j)
	std::cout <<"pbest[j]="<< pbest[j] << std::endl;

    double Lval = L_obj(pbest);
    std::stringstream gc_name, cgc_name;
    gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" << std::setfill('0') << 0 << ".d";
    cgc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_complex_gains_" << std::setfill('0') << 0 << ".d";
    lvg.output_gain_corrections(gc_name.str());
    lvg.output_gains(cgc_name.str());
    
    // Generate summary and ancillary output
    //   Summary file parameter location:
    if (world_rank==0)
    {
      sumout << std::setw(10) << 0;
      for (int k=0; k<nparams; ++k)
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
	std::cout << "V" << j << " rchi2: " << V_rchi2  << " / " << 2*V_data.size() << " - " << Nparam << " - " << Ngains << std::endl;
	// Summary file values
	sumout << std::setw(15) << V_rchi2 / ( 2*int(V_data.size()) - Nparam - Ngains);
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
	     << std::setw(15) << Lval
	     << " "
	     << "V_...dat"
	     << std::endl;
    }
    
    // Run the sampler with the given settings
    if (rep<number_of_rounds) // Only run the desired number of times.
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
