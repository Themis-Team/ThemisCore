/*!
    \file vae_riaf_afss.cpp
    \author Ali Sarertoosi, Avery Broderick
    \date March 2023
    \brief Driver file to run MCMC parameter estimation on Sgr A* using variational autoencoder models trained on RIAFs.
    \todo 
*/


#include "model_image_vae_interpolated_riaf.h"
#include "model_visibility_galactic_center_diffractive_scattering_screen.h"

#include "uncertainty_visibility.h"
#include "uncertainty_visibility_broken_power_change.h"

#include "likelihood_visibility.h"
#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_power_tempered.h"

#include "utils.h"
#include "cmdline_parser.h"
#include "read_data.h"

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

std::vector<std::vector<double>> read_bplfile(std::string file);


int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Prepare command-line parser
  Themis::CMDLineParser argp("vae_riaf_afss","Performs MCMC parameter estimation of EHT data using variational autoencoder\nRIAF models via DEO-tempered automatic factored slice sampling.");

  // Specify command-line arguments
  Themis::CMDLineParser::IntArg Number_temperatures(argp,"-not,--number-of-temperatures","Sets the number of tempering levels.  Defaults to the number of procs.",world_size,false);
  Themis::CMDLineParser::IntArg Number_of_swaps(argp,"-Ns","Sets the number of MCMC steps between each tempering level swap.",10,false);
  Themis::CMDLineParser::IntArg seed(argp,"--seed","Sets the seed for the random number generation.",42,false);

  Themis::CMDLineParser::IntArg number_of_rounds(argp,"-nor,-Nr","Sets the number of DEO rounds.",12,false);
  Themis::CMDLineParser::FloatArg initial_ladder_spacing(argp,"-ils,--initial-ladder-spacing","Sets the geometric factor by which subsequent ladder beta increases.",1.4,false);
  Themis::CMDLineParser::IntArg Temperature_stride(argp,"-ts,--temperature-stride","Sets the number of temperature swaps in the first DEO round.",10,false);

  Themis::CMDLineParser::BoolArg restart_flag(argp,"--restart,--continue","Continue from prior run from an existing MCMC.ckpt file.",false,false);

  Themis::CMDLineParser::FloatArg frequency(argp,"-rf,--frequency","Sets observation frequency in Hz.",230e9,false);
  Themis::CMDLineParser::StringArg model_dir(argp,"-m,--model","Directory containing vae model and ancillary files.","vae_model",false);
  Themis::CMDLineParser::IntArg image_size(argp,"-npx,--image-size","Sets the number of pixels to resize images to. By default, no resizing\nis done.");
  Themis::CMDLineParser::StringArg data_list_file(argp,"-vl","Name of text file with list of visibility data files to fit.",false);
  Themis::CMDLineParser::VariableVectorStringArg data_file_names(argp,"-vd","Add a visibility data file to fit.",false);
  
  Themis::CMDLineParser::BoolArg scatter(argp,"-s,--scatter","Apply Galactic center diffractive scattering screen.",false,false);
  Themis::CMDLineParser::BoolArg reconstruct_gains(argp,"-g,--gains","Perform gain self-calibration.",false,false);
  Themis::CMDLineParser::StringArg array_file(argp,"-a,--array","Indicates which station codes and gain priors to use. Options are\neht2017, eht2018, or a text file with two-letter station codes and\nlog-normal gain amplitude prior.","eht2017",false);

  Themis::CMDLineParser::StringArg fsstart_file(argp,"-f","Name of fit summaries file from which to begin. Must contain the same\nnumber of parameters as current analysis.");
  
  Themis::CMDLineParser::BoolArg model_noise(argp,"-n,--noise","Fit broken-power-change noise model.",false,false);
  Themis::CMDLineParser::StringArg bpl_file(argp,"-bpl","Name of broken-power-change prior file to read.");
  Themis::CMDLineParser::FloatArg threshold_floor(argp,"-tf,--threshold-floor","Lower bound on the logarithmic prior on the threshold noise to add, typically to mitigate refractive scattering.",1e-4,false);
  Themis::CMDLineParser::FloatArg fractional_floor(argp,"-ff,--fractional-floor","Lower bound on the logarithmic prior on the fractional noise (percentage) to add, typically to mitigate non-closing systematic uncertainties.",1e-4,false);

  
  // Parse command line
  argp.parse_args(argc,argv,true);
 
  // Some specifications
  //int Number_temperatures = world_size;  
  //int Number_of_swaps = 10;
  //int seed = 42;

  // Tempering stuff default options
  //size_t number_of_rounds = 10;
  //double initial_ladder_spacing = 1.4;
  int Thin_factor = 1;
  //int Temperature_stride = 10;

  // AFSS adatation parameters
  int init_buffer = 10;
  int window = 0;
  int number_of_adaptation = 0;
  bool save_adapt = true;
  int refresh = 1;

  // Get the list of data files (do this first because this is a common oversight)
  if (data_file_names.is_defined()==false && data_list_file.is_defined()==false)
  {
    std::cerr << "ERROR: Either " << data_file_names.cmdline_flag(0) << " or " << data_list_file.cmdline_flag(0) << " must be passed to specify a data file or list of data files, respectively.\n"
	      << "Try -h or --help for more information.\n";
    std::exit(1);
  }
  std::vector<std::string> v_file_name_list;
  if (data_list_file.is_defined())
    Themis::utils::read_vfile_mpi(v_file_name_list, data_list_file(), MPI_COMM_WORLD);
  if (data_file_names.is_defined())
    for (size_t j=0; j<data_file_names.size(); ++j)
      v_file_name_list.push_back(data_file_names(j));

  // Define model
  Themis::model_image_vae_interpolated_riaf unscattered_image(model_dir());
  if (image_size.is_defined())
    unscattered_image.set_image_size(image_size(),image_size());
  Themis::model_visibility_galactic_center_diffractive_scattering_screen scattered_image( unscattered_image );
  Themis::model_visibility* image_ptr;
  if (scatter())
  {
    image_ptr = &scattered_image;
    std::cerr << "Using scattered image.\n";
  }
  else
  {
    image_ptr = &unscattered_image;
    std::cerr << "Using unscattered image.\n";
  }
  Themis::model_visibility& image = *image_ptr;

  // Define uncertainty model
  Themis::uncertainty_visibility no_noise;
  Themis::uncertainty_visibility_broken_power_change bpl_noise; 
  bpl_noise.constrain_noise_at_4Glambda();
  bpl_noise.logarithmic_ranges();
  Themis::uncertainty_visibility* noise_ptr;
  if (model_noise())
    noise_ptr = &bpl_noise;
  else
    noise_ptr = &no_noise;
  Themis::uncertainty_visibility& noise = *noise_ptr;

  // Output model tag
  unscattered_image.write_model_tag_file();
  
  // Choose gain particulars
  std::vector<std::string> station_codes;
  std::vector<double> station_gain_priors;
  if (array_file()=="eht2017")
  {
    station_codes = Themis::utils::station_codes("uvfits 2017");
    station_gain_priors.resize(station_codes.size(),0.1);
    station_gain_priors[4] = 1.0; // Allow the LMT to vary by 100%
  }
  else if (array_file()=="eht2018")
  {
    station_codes = Themis::utils::station_codes("uvfits 2018");
    station_gain_priors.resize(station_codes.size(),0.1);
    station_gain_priors[2] = 1.0; // Allow the GLT to vary by 100%
  }
  else
  {
    std::ifstream afin(array_file());
    if (!afin.is_open())
    {
      std::cerr << "ERROR: Could not open " << array_file() << '\n';
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
  
  // Read in data files
  std::vector<Themis::data_visibility*> V_data;
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_optimal_complex_gain_visibility*> lvg;
  std::vector<Themis::likelihood_visibility*> lv;
  for (size_t j=0; j<v_file_name_list.size(); ++j)
  {
    V_data.push_back( new Themis::data_visibility(v_file_name_list[j],"HH") );
    V_data[j]->set_default_frequency(frequency());

    if (reconstruct_gains())
    {
      lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(*V_data[j],image,noise,station_codes,station_gain_priors) );
      L.push_back(lvg[j]);
    }
    else
    {
      lv.push_back( new Themis::likelihood_visibility(*V_data[j],image,noise) );
      L.push_back(lv[j]);
    }
  }
  
  /////////////////
  // Set up priors and initial walker ensemble starting positions
  //
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means;
  std::vector<std::string> var_names;
  
  // Add priors for the latent variables
  for (size_t j=0; j<unscattered_image.latent_size(); ++j)
  {
    //P.push_back(new Themis::prior_gaussian(0.0,1.0));
    //P.push_back(new Themis::prior_linear(-10,10));
    //P.push_back(new Themis::prior_truncated_gaussian(0.0,1.0,2.0));
    P.push_back(new Themis::prior_linear(0.01,0.99));   
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

  // Add priors with noise parameters
  if (model_noise())
  {
    if (bpl_file.is_defined())
    {
      double noise_sigma[4];
      double noise_med[4];
      double noise_lo[4], noise_hi[4];
      if (world_rank == 0)
      {
	std::vector<std::vector<double> > params = read_bplfile(bpl_file());
	for ( int nn=1; nn<=4; ++nn )
	{
	  // Interquartile range is about 1.35 * std dev.
	  noise_sigma[nn%4] = (params[nn][4]-params[nn][2])/1.35;
	  noise_med[nn%4] = params[nn][3];
	  noise_lo[nn%4] = params[nn][2];
	  noise_hi[nn%4] = params[nn][4];
	  std::cerr << "Noise BPL intialized at [" << (nn%4) << "] = " << noise_med[nn%4] << " +- " << noise_sigma[nn%4]
		    << "  log: " << std::log(noise_med[nn%4]) << " +- " << noise_sigma[nn%4]/noise_med[nn%4]
		    << '\n';
	}
      }
      //Now broadcast to everyone
      MPI_Bcast(&noise_sigma[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_med[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_lo[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&noise_hi[0], 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // noise threshold
      //P.push_back(new Themis::prior_gaussian(std::log(0.004),1.0));
      //means.push_back(std::log(0.004)); 
      P.push_back(new Themis::prior_linear(std::log(threshold_floor()),std::log(3.0)));
      means.push_back(std::log(std::max(threshold_floor()*1.01,0.004))); 
      // fractional (mimicking non-closing errors)
      //P.push_back(new Themis::prior_gaussian(std::log(0.010),1.0));
      //means.push_back(std::log(0.01));
      P.push_back(new Themis::prior_linear(std::log(fractional_floor()/100.0),std::log(3.0)));
      means.push_back(std::log(std::max(fractional_floor()/100.0*1.01,0.010)));
      // bpc noise amplitude at 4Glambda
      P.push_back(new Themis::prior_linear(std::log(noise_lo[0]),std::log(noise_hi[0])));
      means.push_back(std::log(noise_med[0]));
      // uv distance where two powerlaws break
      P.push_back(new Themis::prior_linear(std::log(noise_lo[1]),std::log(noise_hi[1])));
      means.push_back(std::log(noise_med[1]));
      // long baseline index
      P.push_back(new Themis::prior_linear(noise_lo[2],noise_hi[2]));
      means.push_back(noise_med[2]);
      // short baseline index
      P.push_back(new Themis::prior_linear(1.5,2.5));
      means.push_back(2.0);
    }
    else // Adopt broad range about Sgr A* values
    {
      // noise threshold
      P.push_back(new Themis::prior_gaussian(std::log(0.004),5.0));
      means.push_back(std::log(0.004));
      // fractional (mimicking non-closing errors)
      P.push_back(new Themis::prior_gaussian(std::log(0.010),5.0));
      means.push_back(std::log(0.010));
      // bpc noise amplitude at 4Glambda
      P.push_back(new Themis::prior_linear(std::log(0.0001),std::log(0.1000)));
      means.push_back(std::log(0.018));
      // uv distance where two powerlaws break
      P.push_back(new Themis::prior_linear(std::log(0.01),std::log(10.0)));
      means.push_back(std::log(1.5));
      // long baseline index
      P.push_back(new Themis::prior_linear(1.0,5.0));
      means.push_back(2.5);
      // short baseline index
      P.push_back(new Themis::prior_linear(1.5,2.5));
      means.push_back(2.0);
    }
  }
  
  // Start values, if desired
  if (fsstart_file.is_defined())
  {
    std::ifstream fsin(fsstart_file());
    fsin.ignore(4096,'\n');
    double dtmp;
    fsin >> dtmp;
    for (size_t j=0; j<image.size(); ++j)
      fsin >> means[j];
    fsin.close();
  }

  
  // Check
  if ( (means.size() != P.size())){
    std::cerr << "Error number of means does equal number of priors!\n";
    std::cerr << " means.size: " << means.size()
              << "\n P.size(): " << P.size() << std::endl;
    std::exit(1);
  }
  int nparams = image.size()+noise.size();
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

  // Write a chain of the priors to provide insight into the impact of sampling the latent space
  L_obj.write_prior_chain("prior_chain.dat",10000);
  
  double Lstart = L_obj(means);
  if (world_rank==0)
    std::cerr << "At initialization likelihood is " << Lstart << '\n';

  // Get the numbers of data points and parameters for DoF computations
  int Nparam = image.size()-2.0;
  int Ndata=0, Ngains=0;
  std::vector<int> Ngains_list(L.size(),0);
  for (size_t j=0; j<L.size(); ++j)
  {
    Ndata += 2*V_data[j]->size();
    if (reconstruct_gains())
    {
      Ngains_list[j] = lvg[j]->number_of_independent_gains();
      Ngains += Ngains_list[j];
    }
  }
  int NDoF = Ndata - Nparam - Ngains;
  
  // Get the best point 
  std::vector<double> pbest =  means;

  // Generate a chain
  int Ckpt_frequency = 100;
  // int out_precision = 8;
  int verbosity = 0;
  
  // Create a sampler object
  //Themis::sampler_deo_tempering_MCMC<Themis::sampler_stan_adapt_diag_e_nuts_MCMC> DEO(seed, L_beta, var_names, means.size());
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed(), L_beta, var_names, means.size());

  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "summary.dat");
    
  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat");

  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set annealing schedule
  DEO.set_annealing_schedule(initial_ladder_spacing()); 
  DEO.set_deo_round_params(Number_of_swaps(), Temperature_stride());
  
  //Set the initial location of the sampler
  DEO.set_initial_location(means);
  
  //Get the exploration sampler and set some options
  //Themis::sampler_stan_adapt_diag_e_nuts_MCMC* kexplore = DEO.get_sampler();
  //kexplore->set_max_depth(10);
  Themis::sampler_automated_factor_slice_sampler_MCMC* kexplore = DEO.get_sampler();
  if (number_of_adaptation==0)
    number_of_adaptation = means.size()*means.size()*4;
  kexplore->set_adaptation_parameters(number_of_adaptation, save_adapt);

  if (window==0)
    window = means.size()*means.size()/2;
  kexplore->set_window_parameters(init_buffer, window);

  //////////////////// DEVEL
  //if (Number_temperatures==0)
  //  Number_temperatures = world_size;
  int Number_per_likelihood = world_size/Number_temperatures();
  if (world_rank==0)
  {
    std::cerr << "N temps: " << Number_temperatures() << '\n';
    std::cerr << "world size: " << world_size << '\n';
    std::cerr << "world rank: " << world_rank << '\n';
    std::cerr << "N per L: " << Number_per_likelihood << '\n';
  }
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
  DEO.set_cpu_distribution(Number_temperatures(), Number_per_likelihood);



  // Read a checkpoint, if desired
  //bool restart_flag = true;
  int round_start = 0;
  if (restart_flag() && Themis::utils::isfile("MCMC.ckpt"))
  {
      DEO.read_checkpoint("MCMC.ckpt");
      round_start = DEO.get_round();
      // restart_flag=false;
  }
  else if (restart_flag() && !Themis::utils::isfile("MCMC.ckpt"))
  {
    std::cerr << "!!!Warning no ckpt found starting run from beginning!!!\n";
  }
  
  
  // Start looping over the rounds
  for ( size_t rep = round_start; rep < size_t(number_of_rounds()); ++rep)
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
      for (size_t j=0; j<unscattered_image.latent_size(); ++j)
	sumout << std::setw(15) << "z"+std::to_string(j);
      sumout << std::setw(15) << "flux ratio"
	     << std::setw(15) << "mass ratio"
	     << std::setw(15) << "PA (rad)";
      if (model_noise())
	sumout << std::setw(15) << "n-t"
	       << std::setw(15) << "n-f"
	       << std::setw(15) << "n-sigp"
	       << std::setw(15) << "n-up"
	       << std::setw(15) << "n-ahi"
	       << std::setw(15) << "n-alo";
      for (size_t j=0; j<L.size(); ++j) 
	sumout << std::setw(15) << "V" + std::to_string(j) + " rc2";
      sumout << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << "     FileName(s)"
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
    for (size_t j=0; j<L.size(); ++j)
    {
      std::stringstream gc_name, cgc_name;
      gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" << std::setfill('0') << j << ".d";
      cgc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_complex_gains_" << std::setfill('0') << j << ".d";
      if (reconstruct_gains())
      {
	lvg[j]->output_gain_corrections(gc_name.str());
	lvg[j]->output_gains(cgc_name.str());
      }
    }
    
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
      for (size_t j=0; j<L.size(); ++j)
	sumout << " " << v_file_name_list[j];
      sumout << std::endl;
    }
    
    // Run the sampler with the given settings
    if (rep<size_t(number_of_rounds())) // Only run the desired number of times.
    {
      clock_t start = clock();
      if ( world_rank == 0 )
	std::cerr << "Starting MCMC on round " << rep << std::endl;
      DEO.run_sampler( 1, Thin_factor, refresh, verbosity);
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


/*
 * Reads in a config file and returns a vector with the parameters
 * like the config file
#                      Percentile
#         param           0.5%           2.5%            25%            50%            75%          97.5%          99.5%
              a      0.0187211      0.0228614      0.0618002        0.18952       0.472129       0.934031       0.988063
             u0      0.0313386      0.0797632       0.333633       0.625367        1.26685         3.0245        3.86391
              b       0.954535        1.42388        2.35213        2.86059        3.37849        4.55225        6.14047
              c      0.0600666       0.328347        3.42426        7.14147        10.9416        14.6142        14.9329
          a@4Gl      0.0104484      0.0109439      0.0123039      0.0130988      0.0139344      0.0157996       0.017002

*/
std::vector<std::vector<double> > read_bplfile(std::string cfile)
{
    std::ifstream in(cfile); 
    if (!in.is_open())
    {
        std::cerr << "read_bplfile: Configuration file not found " << cfile << std::endl;
        std::exit(1);
    }
    std::string line;
    //skip first two lines
    std::getline(in, line);
    std::getline(in, line);
    //get first parameter quantiles
    std::string word;
    std::vector<std::vector<double> > params;
    for (int i = 0; i < 5; ++i)
    {
        in >> word; //skip first line since it is a character
        std::vector<double> tmp(7, 0.0);
        for ( int j = 0; j < 7; ++j )
        {
            in >> word;
            tmp[j] = std::stod(word.c_str());
        }
        params.push_back(tmp);
    }
    in.close();
    return params;
}
