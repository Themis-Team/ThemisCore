
#include "data_visibility.h"
#include "utils.h"

#include "model_image_asymmetric_gaussian.h"
#include "model_image_diffractive_scattering_screen.h"

#include "likelihood_visibility.h"
//#include "likelihood_optimal_complex_gain_visibility.h"
#include "likelihood_marginalized_visibility_amplitude.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"

#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include "sampler_deo_tempering_MCMC.h"

#include <mpi.h>
#include <vector>
#include <string>
#include <limits>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);


  // Record options:
  int seed = 842; //42;
  std::string annealing_ladder_file="";
  int window=0;
  int init_buffer=10;
  bool save_warmup = true;
  double initial_spacing = 1.2; //initial geometric spacre_stride;
  int Number_of_steps = 10; 
  int Temperature_stride = 10;
  int thin_factor = 1;  
  int refresh_rate = 1;
  size_t Number_of_reps = 9; //12;
  bool restart_flag = false;
  int round_start = 0;
  int verbosity = 0;
  int Ckpt_frequency = 10; // per swaps
  int number_of_adaption_steps = 2000;
  bool reconstruct_variable_gains = false;
  bool reconstruct_constant_gains = false;
  double gain_prior = 0.2;
  
  std::string fit_summaries_start_file="";
  
  int set=-1;
  
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
    if (opt=="--continue" || opt=="--restart")
    {
      restart_flag = true;
    }
    else if (opt=="-g")
    {
      reconstruct_variable_gains=true;
    }
    else if (opt=="-cg")
    {
      reconstruct_constant_gains=true;
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
    else if (opt=="--set")
    {
      if (k<argc)
        set = atoi(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An int argument must be provided after --set.\n";
        std::exit(1);
      }
    }
    else if (opt=="--gain-prior")
    {
      if (k<argc)
        gain_prior = atof(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An float argument must be provided after --gain-prior.\n";
        std::exit(1);
      }
    }
    else if (opt=="-f")
      if (k<argc)
	fit_summaries_start_file = std::string(argv[k++]);
      else
      {
        if (world_rank==0)
          std::cerr << "ERROR: An filename must be provided after -f.\n";
        std::exit(1);
      }
    else if (opt=="-h" || opt=="--help")
    {
      if (world_rank==0)
        std::cerr << "NAME\n"
                  << "\tDriver executable for fitting the GMVA data for the Galactic center scattering screen\n\n"
                  << "SYNOPSIS"
                  << "\tmpirun -np 40 gcss [OPTIONS]\n\n"
                  << "DESCRIPTION\n"
                  << "\t-h,--help\n"
                  << "\t\tPrint this message.\n"
                  << "\t--seed <int>\n"
                  << "\t\tSets the rank 0 seed for the random number generator to the value given.\n"
                  << "\t\tDefaults to 842.\n"     
                  << "\t-g\n"
                  << "\t\tReconstructs unknown station gains.  Default off.\n"
                  << "\t-cg\n"
                  << "\t\tReconstructs constant unknown station gains.  Default off.\n"
                  << "\t--continue/--restart\n"
                  << "\t\tRestarts a run using the local MCMC.ckpt file.  Note that this restarts at step 0, i.e., does not continue to the next step.\n"
		  << "\t-f <file>"
		  << "\t\tSpecify fit summary file to initialize from.\n"
		  << "\t--gain-prior <float>\n"
		  << "\t\tSets the gain prior to the value specified.  Default 0.2.\n"
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

  
  

  
  // 1. Read in data
  double f0 = 86238703125.0; //86e9;
  Themis::data_visibility_amplitude V1, V2, V3;  
  V1.set_default_frequency(f0);
  //V2.set_default_frequency(f0);
  //V3.set_default_frequency(f0);
  //V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t27.1mJy_tygtd.dat"),"HH");
  //V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t26.7mJy_tygtd.dat"),"HH");
  //V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t7mJy_tygtd.dat"),"HH");
  //V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_t26.7mJy_tygtd.dat"),"HH");
  //V2.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t17.6mJy_tygtd.dat"),"HH");
  //V3.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t9.9mJy_tygtd.dat"),"HH");

  //V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t7mJy_tygtd.dat"),"HH");
  //V2.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t7mJy_tygtd.dat"),"HH");
  //V3.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t7mJy_tygtd.dat"),"HH");


  if (set==0)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t7mJy_tygtd.dat"),"HH");    
  else if (set==1)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t7mJy_tygtd.dat"),"HH");
  else if (set==2)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t7mJy_tygtd.dat"),"HH");
  else if (set==10)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t26.7mJy_tygtd.dat"),"HH");    
  else if (set==11)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t17.6mJy_tygtd.dat"),"HH");
  else if (set==12)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t9.9mJy_tygtd.dat"),"HH");
  else if (set==20)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f0_t26.7mJy_tygtd.dat"),"HH");    
  else if (set==21)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f0_t17.6mJy_tygtd.dat"),"HH");
  else if (set==22)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f0_t9.9mJy_tygtd.dat"),"HH");
  else if (set==30)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t7mJy_tygtd_subGl.dat"),"HH");
  else if (set==31)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t7mJy_tygtd_subGl.dat"),"HH");
  else if (set==32)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t7mJy_tygtd_subGl.dat"),"HH");
  else if (set==40)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f3_t10mJy_tygtd.dat"),"HH");    
  else if (set==41)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f1_t10mJy_tygtd.dat"),"HH");
  else if (set==42)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f1_t10mJy_tygtd.dat"),"HH");
  else if (set==50)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f0_t10mJy_tygtd.dat"),"HH");
  else if (set==60)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MB007_selfcal_scanavg_f0_t7mJy_tygtd.dat"),"HH");    
  else if (set==61)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001A_selfcal_scanavg_f0_t7mJy_tygtd.dat"),"HH");
  else if (set==62)
    V1.add_data(Themis::utils::global_path("gmva_data/SgrA/VM_hops_sgra_MJ001B_selfcal_scanavg_f0_t7mJy_tygtd.dat"),"HH");
  else if (set==100)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_sgra_MB007_scanavg_f3_t7mJy_tygtd.dat"),"HH");
  else if (set==103)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model3_sym-gauss-default-scatt_MB007_scanavg_f3_t7mJy_tygtd.dat"),"HH");
  else if (set==104)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model4_asym-gauss-default-scatt_MB007_scanavg_f3_t7mJy_tygtd.dat"),"HH");
  else if (set==110)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_sgra_MB007_scanavg_f3_t10mJy_tygtd.dat"),"HH");
  else if (set==113)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model3_sym-gauss-default-scatt_MB007_scanavg_f3_t10mJy_tygtd.dat"),"HH");
  else if (set==114)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model4_asym-gauss-default-scatt_MB007_scanavg_f3_t10mJy_tygtd.dat"),"HH");
  else if (set==120)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_sgra_MB007_scanavg_f0_t0mJy_tygtd.dat"),"HH");
  else if (set==123)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model3_sym-gauss-default-scatt_MB007_scanavg_f0_t0mJy_tygtd.dat"),"HH");
  else if (set==124)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model4_asym-gauss-default-scatt_MB007_scanavg_f0_t0mJy_tygtd.dat"),"HH");
  else if (set==160)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_sgra_MB007_scanavg_f0_t7mJy_tygtd.dat"),"HH");
  else if (set==163)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model3_sym-gauss-default-scatt_MB007_scanavg_f0_t7mJy_tygtd.dat"),"HH");
  else if (set==164)
    V1.add_data(Themis::utils::global_path("gmva_data/Simulated/VM_synthetic_model4_asym-gauss-default-scatt_MB007_scanavg_f0_t7mJy_tygtd.dat"),"HH");
  else if (set==-1)
    V1.add_data("./VM.dat","HH"); // Local data set
  else
  {
    std::cerr << "Unknown data set option\n";
    std::exit(1);
  }
    

  
  
  
  
  // 2. Define model
  Themis::model_image_asymmetric_gaussian intrinsic_model;
  intrinsic_model.use_analytical_visibilities();
  Themis::model_image_diffractive_scattering_screen model(intrinsic_model,f0);
  //Themis::model_image& model=intrinsic_model;
  intrinsic_model.write_model_tag_file();
  
  // 3. Create likelihood + priors
  //  3.a. Create start point
  double uas2rad = 1e-6/3600. * M_PI/180.;
  std::vector<double> pstart;
  pstart.push_back(1.0); // I0
  //pstart.push_back(40*uas2rad); // Size (300 uas)
  pstart.push_back(2.507e-10); // Size (300 uas)
  pstart.push_back(0.394978); // A
  pstart.push_back(1.77189); // PA
  pstart.push_back(1.38); // screen theta_maj
  pstart.push_back(0.703); // screen theta_min
  pstart.push_back(81.9*M_PI/180.0); // screen PA
  pstart.push_back(1.38); // screen alpha
  pstart.push_back(7.0); // log10(Inner radius in cm)
  pstart.push_back((2.82*3.086e21)/(5.53*3.086e21)); // M


  if (fit_summaries_start_file=="")
  {
    pstart[0] = 1.0;
    pstart[1] = 2.47743e-10;
    pstart[2] = 0.390872;
    pstart[3] = 1.89917;
    pstart[4] = 1.38;
    pstart[5] = 0.703;
    pstart[6] = 1.42942;
    pstart[7] = 0.803603;
    pstart[8] = 8.2378;
    pstart[9] = 0.509946;
  }
  else
  {
    std::ifstream fss(fit_summaries_start_file);
    if (fss.is_open()==false)
    {
      std::cerr << "ERROR: Could not open " << fit_summaries_start_file << ".\n";
      return 1;
    }
    // Clear header
    fss.ignore(4096,'\n');
    //
    double tmp;
    fss >> tmp;
    for (size_t i=0; i<pstart.size(); ++i)
      fss >> pstart[i];
  }

  if (world_rank==0)
  {
    std::cerr << "Starting point: -----------------------------------------\n" << std::endl;
    for (size_t i=0; i<pstart.size(); ++i)
      std::cerr << std::setw(15) << pstart[i] << std::endl;
    std::cerr << "---------------------------------------------------------\n" << std::endl;
  }
    
  
  
  //  3.b. Create priors
  std::vector<Themis::prior_base*> P;
  if (reconstruct_variable_gains)
    P.push_back(new Themis::prior_linear(0.5,5.0)); // Intensity
  else 
    P.push_back(new Themis::prior_linear(0.9999999,1.0000001)); // Intensity
  P.push_back(new Themis::prior_linear(0.0,1e4*uas2rad)); // Size
  P.push_back(new Themis::prior_linear(0.01,0.99)); // A
  P.push_back(new Themis::prior_linear(0.0,M_PI)); // PA
  P.push_back(new Themis::prior_linear(0.9999999*pstart[4],1.0000001*pstart[4])); // screen_theta_maj
  P.push_back(new Themis::prior_linear(0.9999999*pstart[5],1.0000001*pstart[5])); // screen_theta_min
  P.push_back(new Themis::prior_linear(0.9999999*pstart[6],1.0000001*pstart[6])); // screen_PA
  P.push_back(new Themis::prior_linear(0.01,1.99)); // alpha
  P.push_back(new Themis::prior_linear(5,12)); // Inner radius in cm
  P.push_back(new Themis::prior_linear(0.9999999*pstart[9],1.0000001*pstart[9])); // M

  //  3.b. Create likelihood
  //  3.b.i. Create individual object for each data object
  std::vector<std::string> station_codes;
  station_codes.push_back("AA");
  station_codes.push_back("BR");
  station_codes.push_back("EB");
  station_codes.push_back("FD");
  station_codes.push_back("GB");
  station_codes.push_back("KP");
  station_codes.push_back("LA");
  station_codes.push_back("NL");
  station_codes.push_back("OV");
  station_codes.push_back("PT");
  station_codes.push_back("PV");
  station_codes.push_back("YS");
  // Specify the priors we will be assuming (to 20% by default)
  //std::vector<double> station_gain_priors(station_codes.size(),0.2);
  std::vector<double> station_gain_priors(station_codes.size(),gain_prior);
  
  std::vector<Themis::likelihood_base*> L;
  std::vector<Themis::likelihood_marginalized_visibility_amplitude*> lv;
  std::vector<Themis::likelihood_optimal_gain_correction_visibility_amplitude*> lvg;

  
  if (reconstruct_variable_gains)
  {
    lvg.push_back( new Themis::likelihood_optimal_gain_correction_visibility_amplitude(V1,model,station_codes,station_gain_priors) );
    // lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(V2,model,station_codes,station_gain_priors) );
    // lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(V3,model,station_codes,station_gain_priors) );
    for (size_t j=0; j<lvg.size(); ++j)
      L.push_back(lvg[j]);
  }
  else if (reconstruct_constant_gains)
  {
    // // 3.58472228  8.35138893  8.10138917 << Start times in hr
    // //14.82638884 14.15416718 13.7513895  << End times in hr
    // std::vector<double> tge;
    // tge.push_back(-std::numeric_limits<double>::infinity());
    // tge.push_back(std::numeric_limits<double>::infinity());
    // lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(V1,model,station_codes,station_gain_priors,tge) );
    // lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(V2,model,station_codes,station_gain_priors,tge) );
    // lvg.push_back( new Themis::likelihood_optimal_complex_gain_visibility(V3,model,station_codes,station_gain_priors,tge) );
    // for (size_t j=0; j<lvg.size(); ++j)
    //   L.push_back(lvg[j]);
  }
  else
  {
    lv.push_back( new Themis::likelihood_marginalized_visibility_amplitude(V1,model) );
    //lv.push_back( new Themis::likelihood_marginalized_visibility_amplitude(V2,model) );
    //lv.push_back( new Themis::likelihood_marginalized_visibility_amplitude(V3,model) );
    for (size_t j=0; j<lv.size(); ++j)
      L.push_back(lv[j]);
  }
  
  //   3.b.ii. Create combined object
  std::vector<double> W(L.size(), 1.0);
  Themis::likelihood L_obj(P, L, W);
  Themis::likelihood_power_tempered L_temp(L_obj);

  
  // 4. Create samplers
  //Create the tempering sampler which is templated off of the exploration sampler
  std::vector<double> pbest = pstart;
  std::vector<std::string> var_names;
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(seed, L_temp, var_names, pbest.size());

  //Set the output stream which really just calls the hmc output steam.
  //The exploration sampler handles all the output.
  DEO.set_output_stream("chain.dat", "state.dat", "stan_summary_deov5.dat");
    
  //Sets the output for the annealing summary information
  DEO.set_annealing_output("annealing.dat");

  // If passed an existing ladder
  if (annealing_ladder_file!="")
    DEO.read_initial_ladder(annealing_ladder_file);
  
  // Set a checkpoint
  DEO.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  //If you want to access the exploration sampler to change some setting you can!
  if (window==0)
    window = pbest.size()*pbest.size()/2;
  DEO.get_sampler()->set_window_parameters(init_buffer,window);

  //To run the sampler, we pass not the number of steps to run, but instead the number of 
  //swaps to run in the initial round. This is to force people to have at least one 1 swap the first round.
  DEO.get_sampler()->set_adaptation_parameters( number_of_adaption_steps, save_warmup);
    
  //Now we can also change some options for the sampler itself.
  DEO.set_initial_location(pbest);

  //We can also change the annealing schedule.
  DEO.set_annealing_schedule(initial_spacing);  
  DEO.set_deo_round_params(Number_of_steps,Temperature_stride);
  
  // If continuing
  if (restart_flag)
  {
    DEO.read_checkpoint("MCMC.ckpt");
    round_start = DEO.get_round();
  }


  // Get the numbers of data points and parameters for DoF computations
  std::vector<int> Nd;
  Nd.push_back(V1.size());
  Nd.push_back(V2.size());
  Nd.push_back(V3.size());
  int Ndata = 0;
  Ndata += V1.size();
  //Ndata += V2.size();
  //Ndata += V3.size();
  int Nparam = model.size() - 4; // Take off four fixed screen parameters
  int Ngains = 0;
  std::vector<int> Ngains_list(L.size(),0);

  // Add gain degrees of freedom if present
  for (size_t j=0; j<lvg.size(); ++j)
  {
    Ngains_list[j] = lvg[j]->number_of_independent_gains();
    Ngains += Ngains_list[j];
  }
  
  int NDoF = Ndata - Nparam - Ngains;
  
  
  for (size_t rep=round_start; rep<=Number_of_reps; ++rep)
  {
    if (world_rank==0) 
      std::cerr << "Started round " << rep << std::endl;

    // Generate fit summaries file
    std::stringstream sumoutname;
    sumoutname << "round" << std::setfill('0') << std::setw(3) << rep << "_fit_summaries" << ".txt";
    std::ofstream sumout;
    if (world_rank==0)
    {
      sumout.open(sumoutname.str().c_str());
      sumout << std::setw(10) << "# Index"
	     << std::setw(15) << "A-I"
	     << std::setw(15) << "A-sig"
	     << std::setw(15) << "A-A"
	     << std::setw(15) << "A-phi"
	     << std::setw(15) << "theta_maj"
	     << std::setw(15) << "theta_min"	
	     << std::setw(15) << "PA"	
	     << std::setw(15) << "alpha"	
	     << std::setw(15) << "log10(rin)"	
	     << std::setw(15) << "M"
	     << std::setw(15) << "V1 rc2"
	     << std::setw(15) << "V2 rc2"
	     << std::setw(15) << "V3 rc2"
	     << std::setw(15) << "Total rc2"
	     << std::setw(15) << "log-liklhd"
	     << std::endl;
    }


    // Output the current location
    if (world_rank==0)
      for (size_t j=0; j<pbest.size(); ++j)
	std::cout << pbest[j] << std::endl;

    double Lval = L_obj(pbest);

    // Output gains, if being fitted
    for (size_t j=0; j<lvg.size(); ++j)
    {
      std::stringstream gc_name, cgc_name;
      gc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_gain_corrections_" << std::setfill('0') << j << ".d";
      //cgc_name << "round" << std::setfill('0') << std::setw(3) << rep << "_complex_gains_" << std::setfill('0') << j << ".d";
      lvg[j]->output_gain_corrections(gc_name.str());
      //lvg[j]->output_gains(cgc_name.str());
    }

    // Generate summary and ancillary output
    //   Summary file parameter location:
    if (world_rank==0)
    {
      sumout << std::setw(10) << 0;
      for (size_t k=0; k<model.size(); ++k)
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
	std::cout << "V" << j << " rchi2: " << V_rchi2  << " / " << Nd[j] << " - " << Nparam << " - " << Ngains_list[j] << std::endl;
	// Summary file values
	sumout << std::setw(15) << V_rchi2 / ( Nd[j] - Nparam - Ngains_list[j]);
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
	     << std::endl;
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
