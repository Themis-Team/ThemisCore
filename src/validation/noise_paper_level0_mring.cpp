/*!
  \file validation/noise_paper_level0.cpp
  \author Avery Broderick
  \date November 2021
  \test Gaussian model with visibility amplitude data
  \brief Reads in level0 uncertainty test data (Gaussain errors added directly to complex visibilities) and fits a desired uncertainty prescription.
  
  \details This tests estimates parameters for the asymmetric Gaussian model.
  See reading_data.cpp example for details on reading in EHT data.
*/


#include "data_visibility.h"
#include "model_image_mring.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_fixed_lightcurve.h"
#include "model_image_smooth.h"

#include "likelihood.h"
#include "likelihood_power_tempered.h"
#include "likelihood_visibility.h"
#include "utils.h"
#include "uncertainty_visibility.h"
#include "uncertainty_visibility_loose_change.h"
#include "uncertainty_visibility_power_change.h"
#include "uncertainty_visibility_broken_power_change.h"
#include "sampler_deo_tempering_MCMC.h"
#include "sampler_automated_factor_slice_sampler_MCMC.h"


#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
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
#include <unistd.h>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  int mring_order = 2;
  int test_type = 0;
  if (argc>1) {
    test_type = atoi(argv[1]);
  }  

  Themis::model_image_mring mring(mring_order);
  mring.use_analytical_visibilities();
  Themis::model_image_smooth model(mring);
  
  std::string datafile;
  Themis::uncertainty_visibility *uncer_ptr;
  Themis::uncertainty_visibility uncer_none;
  Themis::uncertainty_visibility_loose_change uncer_lc;
  Themis::uncertainty_visibility_broken_power_change uncer_bpc;
  if (test_type==0) 
  {
    datafile = Themis::utils::global_path("sim_data/NoisePaperData/V_mring_none.dat");
    uncer_ptr = &uncer_none;
  }
  else if (test_type==-1)
  {
    datafile = Themis::utils::global_path("sim_data/NoisePaperData/V_mring_lc.dat");
    uncer_ptr = &uncer_none;
  }
  else if (test_type==1)
  {
    datafile = Themis::utils::global_path("sim_data/NoisePaperData/V_mring_lc.dat");
    uncer_ptr = &uncer_lc;
  }
  else if (test_type==-2)
  {
    datafile = Themis::utils::global_path("sim_data/NoisePaperData/V_mring_bpc.dat");
    uncer_ptr = &uncer_none;
  }
  else if (test_type==2)
  {
    datafile = Themis::utils::global_path("sim_data/NoisePaperData/V_mring_bpc.dat");
    uncer_ptr = &uncer_bpc;
  }
  else 
  {
    if (world_rank==0) 
    {
      std::cerr << "Unknown test type: " << test_type << "\n";
      std::cerr << "Known types are:\n";
      std::cerr << "  0 ... No noise test.\n"; 
      std::cerr << "  1 ... Loose change test.\n";
      std::cerr << " -1 ... Loose change test w/o noise modeling.\n";
      std::cerr << "  2 ... Broken power change test.\n";
      std::cerr << " -2 ... Broken power change test w/o noise modeling.\n"; 
    }
    std::exit(1);
  }
  Themis::uncertainty_visibility& uncer = (*uncer_ptr);
  Themis::data_visibility V_data(datafile,"HH");

  std::vector<Themis::likelihood_base*> L;
  Themis::likelihood_visibility lvis(V_data, model, uncer);
  L.push_back(&lvis);

  //Create the priors
  std::vector<Themis::prior_base*> P;
  std::vector<double> start;
  std::vector<std::string> var_names;


  // 0. Total flux
  P.push_back(new Themis::prior_linear(1e-5, 10.0));
  start.push_back(1.0);
  // 1. Size in uas
  const double uas2rad = M_PI/180.0/3600/1e6;
  P.push_back(new Themis::prior_linear(0.0, 200*uas2rad));
  start.push_back((25.0/2.35)*uas2rad);
  // 2. Mring expansion
  for (int i = 0; i < mring_order; ++i){
      //Real
      P.push_back(new Themis::prior_linear(0.0, 0.9));
      start.push_back(0.1);
      //Imag
      P.push_back(new Themis::prior_linear(-M_PI, M_PI));
      start.push_back(0.0);
  }
  // 3. smoothing size
  P.push_back(new Themis::prior_linear(1.0*uas2rad,20*uas2rad));
  start.push_back(10*uas2rad);
  // 4. smoothing asym
  P.push_back(new Themis::prior_linear(0.00, 1e-6));
  start.push_back(1e-7);
  // 5. smoothing pos ang
  P.push_back(new Themis::prior_linear(-0.001,0.001));
  start.push_back(0.0);

  // Add uncertainty models
  if (uncer.size()>0)
  {
    // Loose change
    P.push_back(new Themis::prior_linear(0.0, 5.0));
    start.push_back(0.01); 
    P.push_back(new Themis::prior_linear(0.0, 1.0));
    start.push_back(0.02);
  }
  if (uncer.size()>2)
  {
    // Broken Power change 
    P.push_back(new Themis::prior_linear(0.0, 1));
    start.push_back(0.05);
    P.push_back(new Themis::prior_linear(0.0, 10.0));
    start.push_back(2.0);
    P.push_back(new Themis::prior_linear(0.0, 10.0));
    start.push_back(3.0);
    P.push_back(new Themis::prior_linear(0.0, 10.0));
    start.push_back(2.0);
  }
  
  //Construct the likelihood
  std::vector<double> W(L.size(), 1.0);
  Themis::likelihood L_obj(P,L,W);
  Themis::likelihood_power_tempered L_beta(L_obj);

  //Create the sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(42, L_beta, var_names, start.size());

  //Create the checkpoint
  DEO.set_checkpoint(100,"MCMC.ckpt");

  //Set adaptation params
  double initial_ladder_spacing = 3.5;
  int nswaps = 10;
  int stride = 10;
  DEO.set_annealing_schedule(initial_ladder_spacing); 
  DEO.set_deo_round_params(nswaps, stride);
  
  //Set the initial location
  DEO.set_initial_location (start);
  
  //Set the tempering output name
  DEO.set_annealing_output("annealing.dat");

  //Set the output file names
  DEO.set_output_stream("chain.dat", "state.dat", "sampler.dat");
  
  int nrounds = 12;
  int thin_factor = 1;
  int refresh = 10;
  int verbosity = 0;
  DEO.run_sampler(nrounds, thin_factor, refresh, verbosity);

  DEO.mpi_cleanup();

  MPI_Finalize();
  return 0;
}


  


  


  
  



