/*!
  \file tests/asymmetric_gaussian_image_comparison.cpp
  \author Avery Broderick
  \date June 2017
  \test Gaussian model with visibility amplitude data
  \brief Fit an asymmetric Gaussian model to visibility amplitude data
  
  \details This tests estimates parameters for the asymmetric Gaussian model.
  See reading_data.cpp example for details on reading in EHT data.
*/

#include "data_visibility.h"
#include "model_image_symmetric_gaussian.h"
#include "model_image_fixed_lightcurve.h"
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


  Themis::model_image_symmetric_gaussian model_static;
  model_static.use_analytical_visibilities();
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian/lc_alpha_minus4_frame_7.2min_correlation_20min_v4.d"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/RandomNoise/LC_random_noise.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/LC_sample_images.h5_lo_0000.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian/V_alpha_h_minus4_beta_minus_2_v1.h5_lo_0000_scanavg_tygtd.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/spatiotemporal_noise_model/thnoise/sample_images.h5/LC_sample_images.h5_lo_0000.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_1e3/thnoise/designer_data_realization.h5/LC_designer_data_realization.h5_lo_0000.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_1e3b/thnoise/designer_data_realization.h5/LC_designer_data_realization.h5_lo_0000.dat"),"HH");
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data/thnoise/designer_data_realization.h5/LC_designer_data_realization.h5_lo_0000.dat"),"HH");  
  //Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_663e3c/thnoise/designer_data_realization.h5/LC_designer_data_realization.h5_lo_0000.dat"),"HH");  
  Themis::model_image_fixed_lightcurve model(model_static,Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_663e3c/thnoise/designer_data_realization.h5/dmc_lightcurve.txt"),"HH");  
  //Themis::model_image& model = model_static;
  
  
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian/V_alpha_minus4_frame_7.2min_correlation_20min_v4.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian/V_alpha_h_minus4_beta_minus_2_v1.h5_lo_0000_scanavg_tygtd.dat"), "HH");  
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/RandomNoise/V_random_noise.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/V_sample_images.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/spatiotemporal_noise_model/thnoise/sample_images.h5/V_sample_images.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_1e3/thnoise/designer_data_realization.h5/V_designer_data_realization.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_1e3b/thnoise/designer_data_realization.h5/V_designer_data_realization.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data/thnoise/designer_data_realization.h5/V_designer_data_realization.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/CorrelatedGaussian2/designer_data_663e3c/thnoise/designer_data_realization.h5/V_designer_data_realization.h5_lo_0000_scanavg_tygtd.dat"), "HH");
  //Themis::data_visibility V_data(Themis::utils::global_path("sim_data/TestUncertainty/RandomNoise/DMC/V_gauss1_scanavg_tygtd.dat"), "HH");

  //Themis::uncertainty_visibility uncer;
  //Themis::uncertainty_visibility_loose_change uncer;
  //Themis::uncertainty_visibility_power_change uncer;
  Themis::uncertainty_visibility_broken_power_change uncer;
  std::vector<Themis::likelihood_base*> L;
  Themis::likelihood_visibility lvis(V_data, model, uncer);
  L.push_back(&lvis);

  //Create the priors
  std::vector<Themis::prior_base*> P;
  std::vector<double> start(model.size(), 0.0);
  std::vector<std::string> var_names;


  //Total flux
  P.push_back(new Themis::prior_linear(1e-5, 10.0));
  start[0] = 1.0;
  //Size in uas
  const double uas2rad = M_PI/180.0/3600/1e6;
  P.push_back(new Themis::prior_linear(0.0, 200*uas2rad));
  start[1] = 25.0*uas2rad;

  // Add uncertainty models
  // Loose change
  P.push_back(new Themis::prior_linear(0.0, 5.0));
  //P.push_back(new Themis::prior_linear(0.009999, 0.01001));
  //P.push_back(new Themis::prior_linear(0.0, 0.000001));
  start.push_back(0.01); 
  //start.push_back(0.0000005); 
  P.push_back(new Themis::prior_linear(0.0, 1.0));
  //P.push_back(new Themis::prior_linear(0.019999, 0.02001));
  //P.push_back(new Themis::prior_linear(0.0, 0.000001));
  start.push_back(0.02);
  //start.push_back(0.0000005); 
  // Power change 
  P.push_back(new Themis::prior_linear(0.0, 1));
  start.push_back(0.0259/std::sqrt(2.0));
  P.push_back(new Themis::prior_linear(0.0, 10));
  start.push_back(3.434);
  P.push_back(new Themis::prior_linear(0.0, 10.0));
  start.push_back(2.096);
  // Broken power change
  P.push_back(new Themis::prior_linear(0.0, 10.0));
  start.push_back(1.899);

  

  //Construct the likelihood
  std::vector<double> W(L.size(), 1.0);
  Themis::likelihood L_obj(P,L,W);
  Themis::likelihood_power_tempered L_beta(L_obj);

  //Create the sampler
  Themis::sampler_deo_tempering_MCMC<Themis::sampler_automated_factor_slice_sampler_MCMC> DEO(58, L_beta, var_names, start.size());

  //Create the checkpoint
  DEO.set_checkpoint(100,"MCMC.ckpt");

  //Set adaptation params
  double initial_ladder_spacing = 1.1;
  int nswaps = 10;
  int stride = 50;
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


  


  


  
  



