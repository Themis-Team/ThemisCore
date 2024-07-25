/*! 
  \file examples/eht_2018_example.cpp
  \author Paul Tiede
  \date Nov, 2018
  \brief Example presented to the EHT collaboration in 2018 that highlights how to use different models.
  
  \details Themis allows a vast variety of models to be compared to
  EHT data. This example shows how to query and fit a few models to 
  proto-eht visibility amplitude data. The models we consider are symmetric Gaussian, cresecent
  model, SED-fitted riaf model from Broderick 2016.
*/

#include "data_visibility_amplitude.h"
#include "model_symmetric_gaussian.h"
#include "likelihood.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "utils.h"
#include "random_number_generator.h"
#include <iostream>
#include <iomanip>
#include <fstream>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/GainTest/TD_SNR2_RapidLMT/gaussian_test_variable_we.d"),"HH");
  
  // Choose the model we want to use!
  Themis::model_symmetric_gaussian image;
  //Blur the image



  
  // Container of base prior class pointers
  double image_scale =  43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0,10.0)); // Itotal
  
  P.push_back(new Themis::prior_linear(0.0,image_scale*10)); // x-dir size
  
  std::vector<double> means, ranges;
  
  
  means.push_back(2.0); //Itot
  means.push_back(0.5*image_scale); //sigma
  ranges.push_back(0.01);
  ranges.push_back(0.01*image_scale);
  

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma$");

  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  //Standard likelihood
  //L.push_back(new Themis::likelihood_visibility_amplitude(dVM,image));

  
  //Likelihood with gains
  std::vector<std::string> station_codes = Themis::utils::station_codes();

  std::vector<double> gains_prior_sigmas(station_codes.size(), 0.2);
  for ( size_t i = 0; i < station_codes.size(); i++)
  {
    if ( station_codes[i] == "Lm")
      gains_prior_sigmas[i] = 1.0;
    else
      gains_prior_sigmas[i] = 0.2;
  }

  Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(dVM,image, station_codes, gains_prior_sigmas);

  L.push_back(&L_ogva);

  std::cout << "Number of independent gains: " << L_ogva.number_of_independent_gains() << std::endl;

  /*
  std::vector<double> tge = L_ogva.get_gain_correction_times();
  std::vector< std::vector<double> > gge = L_ogva.get_gain_corrections();
  if (world_rank==0)
  {
    std::ofstream gout("gain_corrections_o.d");
    for (size_t i=0; i<tge.size()-1; ++i) {
      gout << std::setw(15) << tge[i]-tge[0];
      for (size_t j=0; j<gge[0].size(); ++j)
	gout << std::setw(15) << gge[i][j];
      gout << '\n';
    }
    gout.flush();

    std::ofstream vmout("visamps.d");
    for (size_t j=0, k=0; j<dVM.size(); ++j) {
      if (dVM.datum(j).tJ2000>tge[k])
	      k++;
      vmout << std::setw(15) << dVM.datum(j).tJ2000-tge[0]
	          << std::setw(15) << tge[k]-tge[0]
	          << std::setw(5) << k
	          << std::setw(15) << dVM.datum(j).V
	          << std::setw(15) << dVM.datum(j).err
	          << '\n';
    }
    vmout.flush();
  }
  */





  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);


  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);
  
  
  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 8;
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 5000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 
                              Number_of_processors_per_lklhd);
                              
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                     "Chain-GaussGains.dat", "Lklhd-GaussGains.dat", "Chi2-GaussGains.dat", 
                     means, ranges, var_names, false);


  //Finalize MPI
  MPI_Finalize();
  return 0;
}
