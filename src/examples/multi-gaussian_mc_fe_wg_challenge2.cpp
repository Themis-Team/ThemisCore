/*!
  \file examples/multi-gaussian_mc_fe_wg_challenge2.cpp
  \author
  \date Mar 2018
  \brief Example of fitting a multi-Gaussian image model to visibility amplitude and closure phase data according to the specifications of modeling challenge 2.

  \details This example illustrates how to generate a multi-Gaussian image
  model, include scattering to the intrinsic model image, read-in
  visibility amplitude and closure phase data (as also shown in reading_data.cpp), and
  fitting the model to s simulated data set provided within the challenge. 
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_multigaussian.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include "utils.h"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  //RG: SIMULATED DATA PROVIDED BY MF_FE_WG CHALLENGE 2
  Themis::data_visibility_amplitude SIM_VM_DATA(Themis::utils::global_path("eht_data/mf_fe_wg_challenge2_vm.txt"));
  Themis::data_closure_phase SIM_CP_DATA(Themis::utils::global_path("eht_data/mf_fe_wg_challenge2_cp.txt"));


  // Choose the model to compare
  Themis::model_image_multigaussian image(5);

  // RG: INTRODUCE NEW PARAMETERS N:Nr of Gaussian components, N-1
  // positions, rest the same as before for each Gaussian

  // Container of base prior class pointers
  double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;

  // P.push_back(new Themis::prior_linear(1.,5.)); // Nr of Gaussian components
  P.push_back(new Themis::prior_linear(0.01,5.01)); // Itotal
  P.push_back(new Themis::prior_linear(0.1*image_scale,0.9*image_scale)); // sigma  
  P.push_back(new Themis::prior_linear(-1e-4,1e-4)); // x-dir
  P.push_back(new Themis::prior_linear(-1e-4,1e-4)); // y-dir

  for (size_t n=1; n<5; n++) {
  P.push_back(new Themis::prior_linear(0.01,5.01)); // Itotal
  P.push_back(new Themis::prior_linear(0.1*image_scale,0.9*image_scale)); // sigma  
  P.push_back(new Themis::prior_linear(-0.5*image_scale,0.5*image_scale)); // x-dir size
  P.push_back(new Themis::prior_linear(-0.5*image_scale,0.5*image_scale)); // y-dir size
}
  P.push_back(new Themis::prior_linear(0.,1e-6)); // position angle

  std::vector<double> means, ranges;

  means.push_back(1.0);             // flux
  means.push_back(0.5*image_scale); // width
  means.push_back(0.0);             // x_i
  means.push_back(0.0);             // y_i

  ranges.push_back(1.0);      // flux
  ranges.push_back(0.5*image_scale); // width
  ranges.push_back(0.0); // x_i
  ranges.push_back(0.0); // y_i

  for (size_t i=1; i<5; i++) { // for each Gaussian component
    means.push_back(1.0);             // flux
    means.push_back(0.5*image_scale); // width
    means.push_back(0.0);             // x_i
    means.push_back(0.0);             // y_i

    ranges.push_back(1.0);      // flux
    ranges.push_back(0.5*image_scale); // width
    ranges.push_back(image_scale); // x_i
    ranges.push_back(image_scale); // y_i
  }

  means.push_back(0.);
  ranges.push_back(1e-6);


  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  // var_names.push_back("Nr Gaussians");

  var_names.push_back("Flux_1");var_names.push_back("$\\sigma_1$");var_names.push_back("$x_1$");var_names.push_back("$y_1$");
  var_names.push_back("Flux_2");var_names.push_back("$\\sigma_2$");var_names.push_back("$x_2$");var_names.push_back("$y_2$");
  var_names.push_back("Flux_3");var_names.push_back("$\\sigma_3$");var_names.push_back("$x_3$");var_names.push_back("$y_3$");
  var_names.push_back("Flux_4");var_names.push_back("$\\sigma_4$");var_names.push_back("$x_4$");var_names.push_back("$y_4$");
  var_names.push_back("Flux_5");var_names.push_back("$\\sigma_5$");var_names.push_back("$x_5$");var_names.push_back("$y_5$");

  var_names.push_back("PA");

  // Applying the coordinate transformation on the initial values
  // RG:REMOVE
  Themis::transform_none Trans;
  for(unsigned int i = 0 ; i < means.size(); i++)
  // for(unsigned int i = 0 ; i < 20; i++)
  {
    means[i] = Trans.forward(means[i]);
    ranges[i] = Trans.forward(ranges[i]);
  } 

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;

  for (size_t i=0; i<21; i++ )
  T.push_back(new Themis::transform_none());

  // T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(SIM_VM_DATA,image));

  // Closure Phases Likelihoods
  // double sigma_phi = 3.86;
  double sigma_phi = 1e-5; // in this data set the sigma on CP is not global, should use what's in the data
  // 2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(SIM_CP_DATA,image,sigma_phi));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);


  // Generate a chain
  int Number_of_chains = 128;           // Number of walkers
  int Number_of_temperatures = 4;       
  int Number_of_processors_per_lklhd=1;
  int Number_of_steps = 1e6;           // Total number of MCMC steps before quitting
  int Temperature_stride = 50;          // Communicate among walkers lessons learned about the sampled parameter space every 50 MCMC steps
  int Chi2_stride = 100;                 // Output chi^2 file every 20 MCMC steps

  // Parallelization settings
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Sample the parameter space
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                    "Chain-Ngaussians.dat", "Lklhd-Ngaussians.dat", "Chi2-Ngaussians.dat", 
                    means, ranges, var_names, false);
  

  //Finalize MPI
  MPI_Finalize();
  return 0;
}



/*! 
  \file examples/scattered_gaussian_fitting.cpp
  
  \code
  
  #include "data_visibility_amplitude.h"
  #include "data_closure_phase.h"
  #include "model_image_multigaussian.h"
  #include "sampler_affine_invariant_tempered_MCMC.h"
  #include <mpi.h>
  #include "utils.h"

  int main(int argc, char* argv[])
  {
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

    //RG: SIMULATED DATA PROVIDED BY MF_FE_WG CHALLENGE 2
    Themis::data_visibility_amplitude SIM_VM_DATA(Themis::utils::global_path("eht_data/mf_fe_wg_challenge2_vm.txt"));
    Themis::data_closure_phase SIM_CP_DATA(Themis::utils::global_path("eht_data/mf_fe_wg_challenge2_cp.txt"));


    // Choose the model to compare
    Themis::model_image_multigaussian image(5);

    // RG: INTRODUCE NEW PARAMETERS N:Nr of Gaussian components, N-1
    // positions, rest the same as before for each Gaussian

    // Container of base prior class pointers
    double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
    std::vector<Themis::prior_base*> P;

    // P.push_back(new Themis::prior_linear(1.,5.)); // Nr of Gaussian components
    P.push_back(new Themis::prior_linear(0.01,5.01)); // Itotal
    P.push_back(new Themis::prior_linear(0.1*image_scale,0.9*image_scale)); // sigma  
    P.push_back(new Themis::prior_linear(-1e-4,1e-4)); // x-dir
    P.push_back(new Themis::prior_linear(-1e-4,1e-4)); // y-dir

    for (size_t n=1; n<5; n++) {
    P.push_back(new Themis::prior_linear(0.01,5.01)); // Itotal
    P.push_back(new Themis::prior_linear(0.1*image_scale,0.9*image_scale)); // sigma  
    P.push_back(new Themis::prior_linear(-0.5*image_scale,0.5*image_scale)); // x-dir size
    P.push_back(new Themis::prior_linear(-0.5*image_scale,0.5*image_scale)); // y-dir size
  }
    P.push_back(new Themis::prior_linear(0.,1e-6)); // position angle

    std::vector<double> means, ranges;

    means.push_back(1.0);             // flux
    means.push_back(0.5*image_scale); // width
    means.push_back(0.0);             // x_i
    means.push_back(0.0);             // y_i

    ranges.push_back(1.0);      // flux
    ranges.push_back(0.5*image_scale); // width
    ranges.push_back(0.0); // x_i
    ranges.push_back(0.0); // y_i

    for (size_t i=1; i<5; i++) { // for each Gaussian component
      means.push_back(1.0);             // flux
      means.push_back(0.5*image_scale); // width
      means.push_back(0.0);             // x_i
      means.push_back(0.0);             // y_i

      ranges.push_back(1.0);      // flux
      ranges.push_back(0.5*image_scale); // width
      ranges.push_back(image_scale); // x_i
      ranges.push_back(image_scale); // y_i
    }

    means.push_back(0.);
    ranges.push_back(1e-6);


    // vector to hold the name of variables, if the names are provided it would be added 
    // as the header to the chain file 
    std::vector<std::string> var_names;
    // var_names.push_back("Nr Gaussians");

    var_names.push_back("Flux_1");var_names.push_back("$\\sigma_1$");var_names.push_back("$x_1$");var_names.push_back("$y_1$");
    var_names.push_back("Flux_2");var_names.push_back("$\\sigma_2$");var_names.push_back("$x_2$");var_names.push_back("$y_2$");
    var_names.push_back("Flux_3");var_names.push_back("$\\sigma_3$");var_names.push_back("$x_3$");var_names.push_back("$y_3$");
    var_names.push_back("Flux_4");var_names.push_back("$\\sigma_4$");var_names.push_back("$x_4$");var_names.push_back("$y_4$");
    var_names.push_back("Flux_5");var_names.push_back("$\\sigma_5$");var_names.push_back("$x_5$");var_names.push_back("$y_5$");

    var_names.push_back("PA");

    // Applying the coordinate transformation on the initial values
    // RG:REMOVE
    Themis::transform_none Trans;
    for(unsigned int i = 0 ; i < means.size(); i++)
    // for(unsigned int i = 0 ; i < 20; i++)
    {
      means[i] = Trans.forward(means[i]);
      ranges[i] = Trans.forward(ranges[i]);
    } 

    // Set the variable transformations
    std::vector<Themis::transform_base*> T;

    for (size_t i=0; i<21; i++ )
    T.push_back(new Themis::transform_none());

    // T.push_back(new Themis::transform_none());


    // Set the likelihood functions
    std::vector<Themis::likelihood_base*> L;
    L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(SIM_VM_DATA,image));

    // Closure Phases Likelihoods
    // double sigma_phi = 3.86;
    double sigma_phi = 1e-5; // in this data set the sigma on CP is not global, should use what's in the data
    // 2009
    L.push_back(new Themis::likelihood_marginalized_closure_phase(SIM_CP_DATA,image,sigma_phi));


    // Set the weights for likelihood functions
    std::vector<double> W(L.size(), 1.0);


    // Make a likelihood object
    Themis::likelihood L_obj(P, T, L, W);
    

    // Create a sampler object, here the PT MCMC
    Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);


    // Generate a chain
    int Number_of_chains = 128;           // Number of walkers
    int Number_of_temperatures = 4;       
    int Number_of_processors_per_lklhd=1;
    int Number_of_steps = 1e6;           // Total number of MCMC steps before quitting
    int Temperature_stride = 50;          // Communicate among walkers lessons learned about the sampled parameter space every 50 MCMC steps
    int Chi2_stride = 100;                 // Output chi^2 file every 20 MCMC steps

    // Parallelization settings
    MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

    // Sample the parameter space
    MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-Ngaussians.dat", "Lklhd-Ngaussians.dat", "Chi2-Ngaussians.dat", 
                      means, ranges, var_names, false);
    

    //Finalize MPI
    MPI_Finalize();
    return 0;
  }

  
  \endcode
*/
