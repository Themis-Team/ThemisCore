
#include "data_visibility_amplitude.h"
#include "utils.h"

#include "model_image_symmetric_gaussian.h"
#include "model_image_sum.h"

#include "likelihood_visibility_amplitude.h"

#include "sampler_differential_evolution_tempered_MCMC.h"

#include <mpi.h>
#include <vector>
#include <string>


int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);


  // 1. Read in data
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge2/Ch2_VMs.d"),"HH");

  
  // 2. Define model
  Themis::model_image_symmetric_gaussian image;
  image.use_analytical_visibilities(); // An option for this model

  /* // An example of how to go from 1 to 2 Gaussians
  Themis::model_image_symmetric_gaussian image1, image2;
  image1.use_analytical_visibilities(); // An option for this model
  image2.use_analytical_visibilities(); // An option for this model
  Themis::model_image_sum image;
  image.add(image1);
  image.add(image2);
  */

  // 3. Create likelihood + priors
  //  3.a. Create priors
  std::vector<Themis::prior_base*> P;
  double uas2rad = 1e-6/3600.0 * M_PI/180.;
  P.push_back(new Themis::prior_linear(0.0,10.0)); // Intensity
  P.push_back(new Themis::prior_linear(0.0,1e2*uas2rad)); // Size
  
  //  3.b. Create likelihood
  //  3.b.i. Create individual object for each data object
  Themis::likelihood_visibility_amplitude lva(VM,image);
  //   3.b.ii. Create combined object
  std::vector<Themis::likelihood_base*> L;
  L.push_back(&lva);
  std::vector<double> W(L.size(), 1.0);
  Themis::likelihood L_obj(P, L, W);


  // 4. Create sampler and run
  //  4.a Choose sampler object
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);
  //  4.b Generate and set necessary sampler-specific inputs
  //   4.b.i Magic number options
  int Number_of_steps = 1000;
  int Number_of_chains = 10;
  int Number_of_temperatures = 4;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  ////int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;

  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);

  //   4.b.ii Set up initial ensemble
  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  means.push_back(5.0); // Intensity
  ranges.push_back(2.0); // Intensity
  means.push_back(40*uas2rad); // Size
  ranges.push_back(20*uas2rad); // Size

  //  4.c Run the sampler
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, "Chain.dat", "Lklhd.dat", "Chi2.dat", means, ranges, var_names, restart_flag, out_precision, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}
