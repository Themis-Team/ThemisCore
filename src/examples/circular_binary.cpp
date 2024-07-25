/*!
  \file circular_binary.cpp
  \author Roman Gold
  \date Sep 2017
  \test Test binary model fitting

  \brief Fit visibility magnitude and closure phase data to a simple binary model

  \details Model fitting to a binary model consisting of two point
  sources moving on circular Keplerian orbits projected on the
  sky. Here we mimick the data using two Gaussians that move on a
  circular on the sky (face-on).
*/

#define VERBOSITY (0)

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_circular_binary.h"
// #include "likelihood_closure_phase.h"
// #include "likelihood_visibility_amplitude.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // DATA: STANDARD OLD EHT DATA for now,
  // will change to two Gaussians in face-on binary configuration for this test

  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude VM_d2007("../../eht_data/VM_2007_100.d");
  VM_d2007.add_data("../../eht_data/VM_2007_101.d");
  Themis::data_visibility_amplitude VM_d2009_095("../../eht_data/VM_2009_095.d");
  Themis::data_visibility_amplitude VM_d2009_096("../../eht_data/VM_2009_096.d");
  Themis::data_visibility_amplitude VM_d2009_097("../../eht_data/VM_2009_097.d");

  // Read in closure phases data from
  // 2009
  Themis::data_closure_phase CP_d2009_093("../../eht_data/CP_2009_093.d");
  Themis::data_closure_phase CP_d2009_096("../../eht_data/CP_2009_096.d");
  Themis::data_closure_phase CP_d2009_097("../../eht_data/CP_2009_097.d");
  // 2011
  Themis::data_closure_phase CP_d2011_088("../../eht_data/CP_2011_088.d");
  Themis::data_closure_phase CP_d2011_090("../../eht_data/CP_2011_090.d");
  Themis::data_closure_phase CP_d2011_091("../../eht_data/CP_2011_091.d");
  Themis::data_closure_phase CP_d2011_094("../../eht_data/CP_2011_094.d");
  // 2012
  Themis::data_closure_phase CP_d2012_081("../../eht_data/CP_2012_081.d");
  // 2013
  Themis::data_closure_phase CP_d2013_080("../../eht_data/CP_2013_080.d");
  Themis::data_closure_phase CP_d2013_081("../../eht_data/CP_2013_081.d");
  Themis::data_closure_phase CP_d2013_082("../../eht_data/CP_2013_082.d");
  Themis::data_closure_phase CP_d2013_085("../../eht_data/CP_2013_085.d");
  Themis::data_closure_phase CP_d2013_086("../../eht_data/CP_2013_086.d");


  // MODEL
  double sigma = 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.)));

  std::vector<double> p;
  p.push_back(2.85); // total flux primary
  p.push_back(sigma); // Gaussian width 43 muas
  p.push_back(0.0); // spectral index primary
  
  p.push_back(2.85);; // total flux secondary
  p.push_back(sigma); //  Gaussian width 43 muas
  p.push_back(0.0); // spectral index secondary

  p.push_back(4.e6 * 1.5e5); // total mass in cm
  p.push_back(1.0/127.); // q=1 <=> equal masses // FOR OJ-287 q ~ 1./128.
  p.push_back(10.*2e-7); // separation
  // p.push_back(60.); // separation
  p.push_back(8.); // distance to source (comparable to actual EHT data for this test)
  p.push_back(0.0); // initial phase
  p.push_back(0.2); // inclination mu=cos(i)
  p.push_back(0.0); // PA

  Themis::model_circular_binary BINARY;

  BINARY.generate_model(p);
  // BINARY.use_numerical_visibilities();

  // Themis::datum_closure_phase d_CP(1e8, 1e9, 2e8, 2e9, 10., 1.);
  // Themis::datum_visibility_amplitude d_VA(1e8, 1e9, 2e8, 2e9, 10., 1.);

  // double CP;
  // double VA;
  // CP = BINARY.closure_phase(d_CP, 0);
  // std::cerr << "Generated CLOSURE PHASE:" << CP << std::endl;
  // VA = BINARY.visibility_amplitude(VM_d2007,0.);
  // std::cerr << "Generated VISIBILITY AMPLITUDE:" << VA << std::endl;

  /*
  std::vector<std::vector<double> > a, b, I;
  BINARY.get_image(a,b,I);
  std::cerr << a.size() << " " << b.size() << " " << I.size() << '\n';
  std::cout << "\n\n" << std::endl;;
  */


  // for (int i=0; i<10; ++i)
  // {
  //   double u = 4.e9*i/9.;
  //   Themis::datum_visibility_amplitude d(u,0.0,1.0,0.1);
    
  //   double VA = 0.;
  //   std::complex<double> V;
  //   VA = BINARY.visibility_amplitude(d,0);
    
  //   std::cout << std::setw(15) << u 
  // 	      << std::setw(15) << VA
  // 	      << std::endl;
  // }



  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0, 10.));  // total flux primary
  // double sigma = 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.)));
  P.push_back(new Themis::prior_linear(0.0, 10.*sigma)); // Gaussian width primary 43 muas // sigma
  P.push_back(new Themis::prior_linear(0.,3.)); // spectral index

  P.push_back(new Themis::prior_linear(0.0, 10.));  // total flux secondary
  P.push_back(new Themis::prior_linear(0.0, 10.*sigma)); // Gaussian width secondary 43 muas // sigma
  P.push_back(new Themis::prior_linear(0.,3.)); // spectral index secondary

  P.push_back(new Themis::prior_linear(1.e6 * 1.5e5, 1.e7 * 1.5e5)); // total mass in cm
  P.push_back(new Themis::prior_linear(0.,1.)); // q=1 <=> equal masses , FOR OJ-287 q~1/128
  P.push_back(new Themis::prior_linear(0.,1000.*10.*2e-7)); // separation
  P.push_back(new Themis::prior_linear(5.,10.)); // distance to source (comparable to Sgr A* for this test)
  P.push_back(new Themis::prior_linear(0.,2.*M_PI)); // initial phase
  P.push_back(new Themis::prior_linear(0.,1.)); // inclination mu=cos(i)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // PA



  // Define starting values and ranges for model parameters
  std::vector<double> means, ranges;
  means.push_back(2.85);               // total flux primary
  means.push_back(sigma);             // std dev Gaussian
  means.push_back(0.0);               // spectral index primary
  means.push_back(2.85);               // total flux secondary
  means.push_back(sigma);             // std dev Gaussian secondary
  means.push_back(0.0);               // spectral index secondary
  means.push_back(4.e6 * 1.5e5);     // M
  means.push_back(0.5);                // q
  means.push_back(10.*2e-7);               // separation in [pc]
  means.push_back(8.);              // distance in [kpc]
  means.push_back(M_PI);               // Phi0
  means.push_back(0.5);               // mu=cos(inclination)
  means.push_back(0.);               // position angle

  //  ranges.push_back(5.7);               // total flux primary
  ranges.push_back(0.1*2.85);               // total flux primary
  //  ranges.push_back(2.*sigma);         // std dev Gaussian
  ranges.push_back(0.1*sigma);         // std dev Gaussian
  //  ranges.push_back(3.);               // spectral index primary
  ranges.push_back(0.1);               // spectral index primary
  // ranges.push_back(5.7);               // total flux secondary
  ranges.push_back(0.1*2.85);               // total flux secondary
  ranges.push_back(0.1*sigma);         // std dev Gaussian secondary
  ranges.push_back(1.);               // spectral index secondary
  ranges.push_back(1e11);                // M
  ranges.push_back(0.1);                // q

  ranges.push_back(0.1*10.*2e-7);             // separation in [pc]
  ranges.push_back(1.);                // distance in [kpc]
  ranges.push_back(0.1*1.*M_PI);               // Phi0
  ranges.push_back(0.1);               // mu=cos(inclination)
  ranges.push_back(0.1 * 1.*M_PI);               // position angle

  // ranges.push_back(0.9*10.*2e-7);             // separation in [pc]
  // ranges.push_back(5.);                // distance in [kpc]
  // ranges.push_back(0.1);               // Phi0
  // ranges.push_back(0.1);               // mu=cos(inclination)
  // ranges.push_back(0.1);               // position angle




  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{primary}$");
  var_names.push_back("$\\sigma_{primary}$");
  var_names.push_back("$\\alpha_{primary}$");
  var_names.push_back("$I_{secondary}$");
  var_names.push_back("$\\sigma_{secondary}$");
  var_names.push_back("$\\alpha_{secondary}$");

  var_names.push_back("$M$");
  var_names.push_back("$q$");
  var_names.push_back("$R$");
  var_names.push_back("$d$");
  var_names.push_back("$\\Phi_0$");
  var_names.push_back("$mu=cos(i)$");
  var_names.push_back("$PA$");



  //Set the variable transformations.
  //Here we are using no coordinated transformations on the parameters
  std::vector<Themis::transform_base*> T;

  //Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;

  // Visibility Magnitudes Likelihoods
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2007, BINARY));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_095, BINARY));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_096, BINARY));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_d2009_097, BINARY));

  // Closure Phases Likelihoods
  double sigma_phi = 3.86;
  // 2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_093,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_096,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_097,BINARY,sigma_phi));
  // 2011
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_088,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_090,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_091,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_094,BINARY,sigma_phi));
  // 2012
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2012_081,BINARY,sigma_phi));
  // 2013
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_080,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_081,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_082,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_085,BINARY,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_086,BINARY,sigma_phi));


  //Set the weights for likelihood functions
  std::vector<double> W(1, 1.0);

  //Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  
  // Make Themis sampler object, affine invariant tempered MCMC
  // pass seed of random nr generator is passed to the constructor
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42);




  int nr_of_chains = 6;       // Nr of walkers at each temperature
  int nr_of_temperatures = 6; // Nr of temperatures in the parallel tempering
  int cpu_per_likelihood = 1; // Nr of processes used to evaluate a single likelihood 
  int nr_of_steps = 1e7+1;     // Nr of monte carlo steps 
  int temp_stride = 50;       // Nr of MCMC steps for communication among neighbouring temperatures
  int chi2_stride = 100;    // Frequency of calculating chi squared values
  int ckpt_stride = 1e9;      // Frequency of saving checkpoints

  // Set the cpu distribution on different parallelization levels
  MC_obj.set_cpu_distribution(nr_of_temperatures, nr_of_chains, cpu_per_likelihood);
  // Set checkpointing options
  MC_obj.set_checkpoint(ckpt_stride, "circular_binary.ckpt");
  // Run the sampler with the given settings
  MC_obj.run_sampler(L_obj, nr_of_steps, 
		     temp_stride, chi2_stride, "circular_binary_chain.dat", "circular_binary_lklhd.dat", "circular_binary_chi2.dat", 
		     means, ranges, var_names, false, 0);



  //Finalize MPI
  MPI_Finalize();

  return 0;
}
