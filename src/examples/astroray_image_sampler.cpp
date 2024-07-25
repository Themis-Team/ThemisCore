/*!
  \file astroray_image_sampler.cpp
  \author Roman Gold
  \date June 2017
  \brief Example demonstrating how to drive the external raditaive transfer code ASTRORAY

  \details See model_image_astroray for details.
*/
#define VERBOSITY (0)


#include "model_image_astroray.h"
#include "data_visibility_amplitude.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>


int main(int argc,char* argv[])
{
  MPI_Init(&argc, &argv);
  //int world_rank = MPI::COMM_WORLD.Get_rank();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude d2007("../../eht_data/VM_2007_100.d");
  d2007.add_data("../../eht_data/VM_2007_101.d");
  Themis::data_visibility_amplitude d2009_095("../../eht_data/VM_2009_095.d");
  Themis::data_visibility_amplitude d2009_096("../../eht_data/VM_2009_096.d");
  Themis::data_visibility_amplitude d2009_097("../../eht_data/VM_2009_097.d");


  // Choose the model to compare
  std::vector<double> p;

  // <I>=2.5Jy , <LP>/<I>=3.5%
  p.push_back(1.e5); // rhonor
  p.push_back(60.*M_PI/180.); // inclination angle
  p.push_back(10.); // Te_jet_par
  p.push_back(0.); // Azimuthal viewing (i.e. position) angle

  // best-fit for MAD-disk from https://arxiv.org/abs/1601.05550
  // p.push_back(126198.9381); // rhonor
  // p.push_back(1.73); // inclination angle
  // p.push_back(10.); // Te_jet_par
  // p.push_back(0.); // Azimuthal viewing (i.e. position) angle

  Themis::model_image_astroray ASTRORAY_IMAGE;
  // ASTRORAY_IMAGE.use_numerical_visibilities();
  ASTRORAY_IMAGE.generate_model(p);
  std::vector<std::vector<double> > aN, bN, IN;
  ASTRORAY_IMAGE.get_image(aN,bN,IN);

  Themis::model_ensemble_averaged_scattered_image image(ASTRORAY_IMAGE);

  /*
  for (int i=0; i<10; ++i) {
    double u = 4.e9*i/9.;
    Themis::datum_visibility_amplitude d(u,0.0,1.0,0.1);
    double VA = 0., VB = 0.;
    VA = ASTRORAY_IMAGE.visibility_amplitude(d,0);
    VB = ASTRORAY_IMAGE.visibility_amplitude(d,0);
    // VA = GAUSSIAN.visibility_amplitude(d,0);
    // VB = GAUSSIAN_NUM.visibility_amplitude(d,0);
    std::cout << std::setw(15) << u 
	      << std::setw(15) << VA
	      << std::setw(15) << VB
	      << std::endl;
  }
  */


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(1e5,1e7)); // rhonor
  P.push_back(new Themis::prior_linear(0.0+0.1,M_PI-0.1)); // th (avoid poles)
  P.push_back(new Themis::prior_linear(1.0,1.e3)); // T_e_jet_par
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle

  std::vector<double> means, ranges;
  means.push_back(1.e5);
  means.push_back(60.*M_PI/180.);
  means.push_back(10.);
  means.push_back(0.);
  // What do these mean?
  ranges.push_back(1.0e6);
  ranges.push_back(M_PI/6.);
  ranges.push_back(10.);
  ranges.push_back(0.5*M_PI);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$rho_{norm}$");
  var_names.push_back("$i$");
  var_names.push_back("$T_{e,jet}$");
  var_names.push_back("$d\\phi$");

  // Applying the coordinate transformation on the initial values
  Themis::transform_none Trans;
  for(unsigned int i = 0 ; i < means.size(); i++)
    {
      means[i] = Trans.forward(means[i]);
      ranges[i] = Trans.forward(ranges[i]);
    } 

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 32;
  int Number_of_temperatures = 5;
  int Number_of_steps = 2000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 1);
  MC_obj.run_sampler(L_obj,  
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-astroray-VM-5T.dat", "Lklhd-astroray-VM-5T.dat", 
		     "Chi2-astroray-VM-5T.dat", means, ranges, var_names, false);


  MPI_Finalize();

  return 0;
}
