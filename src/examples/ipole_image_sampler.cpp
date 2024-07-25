/*!
  \file ipole_image_sampler.cpp
  \author Roman Gold, Monika Moscibrodzka
  \date Dec, 2017
  \brief Example demonstrating how to drive the external radiative transfer code IPOLE

  \details See model_image_ipole for details.
*/
#define VERBOSITY (0)


#include "model_image_ipole.h"
#include "data_visibility_amplitude.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include <mpi.h>
#include <memory> 
#include <string>
#include "utils.h"

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

  // Read in closure phases data from
  //2009
  Themis::data_closure_phase CP_d2009_093(Themis::utils::global_path("eht_data/CP_2009_093.d"));
  Themis::data_closure_phase CP_d2009_096(Themis::utils::global_path("eht_data/CP_2009_096.d"));
  Themis::data_closure_phase CP_d2009_097(Themis::utils::global_path("eht_data/CP_2009_097.d"));
  //2011
  Themis::data_closure_phase CP_d2011_088(Themis::utils::global_path("eht_data/CP_2011_088.d"));
  Themis::data_closure_phase CP_d2011_090(Themis::utils::global_path("eht_data/CP_2011_090.d"));
  Themis::data_closure_phase CP_d2011_091(Themis::utils::global_path("eht_data/CP_2011_091.d"));
  Themis::data_closure_phase CP_d2011_094(Themis::utils::global_path("eht_data/CP_2011_094.d"));
  //2012
  Themis::data_closure_phase CP_d2012_081(Themis::utils::global_path("eht_data/CP_2012_081.d"));
  //2013
  Themis::data_closure_phase CP_d2013_080(Themis::utils::global_path("eht_data/CP_2013_080.d"));
  Themis::data_closure_phase CP_d2013_081(Themis::utils::global_path("eht_data/CP_2013_081.d"));
  Themis::data_closure_phase CP_d2013_082(Themis::utils::global_path("eht_data/CP_2013_082.d"));
  Themis::data_closure_phase CP_d2013_085(Themis::utils::global_path("eht_data/CP_2013_085.d"));
  Themis::data_closure_phase CP_d2013_086(Themis::utils::global_path("eht_data/CP_2013_086.d"));



  // Choose the model to compare
  std::vector<double> p;
  
  p.push_back(60.); // th
  p.push_back(1.e18); // Munit
  p.push_back(1.); // trat_j
  p.push_back(1.); // trat_d
  p.push_back(0.); // PA

 
  std::vector<std::vector<double> > aN, bN, IN;
  // Themis::model_image_ipole IPOLE_IMAGE;
  Themis::model_image_ipole image;
  // IPOLE_IMAGE.generate_model(p);
  // IPOLE_IMAGE.get_image(aN,bN,IN);
  image.generate_model(p);
  image.get_image(aN,bN,IN);

  // RG: Need to turn off scattering off when using closure_phase data 
  // Themis::model_ensemble_averaged_scattered_image image(IPOLE_IMAGE);

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0+1.,180.-1.0)); // th (avoid poles)
  P.push_back(new Themis::prior_linear(0.5e18,1.e20)); // Munit
  P.push_back(new Themis::prior_linear(0.1,100.)); //trat_j
  P.push_back(new Themis::prior_linear(0.1,100.)); //trat_d
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // PA

  std::vector<double> means, ranges;
  means.push_back(60.);
  means.push_back(1.e18);
  means.push_back(1.);
  means.push_back(1.);
  means.push_back(0.);

  // What do these mean?
  ranges.push_back(10.);
  ranges.push_back(0.8e18);
  ranges.push_back(0.5);
  ranges.push_back(0.5);
  ranges.push_back(0.05);

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$i$");
  var_names.push_back("$Munit$");
  var_names.push_back("$Rlow$");
  var_names.push_back("$Rhigh$");
  var_names.push_back("$PA$");

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
  T.push_back(new Themis::transform_none());


  // Set the likelihood functions
  // RG: Not sure whether marginalized VA is the right likelihood here...
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));



  //Closure Phases
  double sigma_phi = 3.86;
  //2009
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_093,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_096,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2009_097,image,sigma_phi));
  //2011
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_088,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_090,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_091,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2011_094,image,sigma_phi));
  //2012
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2012_081,image,sigma_phi));
  //2013
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_080,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_081,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_082,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_085,image,sigma_phi));
  L.push_back(new Themis::likelihood_marginalized_closure_phase(CP_d2013_086,image,sigma_phi));



  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object
  Themis::likelihood L_obj(P, T, L, W);
  

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 8;
  int Number_of_temperatures = 4;
  int Number_of_steps = 100000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 1);
  MC_obj.run_sampler(L_obj,  
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-ipole-VM-5T.dat", "Lklhd-ipole-VM-5T.dat", 
		     "Chi2-ipole-VM-5T.dat", means, ranges, var_names, false, 0);


  MPI_Finalize();

  return 0;
}
