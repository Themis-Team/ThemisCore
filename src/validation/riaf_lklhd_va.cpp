/*! 
  \file riaf_lklhd_va.cpp
  \author Jorge A. Preciado
  \date  August 2017
  \brief To be added
  \details To be added
*/

#include "data_visibility_amplitude.h"
#include "model_image_sed_fitted_riaf.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory> 
#include <string>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);


  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude d2007(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  d2007.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  Themis::data_visibility_amplitude d2009_095(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  Themis::data_visibility_amplitude d2009_096(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  Themis::data_visibility_amplitude d2009_097(Themis::utils::global_path("eht_data/VM_2009_097.d"));

  // Create an SED-fitted RIAF model
  Themis::model_image_sed_fitted_riaf intrinsic_image(Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"));
  intrinsic_image.set_image_resolution(64);

  // Scatter the image
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);


  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.0,0.998)); // Spin amplitude
  P.push_back(new Themis::prior_linear(-1,1)); // cos(inclination)
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle


  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;
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


  L_obj.set_mpi_communicator(MPI_COMM_WORLD);

  Themis::Ran2RNG RandomNum_Obj(3);
  

  // Generate image at a random parameter value
  for(int i=1;i<=20;++i)
  {
    std::cout << "Likelihood evaluation no. " << i << "\n" << std::endl;
    std::vector<double> params(3);
    params[0] =  0.1 + 0.02*RandomNum_Obj.rand();
    params[1] = -0.5 + 0.5*RandomNum_Obj.rand();
    params[2] = -156.0/180.0*M_PI + 0.2*M_PI*RandomNum_Obj.rand();

    double Lval = L_obj(params);
    std::cout << "The Likelihood value with Spin = " << params[0]
		<< ", Cos(Inclination) = " << params[1]
		<< ", Position angle = " << params[2]
		<< ",  is: " << Lval
		<< std::endl;
  }

  //Finalize MPI
  MPI_Finalize();
  return 0;
}
