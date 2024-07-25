/*!
  \file validation/asymmetric_gaussian_image_comparison.cpp
  \author Avery E. Broderick
  \date July 2017
  
  \internal
  \validation Validation test for the assymetric Gaussian model with visibility amplitude data.
  \endinternal
  
  \brief Fits an asymmetric Gaussian model to visibility amplitude data 
  for validation purposes.
  
  \details Compares an asymmetric Gaussian model to the visibility amplitude data taken in 2007 and 2009, permitting a day-specific intensity renormalization.  The primary fit result is a measure of the size and asymmetry of the emission region and can be compared to the fit results reported in [Broderick et al. (2011)](http://adsabs.harvard.edu/abs/2011ApJ...735..110B) (see, especially, Fig. 5).  The resulting parameter distribution is:
      
  \image html plots/validation/Asymmetric-Gaussian-Triangle.png "Triangle plot for the marginalized posterior probabilty distribution showing the likely parameter values and associated confidence contours." width=\\textwidth
  
  \n Note that the intensity normalization is solved for analytically in the 
  likelihood_marginalized_visibility_amplitude, and thus the intrinsic 
  normalization is fixed near unity by design.  Also, \f$9.5\times10^{-9}~{\rm rad}=19.6~\mu{\rm as}\f$, making the reconstructed \f$\sigma\f$ and \f$A\f$ consistent with those reported in [Broderick et al. (2011)](http://adsabs.harvard.edu/abs/2011ApJ...735..110B).

  The parameter values associated with the individual chains are:
    
  \image html plots/validation/Asymmetric-Gaussian-Trace.png "Trace plot showing the fluctuations in the parameters for each MCMC chain as a function of MCMC step."
  
  \n The associated likelihoods and \f$\chi^2\f$ of the chains are:
  
  \image html plots/validation/Asymmetric-Gaussian-Likelihood.png "Log-likelihoods of the individual chains as a function of MCMC step."  

  \image html plots/validation/Asymmetric-Gaussian-Chi-squared.png "Chi-squared of the individual chains output periodically throughout the MCMC run."  
    
  \sa See reading_data.cpp example for details on reading in EHT data.
*/


#include "data_visibility_amplitude.h"
#include "model_image_asymmetric_gaussian.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"

/*! \cond */
#include <mpi.h>
#include <memory> 
#include <string>
/*! \endcond */

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in visibility amplitude data from 2007 and 2009
  Themis::data_visibility_amplitude d2007(Themis::utils::global_path("eht_data/VM_2007_100.d"));
  d2007.add_data(Themis::utils::global_path("eht_data/VM_2007_101.d"));
  Themis::data_visibility_amplitude d2009_095(Themis::utils::global_path("eht_data/VM_2009_095.d"));
  Themis::data_visibility_amplitude d2009_096(Themis::utils::global_path("eht_data/VM_2009_096.d"));
  Themis::data_visibility_amplitude d2009_097(Themis::utils::global_path("eht_data/VM_2009_097.d"));


  // Choose the model to compare
  // Themis::model_image_gaussian image;
  Themis::model_image_asymmetric_gaussian intrinsic_image;
  Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);
  
  
  // Container of base prior class pointers
  double image_scale = 3 * 43.e-6 / 3600. /180. * M_PI / (std::sqrt(8.*std::log(2.))) ;
  std::vector<Themis::prior_base*> P;
  P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
  P.push_back(new Themis::prior_linear(0.0,image_scale)); // size sigma
  P.push_back(new Themis::prior_linear(0.1,0.9)); // asymmetry parameters A
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
  
  
  // Prior means and ranges
  std::vector<double> means, ranges;
  means.push_back(1.0);
  means.push_back(0.5*image_scale);
  means.push_back(0.5);
  means.push_back(0.25*M_PI);
  ranges.push_back(1.0e-6);
  ranges.push_back(0.01*image_scale);
  ranges.push_back(0.01);
  ranges.push_back(0.01*M_PI);


  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
  var_names.push_back("$I_{norm}$");
  var_names.push_back("$\\sigma/\\mathrm{rad}$");
  var_names.push_back("$A$");
  var_names.push_back("$\\xi$");
  
  
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
  int Number_of_chains = 128;
  int Number_of_temperatures = 5;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 10000; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 0;

  // Set the CPU distribution
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, 
                              Number_of_processors_per_lklhd);
  

  // Run the Sampler                            
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-AsymmetricGaussian-5T.dat", "Lklhd-AsymmetricGaussian-5T.dat", 
		     "Chi2-AsymmetricGaussian-5T.dat", means, ranges, var_names, false, verbosity);


  // Finalize MPI
  MPI_Finalize();
  return 0;
}
