/*!
	\file validation/crescent_image_comparison.cpp
	\author Jorge A. Preciado
	\date July 2017
		
	\brief Fits a crescent model to visibility amplitude data for validation 
	purposes.
	
	\details Compares a geometric crescent model to the visibility amplitude 
	data taken in 2007 and 2009, permitting a day-specific intensity 
	renormalization. The primary fit result is a measure of the size (\f$ R \f$), 
	the  relative thickness (\f$ \psi \f$), and the degree of symmetry (\f$ \tau \f$)
	of the emission region and can be compared to the fit results reported in 
	[Kamruddin and Dexter 2013](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stt1068).
	
	The resulting parameter distribution is:
		
	\image html plots/validation/Crescent-Triangle.png "Triangle plot for the marginalized posterior probabilty distribution showing the likely parameter values and associated confidence contours."
	
	
	\n Note that the intensity normalization is solved for analytically in the 
	likelihood_marginalized_visibility_amplitude, and thus the intrinsic 
	normalization is fixed near unity by design.
	
	
	The parameter values associated with the individual chains are:
		
	\image html plots/validation/Crescent-Trace.png "Trace plot showing the fluctuations in the parameters for each MCMC chain as a function of MCMC step."
	
	
	\n The associated likelihoods and \f$\chi^2\f$ of the chains are:
	
	
	\image html plots/validation/Crescent-Likelihood.png "Log-likelihoods of the individual chains as a function of MCMC step."  

	\image html plots/validation/Crescent-Chi-squared.png "Chi-squared of the individual chains output periodically throughout the MCMC run."  
		
	\sa See reading_data.cpp example for details on reading in EHT data.
*/


#include "data_visibility_amplitude.h"
#include "model_image_crescent.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"

// Standard Libraries
/// @cond
#include <mpi.h>
#include <memory> 
#include <string>
#include <vector>
/// @endcond



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
	Themis::model_image_crescent intrinsic_image;
	Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);
	
	// Choose the image size
	//intrinsic_image.set_image_resolution(32);
	
	// Use numerical Visibilities
	// intrinsic_image.use_numerical_visibilities();
	
	// Container of base prior class pointers
	// and prior means and ranges
	double crescent_size = 28. * 1.e-6 /3600. /180. * M_PI;
	std::vector<Themis::prior_base*> P;
	std::vector<double> means, ranges;
	P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
	means.push_back(1.0);
	ranges.push_back(1.0e-6);
	
	P.push_back(new Themis::prior_linear(0.01*crescent_size,2.0*crescent_size)); // Overall size R
	means.push_back(crescent_size);
	ranges.push_back(0.01*crescent_size);
	
	P.push_back(new Themis::prior_linear(0.01,0.99)); // psi
	means.push_back(0.10);
	ranges.push_back(0.01);

	P.push_back(new Themis::prior_linear(0.01,0.99)); // tau
	means.push_back(0.10);
	ranges.push_back(0.01);

	P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
	means.push_back(0.4*M_PI);
	ranges.push_back(0.01*M_PI);


	// vector to hold the name of variables, if the names are provided it would be added 
	// as the header to the chain file 
	std::vector<std::string> var_names;
	var_names.push_back("$I_{norm}$");
	var_names.push_back("$R$");
	var_names.push_back("$\\psi$");
	var_names.push_back("$\\tau$");
	var_names.push_back("$\\phi$");
	
	// Applying the coordinate transformation on the initial values
	Themis::transform_none Trans;
	for(unsigned int i = 0 ; i < means.size(); ++i)
	{
		means[i] = Trans.forward(means[i]);
		ranges[i] = Trans.forward(ranges[i]);
	} 



	// Set the likelihood functions
	std::vector<Themis::likelihood_base*> L;
	L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2007,image));
	L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_095,image));
	L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_096,image));
	L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(d2009_097,image));


	// Set the weights for likelihood functions
	std::vector<double> W(L.size(), 1.0);


	// Make a likelihood object
	Themis::likelihood L_obj(P, L, W);


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
	MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
	
	// Set a checkpoint
	MC_obj.set_checkpoint(100,"Crescent.ckpt");
	

	// Run the Sampler                            
	MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-Crescent.dat", "Lklhd-Crescent.dat", 
			   "Chi2-Crescent.dat", means, ranges, var_names, false, 8,verbosity);


	// Finalize MPI
	MPI_Finalize();
	return 0;
}
