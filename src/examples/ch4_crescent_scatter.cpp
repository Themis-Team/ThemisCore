//!!! fitting challenge4 data by crescent model + gaussian scattering prepared by HY

#include "data_visibility_amplitude.h"
#include "model_image_crescent.h"
//#include "model_ensemble_averaged_scattered_image.h"
#include "model_ensemble_averaged_parameterized_scattered_image.h"
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
	Themis::data_visibility_amplitude VM_data(Themis::utils::global_path("eht_data/ch4_VM02.d"));

// Read in closure phases data from
  Themis::data_closure_phase CP_data(Themis::utils::global_path("eht_data/ch4_CP02.d"));

  
	// Choose the model to compare
	Themis::model_image_crescent intrinsic_image;
	Themis::model_ensemble_averaged_parameterized_scattered_image image(intrinsic_image);
	
	// Choose the image size
	//intrinsic_image.set_image_resolution(32);
	
	// Use numerical Visibilities
	// intrinsic_image.use_numerical_visibilities();
	
	// Container of base prior class pointers
	// and prior means and ranges
	double crescent_size = 28. * 1.e-6 /3600. /180. * M_PI;
	std::vector<Themis::prior_base*> P;
	std::vector<double> means, ranges;
	//==for marginalized visibility 
  //P.push_back(new Themis::prior_linear(0.99,1.01)); // Itotal
 	//means.push_back(1.);
	//ranges.push_back(1.0e-6);
 	P.push_back(new Themis::prior_linear(0.,5.)); // Itotal
	means.push_back(3.);
	ranges.push_back(1.0);
	
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
 
 //== the following 7 parameters are for the scattering model
 /*
    - [1]parameters[model.size+0] ... Major/minor axis at pivot frequency.
    - [2]parameters[model.size+1] ... Frequency power law index of major/minor axis.
    - [3]parameters[model.size+2] ... Minor/major axis at pivot frequency.
    - [4]parameters[model.size+3] ... Frequency power law index of minor/major axis.
    - [5]parameters[model.size+4] ... Position angle at pivot frequency.
    - [6]parameters[model.size+5] ... Position angle frequency dependence normalization.
    - [7]parameters[model.size+6] ... Position angle frequency power law index.
 */
 double uas2rad= 1e-6 / 3600. /180. * M_PI;
 //[1]
 	P.push_back(new Themis::prior_linear(0.*uas2rad,50*uas2rad)); //Major/minor axis at pivot frequency
	means.push_back(5.*uas2rad);
	ranges.push_back(1.*uas2rad);
 //[2] 
 	P.push_back(new Themis::prior_linear(-1e-6,1e-6));     // Frequency power law index of major/minor axis=0
	means.push_back(0.);
	ranges.push_back(1e-8);
 //[3]
 	P.push_back(new Themis::prior_linear(0.*uas2rad,50*uas2rad));  //  Minor/major axis at pivot frequency
	means.push_back(5.*uas2rad);
	ranges.push_back(1.*uas2rad);
 //[4] 
 	P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // Frequency power law index of minor/major axis=0
	means.push_back(0.);
	ranges.push_back(1e-8);
 //[5] 
 	P.push_back(new Themis::prior_linear(0.,M_PI)); // Position angle at pivot frequency =xi0
	means.push_back(0.4*M_PI);
	ranges.push_back(0.01*M_PI);
 //[6] 
 	P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // Position angle frequency dependence normalization =0
	means.push_back(0.);
	ranges.push_back(1e-8);
 //[7] 
 	P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // Position angle frequency power law index =0
	means.push_back(0.);
	ranges.push_back(1e-8);
 


	// vector to hold the name of variables, if the names are provided it would be added 
	// as the header to the chain file 
	std::vector<std::string> var_names;
	var_names.push_back("$I_{norm}$");
	var_names.push_back("$R$");
	var_names.push_back("$\\psi$");
	var_names.push_back("$\\tau$");
	var_names.push_back("$\\xi$");
	
	// Applying the coordinate transformation on the initial values
	Themis::transform_none Trans;
	for(unsigned int i = 0 ; i < means.size(); ++i)
	{
		means[i] = Trans.forward(means[i]);
		ranges[i] = Trans.forward(ranges[i]);
	} 



	// Set the likelihood functions
	std::vector<Themis::likelihood_base*> L;
//	L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_data,intrinsic_image));
 	L.push_back(new Themis::likelihood_visibility_amplitude(VM_data,image));
	L.push_back(new Themis::likelihood_closure_phase(CP_data,intrinsic_image));



	// Set the weights for likelihood functions
	std::vector<double> W(L.size(), 1.0);


	// Make a likelihood object
	Themis::likelihood L_obj(P, L, W);


	// Create a sampler object, here the PT MCMC
	Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);


	// Generate a chain
	//int Number_of_chains = 128;
	//int Number_of_temperatures = 5;
 	int Number_of_chains = 64;
	int Number_of_temperatures = 4;
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
			   "Chi2-Crescent.dat", means, ranges, var_names, false, verbosity);


	// Finalize MPI
	MPI_Finalize();
	return 0;
}
