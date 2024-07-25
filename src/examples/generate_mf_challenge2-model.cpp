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

#include "model_image_multigaussian.h"
#include "vrt2.h"
#include <mpi.h>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  Themis::model_image_multigaussian image(5);
  // Themis::model_ensemble_averaged_scattered_image image(intrinsic_image);

	// Specify a set of model parameters
	std::vector<double> parameters;
	parameters.push_back(1.77318);
	parameters.push_back(7.5586e-11);
	parameters.push_back(0.);
	parameters.push_back(0.);
	parameters.push_back(0.804506);
	parameters.push_back(9.14256e-11);
	parameters.push_back(2.6971e-11);
	parameters.push_back(-6.66836e-11);
	parameters.push_back(1.8713);
	parameters.push_back(1.00981e-10);
	parameters.push_back(-9.12107e-11);
	parameters.push_back(9.04753e-11);
	parameters.push_back(0.833491);
	parameters.push_back(9.03774e-11);
	parameters.push_back(9.76864e-11);
	parameters.push_back(-1.25779e-10);
	parameters.push_back(0.901399);
	parameters.push_back(2.35097e-10);
	parameters.push_back(3.57249e-11);
	parameters.push_back(3.52196e-11);
	parameters.push_back(5.17455e-07);

	// Generate a model using the previous parameters
	image.generate_model(parameters);

	// Output the image
	std::vector<std::vector<double> > alpha, beta, I;
	image.get_image(alpha,beta,I);
	
	// Output the image
	
	// Convert from radians to projected GM/c^2 just for convenience
	double rad2M = VRT2::VRT2_Constants::D_SgrA_cm/VRT2::VRT2_Constants::M_SgrA_cm;
	
	// Return the image (not usually required) and output it to a file in the
	// standard pmap format -- a rastered array of ASCII values with the limits
	// and dimensions at the top.  This may be plotted with the plot_vrt2_image.py
	// python script with the result shown.
	std::ofstream imout("Ngaussian_image.d");
	
	imout << "Nx:"
		  << std::setw(15) << rad2M*alpha[0][0]
		  << std::setw(15) << rad2M*alpha[alpha.size()-1].back()
		  << std::setw(15) << alpha.size()
		  << '\n';
		
	imout << "Ny:"
		  << std::setw(15) << rad2M*beta[0][0]
		  << std::setw(15) << rad2M*beta[beta.size()-1].back()
		  << std::setw(15) << beta.size()
		  << '\n';
		
	imout << std::setw(5) << "i"
		  << std::setw(5) << "j"
		  << std::setw(15) << "I (Jy/sr)"
		  << '\n';
		
		
	for (size_t j=0; j<alpha[0].size(); ++j)
		for (size_t i=0; i<alpha.size(); ++i)
			imout << std::setw(5) << i
			      << std::setw(5) << j
			      << std::setw(15) << I[i][j]
			      << '\n';  


  //Finalize MPI
  MPI_Finalize();
  return 0;
}

