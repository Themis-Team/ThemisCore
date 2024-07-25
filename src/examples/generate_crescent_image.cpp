/*!
	\file examples/generate_crescent_image.cpp
	\author Jorge A. Preciado
	\date July 2017
		
	\brief To be added
	
	\details Tobe added
*/


#include "data_visibility_amplitude.h"
#include "model_image_crescent.h"
#include "vrt2.h"

// Standard Libraries
/// @cond
#include <mpi.h>
#include <memory> 
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
/// @endcond



int main(int argc, char* argv[])
{
	// Initialize MPI
	MPI_Init(&argc, &argv);

	// Choose the crescent image model 
	Themis::model_image_crescent crescent_image;
	
	// Set the image resolution
	crescent_image.set_image_resolution(512);
		
	// Specify a set of model parameters
	std::vector<double> parameters;
	parameters.push_back(1.0); 		//Itotal
	parameters.push_back(0.1 * 28 * 1.e-6 / 3600. /180. * M_PI);	// R
	parameters.push_back(0.10);		// psi
	parameters.push_back(0.10);		// tau
	parameters.push_back(0.25*M_PI);		// Position angle
	

	// Generate a model using the previous parameters
	crescent_image.generate_model(parameters);

	// Output the image
	std::vector<std::vector<double> > alpha, beta, I;
	crescent_image.get_image(alpha,beta,I);
	
	// Output the image
	
	// Convert from radians to projected GM/c^2 just for convenience
	double rad2M = VRT2::VRT2_Constants::D_SgrA_cm/VRT2::VRT2_Constants::M_SgrA_cm;
	
	// Return the image (not usually required) and output it to a file in the
	// standard pmap format -- a rastered array of ASCII values with the limits
	// and dimensions at the top.  This may be plotted with the plot_vrt2_image.py
	// python script with the result shown.
	std::ofstream imout("crescent_image.d");
	
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
