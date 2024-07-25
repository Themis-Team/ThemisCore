/*!
  \file examples/generate_movie_shearing_spot.cpp
  \author Paul Tiede
  \date April, 2018
  \brief Example illustrating the creation of a shearing spot movie model.

  \details Themis allows a wide variety of models to be compared to WHT data. This examples show how the query the semi-analytic shearing spot model, which is an explictly time dependent model that captures some aspects of variability in Sgr A*. This script explictly generates a series of images i.e. a movie using the sed of 2010 data run. The output of this script is given by shearing_spot_movie_frame_i.d where i runs from 0 to 9 and is the frames of the movie in pmap format and each frame may be plotted with the python script vrt2_image.py. For example the fifth frame may be plotted using the commands:
  
  $ cd ../../analysis \n
  $ python plot_vrt2_images.py shearing_spot_movie_frame_4.d

  The resulting 10 frame movie is shown below.

  \image html images/plots/shearing_spot.gif "Short movie of shearing spot model."
*/


#include "model_movie_shearing_spot.h"
#include "model_movie_refractive_scattering.h"
#include <mpi.h>
#include <memory>
#include <string>
#include "utils.h"
#include "random_number_generator.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <string>


int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Pick times to observe in J2000
  //Note the number of entries equals the number of image objects created in model_movie_shearing_spot.
  //Here I'll make a 1 hour observation with 10 frames in my movie
  std::vector<double> observing_times;
	
  //Set parameters of movie such as start time, end time and number of images.
  //Set start of my observation to April 1st 2017 00:00:00 in J2000
  double tstart = 0; //In seconds
  double tend = 1*3600; //End time of observation
  int nframes = 24;
  double dt = (tend-tstart)/(nframes); //set frame duration of movie

  //Create vector of time for which you want to create a movie for
  for (int i = 0; i < nframes; i++)
    observing_times.push_back(tstart+i*dt);

  //Set model to the shearing spot class. To do this, I need to specify the file containing the SED fit results, located in 
  // ../../src/VRT2/DataFiles/2010_combined_fit_parameters.d
  std::string sed_image_fitted = "../../src/VRT2/DataFiles/2010_combined_fit_parameters.d";
  //Note tstart does not have to be the first element of observing_times.
  Themis::model_movie_shearing_spot movie(tstart, observing_times, sed_image_fitted);
  //Set image resolution
  double Nrays = 64;
  movie.set_image_resolution(Nrays);

  //Set screen size
  movie.set_screen_size(10);

  //Add background accretion flow
  movie.add_background_riaf();


  //Now scatter the movie
  size_t nModes = 64;
  double uas2rad = 1e-6/3600*M_PI/180.0;
  Themis::model_movie_refractive_scattering<Themis::model_movie_shearing_spot> scattered_movie(movie, nModes, tstart);
  scattered_movie.set_image_resolution(Nrays);
  scattered_movie.set_screen_size(120.0*uas2rad);

  // Choose a specific set of physical parameters at which to create an image.
  std::vector<double> parameters; 
  parameters.push_back(0.2);                 // Spin parameter (0-1)
  parameters.push_back(std::cos(M_PI/3.0));  // Cos(Inclination)
  parameters.push_back(1e8); 					 // Spot density factor
  parameters.push_back(0.5);                 //gaussian spot width
  parameters.push_back(0.0); 							 // Initial time of spot (M)
  parameters.push_back(6.3); 							   // Initial radius of spot (M)
  parameters.push_back(0.0); 							   // Initial azimuthal angle (radians) (-pi-pi)
  parameters.push_back(0.05); 							// Infall rate factor (0-1)
  parameters.push_back(0.99); 							// Subkeplerian factor (0-1)
  parameters.push_back(0.0);

  size_t nsrc = parameters.size();

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  //Now add in the screen parameters
  Themis::GaussianRandomNumberGenerator<Themis::Ran2RNG> grng(42);
  //parameters[nsrc] = 25;
  for ( size_t i = 0; i < nModes*nModes-1; i++ ){
    parameters.push_back(grng.rand());
  }
  if (world_rank==0)
  {

    std::ofstream screen_out("shearing_spot_screen_parameters_test.dat");
    screen_out << "# nModes="<< nModes << std::endl;
    for ( size_t i = nsrc; i < parameters.size(); i++)
      screen_out << parameters[i] << std::endl;
  }

  
  //Generate model movie
  scattered_movie.generate_model(parameters);

  // Returns the movie (usually not required) and output it to a file in the 
  // standard pmap format --  a rastered array of ASCII values with the limits
  // and dimensions as the top. This may be plotted with the plot_vrt2_image.py
  // python script with the results shown.
    std::string basename = "scattered_shearing_spot_frame_";
    for ( size_t l = 0; l < observing_times.size(); ++l)
    {
      //Access the movie frame data
      std::vector< std::vector<double>> I,Iea, alpha, beta;
      scattered_movie.get_movie_frame(observing_times[l], alpha, beta, I);
      scattered_movie.get_ensemble_average_frame(observing_times[l], alpha, beta, Iea);
     

    if (world_rank ==0)
    {
      //Create frame name
      std::stringstream movie_name;
      movie_name << basename << l << ".d";
      std::ofstream movout(movie_name.str().c_str());
      
      movout << "Nx:"
             << std::setw(15) << alpha[0][0]
             << std::setw(15) << alpha[alpha.size()-1].back()
             << std::setw(15) << alpha.size() << std::endl;
      movout << "Ny:"
             << std::setw(15) << beta[0][0]
             << std::setw(15) << beta[beta.size()-1].back()
             << std::setw(15) << beta.size() << std::endl;
      movout << std::setw(5) << "i"
             << std::setw(5) << "j"
	     << std::setw(15) << "I (Jy/sr)"
             << std::setw(15) << "Iea (Jy/sr)\n"; 
      for (size_t m=0; m< alpha[0].size(); ++m)
        for (size_t n=0; n <alpha.size(); ++n)
          movout << std::setw(5) << n
	         << std::setw(5) << m
	         << std::setw(15) << I[n][m]
                 << std::setw(15) << Iea[n][m] << std::endl;
    
      movout.close();
      
    }
  }
  

  //Finalize MPI
  MPI_Finalize();

  return 0;
}

/*! 
  \file examples/generate_movie_shearing_spot.cpp
  \details

  \code

  // Initialize MPI
  MPI_Init(&argc, &argv);

  //Pick times to observe in J2000
	//Note the number of entries equals the number of image objects created in model_movie_shearing_spot.
  //Here I'll make a 1 hour observation with 10 frames in my movie
	std::vector<double> observing_times;
	
  //Set parameters of movie such as start time, end time and number of images.
  //Set start of my observation to April 1st 2017 00:00:00 in J2000
  double tstart = 0; //In seconds
  double tend = 1.2*3600; //End time of observation
  int nframes = 12;
	double dt = (tend-tstart)/(nframes); //set frame duration of movie

  //Create vector of time for which you want to create a movie for
	for (int i = 0; i < nframes; i++)
		observing_times.push_back(tstart+i*dt);

  //Set model to the shearing spot class. To do this, I need to specify the file containing the SED fit results, located in 
  // ../../src/VRT2/DataFiles/2010_combined_fit_parameters.d
	std::string sed_image_fitted = "../../src/VRT2/DataFiles/2010_combined_fit_parameters.d";
  //Note tstart does not have to be the first element of observing_times.
  Themis::model_movie_shearing_spot movie(tstart, observing_times, sed_image_fitted);

  // Choose a specific set of physical parameters at which to create an image.
 	std::vector<double> parameters; 
  parameters.push_back(0.2);                 // Spin parameter (0-1)
  parameters.push_back(std::cos(M_PI/3.0));  // Cos(Inclination)
  parameters.push_back(1e8); 					 // Spot density factor
	parameters.push_back(0.5);                 //gaussian spot width
	parameters.push_back(0); 							 // Initial time of spot (M)
	parameters.push_back(6.3); 							   // Initial radius of spot (M)
	parameters.push_back(0.0); 							   // Initial azimuthal angle (radians) (-pi-pi)
	parameters.push_back(0.95); 							// Infall rate factor (0-1)
	parameters.push_back(0.99); 							// Subkeplerian factor (0-1)
	parameters.push_back(0.0);                //Position angle (which doesn't impact the intrinsic image).

  //Set image resolution
  double Nrays = 64;
  movie.set_image_resolution(Nrays);

	//Set screen size
	movie.set_screen_size(10);

  //Add background accretion flow
  movie.add_background_riaf();
  
  //Generate model movie
	movie.generate_model(parameters);

  // Returns the movie (usually not required) and output it to a file in the 
  // standard pmap format --  a rastered array of ASCII values with the limits
  // and dimensions as the top. This may be plotted with the plot_vrt2_image.py
  // python script with the results shown.
  std::string basename = "shearing_spot_riaf_frame_";
  for ( size_t l = 0; l < observing_times.size(); ++l)
  {
    //Access the movie frame data
    std::vector< std::vector<double>> I, alpha, beta;
    movie.get_movie_frame(observing_times[l], alpha, beta, I);
      
    //Create frame name
    std::stringstream movie_name;
    movie_name << basename << l << ".d";
    std::ofstream movout(movie_name.str().c_str());
      
    movout << "Nx:"
      << std::setw(15) << alpha[0][0]
      << std::setw(15) << alpha[alpha.size()-1].back()
      << std::setw(15) << alpha.size() << std::endl;
	  movout << "Ny:"
      << std::setw(15) << beta[0][0]
      << std::setw(15) << beta[beta.size()-1].back()
      << std::setw(15) << beta.size() << std::endl;
	  movout << std::setw(5) << "i"
	    << std::setw(5) << "j"
	    << std::setw(15) << "I (Jy/sr)"
	    << std::endl;
    for (size_t m=0; m< alpha[0].size(); ++m)
      for (size_t n=0; n <alpha.size(); ++n)
        movout << std::setw(5) << m
	             << std::setw(5) << n
	             << std::setw(15) << I[m][n]
	             << std::endl;
    movout.close();
  }
  

  //Finalize MPI
  MPI_Finalize();
 
  \endcode
*/

