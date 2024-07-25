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
#include <mpi.h>
#include <memory>
#include <string>
#include "utils.h"


int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


  //default parameters values
  std::string basename = "movie_pmap_";
  int Nrays = 64;
  double img_size=10;
  double spin = 0.2;
  double cosInc = 0.5;
  double n0 = 1e7;
  double Rs = 0.5;
  double r0 = 6;
  double phi0 = 0;
  double infall = 0.05;
  double subKep = 0.99;
  int nframes = 6;
  double movie_length_hr = 1;
  bool add_riaf = false;
  double frequency = 230e9;
  for ( int i = 0; i < argc; i++){
    if ( std::string(argv[i]) == "--name")
      basename = argv[++i];
    else if ( std::string(argv[i]) == "-nray")
      Nrays = atoi(argv[++i]);
    else if ( std::string(argv[i]) == "-imgR")
      img_size = atof(argv[++i]);
    else if ( std::string(argv[i]) == "-riaf")
      add_riaf = true;
    else if ( std::string(argv[i]) == "-a")
      spin = atof(argv[++i]);
    else if (std::string(argv[i]) == "-cI")
      cosInc = atof(argv[++i]);
    else if (std::string(argv[i]) == "-n0")
      n0 = atof(argv[++i]);
    else if ( std::string(argv[i]) == "-Rs")
      Rs = atof(argv[++i]);
    else if ( std::string(argv[i]) == "-r0")
      r0 = atof(argv[++i]);
    else if ( std::string(argv[i]) == "-phi0")
      phi0 = atof(argv[++i])*M_PI/180.0;
    else if (std::string(argv[i]) == "-iR")
      infall = atof(argv[++i]);
    else if (std::string(argv[i]) == "-sk")
      subKep = atof(argv[++i]);
    else if (std::string(argv[i]) == "-nf")
      nframes = atoi(argv[++i]);
    else if (std::string(argv[i]) == "-t")
      movie_length_hr = atof(argv[++i]);
    else if (std::string(argv[i]) == "-nu")
      frequency = atof(argv[++i]);
    else if (std::string(argv[i]) == "-h"){
      std::cerr << "Possible arguments:\n"
                  << "--name:      string base the basename of the output.\n" 
                  << "             DEFAULT: movie_pmap_ which outputs movie_pmap_frame-i.d\n"
                  << "-nray:       Number of rays or image resolution in image. DEFAULT: 128\n"
                  << "-imgR:       Image dimensions in units of M. DEFAULT: 10 M\n"
                  << "-nf:         Number of frames in movie. DEFAULT: 6\n"
                  << "-t:          Total duration of movie in HOURS. DEFAULT: 1\n"
                  << "-riaf:       Add riaf background image. DEFAULT no riaf\n"
                  << "-a:          Black hole spin. DEFAULT 0.2\n"
                  << "-cI:         Cosine of inclination. DEFAULT 0.5 i.e. 60 degrees\n"
                  << "-n0:         Spot Density parameters. DEFAULT 1e7\n"
                  << "-Rs:         Initial spot size in M. DEFAULT 0.5 M\n"
                  << "-r0:         Initial radius of spot in M. DEFAULT 6.0\n"
                  << "-phi0:       Initial azimuthal angle of spot in degrees (-180,180). DEFAULT -90\n"
                  << "-iR:         Infall rate parameter. 0 is pure infall, 1.0 is closed orbit. DEFAULT 0.95\n"
                  << "-sk:         Sub-Keplerian factor: DEFAULT: 0.99\n"
                  << "-nu:         Observation band frequency in Hz\n" << std::endl;
      std::exit(1);
    }
  }

  //Pick times to observe in J2000
  //Note the number of entries equals the number of image objects created in model_shearing_spot.
  //Here I'll make a 1 hour observation with 10 frames in my movie
  std::vector<double> observing_times;
	
  //Set parameters of movie such as start time, end time and number of images.
  //Set start of my observation to April 1st 2017 00:00:00 in J2000
  double tstart = 0; //In seconds
  double tend = tstart+movie_length_hr*3600;//200*VRT2::VRT2_Constants::M_SgrA_cm/VRT2::VRT2_Constants::c; //End time of observation
  double dt = (tend-tstart)/(nframes); //set frame duration of movie

  //Create vector of time for which you want to create a movie for
  for (int i = 0; i < nframes; i++)
    observing_times.push_back(tstart+i*dt);

  std::cout << "Number of frames : " << observing_times.size() << std::endl;
  //Set model to the shearing spot class. To do this, I need to specify the file containing the SED fit results, located in 
  // ../../src/VRT2/DataFiles/2010_combined_fit_parameters.d
  std::string sed_image_fitted = Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d");
  //Note tstart does not have to be the first element of observing_times.
  Themis::model_movie_shearing_spot movie(tstart, observing_times, sed_image_fitted,frequency);
  //Set image resolution
  movie.set_image_resolution(Nrays);

  //Set screen size
  movie.set_screen_size(img_size);

  //Add background accretion flow
  if ( add_riaf)
    movie.add_background_riaf();
 
  // Choose a specific set of physical parameters at which to create an image.
  std::vector<double> parameters; 
  parameters.push_back(spin);                 // Spin parameter (0-1)
  parameters.push_back(cosInc);  // Cos(Inclination)
  parameters.push_back(n0); 					 // Spot density factor
  parameters.push_back(Rs);                 //gaussian spot width
  parameters.push_back(0.0); 							 // Initial time of spot (M)
  parameters.push_back(r0); 							   // Initial radius of spot (M)
  parameters.push_back(phi0); 							   // Initial azimuthal angle (radians) (-pi-pi)
  parameters.push_back(infall); 							// Infall rate factor (0-1)
  parameters.push_back(subKep); 							// Subkeplerian factor (0-1)
  parameters.push_back(-156.0*M_PI/180.0);    //Position angle (which doesn't impact the intrinsic image).
  std::vector<string> var_names;
  var_names.push_back("spin");
  var_names.push_back("Cos(Incl)");
  var_names.push_back("n0");
  var_names.push_back("Rs");
  var_names.push_back("t0 (M)");
  var_names.push_back("r0 (M)");
  var_names.push_back("phi0 (rad)");
  var_names.push_back("alphaR");
  var_names.push_back("SubKep");
  var_names.push_back("Pos Ang (rad)");


 
  //Generate model movie
  movie.generate_model(parameters);

  // Returns the movie (usually not required) and output it to a file in the 
  // standard pmap format --  a rastered array of ASCII values with the limits
  // and dimensions as the top. This may be plotted with the plot_vrt2_image.py
  // python script with the results shown.

  if ( world_rank==0){
    //Save parameters to file
    std::stringstream params_name;
    params_name << basename << "params.d";
    std::ofstream outpara(params_name.str().c_str());
    outpara << "##" << std::setw(10) 
	    << "duration "<< std::setw(5)
	    << "nframes " << std::endl;
    outpara << std::setw(10) << (tend-tstart)/3600.0
	    << std::setw(5) << nframes << std::endl;
  
    outpara << "##";
    for ( size_t p = 0; p < var_names.size(); p++)
	outpara << std::setw(15) << var_names[p];
    outpara << std::endl;
    for ( size_t p = 0; p < parameters.size(); p++)
      outpara << std::setw(15) << parameters[p];
    outpara << std::endl;
  }



  for ( size_t l = 0; l < observing_times.size(); ++l)
  {
    //Access the movie frame data
    std::vector< std::vector<double>> I, alpha, beta;
    movie.get_movie_frame(observing_times[l], alpha, beta, I);
    if ( world_rank==0){
      double dxdy = (alpha[1][1]-alpha[0][0])*(beta[1][1]-beta[0][0]);
      double sum = 0;
      //Create frame name
      std::stringstream movie_name;
      movie_name << basename <<"frame-" << l << ".d";
      std::ofstream movout(movie_name.str().c_str());
      movout.precision(12);
      movout << "Nx:"
      	     << std::setw(20) << alpha[0][0]
      	     << std::setw(20) << alpha[alpha.size()-1].back()
      	     << std::setw(20) << alpha.size() << std::endl;
      movout << "Ny:"
      	     << std::setw(20) << beta[0][0]
      	     << std::setw(20) << beta[beta.size()-1].back()
      	     << std::setw(20) << beta.size() << std::endl;
      movout << std::setw(20) << "i"
	     << std::setw(20) << "j"
	     << std::setw(20) << "I (Jy/sr)" << std::endl;

    for (size_t m=0; m< alpha[0].size(); ++m)
      for (size_t n=0; n <alpha.size(); ++n){
        movout << std::setw(20) << m
	       << std::setw(20) << n
	       << std::setw(20) << I[m][n] << std::endl;
	sum += I[m][n]*dxdy;
      }
    movout.close();
    std::cout << "Total flux: " << sum << "Jy\n";
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
	//Note the number of entries equals the number of image objects created in model_shearing_spot.
  //Here I'll make a 1 hour observation with 10 frames in my movie
	std::vector<double> observing_times;
	
  //Set parameters of movie such as start time, end time and number of images.
  //Set start of my observation to April 1st 2017 00:00:00 in J2000
  double tstart = 544276800; //In seconds
  double tend = 544276800 + 3600; //End time of observation
  int number_of_images = 10;
	double dt = (tend-tstart)/(number_of_images-1); //set frame duration of movie

  //Create vector of time for which you want to create a movie for
	for (int i = 0; i < 4; i++)
		observing_times.push_back(tstart+i*dt);

  //Set model to the shearing spot class. To do this, I need to specify the file containing the SED fit results, located in 
  // ../../src/VRT2/DataFiles/2010_combined_fit_parameters.d
	std::string sed_image_fitted = "../../src/VRT2/DataFiles/2010_combined_fit_parameters.d";
  //Note tstart does not have to be the first element of observing_times.
  Themis::model_shearing_spot movie(tstart, observing_times);

  // Choose a specific set of physical parameters at which to create an image.
 	std::vector<double> parameters; 
  parameters.push_back(0.2);                 // Spin parameter (0-1)
  parameters.push_back(std::cos(M_PI/3.0));  // Cos(Inclination)
  parameters.push_back(5.82161e7); 					 // Spot density factor
	parameters.push_back(0.5);                 //gaussian spot width
	parameters.push_back(-2000); 							 // Initial time of spot (M)
	parameters.push_back(7.0); 							   // Initial radius of spot (M)
	parameters.push_back(0.0); 							   // Initial azimuthal angle (radians) (-pi-pi)
	parameters.push_back(0.98); 							// Infall rate factor (0-1)
	parameters.push_back(0.99); 							// Subkeplerian factor (0-1)
	parameters.push_back(0.0);                //Position angle (which doesn't impact the intrinsic image).

  //Set image resolution
  double Nrays = 100;
  movie.set_image_resolution(Nrays);

	//Set screen size
	movie.set_screen_size(12);
  
  //Generate model movie
	movie.generate_model(parameters);

  // Returns the movie (usually not required) and output it to a file in the 
  // standard pmap format --  a rastered array of ASCII values with the limits
  // and dimensions as the top. This may be plotted with the plot_vrt2_image.py
  // python script with the results shown.
  std::string basename = "shearing_spot_frame";
  for ( size_t i = 0; i < observing_times.size(); ++i)
  {
    //Access the movie frame data
    std::vector< std::vector<double>> I, alpha, beta;
    movie.get_movie_frame(observing_times[i], alpha, beta, I);
      
    //Converts from radians to GM/c^2 just for convenience.
    double rad2M = VRT2::VRT2_Constants::D_SgrA_cm/VRT2::VRT2_Constants::M_SgrA_cm;

    //Create frame name
    std::stringstream movie_name;
    movie_name << basename << i << ".d";
    std::ofstream movout(movie_name.str().c_str());
      
    movout << "Nx:"
      << std::setw(15) << rad2M*alpha[0][0]
      << std::setw(15) << rad2M*alpha[alpha.size()-1].back()
      << std::setw(15) << alpha.size() << std::endl;
	  movout << "Ny:"
      << std::setw(15) << rad2M*beta[0][0]
      << std::setw(15) << rad2M*beta[alpha.size()-1].back()
      << std::setw(15) << beta.size() << std::endl;
	  movout << std::setw(5) << "i"
	    << std::setw(5) << "j"
	    << std::setw(15) << "I (Jy/sr)"
	    << std::endl;
    for (size_t j=0; j<alpha[0].size(); ++j)
      for (size_t i=0; i<alpha.size(); ++i)
        movout << std::setw(5) << i
	             << std::setw(5) << j
	             << std::setw(15) << I[i][j]
	             << std::endl;
    movout.close();
  }
  

  //Finalize MPI
  MPI_Finalize();

 
  \endcode
*/

