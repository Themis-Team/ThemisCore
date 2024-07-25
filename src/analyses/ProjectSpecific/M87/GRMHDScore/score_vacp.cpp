/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Hung-Yi Pu
    \date  Nov, 2018
    \brief test model_image_score clas by mcmc runs *without* gain calibration
    \details 
*/

#include "data_visibility_amplitude.h"
#include "model_image_score.h"
#include "model_ensemble_averaged_scattered_image.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  //Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;


   // Read in data
   Themis::data_visibility_amplitude VM(Themis::utils::global_path("sim_data/Challenge4/ch4_VA_im1.d"));
   Themis::data_closure_phase CP(Themis::utils::global_path("sim_data/Challenge4/ch4_CP_im1.d"));


  //===useful constants
  double uas2rad= 1e-6/ 3600. * (M_PI/180.);
  double M_sun =1.99e+33;//g
  double D_pc  =3.086e18;//cm  
  
  //==image information	
  //!!!!note that the image file is specified in model_image_score.cpp!!!
  int    Nray  =160;          // resolution of the image
  double frequency = 230.e+9; // GHz
  double fov = 160.;          // total size of the image (uas)
  double M = 6.2e+9*M_sun;    //black hole mass used for post-processing
  double D = 16.9e+6*D_pc;    //distance used for post-processing
  


  // Choose the model to compare
  Themis::model_image_score image(Nray, M, D, fov, frequency);  
  
  
  
  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  std::vector<double> means, ranges;
  
  
//    Parameter list:\n
//      - parameters[0] ...  V_total
//	- parameters[1] ... (M/R)/ (M0/R0)
//      - parameters[2] ... PA
  
  P.push_back(new Themis::prior_linear(0.3,5.)); // V_total
  means.push_back(1.);  
  ranges.push_back(0.5);
  
  P.push_back(new Themis::prior_linear(0.3,2.)); // (M/R)/ (M0/R0)
  means.push_back(1.);  
  ranges.push_back(0.3);
  
  
  P.push_back(new Themis::prior_linear(-M_PI,M_PI)); // position angle
  means.push_back(10./180.0*M_PI);
  ranges.push_back(0.1);


  // vector to hold the name of variables, if the names are provided it would be added
  // as the header to the chain file
  std::vector<std::string> var_names;
  var_names.push_back("V");
  var_names.push_back("M_ratio");
  var_names.push_back("Position_Angle");

  // Set the variable transformations
  std::vector<Themis::transform_base*> T;
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());
  T.push_back(new Themis::transform_none());



  // Set the likelihood functions
 
  // Set the likelihood functions
     std::vector<Themis::likelihood_base*> L;
       L.push_back(new Themis::likelihood_visibility_amplitude(VM,image));
      
       //Closure Phases
       L.push_back(new Themis::likelihood_closure_phase(CP,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);


  // Make a likelihood object (T and W is not used here)
  Themis::likelihood L_obj(P, T, L, W);


  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);

  // Generate a chain
  int Number_of_chains = 8;
  int Number_of_temperatures = 3;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 10000;
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int out_precision = 8;
  // output all files (Chain, Lklhd, Chi2) for each temperature, if verbosity=1 
  int verbosity = 0;

 
  // if want to start from ckpt, change the file name and turn "false" to "true"
  MC_obj.set_checkpoint(100,"ScoreVA.ckpt");
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
  MC_obj.run_sampler(L_obj,
                      Number_of_steps, Temperature_stride, Chi2_stride, 
                      "Chain-score.dat", "Lklhd-score.dat", 
		     "Chi2-score.dat", means, ranges, var_names, false, out_precision,verbosity); 
              

  //Finalize MPI
  MPI_Finalize();
  return 0;
}
