/*!
  \file analyses/Challenge3/challenge3a_VACPCA_comparison.cpp
  \author Hung-Yi Pu 
  \date Mar 2018
  \brief Example of fitting a multi-Gaussian image model to visibility amplitude and closure phase data according to the specifications of modeling challenge 2.

  \details simply modified from Avery's .cpp for Chanllenge2
*/

#include "data_visibility_amplitude.h"
#include "model_image_multigaussian.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include "random_number_generator.h"

/*! \cond */
#include <mpi.h>
#include <memory> 
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
/*! \endcond */

int main(int argc, char* argv[])
{
  // Initialize MPI
  int world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in challenge data
  // Raw data
  //Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/Challenge2/Ch2_VMs.d"));
  //Themis::data_closure_phase dCP(Themis::utils::global_path("sim_data/Challenge2/Ch2_CPs.d"));
  // SNR-limited data
  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/Challenge3/challenge03a_vtable.d"));
  Themis::data_closure_phase dCP(Themis::utils::global_path("sim_data/Challenge3/challenge03a_btable.wo_trivial.d"));
  Themis::data_closure_amplitude dCA(Themis::utils::global_path("sim_data/Challenge3/challenge03a_ctable.wo_trivial.d"));  

  std::cout << "Printing data:" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank==0)
    for (size_t j=0; j<dVM.size(); ++j) 
      std::cout << "VMdata:"
		<< std::setw(15) << dVM.datum(j).u
		<< std::setw(15) << dVM.datum(j).v
		<< std::setw(15) << dVM.datum(j).V
		<< std::setw(15) << dVM.datum(j).err
		<< std::endl;
  if (world_rank==0)
    for (size_t j=0; j<dCP.size(); ++j) 
      std::cout << "CPdata:"
		<< std::setw(15) << dCP.datum(j).u1
		<< std::setw(15) << dCP.datum(j).v1
		<< std::setw(15) << dCP.datum(j).u2
		<< std::setw(15) << dCP.datum(j).v2
		<< std::setw(15) << dCP.datum(j).CP
		<< std::setw(15) << dCP.datum(j).err
		<< std::endl;
  if (world_rank==0)
    for (size_t j=0; j<dCA.size(); ++j) 
      std::cout << "CAdata:"
		<< std::setw(15) << dCA.datum(j).u1
		<< std::setw(15) << dCA.datum(j).v1
		<< std::setw(15) << dCA.datum(j).u2
		<< std::setw(15) << dCA.datum(j).v2
		<< std::setw(15) << dCA.datum(j).u3
		<< std::setw(15) << dCA.datum(j).v3   
		<< std::setw(15) << dCA.datum(j).CA
		<< std::setw(15) << dCA.datum(j).err
		<< std::endl;   
   
   
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Finished printing data:" << std::endl;

  // Create model
  size_t NG=7;
  if (argc>1) 
    NG = std::atoi(argv[1]);

  if (world_rank==0)
    std::cerr << "Assuming " << NG << " Gaussian components\n";

  Themis::model_image_multigaussian image(NG);


  
  // Container of base prior class pointers
  double u2rad=4.848136811e-12;
  double image_scale = 2e2*1e-6 / 3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;
  // First gaussian
  P.push_back(new Themis::prior_linear(0,10)); // I 
  P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // x=0
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // y=0
  // Remainder of gaussians
  for (size_t j=1; j<NG; ++j)
  {
    P.push_back(new Themis::prior_linear(0,10)); // I 
    P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // x
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // y
  }
  P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // position angle
  
  
  // Prior means and ranges
  std::vector<double> means, ranges;
  means.push_back(5.); //trick: top-down
  means.push_back(1e-1*image_scale);
  means.push_back(0.0);
  means.push_back(0.0);
  ranges.push_back(1e-4);
  //ranges.push_back(0.001);  
  ranges.push_back(1e-4*image_scale);
  ranges.push_back(1e-7*image_scale);
  ranges.push_back(1e-7*image_scale);
  /*
  for (size_t j=1; j<NG; ++j) 
  {
    means.push_back(0.1);
    means.push_back(1e-1*image_scale);
    means.push_back(0.0);
    means.push_back(0.0);
    ranges.push_back(0.01);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
  }
  */
  double cnt=0.;
  for (size_t j=1; j<NG; ++j) 
  {
    cnt=cnt+1.;
    means.push_back(1.);
    means.push_back(1e-1*image_scale);
    //======= initial location
    means.push_back(15.*cnt*u2rad);
    means.push_back(0.0);
    //means.push_back(15.*cnt*u2rad);
        
    //====== initial location at center
    //means.push_back(0.0);
    //means.push_back(0.0);
    
    ranges.push_back(0.01);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
    ranges.push_back(1e-4*image_scale);
  }  
  means.push_back(0.0);
  ranges.push_back(1e-7);

    /*
  if (NG==1) {
    // Best 1 Gaussian based on prior fits
    means[0] = 3.76;
    means[1] = 3.429e-11;
    means[2] = 1.45022e-15;
    means[3] = -1.00068e-15;
    means[4] = 1.49219e-08;
  }
  

  if (NG==2) {
    // Best 2 Gaussians based on prior fits
    means[0] = 3.79;
    means[1] = 3.429e-11;
    means[2] = -2.649e-17;
    means[3] = -4.139e-18;
    means[4] = 3.42;
    means[5] = 5.458e-11;
    means[6] = -1.25e-10;
    means[7] = -2.86e-10;
    means[8] = -3.817e-08;
  }

  if (NG==3) {
    // Best 3 Gaussians based on prior fits
    means[0] = 3.79;
    means[1] = 3.429e-11;
    means[2] = -2.649e-17;
    means[3] = -4.139e-18;
    means[4] = 3.42;
    means[5] = 5.458e-11;
    means[6] = -1.25e-10;
    means[7] = -2.86e-10;
    //means[8] = -3.817e-08;   
    

    means[0] = 1.31716;
    means[1] = 8.30096e-11;
    means[2] = -3.53075e-16;
    means[3] = 1.54491e-16;
    means[4] = 0.798853;
    means[5] = 9.45957e-11;
    means[6] = -1.31409e-10;
    means[7] = 1.47112e-10;
    means[8] = 0.504064;
    means[9] = 1.69079e-10;
    means[10] = -1.04476e-10;
    means[11] = -9.33128e-10;
    means[12] = 2.02454e-07;
    
  }
  */

/*
  if (NG==4) {
    // Best 4 Gaussians based on prior fits
    means[0] = 1.18908;
    means[1] = 8.19623e-11;
    means[2] = 7.78055e-16;
    means[3] = 8.66599e-16;
    means[4] = 0.731208;
    means[5] = 9.28223e-11;
    means[6] = -1.18199e-10;
    means[7] = 1.33615e-10;
    means[8] = 0.241289;
    means[9] = 1.50666e-10;
    means[10] = -5.53723e-10;
    means[11] = 7.15583e-10;
    means[12] = 0.458269;
    means[13] = 1.27292e-10;
    means[14] = -2.75352e-10;
    means[15] = 2.15684e-10;
    means[16] = 1.30882e-07;
  }

  if (NG==5) {
    // Best 5 Gaussians based on prior fits
    means[0] = 1.1854;
    means[1] = 8.19703e-11;
    means[2] = -7.48331e-17;
    means[3] = -1.46241e-16;
    means[4] = 0.749005;
    means[5] = 9.34746e-11;
    means[6] = -1.18598e-10;
    means[7] = 1.33098e-10;
    means[8] = 0.240958;
    means[9] = 1.48486e-10;
    means[10] = -5.50283e-10;
    means[11] = 7.13491e-10;
    means[12] = 0.443694;
    means[13] = 1.27129e-10;
    means[14] = -2.80957e-10;
    means[15] = 2.17804e-10;
    means[16] = 0.000796241;
    means[17] = 3.64976e-12;
    means[18] = -2.28763e-10;
    means[19] = 1.02772e-09;
    means[20] = -1.86576e-08;
  }

  if (NG==6) {
    // Best 6 Gaussians based on prior fits
    means[0] = 1.18001;
    means[1] = 8.1934e-11;
    means[2] = 1.44645e-15;
    means[3] = 1.85493e-15;
    means[4] = 0.725169;
    means[5] = 9.28731e-11;
    means[6] = -1.17014e-10;
    means[7] = 1.31961e-10;
    means[8] = 0.236977;
    means[9] = 1.49743e-10;
    means[10] = -5.57215e-10;
    means[11] = 7.19607e-10;
    means[12] = 0.476488;
    means[13] = 1.27274e-10;
    means[14] = -2.7253e-10;
    means[15] = 2.16866e-10;
    means[16] = 0.000797979;
    means[17] = 5.38368e-13;
    means[18] = -2.27256e-10;
    means[19] = 1.03895e-09;
    means[20] = 0.000554056;
    means[21] = 3.16771e-12;
    means[22] = -5.59751e-10;
    means[23] = 1.91748e-09;
    means[24] = 6.61147e-07;
  }

  if (NG>=7) {
    // Best 7 Gaussians based on prior fits
    means[0] = 1.17873;
    means[1] = 8.1852e-11;
    means[2] = 1.55724e-15;
    means[3] = 7.67429e-16;
    means[4] = 0.733652;
    means[5] = 9.30147e-11;
    means[6] = -1.16971e-10;
    means[7] = 1.32525e-10;
    means[8] = 0.242844;
    means[9] = 1.52169e-10;
    means[10] = -5.66346e-10;
    means[11] = 7.19076e-10;
    means[12] = 0.459886;
    means[13] = 1.2693e-10;
    means[14] = -2.73378e-10;
    means[15] = 2.14639e-10;
    means[16] = 0.000678341;
    means[17] = 6.42594e-12;
    means[18] = -2.27229e-10;
    means[19] = 1.04484e-09;
    means[20] = 0.000708645;
    means[21] = 1.61513e-11;
    means[22] = -5.65326e-10;
    means[23] = 1.90179e-09;
    means[24] = 0.00347352;
    means[25] = 8.00371e-11;
    means[26] = 1.18064e-09;
    means[27] = -3.12968e-10;
    means[28] = -9.36893e-07;
  }
*/

  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;
   /*
   var_names.push_back("I1");
   var_names.push_back("sigma1");
   var_names.push_back("x1");
   var_names.push_back("y1");
   var_names.push_back("I2");
   var_names.push_back("sigma2");
   var_names.push_back("x2");
   var_names.push_back("y2");
   var_names.push_back("I3");
   var_names.push_back("sigma3");
   var_names.push_back("x3");
   var_names.push_back("y3");
   var_names.push_back("P.A.");
   */       
  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  L.push_back(new Themis::likelihood_visibility_amplitude(dVM,image));
  L.push_back(new Themis::likelihood_closure_phase(dCP,image));

  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Create a sampler object, here the PT MCMC
  Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);


#if 1 // Generate a chain for finding best fits

  // Generate a chain
  //int Number_of_chains = 256;
  //int Number_of_temperatures = 8;
  int Number_of_chains = 128;
  int Number_of_temperatures = 8;  
  int Number_of_processors_per_lklhd = 1;
  //int Number_of_steps = 1000000;
  int Number_of_steps = 30000;    
  int Temperature_stride = 50;
  int Chi2_stride = 20;

  // Set the CPU distribution
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);

  // Run the Sampler                            
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-NGtest.dat", "Lklhd-NGtest.dat", 
		     "Chi2-NGtest.dat", means, ranges, var_names, false);


#else // Compute evidence

  // Generate a chain
  int Number_of_chains = 128;
  int Number_of_temperatures = 128;
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 1500; 
  int Temperature_stride = 50;
  int Chi2_stride = 20;
  int verbosity = 1;
  int burn_in = 1000;
  std::vector<double> temperatures(Number_of_temperatures);
  std::vector<std::string> likelihood_files(Number_of_temperatures);

  for(int i = 0; i < Number_of_temperatures; ++i)
    {
      temperatures[i] = 1.0*pow(1.1,i);
      likelihood_files[i] = "Lklhd-NGtest.dat"+std::to_string(i);
    }
  likelihood_files[0] = "Lklhd-NGtest.dat";



  // Set the CPU distribution
  MC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_processors_per_lklhd);
  


  // Run the Sampler                            
  MC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
		     "Chain-NGtest.dat", "Lklhd-NGtest.dat", 
		     "Chi2-NGtest.dat", means, ranges, var_names, false, verbosity, false, temperatures);


  MC_obj.estimate_bayesian_evidence(likelihood_files, temperatures, burn_in);
#endif

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
