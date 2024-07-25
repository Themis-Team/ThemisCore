/*!
  \file analyses/Challenge3/challenge3b_VACP_comparison_mag.cpp
  \author Hung-Yi Pu, Avery Broderick 
  \date Oct 2018
  \brief Example of fitting a multi-Asymmetric Gaussian image model to visibility amplitude and closure phase data according to the specifications of modeling challenge 2.

  \details simply modified from Avery's .cpp for Chanllenge2
*/

#include "data_visibility_amplitude.h"
#include "model_image_multi_asymmetric_gaussian.h"
#include "likelihood.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
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
  //Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/Challenge3/challenge03a_vtable.d"),"HH");
  //Themis::data_closure_phase dCP(Themis::utils::global_path("sim_data/Challenge3/challenge03a_btable.wo_trivial.d"));
  Themis::data_visibility_amplitude dVM(Themis::utils::global_path("sim_data/Challenge3/challenge03b_vtable.d"),"HH");
  Themis::data_closure_phase dCP(Themis::utils::global_path("sim_data/Challenge3/challenge03b_btable.wo_trivial.d"));

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
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Finished printing data:" << std::endl;

  // Create model
  size_t NG=6;
  if (argc>1) 
    NG = std::atoi(argv[1]);

  if (world_rank==0)
    std::cerr << "Assuming " << NG << " Gaussian components\n";

  Themis::model_image_multi_asymmetric_gaussian image(NG);

  
  // Container of base prior class pointers
  //double uas2rad=4.848136811e-12;
  double image_scale = 2e2*1e-6 / 3600. /180. * M_PI;
  std::vector<Themis::prior_base*> P;

  // First gaussian
  P.push_back(new Themis::prior_linear(0,10)); // I 
  P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
  P.push_back(new Themis::prior_linear(0,0.99)); // A
  P.push_back(new Themis::prior_linear(0,M_PI)); // phi
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // x=0
  P.push_back(new Themis::prior_linear(-1e-6*image_scale,1e-6*image_scale));    // y=0
  // Remainder of gaussians
  for (size_t j=1; j<NG; ++j)
  {
    P.push_back(new Themis::prior_linear(0,10)); // I 
    P.push_back(new Themis::prior_linear(0,image_scale)); // sigma
    P.push_back(new Themis::prior_linear(0,0.99)); // A
    P.push_back(new Themis::prior_linear(0,M_PI)); // phi
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // x
    P.push_back(new Themis::prior_linear(-image_scale,image_scale));    // y
  }
  P.push_back(new Themis::prior_linear(-1e-6,1e-6)); // position angle
  
  
  // Prior means and ranges
  std::vector<double> means, ranges;
  means.push_back(5.); //trick: top-down
  means.push_back(1e-1*image_scale);
  means.push_back(0.2);
  means.push_back(0.25*M_PI);
  means.push_back(0.0);
  means.push_back(0.0);

  ranges.push_back(3.0);
  ranges.push_back(3e-2*image_scale);
  ranges.push_back(0.1);
  ranges.push_back(0.1*M_PI);
  ranges.push_back(1e-7*image_scale);
  ranges.push_back(1e-7*image_scale);

  for (size_t j=1; j<NG; ++j) 
  {
    means.push_back(2.0);
    means.push_back(1e-1*image_scale);
    means.push_back(0.2);
    means.push_back(0.25*M_PI);
    means.push_back(0.0);
    means.push_back(0.0);
    ranges.push_back(0.1);
    ranges.push_back(3e-2*image_scale);
    ranges.push_back(0.1);
    ranges.push_back(0.1*M_PI);
    ranges.push_back(2e-1*image_scale);
    ranges.push_back(2e-1*image_scale);
  }
  
  means.push_back(0.0);
  ranges.push_back(1e-7);


  size_t Ndef = 0;

  // 1G solution
  means[0] = 7.817;
  means[1] = 4.49098e-11;
  means[2] = 0.622452;
  means[3] = 0.77254;
  means[4] = -1.78636e-16;
  means[5] = -7.65515e-17;
  Ndef++;

  // 2G solution, try 1
  means[0] = 5.56802;
  means[1] = 3.51676e-11;
  means[2] = 0.587254;
  means[3] = 0.869971;
  means[4] = -8.95671e-16;
  means[5] = 6.47881e-16;
  means[6] = 2.89676;
  means[7] = 4.81853e-11;
  means[8] = 0.246146;
  means[9] = 1.53584;
  means[10] = 2.19304e-10;
  means[11] = 5.03373e-11;
  Ndef++;

  // 2G solutiont, try 2 but with 0-pi
  means[0] = 5.13844;
  means[1] = 2.41702e-11;
  means[2] = 0.827885;
  means[3] = 1.12705;
  means[4] = -6.80312e-16;
  means[5] = 1.45976e-16;
  means[6] = 2.93072;
  means[7] = 4.32253e-11;
  means[8] = 0.53373;
  means[9] = 1.57077;
  means[10] = 2.20606e-10;
  means[11] = 4.83353e-11;

  // 2G solution, final
  means[0] = 4.9655;
  means[1] = 2.61266e-11;
  means[2] = 0.756647;
  means[3] = 1.13634;
  means[4] = 1.67589e-16;
  means[5] = 6.00321e-18;
  means[6] = 3.04345;
  means[7] = 4.59556e-11;
  means[8] = 0.541129;
  means[9] = 1.90417;
  means[10] = 2.2144e-10;
  means[11] = 4.59933e-11;


  // 3G solution, try 1
  means[0] = 4.98593;
  means[1] = 2.70493e-11;
  means[2] = 0.738988;
  means[3] = 1.10627;
  means[4] = -8.04943e-16;
  means[5] = -7.69812e-16;
  means[6] = 2.93703;
  means[7] = 4.59336e-11;
  means[8] = 0.549757;
  means[9] = 1.92554;
  means[10] = 2.21401e-10;
  means[11] = 4.58918e-11;
  means[12] = 0.181242;
  means[13] = 3.92948e-11;
  means[14] = 0.803039;
  means[15] = 0.0384445;
  means[16] = 6.10899e-10;
  means[17] = -4.78257e-10;
  Ndef++;

  // 3G solution, final
  means[0] = 4.77986;
  means[1] = 2.47368e-11;
  means[2] = 0.775474;
  means[3] = 1.13233;
  means[4] = 1.36006e-16;
  means[5] = 3.51149e-16;
  means[6] = 2.61216;
  means[7] = 4.16388e-11;
  means[8] = 0.607715;
  means[9] = 1.88438;
  means[10] = 2.2175e-10;
  means[11] = 4.61453e-11;
  means[12] = 0.634449;
  means[13] = 5.7562e-11;
  means[14] = 0.645542;
  means[15] = 0.000280761;
  means[16] = 6.12934e-10;
  means[17] = -4.76721e-10;

  // 4G 3G solution
  means[0] = 5.21271;
  means[1] = 2.69828e-11;
  means[2] = 0.732296;
  means[3] = 1.06789;
  means[4] = -2.30201e-16;
  means[5] = 4.32343e-16;
  means[6] = 2.00162;
  means[7] = 3.08947e-11;
  means[8] = 0.646863;
  means[9] = 1.69344;
  means[10] = 2.01166e-10;
  means[11] = 5.59442e-11;
  means[12] = 0.409593;
  means[13] = 3.04657e-11;
  means[14] = 0.718486;
  means[15] = 2.24211;
  means[16] = 3.1882e-10;
  means[17] = -1.58884e-11;

  // 4G Solution, final
  means[0] = 5.06937;
  means[1] = 2.5853e-11;
  means[2] = 0.723268;
  means[3] = 1.12759;
  means[4] = 4.61572e-17;
  means[5] = 5.53541e-17;
  means[6] = 1.64937;
  means[7] = 2.81227e-11;
  means[8] = 0.649571;
  means[9] = 1.541;
  means[10] = 1.99148e-10;
  means[11] = 5.42173e-11;
  means[12] = 0.920504;
  means[13] = 3.82054e-11;
  means[14] = 0.659182;
  means[15] = 2.15372;
  means[16] = 2.8619e-10;
  means[17] = 1.34793e-12;
  means[18] = 0.362965;
  means[19] = 4.62392e-12;
  means[20] = 0.989778;
  means[21] = 1.47644;
  means[22] = -7.20001e-11;
  means[23] = -8.119e-11;
  means[24] = 1.21029e-07;
  Ndef++;

  // 4G C3b intermediate
  means[0] = 4.69494;
  means[1] = 2.48582e-11;
  means[2] = 0.738701;
  means[3] = 1.13579;
  means[4] = -5.95838e-16;
  means[5] = -3.74836e-16;
  means[6] = 1.71772;
  means[7] = 2.97297e-11;
  means[8] = 0.632583;
  means[9] = 1.5823;
  means[10] = 1.99736e-10;
  means[11] = 5.29791e-11;
  means[12] = 0.770955;
  means[13] = 3.76689e-11;
  means[14] = 0.723847;
  means[15] = 2.29983;
  means[16] = 2.88844e-10;
  means[17] = 6.84259e-13;
  means[18] = 0.367894;
  means[19] = 5.02908e-12;
  means[20] = 0.98616;
  means[21] = 1.36096;
  means[22] = -6.8984e-11;
  means[23] = -8.07531e-11;
  means[24] = 5.18184e-07;

  // 4G, C3b final
  means[0] = 4.39965;
  means[1] = 2.30641e-11;
  means[2] = 0.770712;
  means[3] = 1.1396;
  means[4] = 1.51005e-16;
  means[5] = 9.17902e-16;
  means[6] = 1.69974;
  means[7] = 2.95874e-11;
  means[8] = 0.63395;
  means[9] = 1.59112;
  means[10] = 2.0053e-10;
  means[11] = 5.19811e-11;
  means[12] = 0.677421;
  means[13] = 3.7838e-11;
  means[14] = 0.702666;
  means[15] = 2.32705;
  means[16] = 2.94045e-10;
  means[17] = -5.0202e-12;
  means[18] = 0.422642;
  means[19] = 5.99918e-12;
  means[20] = 0.982981;
  means[21] = 1.27523;
  means[22] = -6.32065e-11;
  means[23] = -7.71629e-11;
  means[24] = -7.4821e-07;


  for (size_t j=0; j<Ndef; ++j)
  {
    ranges[6*j+0] = 0.01;
    ranges[6*j+1] = 1e-4*image_scale;
    ranges[6*j+2] = 0.01;
    ranges[6*j+3] = 0.01;
    ranges[6*j+4] = 1e-4*image_scale;
    ranges[6*j+5] = 1e-4*image_scale;
  }




  // vector to hold the name of variables, if the names are provided it would be added 
  // as the header to the chain file 
  std::vector<std::string> var_names;

  // Set the likelihood functions
  // Get the EHT standard station codes
  std::vector<std::string> station_codes = Themis::utils::station_codes();
  std::vector<double> gain_prior_sigmas(station_codes.size(),0.2);

  // Hung-Yi Pu -- This is were the M87-specific stuff begins!
  //std::vector<std::string> station_codes = Themis::utils::station_codes("uvfits 2017");
  //std::vector<double> gain_prior_sigmas(station_codes.size(),0.2);
  //gain_prior_sigmas[4] = 1.0; // LMT is poorly calibrated

  //std::vector<double> gain_prior_sigmas(station_codes.size(),1.0);
  Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(dVM,image,station_codes,gain_prior_sigmas);
  /*
  std::vector<double> tge_explicit(0);
  double dt = 600; //  1 minute intervals
  tge_explicit.push_back(dVM.datum(0).tJ2000-0.5*dt);
  for (;tge_explicit.back()<dVM.datum(dVM.size()-1).tJ2000;)
    tge_explicit.push_back(tge_explicit.back()+dt);
  Themis::likelihood_optimal_gain_correction_visibility_amplitude L_ogva(dVM,image,station_codes,gain_prior_sigmas,tge_explicit);
  */

  std::vector<Themis::likelihood_base*> L;
  //L.push_back(new Themis::likelihood_visibility_amplitude(dVM,image));
  L.push_back(&L_ogva);
  L.push_back(new Themis::likelihood_closure_phase(dCP,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);

  // Create a sampler object, here the PT MCMC
  //Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MC_obj(42+world_rank);


  // Make and output an image
  image.use_numerical_visibilities();
  image.generate_model(means);
  std::vector< std::vector<double> > alpha, beta, I;
  image.get_image(alpha,beta,I);
  if (world_rank==0) 
  {
    std::ofstream imout("initial_image.txt");
    for (size_t i=0; i<alpha.size(); i++) {
      for (size_t j=0; j<alpha[0].size(); j++)
	imout << std::setw(15) << alpha[i][j]
	      << std::setw(15) << beta[i][j]
	      << std::setw(15) << I[i][j]
	      << '\n';
      //imout << '\n';
    }
    imout.close();
  }
  image.use_analytical_visibilities();


  // Get initial chi-squared
  L[0]->operator()(means);
  L[1]->operator()(means);
  std::cout << "Visamps chi-squared: " << L[0]->chi_squared(means) << std::endl;
  std::cout << "CPs chi-squared: " << L[1]->chi_squared(means) << std::endl;

  std::cout << "Visamps red. chi-squared: " << L[0]->chi_squared(means)/(double(dVM.size())-double(L_ogva.number_of_independent_gains())) << std::endl;
  std::cout << "CPs red. chi-squared: " << L[1]->chi_squared(means)/double(dCP.size()) << std::endl;

  std::ofstream cpout("cpcomp.txt");
  double cpc2 = 0.0;
  for (size_t j=0; j<dCP.size(); ++j) {
    cpc2 += (dCP.datum(j).CP*dCP.datum(j).CP)/(dCP.datum(j).err*dCP.datum(j).err);
    cpout << std::setw(15) << std::sqrt( dCP.datum(j).u1*dCP.datum(j).u1 + dCP.datum(j).u2*dCP.datum(j).u2 + dCP.datum(j).u3*dCP.datum(j).u3 + dCP.datum(j).v1*dCP.datum(j).v1 + dCP.datum(j).v2*dCP.datum(j).v2 + dCP.datum(j).v3*dCP.datum(j).v3 )
	      << std::setw(15) << dCP.datum(j).CP
	      << std::setw(15) << dCP.datum(j).err
	      << std::setw(15) << image.closure_phase(dCP.datum(j),0)
	      << std::endl;
  }
  std::cout << "CPs chi-squared directly: " << cpc2 << std::endl;


  L_ogva(means);
  std::cout << "Number of independent gains: " << L_ogva.number_of_independent_gains() << std::endl;
  std::vector<double> tge = L_ogva.get_gain_correction_times();
  std::vector< std::vector<double> > gge = L_ogva.get_gain_corrections();
  if (world_rank==0)
  {
    std::ofstream gout("gain_corrections_o.d");
    for (size_t i=0; i<tge.size()-1; ++i) {
      gout << std::setw(15) << tge[i]-tge[0];
      for (size_t j=0; j<gge[0].size(); ++j)
	gout << std::setw(15) << gge[i][j];
      gout << '\n';
    }
    gout.flush();

    std::ofstream vmout("visamps.d");
    for (size_t j=0, k=0; j<dVM.size(); ++j) {
      if (dVM.datum(j).tJ2000>tge[k])
	k++;
      vmout << std::setw(15) << dVM.datum(j).tJ2000-tge[0]
	    << std::setw(15) << tge[k]-tge[0]
	    << std::setw(5) << k
	    << std::setw(15) << dVM.datum(j).V
	    << std::setw(15) << dVM.datum(j).err
	    << '\n';
    }
    vmout.flush();
  }





#if 1 // Generate a chain for finding best fits

  // Generate a chain
  //int Number_of_chains = 256;
  //int Number_of_temperatures = 8;
  int Number_of_chains = 128;
  int Number_of_temperatures = 8;  
  int Number_of_processors_per_lklhd = 1;
  int Number_of_steps = 1000000;
 // int Number_of_steps = 50000;    
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








  /*
  // Now the best fit 5-Gaussian case
  means[0] = 4.22504;
  means[1] = 4.06463e-11;
  means[2] = 0.330625;
  means[3] = 0.836358;
  means[4] = 7.01158e-17;
  means[5] = 4.68029e-17;
  means[6] = 0.474887;
  means[7] = 7.58175e-11;
  means[8] = 0.00462853;
  means[9] = 0.226757;
  means[10] = -3.46461e-10;
  means[11] = 4.42157e-10;
  means[12] = 1.00075;
  means[13] = 3.87488e-11;
  means[14] = 0.0372935;
  means[15] = 1.57076;
  means[16] = 2.41186e-10;
  means[17] = 3.76241e-11;
  means[18] = 0.756412;
  means[19] = 5.32808e-11;
  means[20] = 0.554713;
  means[21] = 0.699277;
  means[22] = -2.81869e-10;
  means[23] = 4.21134e-10;
  means[24] = 0.524351;
  means[25] = 3.4662e-11;
  means[26] = 0.322998;
  means[27] = 0.461097;
  means[28] = -9.06253e-11;
  means[29] = -5.33623e-11;
  means[30] = 6.0754e-08;

  // Try 2
  means[0] = 3.39497;
  means[1] = 1.10037e-11;
  means[2] = 0.943323;
  means[3] = 1.1894;
  means[4] = 8.64306e-16;
  means[5] = -9.60929e-16;
  means[6] = 0.380402;
  means[7] = 1.09311e-11;
  means[8] = 0.968472;
  means[9] = 0.77765;
  means[10] = -3.25901e-10;
  means[11] = 4.31306e-10;
  means[12] = 2.2243;
  means[13] = 2.59154e-11;
  means[14] = 0.746988;
  means[15] = 1.57075;
  means[16] = 2.23785e-10;
  means[17] = 4.12251e-11;
  means[18] = 0.330818;
  means[19] = 2.84466e-11;
  means[20] = 0.966308;
  means[21] = 0.0823439;
  means[22] = -3.03266e-10;
  means[23] = -4.48434e-10;
  means[24] = 1.6395;
  means[25] = 9.99588e-12;
  means[26] = 0.986237;
  means[27] = 0.876564;
  means[28] = 1.09653e-11;
  means[29] = -3.25405e-11;
  means[30] = -9.50002e-07;

  // Try 3
  means[0] = 3.19883;
  means[1] = 7.60436e-12;
  means[2] = 0.968923;
  means[3] = 1.10548;
  means[4] = -9.34443e-16;
  means[5] = -3.06718e-16;
  means[6] = 0.293681;
  means[7] = 1.21792e-11;
  means[8] = 0.950549;
  means[9] = 0.784645;
  means[10] = -3.32846e-10;
  means[11] = 4.29109e-10;
  means[12] = 2.30894;
  means[13] = 2.96378e-11;
  means[14] = 0.635507;
  means[15] = 1.57076;
  means[16] = 2.23826e-10;
  means[17] = 4.1265e-11;
  means[18] = 0.390425;
  means[19] = 2.06643e-11;
  means[20] = 0.984598;
  means[21] = 0.00964079;
  means[22] = -2.99817e-10;
  means[23] = -4.62313e-10;
  means[24] = 1.87572;
  means[25] = 8.84323e-12;
  means[26] = 0.988737;
  means[27] = 0.954498;
  means[28] = 2.34389e-11;
  means[29] = -2.75833e-11;
  means[30] = 9.90983e-07;

  // 6G try 1
  means[0] = 3.21444;
  means[1] = 8.39597e-12;
  means[2] = 0.962469;
  means[3] = 1.10272;
  means[4] = -8.99232e-16;
  means[5] = -3.20103e-16;
  means[6] = 0.29295;
  means[7] = 1.20737e-11;
  means[8] = 0.951053;
  means[9] = 0.776721;
  means[10] = -3.32973e-10;
  means[11] = 4.29061e-10;
  means[12] = 2.29272;
  means[13] = 2.96428e-11;
  means[14] = 0.635837;
  means[15] = 1.57064;
  means[16] = 2.23786e-10;
  means[17] = 4.12707e-11;
  means[18] = 0.39626;
  means[19] = 2.06896e-11;
  means[20] = 0.987832;
  means[21] = 0.0201238;
  means[22] = -2.99811e-10;
  means[23] = -4.62322e-10;
  means[24] = 1.88197;
  means[25] = 8.90459e-12;
  means[26] = 0.988698;
  means[27] = 0.948002;
  means[28] = 2.34331e-11;
  means[29] = -2.75744e-11;
  means[30] = 0.000447702;
  means[31] = 1.12401e-10;
  means[32] = 0.0308753;
  means[33] = 0.750551;
  means[34] = -7.10732e-11;
  means[35] = 1.1156e-10;
  means[36] = 2.89159e-08;
  */

  /*
  // Refine 6G try 2
  means[0] = 3.2298;
  means[1] = 8.76498e-12;
  means[2] = 0.959291;
  means[3] = 1.10116;
  means[4] = -7.52539e-16;
  means[5] = -2.26883e-16;

  //means[6] = 0.296708;
  //means[7] = 1.19664e-11;
  //means[8] = 0.952705;
  //means[9] = 0.777234;
  //means[10] = -3.3301e-10;
  //means[11] = 4.29168e-10;

  means[6] = 2.29269;
  means[7] = 2.96071e-11;
  means[8] = 0.638372;
  means[9] = 1.57068;
  means[10] = 2.23756e-10;
  means[11] = 4.13473e-11;
  means[12] = 0.395476;
  means[13] = 2.07236e-11;
  means[14] = 0.98856;
  means[15] = 0.0246565;
  means[16] = -2.99775e-10;
  means[17] = -4.62449e-10;
  means[18] = 1.8762;
  means[19] = 8.91556e-12;
  means[20] = 0.98879;
  means[21] = 0.94956;
  means[22] = 2.35335e-11;
  means[23] = -2.75876e-11;

  //means[30] = 0.00311233;
  //means[31] = 1.12406e-10;
  //means[32] = 0.0335097;
  //means[33] = 0.757614;
  //means[34] = -7.10622e-11;
  //means[35] = 1.11786e-10;

  means[36] = -2.57109e-08;

  for (size_t j=1; j<4; ++j)
  {
    ranges[6*j+0] = 0.01;
    ranges[6*j+1] = 1e-4*image_scale;
    ranges[6*j+2] = 0.01;
    ranges[6*j+3] = 0.01;
    ranges[6*j+4] = 1e-4*image_scale;
    ranges[6*j+5] = 1e-4*image_scale;
  }
  




  means[0] = 3.2274;
  means[1] = 7.91406e-12;
  means[2] = 0.970213;
  means[3] = 1.18233;
  means[4] = -8.65519e-16;
  means[5] = 7.22766e-16;
  means[6] = 2.44172;
  means[7] = 3.07843e-11;
  means[8] = 0.637724;
  means[9] = 1.57025;
  means[10] = 2.23925e-10;
  means[11] = 4.22015e-11;
  means[12] = 0.291847;
  means[13] = 2.03046e-11;
  means[14] = 0.984109;
  means[15] = 0.00388041;
  means[16] = -3.0745e-10;
  means[17] = -4.57457e-10;
  means[18] = 1.90232;
  means[19] = 1.05543e-11;
  means[20] = 0.981773;
  means[21] = 0.958346;
  means[22] = 2.12402e-11;
  means[23] = -2.78275e-11;
  means[24] = 0.0207601;
  means[25] = 9.40454e-10;
  means[26] = 0.101012;
  means[27] = 0.285953;
  means[28] = -8.85899e-10;
  means[29] = -1.38898e-10;
  means[30] = 0.170434;
  means[31] = 3.57664e-11;
  means[32] = 0.296352;
  means[33] = 1.17741;
  means[34] = 6.30334e-10;
  means[35] = -2.10744e-10;
  means[36] = 5.3188e-07;
  */

#if 0
  // Use Dom's results
  double FWHM2sig = uas2rad / std::sqrt(8.0*std::log(2.0));
  means[0] = 0.03;
  means[1] = 0.39*FWHM2sig;
  means[2] = 32.95*FWHM2sig;
  means[3] = 14.81*M_PI/180;
  means[4] = 0.0;
  means[5] = 0.0;
  
  means[6] = 0.40;
  means[7] = 15.72*FWHM2sig;
  means[8] = 0.30*FWHM2sig;
  means[9] = 0.11*M_PI/180;
  means[10] = 95.42*uas2rad+means[4];
  means[11] = 48.82*uas2rad+means[5];

  means[12] = 4.79;
  means[13] = 9.52*FWHM2sig;
  means[14] = 21.73*FWHM2sig;
  means[15] = 65.74*M_PI/180;
  means[16] = 15.01*uas2rad+means[10];
  means[17] = -15.45*uas2rad+means[11];

  means[18] = 1.28;
  means[19] = 10.39*FWHM2sig;
  means[20] = 31.54*FWHM2sig;
  means[21] = 90.0*M_PI/180;
  means[22] = 35.94*uas2rad+means[16];
  means[23] = -10.16*uas2rad+means[17];

  means[24] = 0.44;
  means[25] = 8.75*FWHM2sig;
  means[26] = 0.28*FWHM2sig;
  means[27] = 44.91*M_PI/180;
  means[28] = 9.17*uas2rad+means[22];
  means[29] = -3.61*uas2rad+means[23];

  means[30] = 1.06;
  means[31] = 37.60*FWHM2sig;
  means[32] = 15.98*FWHM2sig;
  means[33] = 25.71*M_PI/180;
  means[34] = 11.74*uas2rad+means[28];
  means[35] = 12.04*uas2rad+means[29];

  means[36] = 0.0;


  for (size_t j=0; j<NG; ++j)
  {
    double sigM = means[j*6+1];
    double sigm = means[j*6+2];
    double dphi = 0.0;
    if (sigm>sigM) {
      sigM = means[j*6+2];
      sigm = means[j*6+1];
      dphi = 0.5*M_PI;
    }
    double sig = std::sqrt(2.0)*sigm*sigM/std::sqrt(sigm*sigm+sigM*sigM);
    double A = (sigM*sigM-sigm*sigm)/(sigM*sigM+sigm*sigm);
    double phi = -means[j*6+3] + dphi;

    means[j*6+1] = sig;
    means[j*6+2] = A;
    means[j*6+3] = phi+0.5*M_PI;
  }

  for (size_t i=0; i<image.size(); ++i)
    std::cout << "means[" << i << "] = " << means[i] << std::endl;


#endif


#if 0
  // Use Dom's 1G results
  double FWHM2sig = uas2rad / std::sqrt(8.0*std::log(2.0));
  means[0] = 7.81;
  means[1] = 17.10*FWHM2sig;
  means[2] = 35.49*FWHM2sig;
  means[3] = 44.20*M_PI/180;
  means[4] = 0.0;
  means[5] = 0.0;

  means[6] = 0.0;  

  for (size_t j=0; j<NG; ++j)
  {
    double sigM = means[j*6+1];
    double sigm = means[j*6+2];
    double dphi = 0.0;
    if (sigm>sigM) {
      sigM = means[j*6+2];
      sigm = means[j*6+1];
      dphi = 0.5*M_PI;
    }
    double sig = std::sqrt(2.0)*sigm*sigM/std::sqrt(sigm*sigm+sigM*sigM);
    double A = (sigM*sigM-sigm*sigm)/(sigM*sigM+sigm*sigm);
    double phi = means[j*6+3] + dphi;

    means[j*6+1] = sig;
    means[j*6+2] = A;
    means[j*6+3] = phi;//+0.5*M_PI;
  }

  for (size_t i=0; i<image.size(); ++i)
    std::cout << "means[" << i << "] = " << means[i] << std::endl;

#endif

#if 0
  // Use Dom's 2G results
  double FWHM2sig = uas2rad / std::sqrt(8.0*std::log(2.0));
  means[0] = 5.13;
  means[1] = 10.24*FWHM2sig;
  means[2] = 24.65*FWHM2sig;
  means[3] = 63.96*M_PI/180;
  means[4] = 0.0;
  means[5] = 0.0;

  means[6] = 2.87;
  means[7] = 30.85*FWHM2sig;
  means[8] = 15.58*FWHM2sig;
  means[9] = 44.20*M_PI/180;
  means[10] = 42.21*uas2rad+means[4];
  means[11] = 16.67*uas2rad+means[5];
  
  means[12] = 0.0;

  for (size_t j=0; j<NG; ++j)
  {
    double sigM = means[j*6+1];
    double sigm = means[j*6+2];
    double dphi = 0.0;
    if (sigm>sigM) {
      sigM = means[j*6+2];
      sigm = means[j*6+1];
      dphi = 0.5*M_PI;
    }
    double sig = std::sqrt(2.0)*sigm*sigM/std::sqrt(sigm*sigm+sigM*sigM);
    double A = (sigM*sigM-sigm*sigm)/(sigM*sigM+sigm*sigm);
    double phi = means[j*6+3] + dphi;

    means[j*6+1] = sig;
    means[j*6+2] = A;
    means[j*6+3] = phi+0.5*M_PI;
  }

  for (size_t i=0; i<image.size(); ++i)
    std::cout << "means[" << i << "] = " << means[i] << std::endl;

#endif


#if 0
  // 12G try1
  means[0] = 3.21512;
  means[1] = 1.31813e-11;
  means[2] = 0.918784;
  means[3] = 1.21603;
  means[4] = -8.9068e-16;
  means[5] = 9.09731e-16;
  means[6] = 2.44173;
  means[7] = 3.07849e-11;
  means[8] = 0.637705;
  means[9] = 1.57023;
  means[10] = 2.23926e-10;
  means[11] = 4.22009e-11;
  means[12] = 0.291841;
  means[13] = 2.03047e-11;
  means[14] = 0.984055;
  means[15] = 0.00394965;
  means[16] = -3.07451e-10;
  means[17] = -4.57457e-10;
  means[18] = 1.9023;
  means[19] = 1.05539e-11;
  means[20] = 0.981721;
  means[21] = 0.958321;
  means[22] = 2.12403e-11;
  means[23] = -2.78279e-11;
  means[24] = 0.0207329;
  means[25] = 9.40454e-10;
  means[26] = 0.101001;
  means[27] = 0.286015;
  means[28] = -8.85899e-10;
  means[29] = -1.38899e-10;
  means[30] = 0.170388;
  means[31] = 3.57664e-11;
  means[32] = 0.296341;
  means[33] = 1.17741;
  means[34] = 6.30335e-10;
  means[35] = -2.10744e-10;
  means[36] = 0.0113661;
  means[37] = 3.85679e-11;
  means[38] = 0.399246;
  means[39] = 0.0192775;
  means[40] = 5.23203e-10;
  means[41] = -5.90841e-10;
  means[42] = 0.0680222;
  means[43] = 1.14227e-10;
  means[44] = 0.763085;
  means[45] = 1.50633;
  means[46] = 7.59751e-10;
  means[47] = 5.78949e-10;
  means[48] = 0.0481036;
  means[49] = 1.71858e-10;
  means[50] = 0.693767;
  means[51] = 0.0460562;
  means[52] = -1.11443e-10;
  means[53] = -2.08199e-10;
  means[54] = 0.0135727;
  means[55] = 1.85762e-10;
  means[56] = 0.460239;
  means[57] = 0.259406;
  means[58] = 6.48188e-10;
  means[59] = 2.84516e-10;
  means[60] = 0.0166978;
  means[61] = 3.76392e-10;
  means[62] = 0.260772;
  means[63] = 0.591623;
  means[64] = -9.19735e-11;
  means[65] = -9.55013e-10;
  means[66] = 0.0619026;
  means[67] = 2.78709e-10;
  means[68] = 0.136657;
  means[69] = 0.796222;
  means[70] = 9.36667e-11;
  means[71] = -2.39448e-10;
  means[72] = -6.19411e-07;

  means[0] = 3.23697;
  means[1] = 1.20587e-11;
  means[2] = 0.931952;
  means[3] = 1.21206;
  means[4] = -9.29574e-16;
  means[5] = 8.34847e-16;
  means[6] = 2.44177;
  means[7] = 3.07852e-11;
  means[8] = 0.637725;
  means[9] = 1.57021;
  means[10] = 2.23926e-10;
  means[11] = 4.2201e-11;
  means[12] = 0.291836;
  means[13] = 2.03046e-11;
  means[14] = 0.984066;
  means[15] = 0.00396807;
  means[16] = -3.0745e-10;
  means[17] = -4.57457e-10;
  means[18] = 1.90233;
  means[19] = 1.05547e-11;
  means[20] = 0.981723;
  means[21] = 0.958333;
  means[22] = 2.12399e-11;
  means[23] = -2.78276e-11;
  means[24] = 0.0207449;
  means[25] = 9.40454e-10;
  means[26] = 0.101004;
  means[27] = 0.28604;
  means[28] = -8.85899e-10;
  means[29] = -1.38898e-10;
  means[30] = 0.17041;
  means[31] = 3.57666e-11;
  means[32] = 0.296336;
  means[33] = 1.17739;
  means[34] = 6.30335e-10;
  means[35] = -2.10744e-10;
  means[36] = 0.0949092;
  means[37] = 6.7307e-11;
  means[38] = 0.250815;
  means[39] = 0.419543;
  means[40] = 5.211e-10;
  means[41] = -3.35474e-10;
  means[42] = 0.0191964;
  means[43] = 9.58517e-11;
  means[44] = 0.989497;
  means[45] = 1.56297;
  means[46] = 8.58355e-10;
  means[47] = 3.32013e-10;
  means[48] = 0.0521858;
  means[49] = 4.33562e-11;
  means[50] = 0.50697;
  means[51] = 0.0551677;
  means[52] = -4.11826e-10;
  means[53] = -4.21827e-10;
  means[54] = 0.0266608;
  means[55] = 1.29106e-10;
  means[56] = 0.786102;
  means[57] = 1.00924;
  means[58] = 8.34113e-10;
  means[59] = -3.93012e-11;
  means[60] = 0.00398977;
  means[61] = 4.31641e-10;
  means[62] = 0.0811841;
  means[63] = 0.806719;
  means[64] = -4.53488e-10;
  means[65] = -6.24886e-10;
  means[66] = 0.00141608;
  means[67] = 3.27103e-10;
  means[68] = 0.0249592;
  means[69] = 0.967936;
  means[70] = -3.95174e-11;
  means[71] = -5.09067e-10;
  means[72] = -4.74062e-07;

#endif

  /*
  for (size_t j=1; j<NG; ++j)
  {
    ranges[6*j+0] = 1e-5;
    ranges[6*j+1] = 1e-7*image_scale;
    ranges[6*j+2] = 1e-5;
    ranges[6*j+3] = 1e-5;
    ranges[6*j+4] = 1e-7*image_scale;
    ranges[6*j+5] = 1e-7*image_scale;
  }
  */

