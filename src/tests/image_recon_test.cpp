/*!
  \file examples/scattered_gaussian_fitting.cpp
  \author
  \date Apr 2017
  \brief Example of fitting a Gaussian image model to visibility amplitude data.

  \details This example illustrates how to generate a Gaussian image
  model, include scattering to the intrinsic model image, read-in
  visibility amplitude data (as also shown in reading_data.cpp), and
  fitting the model to the eht data. The model can take full advantage
  of the analytically known visibility amplitudes thus making this
  test really fast.
*/

#include "data_visibility_amplitude.h"
#include "data_closure_phase.h"
#include "model_image_raster.h"
#include "likelihood.h"
#include "sampler_affine_invariant_tempered_MCMC.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory> 
#include <string>

#include <iostream>
#include <iomanip>
#include <fstream>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  std::cout << "MPI Initiated - Processor Node: " << world_rank << " executing main." << std::endl;

  // Read in visibility amplitude test data
  //Themis::data_visibility_amplitude VM_data(Themis::utils::global_path("sim_data/ImageReconTest/example_themis_visamp.d"),"HH");
  Themis::data_visibility_amplitude VM_data(Themis::utils::global_path("sim_data/ImageReconTest/VM_test.d"),"HH");
  // Read in closure phase test data
  //Themis::data_closure_phase CP_data(Themis::utils::global_path("sim_data/ImageReconTest/example_themis_cphase.d"),"HH");
  Themis::data_closure_phase CP_data(Themis::utils::global_path("sim_data/ImageReconTest/CP_test.d"),"HH");


  if (world_rank==0)
  {
    std::cout << "\n\n";
    for (size_t j=0; j<10; ++j)
      std::cout << std::setw(15) << VM_data.datum(j).u
		<< std::setw(15) << VM_data.datum(j).v
		<< std::setw(15) << VM_data.datum(j).V
		<< std::setw(15) << VM_data.datum(j).err
		<< std::endl;
    std::cout << "\n\n";
    for (size_t j=0; j<10; ++j)
      std::cout << std::setw(15) << CP_data.datum(j).u1
		<< std::setw(15) << CP_data.datum(j).v1
		<< std::setw(15) << CP_data.datum(j).u2
		<< std::setw(15) << CP_data.datum(j).v2
		<< std::setw(15) << CP_data.datum(j).CP
		<< std::setw(15) << CP_data.datum(j).err
		<< std::endl;
    std::cout << "\n\n";
  }
  
  // Choose raster image model
  double image_scale = 50e-6 / 3600. /180. * M_PI;
  size_t N=8;
  Themis::model_image_raster image(-image_scale,image_scale,N,-image_scale,image_scale,N);



  std::cout << "\n\n";
  
  std::vector<double> params(image.size());
  double x,y;
  double sig = 0.17/0.5*image_scale;
  double norm = std::log(2.5/(2.0*M_PI*sig*sig));
  for (size_t j=0,k=0; j<N; ++j)
    for (size_t i=0; i<N; ++i)
    {
      x = 2.0*image_scale*double(i)/double(N-1) - image_scale;
      y = 2.0*image_scale*double(j)/double(N-1) - image_scale;
      //if (j>0 || (i>0 && i<N-1))
      {
	params[k++] = norm  - (x*x+y*y)/(2.0*sig*sig);

	if (world_rank==0)
	  std::cout << std::setw(15) << x
		    << std::setw(15) << y
		    << std::setw(15) << std::exp(params[k-1])
		    << std::setw(15) << params[k-1]
		    << std::endl;
      }
    }

  image.generate_model(params);

  if (world_rank==0)
  {
  std::cout << "\n\n";

  std::vector< std::vector<double> > alpha,beta,I;
  image.get_image(alpha,beta,I);
  for (size_t i=0; i<alpha.size(); ++i)
    for (size_t j=0; j<alpha[i].size(); ++j)
      std::cout << std::setw(15) << alpha[i][j]
		<< std::setw(15) << beta[i][j]
		<< std::setw(15) << I[i][j]
		<< std::endl;
  



  std::ofstream vmout("vmdata.d");
  for (size_t j=0; j<VM_data.size(); ++j)
    vmout << std::setw(15) << VM_data.datum(j).u
	  << std::setw(15) << VM_data.datum(j).v
	  << std::setw(15) << VM_data.datum(j).V
	  << std::setw(15) << VM_data.datum(j).err
	  << std::setw(15) << image.visibility_amplitude(VM_data.datum(j),0.25*VM_data.datum(j).err)
	  << std::endl;
  }
  
  
  
  
  

  // Container of base prior class pointers
  std::vector<Themis::prior_base*> P;
  for (size_t j=0; j<image.size(); ++j)
    P.push_back(new Themis::prior_linear(-1000,1000)); // Itotal

  std::vector<double> means, ranges;
  std::vector<std::string> var_names;
  for (size_t j=0; j<image.size(); ++j)
  {
    means.push_back(params[j]);
    ranges.push_back(1e-3);
  }


  // Bestfit
  means[0] = 36.050043;
  means[1] = 38.1313;
  means[2] = 42.255916;
  means[3] = 42.630168;
  means[4] = 37.615832;
  means[5] = 38.698061;
  means[6] = 37.21109;
  means[7] = 35.040828;
  means[8] = 40.00294;
  means[9] = 43.977928;
  means[10] = 41.68748;
  means[11] = 41.761671;
  means[12] = 43.020908;
  means[13] = 42.089978;
  means[14] = 42.78998;
  means[15] = 40.034562;
  means[16] = 43.819246;
  means[17] = 44.949236;
  means[18] = 45.475338;
  means[19] = 45.520626;
  means[20] = 44.626365;
  means[21] = 44.213817;
  means[22] = 43.577448;
  means[23] = 42.827452;
  means[24] = 41.110083;
  means[25] = 44.191416;
  means[26] = 45.083425;
  means[27] = 45.827829;
  means[28] = 45.641581;
  means[29] = 44.325596;
  means[30] = 44.501891;
  means[31] = 39.865935;
  means[32] = 43.658938;
  means[33] = 43.100808;
  means[34] = 44.508185;
  means[35] = 45.692789;
  means[36] = 45.915349;
  means[37] = 44.877377;
  means[38] = 44.119206;
  means[39] = 43.421246;
  means[40] = 43.523707;
  means[41] = 43.101254;
  means[42] = 42.178305;
  means[43] = 45.946051;
  means[44] = 45.704421;
  means[45] = 44.417925;
  means[46] = 44.029262;
  means[47] = 40.163296;
  means[48] = 40.032231;
  means[49] = 41.631934;
  means[50] = 44.034272;
  means[51] = 44.946239;
  means[52] = 44.382005;
  means[53] = 41.932641;
  means[54] = 42.824826;
  means[55] = 41.645184;
  means[56] = 36.804701;
  means[57] = 37.739881;
  means[58] = 39.813778;
  means[59] = 40.424688;
  means[60] = 43.242429;
  means[61] = 40.889351;
  means[62] = 39.101258;
  means[63] = 36.425167;

  // bestfit 2.0
  means[0] = 39.177886;
  means[1] = 41.140429;
  means[2] = 41.000209;
  means[3] = 42.195568;
  means[4] = 37.457546;
  means[5] = 36.506714;
  means[6] = 38.834645;
  means[7] = 33.753713;
  means[8] = 41.720458;
  means[9] = 43.213541;
  means[10] = 42.378683;
  means[11] = 39.670117;
  means[12] = 41.75715;
  means[13] = 41.680474;
  means[14] = 40.16861;
  means[15] = 40.977707;
  means[16] = 43.977698;
  means[17] = 44.556677;
  means[18] = 45.296828;
  means[19] = 45.179076;
  means[20] = 44.080167;
  means[21] = 43.222305;
  means[22] = 43.147163;
  means[23] = 42.446549;
  means[24] = 43.311576;
  means[25] = 44.621916;
  means[26] = 44.962515;
  means[27] = 45.885365;
  means[28] = 45.356184;
  means[29] = 43.432199;
  means[30] = 43.840164;
  means[31] = 39.917833;
  means[32] = 43.535937;
  means[33] = 41.53471;
  means[34] = 45.057602;
  means[35] = 45.575924;
  means[36] = 45.81925;
  means[37] = 44.032799;
  means[38] = 44.096072;
  means[39] = 42.776846;
  means[40] = 43.489787;
  means[41] = 42.658099;
  means[42] = 43.756301;
  means[43] = 45.921571;
  means[44] = 45.404824;
  means[45] = 44.175325;
  means[46] = 43.97433;
  means[47] = 42.283816;
  means[48] = 40.68591;
  means[49] = 43.156574;
  means[50] = 44.875967;
  means[51] = 45.038598;
  means[52] = 44.370438;
  means[53] = 43.619775;
  means[54] = 42.977096;
  means[55] = 43.091466;
  means[56] = 37.378832;
  means[57] = 36.960587;
  means[58] = 40.491754;
  means[59] = 38.634717;
  means[60] = 40.294737;
  means[61] = 41.077723;
  means[62] = 43.10041;
  means[63] = 42.054534;


  // Bestfit 3.0
  means[0] = 42.343725;
  means[1] = 42.386704;
  means[2] = 39.699388;
  means[3] = 43.14536;
  means[4] = 39.728453;
  means[5] = 37.791678;
  means[6] = 35.436593;
  means[7] = 32.260004;
  means[8] = 42.684039;
  means[9] = 43.515977;
  means[10] = 43.877036;
  means[11] = 33.763899;
  means[12] = 41.495956;
  means[13] = 41.916711;
  means[14] = 39.442734;
  means[15] = 40.537915;
  means[16] = 44.270871;
  means[17] = 44.994749;
  means[18] = 45.721885;
  means[19] = 45.739604;
  means[20] = 44.615429;
  means[21] = 43.111648;
  means[22] = 42.818373;
  means[23] = 43.037895;
  means[24] = 44.156131;
  means[25] = 45.059855;
  means[26] = 45.436164;
  means[27] = 46.358328;
  means[28] = 45.899131;
  means[29] = 43.991225;
  means[30] = 43.765153;
  means[31] = 41.463521;
  means[32] = 44.242674;
  means[33] = 42.711916;
  means[34] = 45.522057;
  means[35] = 46.013106;
  means[36] = 46.339431;
  means[37] = 44.305716;
  means[38] = 44.433424;
  means[39] = 42.66599;
  means[40] = 44.224752;
  means[41] = 43.796167;
  means[42] = 44.081792;
  means[43] = 46.363772;
  means[44] = 45.902639;
  means[45] = 44.662997;
  means[46] = 44.137611;
  means[47] = 43.424818;
  means[48] = 42.478948;
  means[49] = 44.269987;
  means[50] = 45.461914;
  means[51] = 45.571824;
  means[52] = 44.941958;
  means[53] = 43.79404;
  means[54] = 43.71952;
  means[55] = 43.556636;
  means[56] = 34.754247;
  means[57] = 41.141175;
  means[58] = 42.976773;
  means[59] = 38.469566;
  means[60] = 37.733158;
  means[61] = 41.11005;
  means[62] = 43.135601;
  means[63] = 43.422766;


  // Bestfit 4.0
  means[0] = 42.948786;
  means[1] = 42.855682;
  means[2] = 36.925885;
  means[3] = 43.391534;
  means[4] = 41.47246;
  means[5] = 42.707777;
  means[6] = 32.336866;
  means[7] = 26.698391;
  means[8] = 43.491445;
  means[9] = 43.737418;
  means[10] = 44.505502;
  means[11] = 39.874592;
  means[12] = 40.456776;
  means[13] = 39.48609;
  means[14] = 39.660693;
  means[15] = 41.199558;
  means[16] = 44.153036;
  means[17] = 45.319723;
  means[18] = 45.819223;
  means[19] = 46.141125;
  means[20] = 45.001867;
  means[21] = 43.335388;
  means[22] = 39.94106;
  means[23] = 43.000749;
  means[24] = 44.728668;
  means[25] = 45.161406;
  means[26] = 45.715474;
  means[27] = 46.536917;
  means[28] = 46.399656;
  means[29] = 44.254814;
  means[30] = 43.816054;
  means[31] = 41.912524;
  means[32] = 44.536809;
  means[33] = 43.735601;
  means[34] = 45.540167;
  means[35] = 46.270986;
  means[36] = 46.635048;
  means[37] = 45.152441;
  means[38] = 44.283245;
  means[39] = 43.305591;
  means[40] = 44.645568;
  means[41] = 44.296134;
  means[42] = 44.035321;
  means[43] = 46.469796;
  means[44] = 46.356824;
  means[45] = 44.890139;
  means[46] = 44.398098;
  means[47] = 43.791013;
  means[48] = 43.597791;
  means[49] = 44.886598;
  means[50] = 45.683098;
  means[51] = 45.968158;
  means[52] = 45.188691;
  means[53] = 44.208235;
  means[54] = 44.009507;
  means[55] = 43.719913;
  means[56] = 33.936569;
  means[57] = 43.031046;
  means[58] = 43.717558;
  means[59] = 38.182153;
  means[60] = 35.594843;
  means[61] = 42.034927;
  means[62] = 43.209921;
  means[63] = 43.367497;

  
  // Set the likelihood functions
  std::vector<Themis::likelihood_base*> L;
  //L.push_back(new Themis::likelihood_visibility_amplitude(VM_data,image));
  L.push_back(new Themis::likelihood_marginalized_visibility_amplitude(VM_data,image));
  L.push_back(new Themis::likelihood_closure_phase(CP_data,image));


  // Set the weights for likelihood functions
  std::vector<double> W(L.size(), 1.0);

  // Make a likelihood object
  Themis::likelihood L_obj(P, L, W);


  std::cerr << "Likelihood is " << L_obj(params) << '\n';
  


#if 1
  
  // Create a sampler object, here the PT MCMC
  //Themis::sampler_affine_invariant_tempered_MCMC MC_obj(42+world_rank);
  Themis::sampler_differential_evolution_tempered_MCMC MCMC_obj(42+world_rank);


  // Generate a chain
  int Number_of_steps = 20000;
  int Number_of_chains = N*N*4;           // Number of walkers
  //int Number_of_chains = 240;
  int Number_of_temperatures = 8;
  //int Number_of_temperatures = 40; //16; // 8;
  int Number_of_procs_per_lklhd = 1;
  int Temperature_stride = 50;
  int Chi2_stride = 10;
  int Ckpt_frequency = 500;
  bool restart_flag = false;
  int out_precision = 8;
  int verbosity = 0;
  
  // Set the CPU distribution
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);
  
  // Set a checkpoint
  MCMC_obj.set_checkpoint(Ckpt_frequency,"MCMC.ckpt");
  
  // Set tempering schedule
  MCMC_obj.set_tempering_schedule(1000.,1.,2.0);
  //MCMC_obj.set_tempering_schedule(1000.,1.,1.1);


  // Parallelization settings
  MCMC_obj.set_cpu_distribution(Number_of_temperatures, Number_of_chains, Number_of_procs_per_lklhd);


  if (world_rank==0)
  {
    std::cout << "\n\n";
    for (size_t j=0; j<image.size(); ++j)
      std::cout << std::setw(15) << std::exp(means[j])
		<< std::setw(15) << means[j]
		<< std::endl;

    std::cout << "\n\n";
    for (size_t j=0; j<image.size(); ++j)
      std::cout << std::setw(15) << means[j];
    std::cout << "\n\n";
  }
  

  // Sample the parameter space
  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                    "Chain-image.dat", "Lklhd-image.dat", "Chi2-image.dat", 
		     means, ranges, var_names, restart_flag, out_precision, verbosity);
  

  std::vector<double> pbest = MCMC_obj.find_best_fit("Chain-image.dat","Lklhd-image.dat");
  double Lval = L_obj(pbest);

  if (world_rank==0)
    std::cout << "Lbest = " << Lval << std::endl;

  L[0]->output_model_data_comparison("visibility_amplitude_residuals.d");
  L[1]->output_model_data_comparison("closure_phase_residuals.d");  



  for (size_t j=0; j<image.size(); ++j)
  {
    means[j] = pbest[j];
    ranges[j] = 1e-3;
  }


  MCMC_obj.run_sampler(L_obj, Number_of_steps, Temperature_stride, Chi2_stride, 
                    "Chain-image_B.dat", "Lklhd-image_B.dat", "Chi2-image_B.dat", 
		     means, ranges, var_names, restart_flag, out_precision, verbosity);

  pbest = MCMC_obj.find_best_fit("Chain-image_B.dat","Lklhd-image_B.dat");
  L_obj(pbest);

  L[0]->output_model_data_comparison("visibility_amplitude_residuals.d");
  L[1]->output_model_data_comparison("closure_phase_residuals.d");  



#endif
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
