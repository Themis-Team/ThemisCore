/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Avery Broderick
    \date  Nov, 2018
    \brief test model_image_score clas by mcmc runs *with* gain calibration
    \details Takes a file list generated via something like:
             readlink -f ~/Themis/Themis/sim_data/Score/README.txt ~/Themis/Themis/sim_data/Score/example_image.dat > file_list
*/

#include "data_visibility_amplitude.h"
#include "model_image_score.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
#include "image_family_error.h"
//#include "sampler_affine_invariant_tempered_MCMC.h"
#include "utils.h"
#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Parse the command line inputs
  std::string image_file_list="";
  int istart = 0;
  int iend = -1;
  bool reflect_image=false;
  bool include_theory_errors=false;

  std::string visibility_amplitude_data_file=Themis::utils::global_path("eht_data/ER5/M87_VM3598.d");
  std::string closure_phase_data_file=Themis::utils::global_path("eht_data/ER5/M87_CP3598.d");
    
  
  for (int k=1; k<argc;)
  {
    std::string opt(argv[k++]);

    if (opt=="--start" || opt=="-s")
    {
      if (k<argc)
	istart = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --start, -s.\n";
	std::exit(1);
      }
    }
    else if (opt=="--end" || opt=="-e")
    {
      if (k<argc)
	iend = atof(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --end, -e.\n";
	std::exit(1);
      }
    }
    else if (opt=="--imagefilelist" || opt=="-f")
    {
      if (k<argc)
	image_file_list = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --imagefilelist, -f.\n";
	std::exit(1);
      }
    }
    else if (opt=="--reflect" || opt=="-r")
    {
      reflect_image=true;
      std::cout << "Reflecting images." << std::endl;
    }
    else if (opt=="--theory-error" || opt=="-te")
    {
      include_theory_errors=true;
      std::cout << "Including theoretical variability as independent errors." << std::endl;
    }
    else if (opt=="--visibility-amplitude-data" || opt=="-vad")
    {
      if (k<argc)
	visibility_amplitude_data_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --visiiblity-amplitude-data, -vad.\n";
	std::exit(1);
      }
    }
    else if (opt=="--closure-phase-data" || opt=="-cpd")
    {
      if (k<argc)
	closure_phase_data_file = std::string(argv[k++]);
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: An argument must be provided after --closure-phase-data, -cpd.\n";
	std::exit(1);
      }
    }
    else
    {
      if (world_rank==0)
	std::cerr << "ERROR: Unrecognized option " << opt << "\n";
      std::exit(1);
    }
  }  
  if (image_file_list=="")
  {
    if (world_rank==0)
      std::cerr << "ERROR: No image file list was provided.\n"
		<< "       The image file list is simply a text file\n"
		<< "       with one file name per line containing the\n"
		<< "       *absolute* path to the image files.  The\n"
		<< "       first line must be the README.txt file that\n"
		<< "       describes the image file parameters.\n";
    std::exit(1);
  }

  // Read in the file list
  std::string README_file_name, stmp;
  std::vector<std::string> image_file_names;
  std::fstream ifnin(image_file_list);
  ifnin >> README_file_name;
  for (ifnin >> stmp;  !ifnin.eof(); ifnin >> stmp)
    image_file_names.push_back(stmp);
  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "VA data file:\n\t" << visibility_amplitude_data_file << std::endl;
    std::cout << "CP data file:\n\t" << closure_phase_data_file << std::endl;
    std::cout << "README file:\n\t" << README_file_name << std::endl;
    std::cout << "Image files: (" << image_file_names.size() << ")\n";
    for (size_t i=0; i<image_file_names.size(); ++i)
      std::cout << "\t" << image_file_names[i] << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
  }
  // Sort out the end point
  if (iend<0 || iend>int(image_file_names.size()) )
  {
    if (world_rank==0)
      std::cerr << "WARNING: iend not set or set past end of file list.\n";
    iend = image_file_names.size();
  }
  // Sort out the start point (after iend is sorted)
  if (istart>iend)
  {
    if (world_rank==0)
      std::cerr << "ERROR: istart is set to beyond iend or past end of file list.\n";
    std::exit(1);
  }

  // Prepare the output summary file
  std::stringstream sumoutname;
  sumoutname << "fit_summaries_" << std::setfill('0') << std::setw(5) << istart << "_" << std::setfill('0') << std::setw(5) << iend << ".txt";
  std::ofstream sumout;
  if (world_rank==0)
  {
   sumout.open(sumoutname.str().c_str());
   sumout << std::setw(10) << "# Index"
	  << std::setw(15) << "I (Jy)"
	  << std::setw(15) << "M/D (uas)"
	  << std::setw(15) << "PA (deg)"
	  << std::setw(15) << "VA red. chisq"
	  << std::setw(15) << "CP red. chisq"
	  << std::setw(15) << "red. chisq"
	  << std::setw(15) << "log-liklhd"
	  << "     FileName"
	  << "     VA Data file"
	  << "     CP Data file"
	  << std::endl;
  }


  ////////////////////////////
  // Begin loop over the relevant portion of file_name_list and running chains.
  // ALL items are in the loop to scope potential problems, though this is probably unncessary.
  // This happens in two steps.
  //   1. First element of list is run in longer chain to get good starting location (cutting PA to +-0.5 rad or +-30 deg)
  //   2. All remaining elements are run in shorter chains that exploit more rapid convergence

  double PAguess = M_PI;


  std::ofstream cpout("spat_cp.txt");


  // Read in data
  Themis::data_visibility_amplitude VM_obs(visibility_amplitude_data_file,"HH");
  Themis::data_closure_phase CP_obs(closure_phase_data_file);

  // Prepare theory error computation
  Themis::image_family_static_error theory_error;
  Themis::data_visibility_amplitude VM;
  Themis::data_closure_phase CP;
  if (include_theory_errors)
  {
    std::vector<double> p;
    p.push_back(0.6);
    p.push_back(3.66);
    p.push_back(-72*M_PI/180.);
    theory_error.generate_error_estimates(image_file_names,README_file_name,p);
    VM = theory_error.data_visibility_amplitude(VM_obs);
    CP = theory_error.data_closure_phase(CP_obs);
  }
  else
  {
    VM = VM_obs;
    CP = CP_obs;
  }
  
  for (size_t index=size_t(istart); index<size_t(iend); ++index)
  {

    // Choose the model to compare
    //Themis::model_image_score image(Themis::utils::global_path("sim_data/Score/example_image.dat"),Themis::utils::global_path("sim_data/Score/README.txt"));
    Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image);


    std::vector<double> p(3);
    p[0] = 0.6;
    p[1] = 3.66;
    p[2] = 0.0;

    for (int ipa=0; ipa<36; ++ipa) 
    {
      double PA = ipa * 2.0*M_PI / 36.;

      p[2] = PA;

      image.generate_model(p);

      for (size_t j=0; j<CP.size(); ++j)
      {
	cpout << std::setw(15) << std::sqrt( CP.datum(j).u1*CP.datum(j).u1 + CP.datum(j).v1*CP.datum(j).v1 + CP.datum(j).u2*CP.datum(j).u2 + CP.datum(j).v2*CP.datum(j).v2 + CP.datum(j).u3*CP.datum(j).u3 + CP.datum(j).v3*CP.datum(j).v3 )
	      << std::setw(15) << CP.datum(j).CP
	      << std::setw(15) << CP.datum(j).err
	      << std::setw(15) << image.closure_phase(CP.datum(j),0)
	      << '\n';
      }
      cpout << '\n' << std::endl;

    }
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
