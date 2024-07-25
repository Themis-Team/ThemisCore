/*!
    \file model_image_sed_fitted_riaf.cpp
    \author Avery Broderick
    \date  Nov, 2018
    \brief Generates CP predictions for parameters passed in fit_summaries.txt file for a set of frames
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

  int Number_start_params = 0;
  std::string param_file="";

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
    else if (opt=="--parameter-file" || opt=="-p")
    {
      if (k+1<argc)
      {
	Number_start_params = atoi(argv[k++]);
	param_file = std::string(argv[k++]);
      }
      else
      {
	if (world_rank==0)
	  std::cerr << "ERROR: TWO arguments must be provided after --parameter-file, -p: <# of params to set> and file list.\n";
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
  std::vector< std::vector<double> > start_parameter_list(image_file_names.size());
  if (param_file!="")
  {
    std::fstream pfin(param_file);
    if (!pfin.is_open())
    {
      std::cerr << "ERROR: Could not open " << param_file << '\n';
      std::exit(1);
    }
    std::cerr << "Opened parameter file " << param_file << '\n';
    double dtmp;
    pfin.ignore(4096,'\n');
    // Get first index
    pfin >> dtmp;
    for (size_t j=0; j<start_parameter_list.size() && !pfin.eof(); ++j)
    {
      for (int k=0; k<Number_start_params; k++)
      {
	pfin >> dtmp;
	start_parameter_list[j].push_back(dtmp);
      }
      // Kill remainder of line
      pfin.ignore(4096,'\n');
      // Get next index
      pfin >> dtmp;      
    } 
  }  
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

  


  std::ofstream cpout("cp_fit_distance.txt");  


  ////////////////////////////
  // Begin loop over the relevant portion of file_name_list

  // Read in data
  //Themis::data_visibility_amplitude VM(visibility_amplitude_data_file,"HH");
  Themis::data_closure_phase CP(closure_phase_data_file);


  if (start_parameter_list.size()!=image_file_names.size())
  {
    std::cerr << "ERROR: The number of start parameters from fit_summaries file MUST match the number of frames\n";
    std::cerr << "Found " << start_parameter_list.size() << " parameter sets and " << image_file_names.size() << " image files\n";
    std::exit(1);
  }
  
  for (size_t index=size_t(istart); index<size_t(iend); ++index)
  {

    // Choose the model to compare
    //Themis::model_image_score image(Themis::utils::global_path("sim_data/Score/example_image.dat"),Themis::utils::global_path("sim_data/Score/README.txt"));
    Themis::model_image_score image(image_file_names[index],README_file_name,reflect_image);


    std::vector<double> p(3);
    p[0] = start_parameter_list[index][0];
    p[1] = start_parameter_list[index][1];
    p[2] = start_parameter_list[index][2] * M_PI/180.;



    image.generate_model(p);

    cpout << std::setw(15) << index;
    for (size_t j=0; j<CP.size(); ++j)
    {
      cpout << std::setw(15) << image.closure_phase(CP.datum(j),0);
    }
    cpout << "      " << image_file_names[index]
	  << "      " << closure_phase_data_file
	  << "      " << param_file
	  << std::endl;
  }
    
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
