
#include "image_family_error.h"
#include "data_visibility_amplitude.h"
#include "model_image_score.h"
#include "likelihood_optimal_gain_correction_visibility_amplitude.h"
#include "sampler_differential_evolution_tempered_MCMC.h"
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

  std::cerr << "Finished parsing command lines\n";
  
  Themis::data_visibility_amplitude VM(Themis::utils::global_path("eht_data/ER5/M87_VM3598.d"),"HH");
  Themis::data_visibility_amplitude VM2(Themis::utils::global_path("eht_data/ER5/netcal/3600-hi/wosB/VM_hops_3600_M87+netcal_scanavg.d"),"HH");

  Themis::data_closure_phase CP(Themis::utils::global_path("eht_data/ER5/M87_CP3598.d"));
  Themis::data_closure_phase CP2(Themis::utils::global_path("eht_data/ER5/netcal/3600-hi/wosB/CP_hops_3600_M87+netcal_scanavg.d"));

  std::cerr << "Finished reading data\n";

  std::vector<double> p;
  p.push_back(0.6);
  p.push_back(3.66);
  p.push_back(0.0);

  //Themis::image_family_static_error ifse(image_file_names,README_file_name,p);
  Themis::image_family_static_error ifse;
  ifse.generate_error_estimates(image_file_names,README_file_name,p);

  std::cerr << "Finished creating ifse object\n";

  
  // Create data objects that have the theory error added in quadrature  
  Themis::data_visibility_amplitude VM_new = ifse.data_visibility_amplitude(VM);
  Themis::data_visibility_amplitude VM2_new = ifse.data_visibility_amplitude(VM2);
  Themis::data_closure_phase CP_new = ifse.data_closure_phase(CP);
  Themis::data_closure_phase CP2_new = ifse.data_closure_phase(CP2);

  std::cerr << "Generated errors\n";

  if (world_rank==0)
  {
    std::ofstream dout("data_test.d");
    for (size_t k=0; k<VM.size(); ++k)
      dout << std::setw(15) << VM.datum(k).u
	   << std::setw(15) << VM.datum(k).v
	   << std::setw(15) << VM.datum(k).V
	   << std::setw(15) << VM.datum(k).err
	   << std::setw(5)  << VM.datum(k).Station1 
	   << std::setw(5)  << VM.datum(k).Station2
	   << std::setw(15) << VM.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::setw(15) << VM_new.datum(k).u
	   << std::setw(15) << VM_new.datum(k).v
	   << std::setw(15) << VM_new.datum(k).V
	   << std::setw(15) << VM_new.datum(k).err
	   << std::setw(5)  << VM_new.datum(k).Station1 
	   << std::setw(5)  << VM_new.datum(k).Station2
	   << std::setw(15) << VM_new.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::endl;
    dout << std::endl << std::endl;
    
    for (size_t k=0; k<VM2.size(); ++k)
      dout << std::setw(15) << VM2.datum(k).u
	   << std::setw(15) << VM2.datum(k).v
	   << std::setw(15) << VM2.datum(k).V
	   << std::setw(15) << VM2.datum(k).err
	   << std::setw(5)  << VM2.datum(k).Station1 
	   << std::setw(5)  << VM2.datum(k).Station2
	   << std::setw(15) << VM2.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::setw(15) << VM2_new.datum(k).u
	   << std::setw(15) << VM2_new.datum(k).v
	   << std::setw(15) << VM2_new.datum(k).V
	   << std::setw(15) << VM2_new.datum(k).err
	   << std::setw(5)  << VM2_new.datum(k).Station1 
	   << std::setw(5)  << VM2_new.datum(k).Station2
	   << std::setw(15) << VM2_new.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::endl;
    dout << std::endl << std::endl;
    
    for (size_t k=0; k<CP.size(); ++k)
      dout << std::setw(15) << CP.datum(k).u1
	   << std::setw(15) << CP.datum(k).v1
	   << std::setw(15) << CP.datum(k).u2
	   << std::setw(15) << CP.datum(k).v2
	   << std::setw(15) << CP.datum(k).CP
	   << std::setw(15) << CP.datum(k).err
	   << std::setw(5)  << CP.datum(k).Station1 
	   << std::setw(5)  << CP.datum(k).Station2
	   << std::setw(5)  << CP.datum(k).Station3
	   << std::setw(15) << CP.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::setw(15) << CP_new.datum(k).u1
	   << std::setw(15) << CP_new.datum(k).v1
	   << std::setw(15) << CP_new.datum(k).u2
	   << std::setw(15) << CP_new.datum(k).v2
	   << std::setw(15) << CP_new.datum(k).CP
	   << std::setw(15) << CP_new.datum(k).err
	   << std::setw(5)  << CP_new.datum(k).Station1 
	   << std::setw(5)  << CP_new.datum(k).Station2
	   << std::setw(5)  << CP_new.datum(k).Station3
	   << std::setw(15) << CP_new.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::endl;
    dout << std::endl << std::endl;
    
    for (size_t k=0; k<CP2.size(); ++k)
      dout << std::setw(15) << CP2.datum(k).u1
	   << std::setw(15) << CP2.datum(k).v1
	   << std::setw(15) << CP2.datum(k).u2
	   << std::setw(15) << CP2.datum(k).v2
	   << std::setw(15) << CP2.datum(k).CP
	   << std::setw(15) << CP2.datum(k).err
	   << std::setw(5)  << CP2.datum(k).Station1 
	   << std::setw(5)  << CP2.datum(k).Station2
	   << std::setw(5)  << CP2.datum(k).Station3
	   << std::setw(15) << CP2.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::setw(15) << CP2_new.datum(k).u1
	   << std::setw(15) << CP2_new.datum(k).v1
	   << std::setw(15) << CP2_new.datum(k).u2
	   << std::setw(15) << CP2_new.datum(k).v2
	   << std::setw(15) << CP2_new.datum(k).CP
	   << std::setw(15) << CP2_new.datum(k).err
	   << std::setw(5)  << CP2_new.datum(k).Station1 
	   << std::setw(5)  << CP2_new.datum(k).Station2
	   << std::setw(5)  << CP2_new.datum(k).Station3
	   << std::setw(15) << CP2_new.datum(k).tJ2000-VM.datum(0).tJ2000
	   << std::endl;
    dout << std::endl << std::endl;
  }
  
  //Finalize MPI
  MPI_Finalize();
  return 0;
}
