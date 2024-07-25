/*!
    \file sgrA_riaf_fitting.cpp
    \author Paul Tiede, Avery Broderick
    \date May, 2020
    \brief Driver file for SgrA analysis with Broderick et al. 2016 RIAF model which is static
    \details Takes file lists generated via something like:
	     These must be passed a -v <file> which contains a list of files to be read in.
*/


#include "utils.h"
#include "cmdline_parser.h"

#include "interpolator1D.h"
#include "model_image_general_riaf.h"
#include "model_image_sum.h"
#include "model_image_symmetric_gaussian.h"
#include "vrt2.h"

#include <mpi.h>
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <complex>

int main(int argc, char* argv[])
{
  
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << "MPI Initiated in rank: " << world_rank << " of " << world_size << std::endl;

  // Fixed options
  int Number_start_params = 5;
  const double mass = 4.15e6;
  double ne_initial = 2e8;
  const double infall = 0.05;
  const double pos = 0.0;

  // Prepare command-line parser
  Themis::CMDLineParser argp("general_riaf_generate_image_library","Performs ray tracing to generate a library of images.");

  Themis::CMDLineParser::IntArg npix (argp,"-N,--npix","Sets the initial (before refines) number of image pixels.",40,false);
  Themis::CMDLineParser::IntArg nref (argp,"-r,--refines","Sets the number of refine passes to perform.",3,false);
  Themis::CMDLineParser::FloatArg fov(argp,"-fov,--field-of-view","Sets the field of view of the images in M/D.",60,false);
  Themis::CMDLineParser::StringArg param_file(argp,"-p","Name of file with list of parameters in fit summary format.",true);
  Themis::CMDLineParser::IntArg start_line (argp,"-s,--start","Sets *line* at which to begin.",0,false);
  Themis::CMDLineParser::IntArg end_line (argp,"-e,--end","Sets *line* after that at which to end.",false);
  Themis::CMDLineParser::StringArg outname(argp,"-o,--out-prefix","Prefix of output name.  Files will be named <prefix>_<index>.dat","riaf",false);
  Themis::CMDLineParser::FloatArg flux_target(argp,"-F,--flux-target","Sets the target flux in Jy.",2.5,false);
  Themis::CMDLineParser::IntArg procs_per_model(argp,"-mp,--procs-per-model","Sets the number of processes to use per model evaluation.  Parallelizes library construction on remaining dimension.",4,false);

  // Parse command line
  argp.parse_args(argc,argv,true);
  
  // Set and fill the start parameters if provided
  // Assumes has the same format as the fit_summaries.txt file (header, index, parameters, then other items)
  std::vector<int> index_list;
  std::vector<std::vector<double> > start_parameter_list;
  std::ifstream pfin(param_file());
  if (!pfin.is_open())
  {
    std::cerr << "ERROR: Could not open " << param_file() << '\n';
    std::exit(1);
  } 
  double dtmp;
  int itmp;
  pfin.ignore(4096,'\n'); // kill header
  while (pfin>>itmp) {
    index_list.push_back(itmp);
    // process string ...
    std::vector<double> plist(Number_start_params, 0.0);
    for (int k=0; k<Number_start_params; k++)
    {
      pfin >> dtmp;
      plist[k] = dtmp;
    }
    pfin.ignore(4096,'\n'); // kill the rest of the line
    start_parameter_list.push_back(plist);
  }
  pfin.close();
  size_t end_line_index = start_parameter_list.size();
  if (end_line.is_defined())
    end_line_index = std::min(size_t(end_line()),end_line_index);
  //  Output these for check
  if (world_rank==0)
  {
    std::cout << "---------------------------------------------------\n" << std::endl;
    std::cout << "Npix: " << npix() << std::endl
              << "Nref: " << nref() << std::endl
	      << "FOV: " << fov() << std::endl
	      << "start line: " << start_line() << std::endl
	      << "end line:   " << end_line_index << std::endl
	      << "fs file: " << param_file() << "\n";
    for (size_t p = size_t(start_line()); p < std::min(size_t(5),std::min(end_line_index,start_parameter_list.size())); ++p)
    {
      std::cout << "\t";
      std::cout << std::setw(10) << index_list[p];
      for (size_t j=0; j< start_parameter_list[p].size(); ++j)
	std::cout << std::setw(15) << start_parameter_list[p][j];
      std::cout << std::endl;  
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------------------\n" << std::endl;
  }

  
  VRT2::SgrA_PolintDiskModelParameters2010 sdmp(Themis::utils::global_path("src/VRT2/DataFiles/2010_combined_fit_parameters.d"),2,3);

  // Make new communicator for groups of CPU's
  MPI_Comm model_comm;
  int model_color = world_rank/procs_per_model();
  int number_of_model_colors = world_size/procs_per_model();
  MPI_Comm_split(MPI_COMM_WORLD, model_color, world_rank, &model_comm);
  
  Themis::model_image_general_riaf riaf;
  riaf.set_image_resolution(npix(), nref());
  riaf.set_screen_size(fov()/2.0);
  riaf.set_mpi_communicator(model_comm);

  int mc_size, mc_rank;
  MPI_Comm_rank(model_comm,&mc_rank);
  MPI_Comm_size(model_comm,&mc_size);
  
  //RIAF model plus a large scale gaussian
  for ( size_t p = size_t(start_line()); p < std::min(end_line_index,start_parameter_list.size()); ++p )
  {
    // If this is not the current processes job, skip
    if ( int(p)%number_of_model_colors != model_color )
      continue;
    
    std::vector<double> params(riaf.size(), 0.0);
    params[0] = mass;
    params[1] = start_parameter_list[p][0];
    params[2] = start_parameter_list[p][1];
    sdmp.reset(std::fabs(params[1]), std::acos(params[2])*180.0/M_PI);
    params[3] = sdmp.ne_norm();
    if ( params[1] < 0 )
      ne_initial = sdmp.ne_norm()*5;
    else
      ne_initial = sdmp.ne_norm();

    params[4] = sdmp.ne_index();
    params[5] = start_parameter_list[p][2];
    //params[6] = 8e10;//sdmp.Te_norm(); // Not sure why this was set to 8e10!
    params[6] = sdmp.Te_norm();
    params[7] = sdmp.Te_index();
    params[8] = sdmp.ne_norm()*start_parameter_list[p][3];
    params[9] = sdmp.nnth_index();
    params[10] = start_parameter_list[p][2];
    params[11] = infall;
    params[12] = start_parameter_list[p][4];
    params[13] = pos;




    if ( mc_rank == 0 ){
      std::cout << "Generating model " << p 
                << "/" << start_parameter_list.size() << std::endl;
      std::cout << "Parameters: \n";
      for ( size_t ii = 0; ii < params.size(); ++ii )
        std::cout << std::setw(15) << params[ii];
      std::cout << std::endl;
      //std::cout << "-----------------------------------------------------------" << std::endl;
    }

    //riaf.set_image_resolution(npix, std::max(nref-1.0, 0.0));
    riaf.set_image_resolution(npix(), 0);
    double dmin = -4;
    double dmax = 4;
    double dfactor = 0.2;
    double dflux_min = 1.0;
    double dflux_max = 1.0;
    while (dflux_min*dflux_max > 0){
      params[3] = ne_initial*std::pow(10.0,dmax);
      params[8] = ne_initial*start_parameter_list[p][3]*std::pow(10.0,dmax);
      riaf.generate_model(params);
      //riaf.generate_complex_visibilities();
      dflux_max = riaf.flux()-flux_target();
    
      params[3] = ne_initial*std::pow(10.0,dmin);
      params[8] = ne_initial*start_parameter_list[p][3]*std::pow(10.0,dmin);
      riaf.generate_model(params);
      //riaf.generate_complex_visibilities();
      dflux_min = riaf.flux()-flux_target();
      if (dflux_min > 0){
        dmax = dmin;
        //dmin /= 2.0;
	dmin -= 4;
      }
      if (dflux_max < 0){
        dmin = dmax;
        //dmax *= 2.0;
	dmax += 4;
      }
      if (dflux_min*dflux_max > 0){
        if (mc_rank == 0)
          std::cout << "Not bracketed!\n"
		    << "\tTarget:          " << std::setw(15) << flux_target() << '\n'
		    << "\tUpper flux/dval: " << std::setw(15) << flux_target()+dflux_max << std::setw(15) << dmax << '\n'
		    << "\tLower flux/dval: " << std::setw(15) << flux_target()+dflux_min << std::setw(15) << dmin << std::endl;
      }
    }
    int nmax = 20;
    int n = 0;
    double dmid;
    while (n < nmax){
      dmid = (dmax+dmin)/2.0;
      params[3] = ne_initial*std::pow(10.0,dmid);
      params[8] = ne_initial*start_parameter_list[p][3]*std::pow(10.0,dmid);
      riaf.generate_model(params);
      //riaf.generate_complex_visibilities();
      double dflux_mid = riaf.flux()-flux_target();
      // if (mc_rank == 0){
      //    std::cout << "flux " << n << " : " << riaf.flux() << std::endl;
      // }
      if (dflux_mid*dflux_min > 0)
        dmin = dmid;
      else
        dmax = dmid;
      if (std::fabs(dflux_mid) < 1e-2 || (dmax-dmin)< 1e-6){
        break;
      }
      n++;
    }
    dfactor = std::pow(10.0,dmid);
    //dfactor -= 0.01; /// Why?
    //dfactor -= 0.1;
    /*
    if (ldf>fluxes[0] && ldf<fluxes[fluxes.size()-1]) {
      std::vector<double>::const_iterator p = std::lower_bound(fluxes.begin(),fluxes.end(),ldf);
      // p should now be an iterator to the first value less than x
      size_t i = p - fluxes.begin() - 1;
      double dlf = (ldf-fluxes[i])/(fluxes[i+1]-fluxes[i]);
      dfactor =  (  dlf*dfactors[i+1] + (1.0-dlf)*dfactors[i] );
    }
    else if (ldf<=fluxes[0])
      dfactor =  dfactors[0] + (dfactors[1]-dfactors[0])/(fluxes[1]-fluxes[0])*(ldf-fluxes[0]);
    else
      dfactor =  dfactors[dfactors.size()-1] + 
                (dfactors[dfactors.size()-1]-dfactors[dfactors.size()-2])/
                (fluxes[fluxes.size()-1]-fluxes[fluxes.size()-2])*
                (ldf-fluxes[dfactors.size()-1]);
    */
    

    if (mc_rank == 0)
      std::cout << "Correction dfactor: " << dfactor << std::endl;
    //Update ne_initial to be a better match for the next image
    ne_initial *= dfactor;

    //Reset resolutions and other stuff
    params[3] = ne_initial;
    params[8] = ne_initial*start_parameter_list[p][3];
    riaf.set_image_resolution(npix(), nref());
    riaf.generate_model(params);
    riaf.generate_complex_visibilities();
    
    std::stringstream outstream;
    outstream << outname() << "_"
              << std::setfill('0')
              << std::setw(8)
              << index_list[p]
	      << ".dat";
    if ( mc_rank == 0 ){
      std::cout << "Saving image. Final flux: " << riaf.flux() << std::endl;
      riaf.output_image(outstream.str());
    }
  }

  //Finalize MPI
  MPI_Finalize();
  return 0;
  
}
