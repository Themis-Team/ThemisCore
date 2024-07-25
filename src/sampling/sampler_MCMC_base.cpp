/*!
  \file sampler_MCMC_base.cpp
  \author Paul Tiede
  \brief Implementation file for the abstract mcmc base class. 
  \details This forms the basis/interface for all samplers using mcmc. 
*/

#include "sampler_MCMC_base.h"
#include <mpi.h>
#include <sstream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <iomanip>


namespace Themis
{
  sampler_MCMC_base::sampler_MCMC_base(int seed, likelihood& L, 
                                       std::vector<std::string> var_names, size_t dimension)
  : _seed(seed),_rng(seed), _step_count(0), _ckpt_stride(-1), _sum_lklhd(0.0),  _comm(MPI_COMM_WORLD)
  { 
    _var_names.resize(0);
    if ( var_names.size() != dimension )
    {
      std::cout << "Warning var_names is has the wrong dimension, filling it with the default names.\n";
      for ( size_t i = 0; i < dimension; ++i )
      {
        _var_names.push_back("p"+std::to_string(i));
      }
    }
    else
    {
      _var_names = var_names;
    }
    _L = &L;
    _dimension = dimension;
  }



  void sampler_MCMC_base::set_output_stream(std::string chain_out, std::string state_out,
                                            std::string sampler_out,
                                            int output_precision)
  {
    _chain_file = chain_out;
    _state_file = state_out;
    _sampler_file = sampler_out;
    _output_precision = output_precision;

  }

  void sampler_MCMC_base::set_checkpoint(int ckpt_stride, std::string ckpt_file)
  {
    _ckpt_stride = ckpt_stride;
    _ckpt_file = ckpt_file;
  }


  void sampler_MCMC_base::read_checkpoint(std::string ckpt_file)
  {
    std::cout << "Reading in checkpoint file " << ckpt_file << std::endl;
    std::ifstream ckpt_in(ckpt_file.c_str());
    if (!ckpt_in.is_open()){
      std::cerr << "Checkpoint file " << ckpt_file << " does not exist!\n";
      std::exit(1);
    }
    read_checkpoint(ckpt_in);
  }

  void sampler_MCMC_base::set_mpi_communicator(MPI_Comm comm)
  {
    _comm = comm;
    _L->set_mpi_communicator(_comm);
  }

  void sampler_MCMC_base::write_chain_header()
  {
    _chain_out << "#";
    for ( size_t i = 0; i < _var_names.size(); ++i )
      _chain_out << std::setw(15) << _var_names[i];
    _chain_out << std::endl;
  }

  void sampler_MCMC_base::write_state_header()
  {
    _state_out << "#";
    _state_out << std::setw(15) << "log-lklhd" << std::endl;
  }

  std::vector<double> sampler_MCMC_base::find_best_fit() const
  {
    return this->find_best_fit(_chain_file, _state_file);
  }

  std::vector<double> sampler_MCMC_base::find_best_fit(std::string chain_file, std::string state_file) const
  {
    std::vector<double> pBest;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<double> parameters;
  
    if (world_rank==0)
    {
      // Read likelihood file
      std::fstream lin(state_file.c_str());
      if (!lin.is_open())
      {
        std::cerr << "sampler_MCMC_base: Can't open state file " << state_file << std::endl;
        std::exit(1);
      }
      double L, Lmax;
      int indmax=0;
      std::string line;
      std::getline(lin, line);
      //skip first line if starts with #
      if (line.rfind("#",0)==0)
        std::getline(lin,line);
      if (line.rfind("#",0)==0)
        std::getline(lin,line);
      std::istringstream stream(line);
      stream >> L;
      Lmax = L;
      int index = 0;
      while (std::getline(lin, line)){
        std::istringstream sin(line);
        sin >> L;
        index++;
        if ( L > Lmax )
        {
          indmax = index;
          Lmax = L;
        }
      }
      lin.close();

      //Now read in the line with the maximum likelihood.
      
      std::fstream cin(chain_file.c_str());
      if (!cin.is_open())
      {
        std::cerr << "sampler_MCMC_base: Can't open chain file " << chain_file << std::endl;
        std::exit(1);
      }
      std::string parameter_line;
      //indmax+=1;
      std::getline(cin, parameter_line);
      if (parameter_line.rfind("#",0)==0)
        std::getline(cin,parameter_line);
      if (line.rfind("#",0)==0)
        std::getline(cin,parameter_line);
      for (int index=0; index<indmax+1; ++index)
        getline(cin,parameter_line);
      std::istringstream iss(parameter_line);
      double tmp;
      iss >> tmp;
      do
      {
        parameters.push_back(tmp);
        iss >> tmp;
      }
      while(iss);
      std::cout << "Lmax = " << Lmax << std::endl;

    }

    int N;
    if (world_rank==0)
      N = int(parameters.size());
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD );

    double *dbuff = new double[N];
    if (world_rank==0)
      for (int j=0; j<N; ++j)
        dbuff[j] = parameters[j];
    MPI_Bcast(&dbuff[0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    parameters.resize(0);
    for (int j=0; j<N; ++j)
      parameters.push_back(dbuff[j]);

    return parameters;
    
  }

  


}//end Themis

