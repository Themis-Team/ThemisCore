/*!
  \file sampler_grid_search.cpp
  \author Mansour Karami
  \brief Implementation file for the sampler_grid_search class
*/


#include "sampler_grid_search.h"
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <iomanip>

#include "stop_watch.h"
#include <sstream>

#define MASTER 0

namespace Themis{

  sampler_grid_search::sampler_grid_search()
  {
  }

  sampler_grid_search::~sampler_grid_search()
  {    
  }

  void sampler_grid_search::set_cpu_distribution(int num_batches, int num_likelihood)
  {
    BNum = num_batches;
    LNum = num_likelihood;
    default_cpu_distribution = false;
    
  }

  void sampler_grid_search::run_sampler(likelihood _L, std::vector<double>& range_min, std::vector<double>& range_max, std::vector<int>& num_samples, std::string output_file, int output_precision)
  {

    
    // Setting up MPI
    int size, rank;
    MPI_Status Stat;
    //MPI_Request Req;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

 
    MPI_Comm B_COMM;   //B_COMM communicates between different batches
    int B_size, B_rank;
    MPI_Comm L_COMM;   //L_COMM communicates between processors calculating a single likelihood instance in parallel 
    int L_size, L_rank;

    // Creating new MPI communicators
    int flavour = rank / (LNum);
    MPI_Comm_split(MPI_COMM_WORLD, flavour, rank, &L_COMM);
    MPI_Comm_rank(L_COMM, &L_rank);
    MPI_Comm_size(L_COMM, &L_size);  

    int color = rank % (LNum);
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &B_COMM);
    MPI_Comm_rank(B_COMM, &B_rank);
    MPI_Comm_size(B_COMM, &B_size);  

    //Set the likelihood commmunicator 
    _L.set_mpi_communicator(L_COMM);

    //Open the output file
    std::ofstream out, ckpt;
    if(rank == MASTER)
      {
	out.open(output_file.c_str(), std::ios::out);
	out.precision(output_precision);
      }

    int dim = num_samples.size(); //Number of dimentions
    long int tot_samples = num_samples[0]; //Total number of samples 
    for (int i = 1; i < dim; i++)
      {
	tot_samples *= num_samples[i];
      }

    //Construct the parameter array
    double** param = new double*[dim];
    for(int i = 0; i < dim; ++i)
      param[i] = new double[tot_samples];


    //Fill in the parameter array
    for(int j = 0; j < dim; j++)
      {
	long int stride = 1;
	for(int k = 0; k < j; k++)
	  stride *= num_samples[k];
	for (int i = 0; i < tot_samples; i++)
	  {
	  param[j][i] = range_min[j] + ((i/stride)%num_samples[j]) * (range_max[j] - range_min[j]) / num_samples[j];
	  }

      } 


    //Main loop to calculate the likelihoods
    int start = B_rank * (tot_samples)/B_size; //Start index of the parameter vector for each process
    int end = (B_rank+1) * (tot_samples)/B_size; //Ending index of the parameter vector for each process
    //int batch_size = (tot_samples)/B_size; //Parameter batch size for each process

    std::vector<double> state(dim); // Vector in the parameter space to calculate the chi_squared at
    std::vector<double> buff(dim+1); 
    //double likelihood_value = 0.0; 
    double chi_squared_value = 0.0;

    // Main loop to evaluate the chi squared values at grid points
    for(int i = start; i < end; i++)
      {
	state.resize(dim);
	for(int j = 0; j < dim; j++)
	  state[j] = param[j][i];
	chi_squared_value = _L(state);
	state.push_back(chi_squared_value);

	// All batches send their samples to the master batch
	if((L_rank == MASTER) && (B_rank > MASTER))
	  {
	    MPI_Send(&state[0], dim+1, MPI_DOUBLE, MASTER, B_rank, B_COMM);
	  }

	// Master batch receives the samples form all the other batches
	// and writes to the output
	for(int j = 1; j < B_size; j++)
	  {   
	    if(rank == MASTER)
	      {
		MPI_Recv(&buff[0], dim+1, MPI_DOUBLE, j, j, B_COMM, &Stat);
		for(int k = 0; k < dim+1; k++)
		  {
		    out <<  buff[k] << "   ";
		  }
		out << std::endl;
	      }
	  }

	// Master process writes the batch calculated by it's own batch
	if(rank == MASTER)
	  {
	    for(int k = 0; k < dim + 1; k++)
	      {
		out << state[k] << "   ";
	      }
	    out << std::endl;
	  }
      } 
   

  }



};
