/*!
  \file sampler_affine_invariant_tempered_MCMC.cpp
  \author Mansour Karami
  \brief Implementation file for the sampler_affine_invariant_tempered_MCMC class
*/


#include "sampler_affine_invariant_tempered_MCMC.h"
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
#include <limits>
#include "stop_watch.h"
#include <sstream>

#define MASTER 0

namespace Themis{

  sampler_affine_invariant_tempered_MCMC::sampler_affine_invariant_tempered_MCMC(int seed): _rng(seed), _grng(int(RAND_MAX*_rng.rand()))
  {
    // Use the default cpu distributio
    default_cpu_distribution = true;
    _ckpt_file = "sampler.ckpt";
    _ckpt_stride = 10;

  }

  sampler_affine_invariant_tempered_MCMC::~sampler_affine_invariant_tempered_MCMC()
  {
    
  }
  
  // Function to set the cpu distribution for MPI parallelization 
  void sampler_affine_invariant_tempered_MCMC::set_cpu_distribution(int num_temperatures, int num_walkers, int num_likelihood)
  {
    TNum = num_temperatures;
    ChNum = num_walkers;
    LKLHD_Num = num_likelihood;
    default_cpu_distribution = false;
    
  }

  // Function to set checkpointing options
  void sampler_affine_invariant_tempered_MCMC::set_checkpoint(int ckpt_stride, std::string ckpt_file)
  {
    _ckpt_stride = ckpt_stride;
    _ckpt_file = ckpt_file;
    
  }

  // Function to calculate the Bayesian evidence
  void sampler_affine_invariant_tempered_MCMC::estimate_bayesian_evidence(std::vector<std::string> file_names, std::vector<double> temperatures, int burn_in)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    std::ofstream result;
    std::ifstream likelihood_in;  
    std::vector<double> U(file_names.size(),0.0);
    double log_evidence = 0.0, err = 0.0, err2 = 0.0;

    if(rank ==  MASTER)
      {
	if(temperatures.size() != file_names.size())
	  {
	    std::cerr << "Bayesian evidence calculation failed. The Number of likelihood files should match the number of temperatures." << std::endl;
	    return;
	  }

	result.open("Bayesian_evidence.txt", std::ios::out);

	//Calculate the expectation value of the log-likelihood U=E(log(likelihood)) at each temperature 
	for(size_t i = 0; i < file_names.size(); ++i)
	  {
	    likelihood_in.open(file_names[i].c_str(), std::ios::in);
	    if(!likelihood_in.is_open())
	      {
		std::cerr << "Likelihhod file doesn't exist. Cannot calculate the Bayesian evidence." << std::endl;
		return;
	      }

	    std::string line, word;
	    int likelihood_num = 0, line_num = 0;
	    while(getline(likelihood_in, line))
	      {
		line_num += 1;
		
		if(line_num > burn_in)
		  {
		    std::istringstream iss(line);	      
		    while(iss >> word)
		      {
			likelihood_num += 1;
			U[i] += std::stod(word.c_str());
		      }
		  }
	      }
	    
	    result << "U[" << i << "]:  " << U[i]/likelihood_num <<  std::endl;
	    U[i] /= likelihood_num;

	    likelihood_in.close();
	  }

	//Integrate over beta=1/T to estimate the evidence
	for(size_t i = 0; i < temperatures.size()-1; ++i)
	  {
	    result << "T:  " << temperatures[i] << "  d[1/t]:  " << 1.0/temperatures[i] - 1.0/temperatures[i+1] 
	    	      << "  d(log-evidence):  " << 0.5*(U[i] + U[i+1])*(1.0/temperatures[i] - 1.0/temperatures[i+1])<<  std::endl;
	    log_evidence += 0.5*(U[i] + U[i+1])*(1.0/temperatures[i] - 1.0/temperatures[i+1]);

	    //std::cout << "log-evidence: " << log_evidence << " Err: " << err << std::endl; 
	  }

	for(size_t i = 0; i < temperatures.size()-2; i+=2)
	  {
	    err += (pow(1.0/temperatures[i] - 1.0/temperatures[i+2], 2.0) / 12.0) * ((U[i+2] - U[i+1])/ (1.0/temperatures[i+2] - 1.0/temperatures[i+1]) - (U[i+1] - U[i])/ (1.0/temperatures[i+1] - 1.0/temperatures[i]));
	    err2 += 0.5*(U[i] + U[i+2])*(1.0/temperatures[i] - 1.0/temperatures[i+2]);
	  }

	result << "log(Bayesian evidence): " << log_evidence <<  " Error:   " << fabs(err) << " Error:  " <<  fabs(log_evidence - err2)  <<  std::endl;
	result.close();
      }


      
    return;
  }


  //Parallel tempered ensemble sampler MCMC routine
  void sampler_affine_invariant_tempered_MCMC::run_sampler(likelihood _L, 
							     int length, int temp_stride, int chi2_stride, 
							     std::string chain_file, std::string lklhd_file,
							     std::string chi2_file, std::vector<double> means, 
							   std::vector<double> ranges, std::vector<std::string> var_names,
							   bool continue_flag, int output_precision, int verbosity, bool adaptive_temperature, std::vector<double> temperatures)
  {


  

  // Number of dimentions in the parameter space to be explored
  dim = means.size();


  // Number of steps after which different temperatures communicate
  int stride = temp_stride;
  
  int size, rank;
  MPI_Status Stat, Stat2, Stat3, Stat4;
  MPI_Request Req, Req2, Req3;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(default_cpu_distribution)
    {
      // Number of temperatures
      //TNum = temperature_number;
      TNum = 4;

      // Number of chains in each temperature
      //ChNum = chain_number;
      ChNum = dim * 4;
      
      LKLHD_Num = 1;
    }

  LKLHD_Num = size / ((ChNum/2)*TNum); // Number of processes assigned to each likelihood calculation 
  if (LKLHD_Num==0) // At least one proc for likelihood computation
      LKLHD_Num=1;



  //Test whether we have the right number of CPUs
  if((size%TNum != 0) || (ChNum%2 != 0) || ((size%(TNum*ChNum/2) != 0) && ((TNum*ChNum/2)%size != 0)))
    {
      if(rank == 0)
	{
	  std::cerr << "###############################" << std::endl;
	  std::cerr << "Error: invalid number of CPU's!" << std::endl;
	  //std::cerr << "Default setup needs " << TNum * ChNum /2 << "  processes to run."<< std::endl;
	  std::cerr << "Number of MPI processes should be a multiple of number of temperatures." << std::endl;
	  std::cerr << "Number of walkers has to be an even number." << std::endl;
	  std::cerr << "And number of MPI processes has to satisfy: " << std::endl;
	  std::cerr << "# Walkers * # Temps / (2.0 * # CPUs) = n or 1/n for an integer n." << std::endl;
	  std::cerr << "The minimum number of CPU's for the current configuration is: " <<  TNum << "." << std::endl;
	  std::cerr << "###############################" << std::endl;
	}
      return;
    }

  ////////// Splitting the MPI_COMM_WORLD communicator ///////////

  MPI_Comm E_COMM;   //E_COMM communicates between chains of the same temperature
  int E_size, E_rank;
  MPI_Comm T_COMM;   //T_COMM communicates between corresponding chains of different temperatures
  int T_size, T_rank;
  MPI_Comm L_COMM;   //L_COMM communicates between processors calculating a single likelihood instance in parallel 
  int L_size, L_rank;
  MPI_Comm C_COMM;   //C_COMM communicates between corresponding cpus associated with each walker
  int C_size, C_rank;

  int color, flavour, hue, saturation; // Parameters that determine to which new communicator each process will belong

 
  //Make new communicator for groups of CPU's
  //E_COMM communicates between chains of the same temperature
  //color = rank / ((ChNum/2)*LKLHD_Num);
  color = rank / (size/TNum);
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &E_COMM);
  MPI_Comm_rank(E_COMM, &E_rank);
  MPI_Comm_size(E_COMM, &E_size);  

  //T_COMM communicates between corresponding chains of different temperatures
  //flavour = rank % ((ChNum/2)*LKLHD_Num);
  flavour = rank % (size/TNum);
  MPI_Comm_split(MPI_COMM_WORLD, flavour, rank, &T_COMM);
  MPI_Comm_rank(T_COMM, &T_rank);
  MPI_Comm_size(T_COMM, &T_size);
  
  //L_COMM communicates between processors calculating a single likelihood instance in parallel 
  hue = E_rank / LKLHD_Num;
  MPI_Comm_split(E_COMM, hue, E_rank, &L_COMM);
  MPI_Comm_rank(L_COMM, &L_rank);
  MPI_Comm_size(L_COMM, &L_size); 
    
  //C_COMM communicates between corresponding cpus associated with each walker
  //  if(LKLHD_Num > 1 )    
  //{  
  saturation = E_rank % LKLHD_Num;
  MPI_Comm_split(E_COMM, saturation, E_rank, &C_COMM);
  MPI_Comm_rank(C_COMM, &C_rank);
  MPI_Comm_size(C_COMM, &C_size); 
  //}
  //else
  //{
  //MPI_Comm_dup(E_COMM, &C_COMM);
  //}
    
 
  //Set the likelihood commmunicator 
  _L.set_mpi_communicator(L_COMM);

  /*//SC_COMM Start-up communicator. Splits the cpu's to initialize the likelihoods
  //Communicates between different chains
  MPI_Comm SC_COMM;
  int SC_size, SC_rank;
  int scolor = rank % (ChNum/2);
  
  MPI_Comm_split(MPI_COMM_WORLD, scolor, rank, &SC_COMM);
  MPI_Comm_rank(SC_COMM, &SC_rank);
  MPI_Comm_size(SC_COMM, &SC_size); 

  //SL_COMM Start-up communicator. Splits the cpu's to initialize the likelihoods
  //Communicates between the processes calculating a single likelihood instance
  MPI_Comm SL_COMM;
  int SL_size, SL_rank;
  int sflavour = rank / (ChNum/2);
  
  MPI_Comm_split(MPI_COMM_WORLD, sflavour, rank, &SL_COMM);
  MPI_Comm_rank(SL_COMM, &SL_rank);
  MPI_Comm_size(SL_COMM, &SL_size); */


  //  std::cout << rank << "  " << SL_rank << "  " << SC_rank << std::endl; 
  for(int k =-1; k< size; k++)
    {
      if (k<0 && rank==0)
	std::cout << "W  E  T  L  C" << std::endl;
      if(rank == k)
	std::cout << rank << "  " << E_rank << "  " << T_rank << "  " << L_rank << "  " << C_rank << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
    }





  //Set hyperparmeters for adaptive temperature adjustments
  t0 = 1000.0/ChNum;
  nu = 100.0/ChNum;
  //Temperature and swap acceptance arrays
  double T[TNum], A[TNum - 1];
  
  //Initialize the temperatures
  if(temperatures.size() == 0)
    {
      double f = 5.0;
      T[0] = 1.0;
      for(int i = 1; i < TNum; ++i)
	T[i] = T[i-1] * f; 
    }
  else if (temperatures.size() != (size_t)TNum)
    {
      std::cerr << "The size of the temperature vector doesn't match the number of temperatures provided." << std::endl;
      return;
    }
  else
    {
      T[0] = 1.0;
      for(int i = 1; i < TNum; ++i)
	{
	  T[i] = temperatures[i];
	  //std::cout <<  "T[" << i << "]: " << T[i] << std::endl;
	}
    }

  //variable declarations
  int j, start_index = 0;
  double z, q, L, r;
  double preL[ChNum], neiL[ChNum];
  double chi2_vec[ChNum];
  double  swapL, neiC;
  double Y[dim];
  std::vector<double> pos(dim), chi2_arg(dim);
  double neiS[dim];
  double preS[ChNum][dim];

  //input/output streams
  std::ofstream out, out2, out3, ckpt, ckptbak;
  std::ifstream fin, ckptin;  

  //Set the likelihood commmunicator 
  //_L.set_mpi_communicator(S_COMM);

  //Initialize the chains 
  int start, sizeS, sizeL, end;
  //double LKLHD_value = 0.0;


  if(continue_flag) //Continue the chains from the last steps of provided chain file if continue_flag is set to True
    {
      if(rank == MASTER) //Only the master process should access the chain file
	{
	  double buff[ChNum][dim];
	  ckptin.open(_ckpt_file.c_str(), std::ios::in);
	  if(!ckptin.is_open())
	    {
	      std::cerr << "Checkpoint file doesn't exist!" << std::endl;
	      std::cerr << "Cannot restart the sampler." << std::endl;
	      return;

	    }

	  std::string line, word, dummy;
	  //int LineNum = 0; // Counter for the number of lines
	  //int WordNum = 0; //Counter of the words in the line

	  getline(ckptin, line);
	  std::istringstream iss(line);	      
	  iss >> word;
	  start_index = std::stoi(word.c_str());
	  // Read in the state vectors for walkers at T = 1
	  for(int n = 0; n < ChNum; n++)	    
	    {
	      getline(ckptin, line);                 	 
	      std::istringstream iss(line);	      
	      iss >> dummy;
	      iss >> dummy;
	      T[0] = 1;

	      for(int i = 0; i < dim; i++)		
		{
		  iss >> word;
		  preS[n][i] = std::stod(word.c_str());
		  //std::cout << preS[n][i] << "  ";
		}
	      //std::cout << std::endl;
	    }

	  // Read in the state vectors and temperatures for walkers at high Temperatures [T > 1] 
	  for(int t_cnt = 1; t_cnt < TNum; t_cnt ++)
	    {
	      
	      for(int n = 0; n < ChNum; n++)	    
		{
		  getline(ckptin, line);                 	 
		  std::istringstream iss(line);	      
		  iss >> word;
		  iss >> dummy;

		  // Set the temperatures
		  T[t_cnt] = std::stod(word.c_str());

		  for(int i = 0; i < dim; i++)		
		    {
		      iss >> word;
		      buff[n][i] = std::stod(word.c_str());
		      //std::cout << preS[n][i] << "  ";
		    }
		  //std::cout << std::endl;
		}
	      //MPI_Bcast(&preS[0][0], ChNum*dim, MPI_DOUBLE, 0, E_COMM);
	      MPI_Send(&buff[0][0], dim*ChNum, MPI_DOUBLE, t_cnt, t_cnt, T_COMM);
	    } 
	  ckptin.close();
	}
	  // Check wheather the chain does exist to use for initializing the position of walkers
	  /*std::ifstream fin(chain_file.c_str(), std::ios::in);
	  if(!fin.is_open())
	    {
	      std::cerr << "Chain file doesn't exist!" << std::endl;
	      std::cerr << "Cannot continue the MCMC chain." << std::endl;
	      return;
	    }

	  std::string line, word;
	  int LNum = 0; // Counter for the number of lines
	  while(getline(fin, line))	    
	    {                                                                                                                             
	      if(line[0] != '#') // Ignore the comment lines (starting with "#")
		{                          	      
		  std::istringstream iss(line);	      
		  for(int i = 0; i < dim; i++)		
		    {
		      iss >> word;
		      preS[LNum%ChNum][i] = std::stod(word.c_str()); 
		    }

		  LNum += 1;
		}
	    }
	  fin.close();*/

	  //Apply coordinate transformation on the parameters if we have to
	  /*for(int c = 0; c < ChNum; c++)
	    { 
	      for(int i = 0; i < dim; i++)
		{
		  pos[i] = preS[c][i];
		  if(_L.transform_state())
		    _L.forward_transform(pos);
		  preS[c][i] = pos[i];
		}
		}*/
     
      if((C_rank == 0) && (L_rank == 0) && (T_rank > 0))
	{
	  MPI_Recv(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, T_rank, T_COMM, &Stat);
	}
      MPI_Bcast(&start_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&T[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&preS[0][0], ChNum*dim, MPI_DOUBLE, 0, E_COMM);

      for(int c = 0; c < ChNum; c++)
	{ 
	  for(int i = 0; i < dim; i++)
	    {
	      pos[i] = preS[c][i];
	      if(_L.transform_state())
		_L.forward_transform(pos);
	      preS[c][i] = pos[i];
	    }
	}

      /*if((C_rank == 1) && (L_rank == 1))
	{
	  std::cout << T[T_rank] << std::endl;
	  //for(int n = 0; n < ChNum; ++n)
	  //std::cout << std::setw(3) << T_rank << "  " << std::setw(3) << C_rank << "  " << preS[n][0] << "  " << preS[n][1]<<std::endl;
	  }*/

       //MPI_Recv(&Temp, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);

      //return;
      // Broadcast the initial position of the walkers to all the processes 
      //MPI_Bcast(&preS[0][0], ChNum*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      //Parallelizing the likelihood initialization 
      


      // Calculate the likelihoods for the initial position of the walkers
      // This is done in two steps, half of the walkers at a time 
      //Split the start-up workload into chunks
      start = C_rank*(ChNum/2)/C_size;
      end = (C_rank+1)*(ChNum/2)/C_size;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;      

      for(int i  = start; i < end; ++i)
	{
	  for( j = 0; j < dim; ++j)
	    {
	      pos[j] = preS[i][j];
	    }
	  
	  preL[i] = _L(pos);
	  MPI_Bcast(&preL[i], 1, MPI_DOUBLE, 0, L_COMM);
	}
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);
      

      start = C_rank*(ChNum/2)/C_size+ChNum/2;
      end = (C_rank+1)*(ChNum/2)/C_size+ChNum/2;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;

      for(int i  = start; i < end; ++i)
	{
	  for( j = 0; j < dim; ++j)
	    {
	      pos[j] = preS[i][j];
	    }
	  
	  preL[i] = _L(pos);
	  MPI_Bcast(&preL[i], 1, MPI_DOUBLE, 0, L_COMM);
	}

      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);

      // Open the output files to append the new samples
      // e.g. the cahin file, likelihood file and chi-squared file
      if (rank == MASTER) {
	out.open(chain_file.c_str(), std::ios::app);
	out2.open(lklhd_file.c_str(), std::ios::app);
	out3.open(chi2_file.c_str(), std::ios::app);
	out.precision(output_precision);
	out2.precision(output_precision);
	out3.precision(output_precision);
	//out4.open("Tempering.d", std::ios::out);
      }
      // Open output files for high temperature chains
      if (verbosity == 1)
	{
	  if ((T_rank != MASTER) & (E_rank == 0)) 
	    {
	      out.open((chain_file+std::to_string(T_rank)).c_str(), std::ios::out);
	      out2.open((lklhd_file+std::to_string(T_rank)).c_str(), std::ios::out);
	      out.precision(output_precision);
	      out2.precision(output_precision);
	      //out3.open((chi2_file+std::to_string(T_rank)).c_str(), std::ios::out);
	      //out4.open("Tempering.d", std::ios::out);
	    }
	}
    }

  // Initialize the chains according to the specified means and ranges
  else{ 
    start = C_rank*(ChNum/2)/C_size;
    end = (C_rank+1)*(ChNum/2)/C_size;
    sizeS = dim*(ChNum/2)/C_size;
    sizeL = (ChNum/2)/C_size;      

 
    //std::cout << "test1: " << start << "   " << end << "  " << sizeL << std::endl;

    for(int i  = start; i < end; ++i)
    {
      do {
	for( j = 0; j < dim; ++j)
	{
	  //preS[i][j] = RndGaussian(mean+0.1*T_rank, std, true);
	  if(L_rank == 0)
	    preS[i][j] = RndGaussian(means[j], 0.5*ranges[j], false);

	  /*
	  for (int r=0; r<size; r++) {
	    if (rank==r) {
	      std::cout << '(' << i << ',' << j << ") "
			<< "Before: Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
			<< " L_rank: " << std::setw(3) << L_rank
			<< " C_rank: " << std::setw(3) << C_rank
			<< " T_rank: "<< std::setw(3) << T_rank
			<< " E_rank: " << std::setw(3) << E_rank
			<< " Position:  ";
	      for (int k = 0; k< dim; k++)
		std::cout << std::setw(15) << preS[i][k];
	      std::cout << std::endl;
	    }
	    MPI_Barrier(MPI_COMM_WORLD);
	  }
	  */	  
	  
	  MPI_Bcast(&preS[i][j], 1, MPI_DOUBLE, 0, L_COMM);

	  /*
	  for (int r=0; r<size; r++) {
	    if (rank==r) {
	      std::cout << '(' << i << ',' << j << ") "
			<< "After: Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
			<< " L_rank: " << std::setw(3) << L_rank
			<< " C_rank: " << std::setw(3) << C_rank
			<< " T_rank: "<< std::setw(3) << T_rank
			<< " E_rank: " << std::setw(3) << E_rank
			<< " Position:  ";
	      for (int k = 0; k< dim; k++)
		std::cout << std::setw(15) << preS[i][k];
	      std::cout << std::endl;
	    }
	    MPI_Barrier(MPI_COMM_WORLD);
	  }
	  */
	  
	  pos[j] = preS[i][j];
	  
	  
	}
	
	if(_L.transform_state())
	  _L.forward_transform(pos);
	for( j = 0; j < dim; ++j)
	  preS[i][j] = pos[j];
	
	
	//preL[i] = i; //_L(pos);
	preL[i] = _L(pos);

	MPI_Bcast(&preL[i], 1, MPI_DOUBLE, 0, L_COMM);

	/*
	for (int r=0; r<size; r++) {
	  if (rank==r) {
	    std::cout << "Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
		      << " L_rank: " << std::setw(3) << L_rank
		      << " C_rank: " << std::setw(3) << C_rank
		      << " T_rank: "<< std::setw(3) << T_rank
		      << " E_rank: " << std::setw(3) << E_rank
		      << "  L= " << std::setw(3) << preL[i] 
		      << " Chi2= " << std::setw(3) << 1.0 //_L.chi_squared(pos)
		      << " Position:  ";
	    for (int k = 0; k < dim; k++)
	      std::cout << std::setw(15) << pos[k];
	    std::cout << std::endl;
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	}
	*/
	
	//cout << preL[i] << endl;
      } while (preL[i]==-std::numeric_limits<double>::infinity());
    }
    MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
    MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);


    for (int r=0; r<size; r++) {
      if (rank==r) {
	//std::cout << "Midished Initializing rank " << std::setw(3) << rank
	//	  << " L_rank: " << std::setw(3) << L_rank
	//	  << " C_rank: " << std::setw(3) << C_rank
	//	  << " T_rank: "<< std::setw(3) << T_rank
	//	  << " E_rank: " << std::setw(3) << E_rank
	//	  << " preSL:  ";
	for (int i=0; i<ChNum; ++i) {
	  for (int k = 0; k < dim; k++)
	    std::cout << std::setw(15) << preS[i][k];
	  std::cout << std::setw(15) << preL[i];
	}
	std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }


    start = C_rank*(ChNum/2)/C_size+ChNum/2;
    end = (C_rank+1)*(ChNum/2)/C_size+ChNum/2;
    sizeS = dim*(ChNum/2)/C_size;
    sizeL = (ChNum/2)/C_size;
    //if (rank == 0)
    //std::cout << "C_size: " << C_size << "ChNum/2:  " << ChNum/2 << std::endl;
    //std::cout << "test2: " << start << "   " << end << "  " << sizeL << std::endl;

    for(int i  = start; i < end; ++i)
    {
      do {
	for( j = 0; j < dim; ++j)
	{
	  //preS[i][j] = RndGaussian(mean+0.1*T_rank, std, true);
	  if(L_rank == 0)
	    preS[i][j] = RndGaussian(means[j], 0.5*ranges[j], false);

	  /*
	  for (int r=0; r<size; r++) {
	    if (rank==r) {
	      std::cout << '(' << i << ',' << j << ") "
			<< "Before: Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
			<< " L_rank: " << std::setw(3) << L_rank
			<< " C_rank: " << std::setw(3) << C_rank
			<< " T_rank: "<< std::setw(3) << T_rank
			<< " E_rank: " << std::setw(3) << E_rank
			<< " Position:  ";
	      for (int k = 0; k< dim; k++)
		std::cout << std::setw(15) << preS[i][k];
	      std::cout << std::endl;
	    }
	    MPI_Barrier(MPI_COMM_WORLD);
	  }	  
	  */
	  
	  MPI_Bcast(&preS[i][j], 1, MPI_DOUBLE, 0, L_COMM);

	  /*
	  for (int r=0; r<size; r++) {
	    if (rank==r) {
	      std::cout << '(' << i << ',' << j << ") "
			<< "After: Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
			<< " L_rank: " << std::setw(3) << L_rank
			<< " C_rank: " << std::setw(3) << C_rank
			<< " T_rank: "<< std::setw(3) << T_rank
			<< " E_rank: " << std::setw(3) << E_rank
			<< " Position:  ";
	      for (int k = 0; k< dim; k++)
		std::cout << std::setw(15) << preS[i][k];
	      std::cout << std::endl;
	    }
	    MPI_Barrier(MPI_COMM_WORLD);
	  }
	  */
	  
	  pos[j] = preS[i][j];
	}
	  
	if(_L.transform_state())
	  _L.forward_transform(pos);
	for( j = 0; j < dim; ++j)
	  preS[i][j] = pos[j];
	
	//preL[i] = i; //_L(pos);
	preL[i] = _L(pos);

	MPI_Bcast(&preL[i], 1, MPI_DOUBLE, 0, L_COMM);

	/*
	for (int r=0; r<size; r++) {
	  if (rank==r) {
	    std::cout << "Initializing rank " << std::setw(3) << MPI::COMM_WORLD.Get_rank()
		      << " L_rank: " << std::setw(3) << L_rank
		      << " C_rank: " << std::setw(3) << C_rank
		      << " T_rank: "<< std::setw(3) << T_rank
		      << " E_rank: " << std::setw(3) << E_rank
		      << "  L= " << std::setw(3) << preL[i] 
		      << " Chi2= " << std::setw(3) << 1.0 //_L.chi_squared(pos)
		      << " Position:  ";
	    for (int k = 0; k < dim; k++)
	      std::cout << std::setw(15) << pos[k];
	    std::cout << std::endl;
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	}
	*/
	
	  //cout << preL[i] << endl;
      } while (preL[i]==-std::numeric_limits<double>::infinity());
    }
    //MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
    //MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);
    MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[ChNum/2][0], sizeS, MPI_DOUBLE, C_COMM);
    MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);

    
    for (int r=0; r<size; r++) {
      if (rank==r) {
	std::cout << "Finished Initializing rank " << std::setw(3) << rank
		  << " L_rank: " << std::setw(3) << L_rank
		  << " C_rank: " << std::setw(3) << C_rank
		  << " T_rank: "<< std::setw(3) << T_rank
		  << " E_rank: " << std::setw(3) << E_rank
		  << " preSL:  ";
	for (int i=0; i<ChNum; ++i) {
	  for (int k = 0; k < dim; k++)
	    std::cout << std::setw(15) << preS[i][k];
	  std::cout << std::setw(15) << preL[i];
	}
	std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }



    
    // Open the output files    
    if (rank == MASTER) {
      out.open(chain_file.c_str(), std::ios::out);
      out2.open(lklhd_file.c_str(), std::ios::out);
      out3.open(chi2_file.c_str(), std::ios::out);
      out.precision(output_precision);
      out2.precision(output_precision);
      out3.precision(output_precision);
      //out4.open("Tempering.d", std::ios::out);
    }
    
    // Open output files for high temperature chains
    if (verbosity == 1)
      {
	if ((T_rank != MASTER) & (E_rank == 0)) 
	  {
	    out.open((chain_file+std::to_string(T_rank)).c_str(), std::ios::out);
	    out << "# ";
	    for(size_t i = 0; i < var_names.size(); i++)
	      {
		out << var_names[i] << "  ";
	      }
	    out << std::endl;
	    out2.open((lklhd_file+std::to_string(T_rank)).c_str(), std::ios::out);
	    out.precision(output_precision);
	    out2.precision(output_precision);
	    //out2.open(lklhd_file.c_str(), std::ios::out);
	    //out3.open(chi2_file.c_str(), std::ios::out);
	    //out4.open("Tempering.d", std::ios::out);
	  }	
      }

    //If the varibale names are provided write them as the header for the chain file
    if((rank == MASTER) && (var_names.size() > 0))
      {
	out << "# ";
	for(size_t i = 0; i < var_names.size(); i++)
	  {
	    out << var_names[i] << "  ";
	  }
	out << std::endl;
      }
  }



  /*
  // AEB DEBUG
  for (int j=0; j<10; ++j)
  {
    std::cerr << "Rank " << rank << " finished: " << j << std::endl;
    std::cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI::COMM_WORLD.Barrier();
  }
  return;
  */

  /////////////////////// Main MCMC loop //////////////////////////
  //Run MCMC, loop over number of steps

  //Set the likelihood communicator for the actual chain
  //  _L.set_mpi_communicator(L_COMM);

  /*  if(rank == MASTER)
    {
      for(int k  = 0; k < ChNum ; ++k)
	std::cout << k << "  " << preS[k][0] << std::endl;
	}*/

  Themis::StopWatch sw;
  
  for(int i = start_index + 1; i < start_index + 1 + length; ++i)
    {
      std::stringstream swlblpre,swlblpost;
      swlblpre << "At MCMC step: " << i << " Temperature rank: " << T_rank << ".  Took ";
      swlblpost << " seconds since last step.";
      if (L_rank == 0 &&  C_rank == 0)
	sw.print_lap(std::cout,swlblpre.str(),swlblpost.str());

      //std::cout << "\r" << "At MCMC step:  " <<  i << "      " << std::endl; 

      //Specify the chains each CPU is responsible for
      //and the size of likelihhood array and state matrix
      start = C_rank*(ChNum/2)/C_size;
      end = (C_rank+1)*(ChNum/2)/C_size;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;
      
      for(int k  = start; k < end ; ++k)
	{
    
	  j = RndUnint(ChNum/2, ChNum - 1);
	  z = RndGz(2.0);

	  for(int n = 0; n < dim; ++n)
	    {
		Y[n] = preS[j][n] + z * (preS[k][n] - preS[j][n]);
	    }


	  MPI_Bcast(&Y[0], dim, MPI_DOUBLE, 0, L_COMM);

	  for(int n = 0; n < dim; ++n)
	    pos[n] = Y[n];
	  
	
	  /*if(L_rank == 0)
	    {
	      std::cout << "Trial step:  "<< rank << "  " << T_rank << "  " << E_rank << "  " << C_rank  << "  Position: ";
	      for(int n = 0; n < dim; ++n)
		std::cout << Y[n] << "  ";
	      std::cout << " L[" << k << "] = " << L << std::endl;
	    }*/
	  L = _L(pos);


	  //q = pow(z, dim - 1) * exp((-L + preL[k])/T[T_rank]);
	  q = log(pow(z, dim - 1)) + ((L - preL[k])/T[T_rank]);
	  r = log(RndUni(0.0, 1.0));

	  MPI_Bcast(&q, 1, MPI_DOUBLE, 0, L_COMM);
	  MPI_Bcast(&r, 1, MPI_DOUBLE, 0, L_COMM);

	  if( r <= q)
	    {
	      for(int m = 0; m < dim; ++m)
		{
		  preS[k][m] = Y[m];
		}
	      preL[k] = L;
	    }

	  // Calculate the chi2_values every "chi2_stride" steps
	  if(i%chi2_stride == 0)
	    {
	      for(int m = 0; m < dim; ++m)
		{
		  chi2_arg[m] = preS[k][m];
		}
	      //std::cout << "CP at step: " << i << " Rank:  "<< rank << " T_rank:  " << T_rank << " k: " << k
	      //<< " Parameters:  " << chi2_arg[0] << "  " << chi2_arg[1] << "  " << chi2_arg[2]
	      //<< " Pos:  " << pos[0] << "  " << pos[1] << "  " << pos[2] << std::endl;
	      
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
	      //chi2_vec[k] = _L.chi_squared(pos);
	    }
	  //else
	  //std::cout << "At step: " << i << " Rank:  "<< rank << " T_rank:  " << T_rank << " k: " << k << " j:  " << j << " L:  " << L << " preL:  "<< preL[k] << " chi2_vec: " << chi2_vec[k] << std::endl;
  
	}
      
      
      //Communicate the new likelihoods ans states between chains of the same temperature
      MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&chi2_vec[start], sizeL, MPI_DOUBLE, &chi2_vec[0], sizeL, MPI_DOUBLE, C_COMM);

      //Specify the chains each CPU is responsible for
      //and the size of likelihhood array and state matrix
      start = C_rank*(ChNum/2)/C_size+ChNum/2;
      end = (C_rank+1)*(ChNum/2)/C_size+ChNum/2;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;
      
      for(int k  = start; k < end ; ++k)
	{
	  
	  j = RndUnint(0, ChNum/2 - 1);
	  z = RndGz(2.0);

	  for(int n = 0; n < dim; ++n)
	    {
		Y[n] = preS[j][n] + z * (preS[k][n] - preS[j][n]);
	    }
	  MPI_Bcast(&Y[0], dim, MPI_DOUBLE, 0, L_COMM);

	  for(int n = 0; n < dim; ++n)
	    pos[n] = Y[n];

	  /*if(L_rank == 0)
	    {
	      std::cout << "Trial step:  "<< rank << "  " << T_rank << "  " << E_rank << "  " << C_rank  << "  Position: ";
	      for(int n = 0; n < dim; ++n)
		std::cout << Y[n] << "  ";
	      std::cout << " L[" << k << "] = " << L << std::endl;
	      }*/	
	  
	  L = _L(pos);

	  
	  //q = pow(z, dim - 1) * exp((-L + preL[k])/T[T_rank]);
	  q = log(pow(z, dim - 1)) + ((L - preL[k])/T[T_rank]);
	  r = log(RndUni(0.0, 1.0));

	  MPI_Bcast(&q, 1, MPI_DOUBLE, 0, L_COMM);
	  MPI_Bcast(&r, 1, MPI_DOUBLE, 0, L_COMM);

	  if( r <= q)
	    {
	      for(int m = 0; m < dim; ++m)
		{
		  preS[k][m] = Y[m];
		}
	      preL[k] = L;
	    }

	  // Calculate the chi2_values every "chi2_stride" steps
	  if(i%chi2_stride == 0)
	    {
	      for(int m = 0; m < dim; ++m)
		{
		  chi2_arg[m] = preS[k][m];
		}
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
	    }
	  //else
	  //cout << "B:  "<< rank << "  " << T_rank << "  " << k << "  " << j <<  "  " << L << "  "<< preL[k] <<endl;

	}

      //Communicate the new likelihoods ans states between chains of the same temperature
      MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[ChNum/2][0], sizeS, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&chi2_vec[start], sizeL, MPI_DOUBLE, &chi2_vec[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);

      if( (i % stride == stride - 1))
	{
	  //if( (T_rank > 0) && (E_rank == 0))
	    
	  //Communicate likelihoods between adjacent temperature chains
	  if( (T_rank < T_size - 1) && (C_rank == 0))
	    {
	      //MPI_Isend(&preL[0], ChNum, MPI_DOUBLE, T_rank + 1, 0, T_COMM, &Req);
	      MPI_Send(&preL[0], ChNum, MPI_DOUBLE, T_rank + 1, 0, T_COMM);
	    }
	  
	  if( (T_rank > 0) && (C_rank == 0))
	    {
	      MPI_Recv(&neiL[0], ChNum, MPI_DOUBLE, T_rank - 1, 0, T_COMM, &Stat);
	    }
	  //MPI_Wait(&Req, &Stat);

      
      //Decide if we need to swap the states and communicate back
      //to the adjacent temperature chain
	  int swap[ChNum], nswap[ChNum];
	  for (int i2 = 0; i2 < ChNum; ++i2)
	    {
	      nswap[i2] = 0;
	      swap[i2] = 0;
	    }
	  //std::fill_n(swap, ChNum, 2);
	  //std::fill_n(nswap, ChNum, 2);
	  if((T_rank > 0) && (C_rank == 0) && (L_rank == 0))
	    {
	      for(int m = 0; m < ChNum; ++m)
		{
		  double alpha = log(RndUni(0.0, 1.0));
		  //double beta = std::min(1.0, pow(exp(-(neiL[m] - preL[m])) ,(1.0/T[T_rank] - 1.0/T[T_rank-1])));
		  //double beta = std::min(1.0, pow(exp((neiL[m] - preL[m])) ,(1.0/T[T_rank] - 1.0/T[T_rank-1])));
		  double beta = std::min(0.0 , (neiL[m] - preL[m]) * (1.0/T[T_rank] - 1.0/T[T_rank-1]));

		  if(alpha < beta)
		    swap[m] =  1;
	       
		}
	      MPI_Send(&swap[0], ChNum, MPI_INT, T_rank - 1, 1, T_COMM);
	      
	    }
	  if((T_rank < T_size - 1) && (C_rank == 0) && (L_rank == 0))
	    {
	      MPI_Recv(&nswap[0], ChNum, MPI_INT, T_rank + 1, 1, T_COMM, &Stat); 
	    }

	  
	  //if(i == 799 && E_rank == 0)
	    //for(int m = 0; m < ChNum; ++m)
	      //cout << T_rank  << "  " << swap[m] << "  " << nswap[m] << "  " << m << endl; 
	  
      
	  //Now that it turns out we want to swap states let's actually 
	  //swap states between adjacent temperature chains
	  if( (T_rank > 0) && (C_rank == 0) && (L_rank == 0))
	    {
	      A[T_rank - 1] = 0.0;  
	      for(int m = 0; m < ChNum; ++m)
		{
		  if((swap[m]) == 1)
		    {
		      A[T_rank - 1] += 1.0;
		      //cout << "Swapped!" << endl;
		      MPI_Isend(&preS[m][0], dim, MPI_DOUBLE, T_rank - 1, m, T_COMM, &Req);
		      MPI_Recv(&neiS[0], dim, MPI_DOUBLE, T_rank - 1, m, T_COMM, &Stat);
		      MPI_Wait(&Req, &Stat);

		      MPI_Isend(&preL[m], 1, MPI_DOUBLE, T_rank - 1, m+ChNum, T_COMM, &Req2);
		      MPI_Recv(&swapL, 1, MPI_DOUBLE, T_rank - 1, m+ChNum, T_COMM, &Stat2);
		      MPI_Wait(&Req2, &Stat2);

		      MPI_Isend(&chi2_vec[m], 1, MPI_DOUBLE, T_rank - 1, m+2*ChNum, T_COMM, &Req3);
		      MPI_Recv(&neiC, 1, MPI_DOUBLE, T_rank - 1, m+2*ChNum, T_COMM, &Stat3);
		      MPI_Wait(&Req3, &Stat3);


		      preL[m] = swapL;
		      chi2_vec[m] = neiC;
		      for(int o = 0; o < dim; ++o)
			preS[m][o] = neiS[o];
		    }
		}
	      A[T_rank - 1] /= ChNum;
	      //MPI_Isend(&A[T_rank - 1], 1, MPI_DOUBLE, T_rank - 1, 20000, T_COMM, &Req);
	      MPI_Send(&A[T_rank - 1], 1, MPI_DOUBLE, T_rank - 1, ChNum*3, T_COMM);
	      //cout << A[T_rank - 1] << endl;
	    }
	  //if((E_rank == 0) && (T_rank > 0))
	    //{
	      //cout << A[T_rank - 1] << endl;
	    //MPI_Bcast(&A[T_rank - 1], 1, MPI_DOUBLE, T_rank, T_COMM );
	    //}

	  
	  if( (T_rank < T_size - 1) && (C_rank == 0) && (L_rank == 0))
	    {
	      for(int m = 0; m < ChNum; ++m)
		{
		  //if((nswap[m] == 1))
		  if((nswap[m]) == 1)
		    {
		      MPI_Recv(&neiS[0], dim, MPI_DOUBLE, T_rank + 1, m, T_COMM, &Stat);
		      MPI_Send(&preS[m][0], dim, MPI_DOUBLE, T_rank + 1, m, T_COMM);


		      MPI_Recv(&swapL, 1, MPI_DOUBLE, T_rank + 1, m+ChNum, T_COMM, &Stat2);
		      MPI_Send(&preL[m], 1, MPI_DOUBLE, T_rank + 1, m+ChNum, T_COMM);


		      MPI_Recv(&neiC, 1, MPI_DOUBLE, T_rank + 1, m+2*ChNum, T_COMM, &Stat3);
		      MPI_Send(&chi2_vec[m], 1, MPI_DOUBLE, T_rank + 1, m+2*ChNum, T_COMM);

		      for(int o = 0; o < dim; ++o)
			preS[m][o] = neiS[o];
		      preL[m] = swapL;
		      chi2_vec[m] = neiC;
	
		    }
		}
	      MPI_Recv(&A[T_rank], 1, MPI_DOUBLE, T_rank + 1, ChNum*3, T_COMM, &Stat4);
	    }
	  
	  //Broadcast the new state to all chains within the same temperature  
	  //MPI_Bcast(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, C_COMM);
	  //MPI_Bcast(&preL[0], ChNum, MPI_DOUBLE, 0, C_COMM);
	  MPI_Bcast(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, E_COMM);
	  MPI_Bcast(&preL[0], ChNum, MPI_DOUBLE, 0, E_COMM);

	  //std::cout << T[T_rank] << "  " << A[T_rank] <<  "  " << A[T_rank-1] << "  " << T_rank <<  std::endl;
	  //if(E_rank == 0)
	  //MPI_Bcast(&A[T_rank - 1], 1, MPI_DOUBLE, 2, T_COMM );
	  //cout << "After sync:   " << A[T_rank -1] <<  "  " << A[T_rank] <<  "  " << T_rank<<endl;
	  //Update the temperatures
	  //if((T_rank > 0) && (C_rank == 0) && (T_rank < TNum - 1))
	  if((T_rank > 0) && (C_rank == 0) && (L_rank == 0))
	    {
	      if (T_rank == TNum - 1)
		A[T_rank] = 0.5;

	      std::cout  << "Tempering acceptance rate:  " << A[T_rank - 1] << " Temperature:  " << T[T_rank] << std::endl;
	      //if(L_rank == 0)
	      //out4 << A[T_rank - 1] << "  " << T[T_rank] << std::endl;

	      double k = (t0/((i/stride)+t0))/nu;
	      double S;
	      S = log(T[T_rank] - T[T_rank-1]);
	      S += k * (A[T_rank-1] - A[T_rank]);
	      if(adaptive_temperature)
		T[T_rank] =  T[T_rank-1] + exp(S);
	      //cout <<"Afetr sync:   " <<A[T_rank - 1] <<"  " <<  A[T_rank]<< "  " << T[T_rank] << endl;
	      //Broadcast the new temperature
	      //cout << T_rank << " " << exp(S)  << "  " << k << "  " << T[T_rank] << endl;
	      // cout << T[T_rank] << endl;
	      
	    }
	  MPI_Bcast(&T[T_rank], 1, MPI_DOUBLE, 0, C_COMM );
	  //cout << i << "  " << E_rank << "  " << T_rank << "  " << T[T_rank] << endl;
	  
	  //if(Tndx == 0)
	  //cout << rank << "  " <<A[T_rank-1] << "  " << T[T_rank] << endl;
	  //std::cout << T_rank << "  " << T[T_rank] << std::endl;
	  
	}
      
      //Write the new step to the file
      if(rank == MASTER)
	{
	  //write the parameters and likelihoods to output files
	  for(int k = 0; k < ChNum; ++k)
	    {
	      for(int n = 0; n < dim; ++n)
		{
		  out << preS[k][n] << "  ";
		  
		}
	      out << std::endl;
	      out2 << preL[k] << " " ;
	    }
	  out2 << std::endl; 

	  //write the chi2 values to output file
	  if(i%chi2_stride == 0)
	    {
	      for(int n = 0; n < ChNum; ++n)
		{
		  out3 << chi2_vec[n] << "  ";
		}
	      out3 << std::endl;
	    }       
	   
	}



      //Write the new step to the file
      if(verbosity == 1)
	{
	  if((T_rank != MASTER) && (E_rank == 0))
	    {
	      //write the parameters and likelihoods to output files
	      for(int k = 0; k < ChNum; ++k)
		{
		  for(int n = 0; n < dim; ++n)
		    {
		      out << preS[k][n] << "  ";
		      
		    }
		  out << std::endl;
		  out2 << preL[k] << " " ;
		}
	      out2 << std::endl; 
	    }
	}


      //////////////////////// checkpoint generation //////////////////
      // Write the state vectors (for all temperatures) to the checkpoint file        
      if(i%_ckpt_stride == _ckpt_stride - 1)
	{
	  double buff[ChNum][dim];
	  double Temp;

	  // Master writes the information for the lowet temperature [T = 1] to the checkpoint
	  if (rank == MASTER)
	    {
	      ckpt.open(_ckpt_file.c_str(), std::ios::out);
	      ckpt.precision(output_precision);
	      ckpt << i << std::endl;
	      for(int n_c = 0; n_c < ChNum; ++n_c)
		{
		  ckpt << T[0] << "  " << n_c << "  ";
		  for(int n_d = 0; n_d < dim; ++n_d) 
		    ckpt << preS[n_c][n_d] << "  ";      
		  ckpt << std::endl;
		}
	      
	    }
	  // Master collects the information form high temperature walkers and writes to the checkpoint
	  for(int n_t = 1; n_t < T_size; n_t++)
	    {
	      
	      if ((T_rank == n_t) && (C_rank == 0) && (L_rank == 0))
		{
		  MPI_Send(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, n_t, T_COMM);
		  MPI_Send(&T[n_t], 1, MPI_DOUBLE, 0, T_size+n_t, T_COMM);
		}
	     
		if(rank == MASTER)
		  {
		    MPI_Recv(&buff[0][0], dim*ChNum, MPI_DOUBLE, n_t, n_t, T_COMM, &Stat);
		    MPI_Recv(&Temp, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);
		    for(int n_c = 0; n_c < ChNum; ++n_c)
		      {
			ckpt << Temp << "  " << n_c << "  ";
			for(int n_d = 0; n_d < dim; ++n_d) 
			  ckpt << buff[n_c][n_d] << "  ";      
			ckpt << std::endl;
		      }
		  }
		MPI_Barrier(T_COMM);
	    }
	 
	  // Make a back-up copy of the checkpoint file
	  // This is for the unlikely scenario where the system is interrupted during writing of
	  // the checkpoint file
	  if(rank == MASTER)
	    {
	      ckpt.close();
	      ckptin.open(_ckpt_file.c_str(), std::ios::in);
	      ckptbak.open((_ckpt_file+".bak").c_str(), std::ios::out);
	      ckptbak.precision(output_precision);
	      std::string line;
	      while(getline(ckptin, line))
		{
		  ckptbak << line << std::endl;
		}
	      ckptin.close();
	      ckptbak.close();
	    }
	}
	 

	  
 

    }
  // Master process closes the output files
  if (rank == MASTER) 
    {
      out.close();
      out2.close();
      out3.close();
      //out4.close();
    }
  
  // Marks the communicator objects for deallocation
  MPI_Comm_free(&E_COMM);
  MPI_Comm_free(&C_COMM);
  MPI_Comm_free(&L_COMM);
  MPI_Comm_free(&T_COMM);

  // Unset the communicator for the likelihood
  _L.set_mpi_communicator(MPI_COMM_WORLD);
  
  }
 


/*// Generate random numbers with uniform distribution
double sampler_affine_invariant_tempered_MCMC::RndUni(double a, double b)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);
  return dis(gen);
}

// Generate integer random numbers with uniform distribution
int sampler_affine_invariant_tempered_MCMC::RndUnint(int a, int b)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(a, b);
  return dis(gen);
}


// Generate random numbers with g(z) distribution function
//g(z) = 1/sqrt(z) for 1/a<z<a
double sampler_affine_invariant_tempered_MCMC::RndGz(double a)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1/a, a);
  bool next = true;
  double z;
  do{
  z = dis(gen);
  std::uniform_real_distribution<> dis2(0.0, sqrt(a));  
  if(dis2(gen) < 1/sqrt(z))
    next = false;
  }while(next);
  return z;
}

// Generate random numbers with normal distribution using Marsaglia polar method
double sampler_affine_invariant_tempered_MCMC::RndGaussian(double mean, double stDev, bool CONTINUE)
{
    static int count = 0;
    static double nextGaussianVal;
    double firstGaussianVal, v1, v2, s;

    if ( (count == 0) || (CONTINUE == false) ) {
       do { 
	 v1 = (2.0 * (double)rand())/(1.0 + RAND_MAX) - 1;   // between -1.0 and 1.0
	 v2 = (2.0 * (double)rand())/(1.0 + RAND_MAX) - 1;   // between -1.0 and 1.0
         s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);
        double multiplier = sqrt(-2 * log(s)/s );
        nextGaussianVal = mean + stDev * v2 * multiplier;
        firstGaussianVal = mean + stDev * v1 * multiplier;
        count = 1;
        return firstGaussianVal;
    }

    count = 0;
    return nextGaussianVal;
    }*/

// Generate random numbers with uniform distribution
double sampler_affine_invariant_tempered_MCMC::RndUni(double a, double b)
{
  return (b-a)*_rng.rand()+a;

  /*
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);
  return dis(gen);
  */
}

// Generate integer random numbers with uniform distribution
int sampler_affine_invariant_tempered_MCMC::RndUnint(int a, int b)
{
  return int( (b-a)*_rng.rand()+a );

  /*  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(a, b);
  return dis(gen);
  */
}


// Generate random numbers with g(z) distribution function
//g(z) = 1/sqrt(z) for 1/a<z<a
double sampler_affine_invariant_tempered_MCMC::RndGz(double a)
{
  double y = (std::sqrt(a)-std::sqrt(1.0/a))*_rng.rand() + std::sqrt(1.0/a);
  return ( y*y );

  /*
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1/a, a);
  bool next = true;
  double z;
  do{
  z = dis(gen);
  std::uniform_real_distribution<> dis2(0.0, sqrt(a));  
  if(dis2(gen) < 1/sqrt(z))
    next = false;
  }while(next);
  return z;
  */
}

// Generate random numbers with normal distribution using Marsaglia polar method
double sampler_affine_invariant_tempered_MCMC::RndGaussian(double mean, double stDev, bool CONTINUE)
{
  return (stDev*_grng.rand()+mean);

  /*
    static int count = 0;
    static double nextGaussianVal;
    double firstGaussianVal, v1, v2, s;

    if ( (count == 0) || (CONTINUE == false) ) {
       do { 
	 v1 = (2.0 * (double)rand())/(1.0 + RAND_MAX) - 1;   // between -1.0 and 1.0
	 v2 = (2.0 * (double)rand())/(1.0 + RAND_MAX) - 1;   // between -1.0 and 1.0
         s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);
        double multiplier = sqrt(-2 * log(s)/s );
        nextGaussianVal = mean + stDev * v2 * multiplier;
        firstGaussianVal = mean + stDev * v1 * multiplier;
        count = 1;
        return firstGaussianVal;
    }

    count = 0;
    return nextGaussianVal;
  */
}






  /*  
// Box-Muller Gaussian Random Variables
template<class T>
GaussianRandomNumberGenerator<T>::GaussianRandomNumberGenerator(int seed)
: _compute_new(true), _rng(seed)
{
}
template<class T>
double GaussianRandomNumberGenerator<T>::rand()
{
  if (_compute_new)
  {
    _compute_new = false;
    double x1, x2;
    x1 = std::sqrt( -2.0*std::log(_rng.rand()) );
    x2 = 2.0*M_PI*_rng.rand();

    _r1 = x1*cos(x2);
    _r2 = x1*sin(x2);

    return _r1;
  }
  else
  {
    _compute_new = true;

    return _r2;
  }
}

// NR Rejection for generating Poisson Random Variables
template<class T>
PoissonRandomNumberGenerator<T>::PoissonRandomNumberGenerator(int seed, double mean)
  : _rng(seed), _mean(mean)
{
  if (_mean<12.0)
  {
    _g = std::exp(-_mean);
  }
  else
  {
    _sq = std::sqrt(2.0*_mean);
    _alxm = std::log(_mean);
    _g = _mean*_alxm-gammln(_mean+1.0);
  }
}
template<class T>
double PoissonRandomNumberGenerator<T>::rand()
{
  static double sq,alxm;
  double em,t,y;
  
  if (_mean < 12.0)
  {
    em = -1;
    t=1.0;
    do {
      ++em;
      t *= _rng.rand();
    } while (t > _g);
  }
  else
  {
    do {
      do {
	y=std::tan(M_PI*_rng.rand());
	em=_sq*y+_mean;
      } while (em < 0.0);
      em=std::floor(em);
      t=0.9*(1.0+y*y)*std::exp(em*_alxm-gammln(em+1.0)-_g);
    } while (_rng.rand() > t);
  }

  return em;
}
template<class T>
double PoissonRandomNumberGenerator<T>::gammln(double xx)
{
  double x,y,tmp,ser;
  static double cof[6]={76.18009172947146,-86.50532032941677,
			24.01409824083091,-1.231739572450155,
			0.1208650973866179e-2,-0.5395239384953e-5};
  int j;
  
  y=x=xx;
  tmp=x+5.5;
  tmp -= (x+0.5)*std::log(tmp);
  ser=1.000000000190015;

  for (j=0;j<=5;j++)
    ser += cof[j]/++y;

  return -tmp+log(2.5066282746310005*ser/x);
}
template<class T>
double PoissonRandomNumberGenerator<T>::factrl(int n)
{
  static int ntop=4;
  static double a[33]={1.0,1.0,2.0,6.0,24.0};
  int j;
  
  if (n > 32)
    return std::exp(gammln(n+1.0));
  
  while (ntop<n) {
    j=ntop++;
    a[ntop]=a[j]*ntop;
  }
  return a[n];
}
template<class T>
double PoissonRandomNumberGenerator<T>::prob(int n)
{
  return std::pow(_mean,n)*std::exp(-_mean)/factrl(n);
}
  */  


std::vector<double> sampler_affine_invariant_tempered_MCMC::find_best_fit(std::string chain_file, std::string lklhd_file)
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<double> parameters;
  
  if (world_rank==0)
  {
    // Read likelihood file
    std::fstream lin(lklhd_file.c_str());
    if (!lin.is_open())
    {
      std::cerr << "sampler_differential_evolution_tempered_MCMC: Can't open likelihood file " << lklhd_file << std::endl;
      std::exit(1);
    }
    double L, Lmax;
    int indmax=0;
    lin >> Lmax >> L;
    for (int index=1; !lin.eof(); ++index)
    {
      if (L>Lmax)
      {
	indmax=index;
	Lmax = L;
      }
      lin >> L;
    }
    lin.close();
    
    std::fstream cin(chain_file.c_str());
    if (!cin.is_open())
    {
      std::cerr << "sampler_differential_evolution_tempered_MCMC: Can't open chain file " << chain_file << std::endl;
      std::exit(1);
    }
    std::string parameter_line;
    for (int index=0; index<=indmax; ++index)
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

  
};


