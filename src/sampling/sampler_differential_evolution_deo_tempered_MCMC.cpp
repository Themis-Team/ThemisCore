/*!
  \file sampler_differential_evolution_deo_tempered_MCMC.cpp
  \author Paul Tiede
  \brief Implementation file for the sampler_differential_evolution_deo_tempered_MCMC class
*/


#include "sampler_differential_evolution_deo_tempered_MCMC.h"
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

  sampler_differential_evolution_deo_tempered_MCMC::sampler_differential_evolution_deo_tempered_MCMC(int seed)
    : _nrounds(1), _b(2), _initial_spacing(2.0), _rng(seed), _grng(int(RAND_MAX*_rng.rand()))
  {
    // Use the default cpu distributio
    default_cpu_distribution = true;
    _no_initial_ladder = true;
    _ckpt_file = "sampler.ckpt";
    _ckpt_stride = 10;

  }

  sampler_differential_evolution_deo_tempered_MCMC::~sampler_differential_evolution_deo_tempered_MCMC()
  {    
  }
  
  // Function to set the cpu distribution for MPI parallelization 
  void sampler_differential_evolution_deo_tempered_MCMC::set_cpu_distribution(int num_temperatures, int num_walkers, int num_likelihood)
  {
    if (num_temperatures < 1)
    {
      std::cerr << "Must have at least one replica in tempering, i.e. num_replicas >= 1\n";
      std::exit(1);
    }
    TNum = num_temperatures;
    ChNum = num_walkers;
    LKLHD_Num = num_likelihood;
    default_cpu_distribution = false;
  }

  // Function to set the tempering schedule
  void sampler_differential_evolution_deo_tempered_MCMC::set_annealing_schedule(int num_rounds, int geometric_increase, double initial_spacing)
  {
    if (num_rounds < 1 || geometric_increase < 1)
    {
      std::cerr << "set_annealing_schedule: Number of round and geometric increase must be greater than 1\n";
      std::exit(1);
    }
    _nrounds = num_rounds;
    _b = geometric_increase;
    _initial_spacing = initial_spacing;
  }

  // Function to set checkpointing options
  void sampler_differential_evolution_deo_tempered_MCMC::set_checkpoint(int ckpt_stride, std::string ckpt_file)
  {
    _ckpt_stride = ckpt_stride;
    _ckpt_file = ckpt_file;
    
  }



  // Function to calculate the Bayesian evidence
  void sampler_differential_evolution_deo_tempered_MCMC::estimate_bayesian_evidence(std::vector<std::string> file_names, std::vector<double> temperatures, int burn_in)
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
		std::cerr << "Likelihood file doesn't exist. Cannot calculate the Bayesian evidence." << std::endl;
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
  
  double sampler_differential_evolution_deo_tempered_MCMC::find_beta(Interpolator1D& fLambda, int k, double Lambda, double eps, int MAX_ITR)
  {
    int i = 0;
    double N = TNum-1;
    double betaMax = 1.0;
    double betaMin = 0.0;
    double c = 0.0;
    double fMax = fLambda(betaMax) - k/N*Lambda;
    double fMin = fLambda(betaMin) - k/N*Lambda;

    do{
      i++;
      double error = (betaMax-betaMin)/2;
      c = (betaMax+betaMin)/2.0;
      if (c/1000 < eps)
	eps /= 10.0;
      double fMid = fLambda(c) - k/N*Lambda;
      if (error < eps || fMid==0.0)
        return c;


      if (fMid*fMax > 0 )
      {
        betaMax = c;
        fMax = fMid;
      }
      else if (fMid*fMin>0)
      {
        betaMin = c;
        fMin = fMid; 
      }
    }while(i < MAX_ITR);

    std::cerr << "Maximum iterations reached in find_temp beta might be wrong\n";
    return c;
  
  }

  void sampler_differential_evolution_deo_tempered_MCMC::read_initial_ladder(std::string annealing_file)
  {
    _old_annealing_file = annealing_file;
    _no_initial_ladder = false;
  }

  double sampler_differential_evolution_deo_tempered_MCMC::update_annealing_params(double *beta, double R[])
  {
    std::vector<double> Lambda(TNum, 0.0);
    std::vector<double> betaRev(TNum, 0.0);
    for ( int i = 0; i < TNum; ++i )
    {
      betaRev[i] = beta[TNum-1-i];
      for (int j = 0; j < i; ++j)
        Lambda[i] += R[TNum-1-j-1];
      //std::cerr << "adaption:    " << betaRev[i] << "    " << R[TNum-1-i] << "     " << Lambda[i] << std::endl;
    }


    Interpolator1D fLambda(betaRev,Lambda, "mcubic");
    for ( int k = 1; k < TNum-1; ++k)
    {
      beta[TNum-1-k] = find_beta(fLambda, k, Lambda[TNum-1]);  
      //std::cerr << "Found "<< k << " beta " << beta[TNum-1-k] << std::endl;
    }
    beta[0] = 1.0;
    beta[TNum-1] = 0.0;
    return 1.0/(2.0*(1.0+Lambda[TNum-1]));
  }


  //Parallel tempered ensemble sampler MCMC routine
  void sampler_differential_evolution_deo_tempered_MCMC::run_sampler(likelihood _L, 
								 int start_length, int thin_stride, int temp_stride, int chi2_stride, 
								 std::string chain_file, std::string lklhd_file,
								 std::string chi2_file, std::string annealing_file,
                                                                 std::vector<double> means, 
								 std::vector<double> ranges, std::vector<std::string> var_names,
								 bool continue_flag, int output_precision, int verbosity, std::vector<double> inverse_temperatures)
  {
    if (thin_stride==0){
      thin_stride = 1;
    }

    if (temp_stride > start_length){
      std::cerr << "ERROR: The swapping stride must be smaller than the number of steps in the initial round!\n";
      std::cerr << "Please change this in the run_sampler command\n";
      std::exit(1);
    }
    
    
    // Number of dimensions in the parameter space to be explored
    dim = means.size();

    
    // Number of steps after which different temperatures communicate
    int stride = temp_stride;
    
    int size, rank;
    MPI_Status Stat, Stat2, Stat3;
    MPI_Request Req, Req2, Req3;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    /*
    // Start the tempering level output stream for debugging AEB
    std::stringstream myname;
    myname << "tempering_info_" << std::setfill('0') << std::setw(4) << rank << ".txt";
    std::ofstream tempout(myname.str().c_str());
    */

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

    if (rank==0)
      std::cout << "Starting length is " << start_length << std::endl
                << "and we are using " << _nrounds << " round\n"
                << "with a geometric increase factor of " << _b << std::endl
                << "meaning the last round will have " <<  start_length*std::pow(_b,(_nrounds-1)) 
                << " steps so plan accordingly! \n";

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
    
    /*
    MPI_Comm E_COMM;   //E_COMM communicates between chains of the same temperature
    int E_size, E_rank;
    MPI_Comm T_COMM;   //T_COMM communicates between corresponding chains of different temperatures
    int T_size, T_rank;
    MPI_Comm L_COMM;   //L_COMM communicates between processors calculating a single likelihood instance in parallel 
    int L_size, L_rank;
    MPI_Comm C_COMM;   //C_COMM communicates between corresponding cpus associated with each walker
    int C_size, C_rank;
    */


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
      if (rank == k)
	std::cout << rank << "  " << E_rank << "  " << T_rank << "  " << L_rank << "  " << C_rank << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
    }

    
    //Temperature and swap rejection arrays
    double beta[TNum], R[TNum];
    for ( int i = 0; i < TNum; ++i )
      R[i] = 0.0;

    /*
    // AEB 
    tempout << "t0= " << t0 << "  nu= " << nu << std::endl;
    tempout << "report: " << rank << std::endl;
    */

    
    //Initialize the temperatures
    if(inverse_temperatures.size() == 0 && _no_initial_ladder){
      beta[0] = 1.0;
      for(int it = 1; it < TNum; ++it)
	beta[it] = beta[it-1]/_initial_spacing; 
    }else if (inverse_temperatures.size() != (size_t)TNum && _no_initial_ladder){
      std::cerr << "The size of the inverse_temperatures vector doesn't match the number of temperatures provided." << std::endl;
      return;
    }else if (_no_initial_ladder){
      beta[0] = 1.0;
      for(int it = 1; it < TNum; ++it)
      {
	beta[it] = inverse_temperatures[it];
	//std::cout <<  "T[" << i << "]: " << T[i] << std::endl;
      }
    }else{
      if (rank == MASTER){
        std::ifstream ain;
	ain.open(_old_annealing_file.c_str(), std::ios::in);
	if(!ain.is_open()){
	  std::cerr << "Annealing file " << _old_annealing_file << " doesn't exist!" << std::endl;
	  std::cerr << "Cannot restart the sampler." << std::endl;
	  return;
	}
        //Read in the annealing file
        int ntemp;
        std::string line, word, dummy;
        std::getline(ain, line);
        std::istringstream iss(line);
        iss >> word;
        iss >> word;
        iss >> word;
        iss >> word;
        ntemp = std::stoi(word.c_str());
        std::cout << "Reading initial tempering ladder from: " << annealing_file << std::endl;
        if (ntemp != TNum){
          std::cerr << "Annealing file must have the same number of temperatures as requested!\n"
                    << "Expected Temp = " << TNum << std::endl
                    << "File Temps = " << ntemp << std::endl;
          std::exit(1);
        }
        std::getline(ain,line);
        int roundd = 0;
        int irt = 0;
        while (std::getline(ain, line)){
          std::stringstream iss(line);
          iss >> word;
          int newround = std::stoi(word.c_str());
          iss >> word;
          double beta_read = std::stod(word.c_str());
          iss >> word;
          if (roundd == newround){
            beta[irt] = beta_read;
            irt++;
          }else{
            roundd = newround;
            irt = 0;
            beta[irt] = beta_read;
            irt = 1;
          }
        }
        ain.close();

      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    beta[TNum-1]=0.0;
    MPI_Bcast(&beta[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    /*
    std::ofstream tempout("deo_foo");
    if (rank==0) {
      tempout << "Start beta values:\n";
      for (int j=0; j<TNum; ++j)
	tempout << std::setw(15) << beta[j];
      tempout << std::endl;
    }
    */

    //variable declarations
    int j, j2, start_index = 0, start_round=0;
    double z, q, L, r;
    double preL[ChNum], neiL[ChNum]; //Stores the old and new likelhood value for all the chains for a give temp level.
    double chi2_vec[ChNum]; //Stores the chi2 for a temp level
    double  swapL, neiC;
    double Y[dim]; //Stores the proposed point according to the differential evolutions step proposal.
    std::vector<double> pos(dim), chi2_arg(dim);
    double neiS[dim];
    double preS[ChNum][dim]; //Stores the positions of each chain at a temperature for all the walkers of that temp

    //input/output streams
    //out = chain; out2 = lklhd; out3 = chi2; out4 = annealing, out5 = annealing_summary;
    std::ofstream out, out2, out3,out4,out5, ckpt, ckptbak;
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
	start_round = std::stoi(word.c_str());
        iss >> word;
        start_index = std::stoi(word.c_str());
	
	// Read in the state vectors for walkers at T = 1
	for(int n = 0; n < ChNum; n++)	    
	{
	  getline(ckptin, line);                 	 
	  std::istringstream iss(line);	      
	  iss >> dummy;
	  iss >> dummy;
	  beta[0] = 1;
          R[0] = std::stod(dummy.c_str());
          iss >> dummy;
          
	  
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
	    beta[t_cnt] = std::stod(word.c_str());
	    iss >> word;
            R[t_cnt-1] = std::stod(word.c_str());
            iss >> dummy;
	    
	    for(int i = 0; i < dim; i++)		
	    {
	      iss >> word;
	      buff[n][i] = std::stod(word.c_str());
	      //std::cout << preS[n][i] << "  ";
	    }
	    //std::cout << std::endl;
	  }
          std::cout << "Reading in the checkpoint file\n";
          std::cout << "On T " << t_cnt << " I think beta is " << beta[t_cnt] << " with a rejection rate of " << R[t_cnt-1] << std::endl;
	  //MPI_Bcast(&preS[0][0], ChNum*dim, MPI_DOUBLE, 0, E_COMM);
	  MPI_Send(&buff[0][0], dim*ChNum, MPI_DOUBLE, t_cnt, t_cnt, T_COMM);
	} 
	ckptin.close();
      }
     
      if((C_rank == 0) && (L_rank == 0) && (T_rank > 0))
      {
	MPI_Recv(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, T_rank, T_COMM, &Stat);
      }
      MPI_Bcast(&start_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&start_round, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&beta[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&R[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&preS[0][0], ChNum*dim, MPI_DOUBLE, 0, E_COMM);

      if ( rank==MASTER ){
        R[0] = 0.0; //For checkpoint consistency when outputting.
      }

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

      for(int i  = start; i < end; ++i){
	for( j = 0; j < dim; ++j){
	  pos[j] = preS[i][j];
	}
	
	preL[i] = _L(pos);
	MPI_Bcast(&preL[i], 1, MPI_DOUBLE, 0, L_COMM);
      }
      
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);

      // Open the output files to append the new samples
      // e.g. the cahin file, likelihood file and chi-squared file
      if (rank == MASTER) {
	out.open((chain_file+".r"+std::to_string(start_round)).c_str(), std::ios::app);
	out2.open((lklhd_file+".r"+std::to_string(start_round)).c_str(), std::ios::app);
	out3.open((chi2_file+".r"+std::to_string(start_round)).c_str(), std::ios::app);
        out4.open(annealing_file.c_str(), std::ios::app);
        out5.open((annealing_file+".summary").c_str(), std::ios::app);
	out.precision(output_precision);
	out2.precision(output_precision);
	out3.precision(output_precision);
	out4.precision(output_precision);
	out5.precision(output_precision);
      }

      // Open output files for high temperature chains
      if (verbosity == 1){
	if ((T_rank != MASTER) & (E_rank == 0)) {
	  out.open((chain_file+".r"+std::to_string(start_round)+std::to_string(T_rank)).c_str(), std::ios::out);
	  out2.open((lklhd_file+".r"+std::to_string(start_round)+std::to_string(T_rank)).c_str(), std::ios::out);
	  out.precision(output_precision);
	  out2.precision(output_precision);
          //out3.open(chi2_file.c_str(), std::ios::out);
	}
      }
    }
    else
    {
      start = C_rank*(ChNum/2)/C_size;
      end = (C_rank+1)*(ChNum/2)/C_size;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;      

      //std::cout << "test1: " << start << "   " << end << "  " << sizeL << std::endl;
      
      for(int i  = start; i < end; ++i){
        long ntries = 0;
	do{
	  for( j = 0; j < dim; ++j){
	    //preS[i][j] = RndGaussian(mean+0.1*T_rank, std, true);
	    if(L_rank == 0){
	      preS[i][j] = RndGaussian(means[j], 0.5*ranges[j], false);
            }
            //Keep track of the number of attempts to initialize the chain so we don't get hanging.
            ntries++;
            if (ntries > long(1e6)){
              if (rank==MASTER)
	      {
                std::cerr << "ERROR: Stuck in initialization! Check your prior bounds\n";
		std::exit(1);
	      }
            }
	    
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
	
	  if(_L.transform_state()){
	    _L.forward_transform(pos);
          }
	  for( j = 0; j < dim; ++j){
	    preS[i][j] = pos[j];
          }
	  
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
      
      MPI_Barrier(MPI_COMM_WORLD);
      
      MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);

      for (int r=0; r<size; r++) {
	if (rank==r) {
	  std::cout << "Midished Initializing rank " << std::setw(3) << rank
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


      start = C_rank*(ChNum/2)/C_size+ChNum/2;
      end = (C_rank+1)*(ChNum/2)/C_size+ChNum/2;
      sizeS = dim*(ChNum/2)/C_size;
      sizeL = (ChNum/2)/C_size;
      //if (rank == 0)
      //std::cout << "C_size: " << C_size << "ChNum/2:  " << ChNum/2 << std::endl;
      //std::cout << "test2: " << start << "   " << end << "  " << sizeL << std::endl;

      for(int i  = start; i < end; ++i){
        long ntries = 0;
	do{
	  for( j = 0; j < dim; ++j){
	    //preS[i][j] = RndGaussian(mean+0.1*T_rank, std, true);
	    if(L_rank == 0){
	      preS[i][j] = RndGaussian(means[j], 0.5*ranges[j], false);
            }
            //Keep track of the number of attempts to initialize the chain so we don't get hanging.
            ntries++;
            if (ntries > long(1e6)){
              if (rank==MASTER)
	      {
                std::cerr << "ERROR: Stuck in initialization! Check your prior bounds\n";
		std::exit(1);
	      }
            }
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
	  
	  if(_L.transform_state()){
	    _L.forward_transform(pos);
          }
	  for( j = 0; j < dim; ++j){
	    preS[i][j] = pos[j];
          }
	
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
	}while (preL[i]==-std::numeric_limits<double>::infinity());
      }
      //MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
      //MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[ChNum/2][0], sizeS, MPI_DOUBLE, C_COMM);
      MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);


      for (int r=0; r<size; r++){
	if (rank==r){
	  std::cout << "Finished Initializing rank " << std::setw(3) << rank
		    << " L_rank: " << std::setw(3) << L_rank
		    << " C_rank: " << std::setw(3) << C_rank
		    << " T_rank: "<< std::setw(3) << T_rank
		    << " E_rank: " << std::setw(3) << E_rank
		    << " preSL:  ";
	  for (int i=0; i<ChNum; ++i){
	    for (int k = 0; k < dim; k++){
	      std::cout << std::setw(15) << preS[i][k];
            }
	    std::cout << std::setw(15) << preL[i];
	  }
	  std::cout << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
      }
      


      // Open the output files    
      if (rank == MASTER){
	out.open((chain_file+".r0").c_str(), std::ios::out);
        out << "#chainfmt ensemble\n";
	out2.open((lklhd_file+".r0").c_str(), std::ios::out);
        out2 << "#lklhdfmt ensemble\n";
	out3.open((chi2_file+".r0").c_str(), std::ios::out);
	out4.open(annealing_file.c_str(), std::ios::out);
	out4 << "#" 
             << " nTemp = " << TNum << std::endl
             << "#"
             << std::setw(10) << "round"
             << std::setw(15) << "Beta"
             << std::setw(15) << "R" << std::endl;
	out5.open((annealing_file+".summary").c_str(), std::ios::out);
        out5 << "#"
             << std::setw(5) << "round"
             << std::setw(15) << "Lambda"
             << std::setw(15) << "rt_opt"
             << std::setw(15) << "E"
             << std::setw(15) << "rt_est"
             << std::setw(15) << "Avg(R)"
             << std::setw(15) << "Std(R)" << std::endl;
        out.precision(output_precision);
	out2.precision(output_precision);
	out3.precision(output_precision);
        out4.precision(output_precision);
        out5.precision(output_precision);
	//out4.open("Tempering.d", std::ios::out);
      }

      // Open output files for high temperature chains
      if (verbosity == 1){
	if ((T_rank != MASTER) & (E_rank == 0)) {
	  out.open((chain_file+".r0."+std::to_string(T_rank)).c_str(), std::ios::out);
	  out << "# ";
	  for(size_t i = 0; i < var_names.size(); i++){
	    out << var_names[i] << "  ";
	  }
	  out << std::endl;
	  out2.open((lklhd_file+".r0."+std::to_string(T_rank)).c_str(), std::ios::out);
	  out.precision(output_precision);
	  out2.precision(output_precision);
	  //out3.open(chi2_file.c_str(), std::ios::out);
	  //out4.open("Tempering.d", std::ios::out);
	}
      }
      
      //If the variable names are provided write them as the header for the chain file
      if((rank == MASTER) && (var_names.size() > 0)){
	out << "# ";
	for(size_t i = 0; i < var_names.size(); i++){
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

    Themis::StopWatch sw;
    
    double gamma0 = 2.38/sqrt(2*dim); //Initialize gamma0
    double gamma = gamma0;
    int jump_stride = 100; //Number of steps after which gamma is set to one to facilitate mode jumping
    double acceptance = 0.25; //Optimal acceptance for high dimensional problems
    double local_acceptance;
    double temp_acceptance=0.0;
    bool even_swap = true; //Start with an even swap so 2->3 4->5 and so on

    //get parameter bounds
    std::vector<double> lbounds(dim, 0.0), ubounds(dim, 0.0);
    for ( int n = 0; n < dim; ++n  ){
      lbounds[n] = _L.priors()[n]->lower_bound();
      ubounds[n] = _L.priors()[n]->upper_bound();
    }

    for ( int ir = start_round; ir < _nrounds;  ++ir){
      //Start of round i
      int length = start_length*std::pow(_b,ir);
      int nr_swap = length/temp_stride; //number of swaps to do
      for(int i = start_index + 1; i < length; ++i){
        std::stringstream swlblpre,swlblpost;
        
        swlblpre << "At MCMC step: " << i << " in round " << ir 
                 <<" Temperature rank: " << T_rank << ".  Took ";
        swlblpost << " seconds since last step.";
        if (L_rank == 0 &&  C_rank == 0 && T_rank ==0 )
	  sw.print_lap(std::cout,swlblpre.str(),swlblpost.str());
        //swlblpre << "At MCMC step: " << i << ".  Took ";
        //swlblpost << " seconds since last step.";
        //if (rank==0)
        //sw.print_lap(std::cout,swlblpre.str(),swlblpost.str());
  

        //Specify the chains each CPU is responsible for
        //and the size of likelihhood array and state matrix
        start = C_rank*(ChNum/2)/C_size;
        end = (C_rank+1)*(ChNum/2)/C_size;
        sizeS = dim*(ChNum/2)/C_size;
        sizeL = (ChNum/2)/C_size;

        //Rescale jump proposals to get optimal acceptance ratio
        if(acceptance < 0.2){
	  gamma0 *= 0.9;
        }else if (acceptance > 0.31){
	  gamma0 *= 1.1;
        }else{
	  gamma0 *= sqrt(acceptance/0.25);
        }

        //Reset the acceptance for the current step
        local_acceptance = 0.0;
        if (T_rank != TNum-1){
          for(int k  = start; k < end ; ++k){
	    j = RndUnint(ChNum/2, ChNum - 1);
	    do{
	      j2 = RndUnint(ChNum/2, ChNum - 1);
	    }while(j2 == j);

	    z = RndGaussian(0.0, 0.01, false);
	
	    gamma = gamma0 * (1.0 + z);
	    if(i%jump_stride == 0){
	      gamma = 1.0;
            }
	    //std::cout << "Gamma: " << gamma << "  " << gamma0 << " Acc:   " << acceptance<<std::endl;
	    for(int n = 0; n < dim; ++n){
	      Y[n] = preS[k][n] + gamma * (preS[j2][n] - preS[j][n]);
	    }
	
	    MPI_Bcast(&Y[0], dim, MPI_DOUBLE, 0, L_COMM);
	
	    for(int n = 0; n < dim; ++n)
	      pos[n] = Y[n];
	
	
	    L = _L(pos);
	    //q = pow(z, dim - 1) * exp((-L + preL[k])/T[T_rank]);
	    q = ((L - preL[k])*beta[T_rank]);
	    r = log(RndUni(0.0, 1.0));

	    MPI_Bcast(&q, 1, MPI_DOUBLE, 0, L_COMM);
	    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, L_COMM);

	    if( r <= q){
	      for(int m = 0; m < dim; ++m){
	        preS[k][m] = Y[m];
	      }
	      preL[k] = L;
	      if(L_rank == 0 )
	        local_acceptance += 1.0/ChNum;
	    }
          

	    // Calculate the chi2_values every "chi2_stride" steps
	    if(i%chi2_stride == 0){
	      for(int m = 0; m < dim; ++m){
	      chi2_arg[m] = preS[k][m];
	      }
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
              
	    }
          }
        }else{
          for ( int k = start; k < end; ++k ){
            //Reference distribution is flat! So lets just draw IID from it.
            for ( int n = 0; n < dim; ++n ){
              double lower = lbounds[n];
              double upper = ubounds[n];
              if ( (std::isfinite(lower) && std::isfinite(upper)) ){
                pos[n] = lower + RndUni(0.0,1.0)*(upper - lower);
              }else if (std::isfinite(lower)){
                pos[n] = lower + RndUni(0.0,1e10);
              }else if (std::isfinite(upper)){
                pos[n] = upper - RndUni(0.0,1e10);
              }else{
                pos[n] = RndUni(-1e10,1e10);
              }
            }
          

            //Check if L is indeed flat
            L = _L(pos); 
	    if(_L.transform_state()){
	      _L.forward_transform(pos);
            }
            if ( (L-preL[k])*beta[T_rank] != 0.0){
              std::cerr << "THere is structure in the reference something is wrong!\n";
              std::cerr << beta[T_rank] << " is beta";
              std::cerr << "New L = " << L << std::endl
                        << "Old L = " << preL[k] << std::endl;
              std::exit(1);
            }
            for (int n = 0; n < dim; ++n){
              preS[k][n] = pos[n];
            }
            preL[k] = L;
            acceptance = 0.25;
	    
	    if(i%chi2_stride == 0){
	      for(int m = 0; m < dim; ++m){
	      chi2_arg[m] = preS[k][m];
	      }
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
	    }
          }
        }
	
        
      
        //Communicate the new likelihoods and states between chains of the same temperature
        MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[0][0], sizeS, MPI_DOUBLE, C_COMM);
        MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[0], sizeL, MPI_DOUBLE, C_COMM);
        MPI_Allgather(&chi2_vec[start], sizeL, MPI_DOUBLE, &chi2_vec[0], sizeL, MPI_DOUBLE, C_COMM);
      
        //Specify the chains each CPU is responsible for
        //and the size of likelihhood array and state matrix
        start = C_rank*(ChNum/2)/C_size+ChNum/2;
        end = (C_rank+1)*(ChNum/2)/C_size+ChNum/2;
        sizeS = dim*(ChNum/2)/C_size;
        sizeL = (ChNum/2)/C_size;
        if ( T_rank != TNum - 1  ){
          for(int k  = start; k < end ; ++k){
	    j = RndUnint(0, ChNum/2 - 1);
	    do{
	      j2 = RndUnint(0, ChNum/2 - 1);
	    }while(j2 == j);

	    z = RndGaussian(0.0, 0.01, false);
	
	    gamma = gamma0 * (1.0 + z);
	    if(i%jump_stride == 0)
	      gamma = 1.0;
	
	    for(int n = 0; n < dim; ++n){
	      Y[n] = preS[k][n] + gamma * (preS[j2][n] - preS[j][n]);
	    }
	
	    MPI_Bcast(&Y[0], dim, MPI_DOUBLE, 0, L_COMM);
	
	    for(int n = 0; n < dim; ++n)
	      pos[n] = Y[n];
	
	    L = _L(pos);
	    //q = pow(z, dim - 1) * exp((-L + preL[k])/T[T_rank]);
	    q = ((L - preL[k])*beta[T_rank]);
	    r = log(RndUni(0.0, 1.0));
	
	    MPI_Bcast(&q, 1, MPI_DOUBLE, 0, L_COMM);
	    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, L_COMM);
	
	    if( r <= q){
	      for(int m = 0; m < dim; ++m){
	        preS[k][m] = Y[m];
	      }
	      preL[k] = L;
	      if(L_rank == 0)
	        local_acceptance += 1.0/ChNum;
	    }
	  // Calculate the chi2_values every "chi2_stride" steps
	    if(i%chi2_stride == 0){
	      for(int m = 0; m < dim; ++m){
	      chi2_arg[m] = preS[k][m];
	      }
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
	    }
          } 
        }else{
          for ( int k = start; k < end; ++k ){
            //Reference distribution is flat! So lets just draw IID from it.
            for ( int n = 0; n < dim; ++n ){
              double lower = lbounds[n];
              double upper = ubounds[n];
              if ( (std::isfinite(lower) && std::isfinite(upper)) ){
                pos[n] = lower + RndUni(0.0,1.0)*(upper - lower);
              }else if (std::isfinite(lower)){
                pos[n] = lower + RndUni(0.0,1e10);
              }else if (std::isfinite(upper)){
                pos[n] = upper - RndUni(0.0,1e10);
              }else{
                pos[n] = RndUni(-1e10,1e10);
              }
            }
	    MPI_Bcast(&pos[0], dim, MPI_DOUBLE, 0, L_COMM);
            //Check if L is indeed flat
            L = _L(pos);

	    if(_L.transform_state()){
	      _L.forward_transform(pos);
            }
            for (int n = 0; n < dim; ++n){
              preS[k][n] = pos[n];
            }
            preL[k] = L;
            acceptance = 0.25;
          
	    // Calculate the chi2_values every "chi2_stride" steps
	    if(i%chi2_stride == 0){
	      for(int m = 0; m < dim; ++m){
	      chi2_arg[m] = preS[k][m];
	      }
	      chi2_vec[k] = _L.chi_squared(chi2_arg);
	    }
	  }


        
        }
        //Communicate the new likelihoods ans states between chains of the same temperature
        MPI_Allgather(&preS[start][0], sizeS, MPI_DOUBLE, &preS[ChNum/2][0], sizeS, MPI_DOUBLE, C_COMM);
        MPI_Allgather(&preL[start], sizeL, MPI_DOUBLE, &preL[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);
        MPI_Allgather(&chi2_vec[start], sizeL, MPI_DOUBLE, &chi2_vec[ChNum/2], sizeL, MPI_DOUBLE, C_COMM);
      
        //Calculate the acceptance and communicate to all threads
        MPI_Allreduce(&local_acceptance, &acceptance, 1, MPI_DOUBLE, MPI_SUM, C_COMM);
        MPI_Bcast(&acceptance, 1, MPI_DOUBLE, 0, L_COMM);
      
        //Temp swapping state
        //SWAP NOW
        if( (i % stride == stride - 1)){
          //Needed because all the chains need to have moved stride steps before communicating?
          MPI_Barrier(T_COMM);
          
          even_swap = !even_swap;
          bool even_temp = (T_rank%2==0);
	  //Communicate likelihoods between adjacent temperature chains
	  // From smaller T_rank to larger T_rank, corresponding to from higher beta to lower beta
          if ((T_rank < T_size - 1) && (C_rank == 0))
	    MPI_Isend(&preL[0], ChNum, MPI_DOUBLE, T_rank + 1, T_rank+1, T_COMM, &Req);
	  
	
	  if((T_rank > 0) && (C_rank == 0))
	    MPI_Recv(&neiL[0], ChNum, MPI_DOUBLE, T_rank - 1, T_rank, T_COMM, &Stat);
	  //MPI_Wait(&Req, &Stat);	

	  //Decide if we need to swap the states and communicate back
	  //to the adjacent temperature chain
	  int swap[ChNum], nswap[ChNum];
	  for (int i2 = 0; i2 < ChNum; ++i2){
	    nswap[i2] = 0;
	    swap[i2] = 0;
	  }
	  if((T_rank > 0) && (C_rank == 0) && (L_rank == 0)){
          
            temp_acceptance = 0.0;
	    for(int m = 0; m < ChNum; ++m){
	      double alpha = RndUni(0.0, 1.0);
              double mhRatio = std::exp(std::min(0.0 , (neiL[m] - preL[m]) * (beta[T_rank] - beta[T_rank-1])));
	      temp_acceptance += mhRatio;
	    
	      if(alpha < mhRatio && ((!even_swap&&even_temp)||(even_swap&&!even_temp)))
	        swap[m] =  1; 
	    
              /*
	      tempout << '|' 
		      << std::setw(15) << even_swap
		      << std::setw(15) << T_rank 
		      << std::setw(15) << m
	              << std::setw(15) << alpha
		      << std::setw(15) << mhRatio
		      << std::setw(15) << beta[T_rank]
		      << std::setw(15) << beta[T_rank-1]
                      << std::setw(15) << preL[m]
                      << std::setw(15) << neiL[m]
		      << std::setw(15) << swap[m];
              */
	    
	    }
            temp_acceptance /= ChNum;

	    std::cout << std::setw(10) << ir <<  std::setw(10) << i << "  Tempering acceptance rate:  " << temp_acceptance << "   beta:  " << beta[T_rank] << std::endl;
	    // AEB
	    //tempout << std::endl;
	    MPI_Isend(&swap[0], ChNum, MPI_INT, T_rank - 1, T_rank, T_COMM, &Req);
	  }
	  if((T_rank < T_size - 1) && (C_rank == 0) && (L_rank == 0)){
	    MPI_Recv(&nswap[0], ChNum, MPI_INT, T_rank + 1, T_rank+1, T_COMM, &Stat); 
	  }

	  
	  //if(i == 799 && E_rank == 0)
	  //for(int m = 0; m < ChNum; ++m)
	  //cout << T_rank  << "  " << swap[m] << "  " << nswap[m] << "  " << m << endl; 
	  
      
	  //Now that it turns out we want to swap states let's actually 
	  //swap states between adjacent temperature chains
	  if( (T_rank > 0) && (C_rank == 0) && (L_rank == 0)){
	    /*
	    // AEB
	    tempout << std::setw(5) << T_rank << " swap dets2:";
	    */

	    R[T_rank - 1] += 1-temp_acceptance;
	    for(int m = 0; m < ChNum; ++m){
	      if((swap[m]) == 1){
                //tempout << "For deo swap " << even_swap << " on temp rank " << T_rank-1 << "\n";
	        MPI_Isend(&preS[m][0], dim, MPI_DOUBLE, T_rank - 1, m, T_COMM, &Req);
	        MPI_Recv(&neiS[0], dim, MPI_DOUBLE, T_rank - 1, m, T_COMM, &Stat);
	        MPI_Wait(&Req, &Stat);
	      
	        MPI_Isend(&preL[m], 1, MPI_DOUBLE, T_rank - 1, m+ChNum, T_COMM, &Req2);
	        //Already have neiL!
                MPI_Wait(&Req2, &Stat2);
	      
	        MPI_Isend(&chi2_vec[m], 1, MPI_DOUBLE, T_rank - 1, m+2*ChNum, T_COMM, &Req3);
	        MPI_Recv(&neiC, 1, MPI_DOUBLE, T_rank - 1, m+2*ChNum, T_COMM, &Stat3);
	        MPI_Wait(&Req3, &Stat3);
	    
	        
	        /*
                tempout << '|' 
		        << std::setw(15) << preL[m]
		        << std::setw(15) << swapL;
	        */

	        preL[m] = neiL[m];
	        chi2_vec[m] = neiC;
	        for(int o = 0; o < dim; ++o)
		  preS[m][o] = neiS[o];
	      }
	    }
	    //tempout << std::endl;
	  
	    //MPI_Isend(&A[T_rank - 1], 1, MPI_DOUBLE, T_rank - 1, 20000, T_COMM, &Req);
	    //MPI_Send(&R[T_rank - 1], 1, MPI_DOUBLE, T_rank - 1, ChNum*3, T_COMM);
	    //cout << A[T_rank - 1] << endl;
	  }
	  //if((E_rank == 0) && (T_rank > 0))
	  //{
	  //cout << A[T_rank - 1] << endl;
	  //}
	
	
	  if( (T_rank < T_size - 1) && (C_rank == 0) && (L_rank == 0))
	  {
	    for(int m = 0; m < ChNum; ++m)
	    {
	      //if((nswap[m] == 1))
	      if((nswap[m]) == 1)
	      {
		MPI_Recv(&neiS[0], dim, MPI_DOUBLE, T_rank + 1, m, T_COMM, &Stat);
		MPI_Isend(&preS[m][0], dim, MPI_DOUBLE, T_rank + 1, m, T_COMM, &Req);
		
		MPI_Recv(&swapL, 1, MPI_DOUBLE, T_rank + 1, m+ChNum, T_COMM, &Stat2);
		
		MPI_Recv(&neiC, 1, MPI_DOUBLE, T_rank + 1, m+2*ChNum, T_COMM, &Stat3);
		MPI_Isend(&chi2_vec[m], 1, MPI_DOUBLE, T_rank + 1, m+2*ChNum, T_COMM, &Req3);
		
		for(int o = 0; o < dim; ++o)
		  preS[m][o] = neiS[o];
		preL[m] = swapL;
		chi2_vec[m] = neiC;
		
	      }
	    }
	    //MPI_Recv(&R[T_rank-1], 1, MPI_DOUBLE, T_rank + 1, ChNum*3, T_COMM, &Stat4);
	  }
	
	  //Broadcast the new state to all chains within the same temperature  
	  MPI_Bcast(&preS[0][0], dim*ChNum, MPI_DOUBLE, 0, E_COMM);
	  MPI_Bcast(&preL[0], ChNum, MPI_DOUBLE, 0, E_COMM);
          
	
          
        
        }//End the swap 
        //MPI_Barrier(E_COMM);
        //Write the new step to the file
        if(rank == MASTER)
        {
          if (i%thin_stride==0){
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
          if (i%thin_stride==0){
	    if((T_rank != MASTER) && (E_rank == 0)){
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
        }
      

        //////////////////////// checkpoint generation //////////////////
        // Write the state vectors (for all temperatures) to the checkpoint file        
        if(i%_ckpt_stride == _ckpt_stride - 1)
        {
	  double buff[ChNum][dim];
	  double Temp;
          double Rej;
	
	  // Master writes the information for the lowest temperature [T = 1] to the checkpoint
	  if (rank == MASTER)
	  {
	    ckpt.open(_ckpt_file.c_str(), std::ios::out);
	    ckpt.precision(output_precision);
	    ckpt << std::setw(10) << ir 
                 << std::setw(10) << i << std::endl;
	    for(int n_c = 0; n_c < ChNum; ++n_c)
	    {
	      ckpt << beta[0] << "  " << R[0] << "  "  << n_c << "  ";
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
	      MPI_Send(&beta[n_t], 1, MPI_DOUBLE, 0, T_size+n_t, T_COMM);
	      MPI_Send(&R[n_t-1], 1, MPI_DOUBLE, 0, T_size+n_t, T_COMM);
	    }
	  
	    if(rank == MASTER)
	    {
	      MPI_Recv(&buff[0][0], dim*ChNum, MPI_DOUBLE, n_t, n_t, T_COMM, &Stat);
	      MPI_Recv(&Temp, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);
	      MPI_Recv(&Rej, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);
	      for(int n_c = 0; n_c < ChNum; ++n_c)
	      {
	        ckpt << Temp << "  " << Rej << "  "  << n_c << "  ";
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
      // Master process closes the output files and reopens for the next round
      
      if (rank == MASTER) 
      {
        out.close();
        out2.close();
        out3.close();
	if (ir+1 < _nrounds)
	{
	  out.open((chain_file+".r"+std::to_string(ir+1)).c_str(), std::ios::out);
	  //Write variable name header
	  if (var_names.size() >0){
            out << "# ";
	    for ( size_t iv = 0; iv<var_names.size(); ++iv )
	      out << var_names[iv] << "  ";
	    out << std::endl;
	  }
	
	  out2.open((lklhd_file+".r"+std::to_string(ir+1)).c_str(), std::ios::out);
	  out3.open((chi2_file+".r"+std::to_string(ir+1)).c_str(), std::ios::out);
	}
      }
      if (verbosity == 1){
	if ((T_rank != MASTER) & (E_rank == 0)) {
          out.close();
          out2.close();
	  if ( ir+1 < _nrounds ){
            out.open((chain_file+".r"+std::to_string(ir+1)+"."+std::to_string(T_rank)).c_str(), std::ios::out);
	    out << "# ";
	    for(size_t i = 0; i < var_names.size(); i++){
	      out << var_names[i] << "  ";
	    }
	    out << std::endl;
	    out2.open((lklhd_file+".r"+std::to_string(ir+1)+"."+std::to_string(T_rank)).c_str(), std::ios::out);
	    out.precision(output_precision);
	    out2.precision(output_precision);
	  }  
        }
      }
    
      
      //Save the temperature information
      // Master collects the information form high temperature walkers and writes to the checkpoint
      double betaEnd, Rend;
      if (rank == MASTER)
      {
        out4 << std::setw(10) << ir 
             << std::setw(15) << beta[0];
      }
      //Save the average rejection rates for that round
      for(int n_t = 1; n_t < T_size; n_t++)
      {
        if ((T_rank == n_t) && (C_rank == 0) && (L_rank == 0))
        {
	  MPI_Send(&R[n_t-1],  1, MPI_DOUBLE, 0, T_size+n_t, T_COMM);
	  MPI_Send(&beta[n_t], 1, MPI_DOUBLE, 0, 2*T_size+n_t, T_COMM);

        }
	  
        if(rank == MASTER)
        {
	  MPI_Recv(&Rend, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);
	  MPI_Recv(&betaEnd, 1, MPI_DOUBLE, n_t, 2*T_size+n_t, T_COMM, &Stat);
          beta[n_t] = betaEnd;
          R[n_t-1] = Rend/nr_swap;
	
          out4 << std::setw(15) << Rend/nr_swap << std::endl  
               << std::setw(10) << ir
	       << std::setw(15) << beta[n_t];

        }
      }
      if (rank==MASTER)
	out4 << std::setw(15) << 0.0 << std::endl;
      //Now calculate the updated temperature ladder
        //Will do this on the master rank since we have already communicated with it for the output.
      if (rank==MASTER)
      {
        std::cout << "Updating annealing schedule\n";
        double oroundtrip = update_annealing_params(beta, R);
        double mean_rej = 0.0, std_rej = 0.0, E = 0.0;
        for ( int tt = 0; tt < TNum-1; ++tt )
        {
          E += R[tt]/(1.0-R[tt]+1e-10);
          mean_rej += R[tt];
        }
        double Lambda = mean_rej;
        mean_rej /= (TNum-1.0);
        for ( int tt = 0; tt < TNum-1; ++tt )
          std_rej = (R[tt]-mean_rej)*(R[tt]-mean_rej);
        std_rej = std::sqrt(std_rej/(TNum-2.0)); 
          
        out5 << std::setw(5)  << ir 
             << std::setw(15) << Lambda
             << std::setw(15) << oroundtrip
             << std::setw(15) << E
             << std::setw(15) << 1.0/(2.0+2.0*E)
             << std::setw(15) << mean_rej
             << std::setw(15) << std_rej << std::endl;
        std::cout << "Done round "<< ir << " estimate round trip rate: " << 1.0/(2.0+2.0*E) << std::endl
                  << "I estimate that you have completed: " << (length/stride)/(2.0+2.0*E) 
                  << " round trips, hopefully this is greater than 1\n";
        //Broadcast the updated temperatures to everyone.
        //MPI_Bcast(&beta[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Bcast(&beta[0], TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      //MPI_Bcast(&beta[0], TNum, MPI_DOUBLE, 0, T_COMM);
      for (int ti = 0; ti < TNum; ++ti)
        R[ti] = 0.0;

      start_index = 0; //Reset the start index

    }
    //Close the output stream
    if (rank==MASTER){
      out4.close();
      out5.close();
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
double sampler_differential_evolution_deo_tempered_MCMC::RndUni(double a, double b)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);
  return dis(gen);
}

// Generate integer random numbers with uniform distribution
int sampler_differential_evolution_deo_tempered_MCMC::RndUnint(int a, int b)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(a, b);
  return dis(gen);
}


// Generate random numbers with g(z) distribution function
//g(z) = 1/sqrt(z) for 1/a<z<a
double sampler_differential_evolution_deo_tempered_MCMC::RndGz(double a)
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
double sampler_differential_evolution_deo_tempered_MCMC::RndGaussian(double mean, double stDev, bool CONTINUE)
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
double sampler_differential_evolution_deo_tempered_MCMC::RndUni(double a, double b)
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
int sampler_differential_evolution_deo_tempered_MCMC::RndUnint(int a, int b)
{
  return int( (b-a)*_rng.rand()+a );

  /*  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(a, b);
  return dis(gen);
  */
}



// Generate random numbers with normal distribution using Marsaglia polar method
double sampler_differential_evolution_deo_tempered_MCMC::RndGaussian(double mean, double stDev, bool CONTINUE)
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




std::vector<double> sampler_differential_evolution_deo_tempered_MCMC::find_best_fit(std::string chain_file, std::string lklhd_file)
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
      std::cerr << "sampler_differential_evolution_deo_tempered_MCMC: Can't open likelihood file " << lklhd_file << std::endl;
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
      std::cerr << "sampler_differential_evolution_deo_tempered_MCMC: Can't open chain file " << chain_file << std::endl;
      std::exit(1);
    }
    std::string parameter_line;
    //std::cout << "Indmax = " << indmax  << std::endl;
    //for (int index=0; index<=indmax; ++index)
    //indmax+=2;
    indmax+=1;
    for (int index=0; index<indmax; ++index)
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

};


