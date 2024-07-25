/*!
  \file sampler_deo_tempering_MCMC.h
  \author Paul Tiede
  \brief Header file for sampler_deo_tempering_MCMC class.
  \details This uses any of the sampler_MCMC_base classes to define a tempered sampler.
  The tempering scheme follows Syed 2019, and uses deterministic even odd swap scheme to 
  enable massive parallelization.
*/

#include "sampler_MCMC_base.h"
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
#include "likelihood_power_tempered.h"
#include "interpolator1D.h"


#define MASTER 0

#ifndef THEMIS_SAMPLER_DEO_TEMPERING_MCMC_H
#define THEMIS_SAMPLER_DEO_TEMPERING_MCMC_H

namespace Themis{
/*!
  \class sampler_deo_tempering_MCMC
  \brief Parallel tempered Markov Chain Monte Carlo using DEO swapping scheme.

  \details Runs a parallel tempered mcmc chain using a exploration kernel based
  on the sampler_exploration_mcmc_base class. This uses the parallel tempering
  scheme from Syed et al. (2020), which is based on the deterministic even-odd
  swapping scheme, which is a non-reversible markov chain. For more information
  on the tempering please see Syed 2020). 
*/
template<class S>
class sampler_deo_tempering_MCMC : public sampler_MCMC_base
{
public:
  
  /*! 
    \brief Class constructor, takes a reference to the exploration kernel to be used.
  */
  sampler_deo_tempering_MCMC(int seed, likelihood_power_tempered& L, 
                             std::vector<std::string> var_names, size_t dimension);
  ~sampler_deo_tempering_MCMC(){};



  /*
    \brief sets starting location of the sampler
  */
  void set_initial_location(std::vector<double> initial_parameters);
  

  /*!
    \brief Runs DEO using where you specify the starting run length and adaption parameters. After each round each number if increased by the geometric increase factor specified in set_annealing_schedule.
    \param number_of_rounds How many rounds you want to run the sampler for. You can run this many times to build up rounds.
    \param thin The thinning rate of the chain. Every thin'd step will be saved.
    \param refresh How often cout is updated. If < 0 it is never updated.
    \param verbosity If 1 it will save the chain information for all tempering levels.
    \note The total number of samples will be given by start_samples*swap_stride.
  */
  void run_sampler(int number_of_rounds, 
                   int thin, int refresh,
                   int verbosity = 0);


  
  /*!
    \brief Provides access to the exploration sampler so that you can change options for it. Returns a pointer to the sampler. 
  */
  S* get_sampler(){return &_sampler;};

  /*!
    \brief Returns the current round of the sampler.
  */
  int get_round() const{return _current_round;};


  /*!
    \brief Sets the output streams for the sampler. This is the same function as 
    for sampler_MCMC_base.
    \note For DEO every chain,state,sampler file will be prefixed with roundXXX_ to denote which
    deo round it is on.
  */
  void set_output_stream(std::string chain_file, std::string state_file,
                         std::string sampler_file,
                         int output_precision=8);




  /*!
    \brief Sets the annealing schedule to be used. 
    \param initial_length The number of swaps to do in the first round..
    \param swap_stride The number of steps to take before we swap temperatures.
    Typically this should be at least 5.
    \note The default is 10 swaps initially and a swap stride of 25. This is probably too 
    short.
  */
  void set_deo_round_params(int initial_length, int swap_stride);
  
  /*!
    \param initial_spacing The initial geometric spacing for the ladder. That is, the 
    inverse temperature for the ith rung will be given by 1.0/geometric_increase^i.
    \param geometric_increase The geometric increase for the number of steps to be used
    for each round. This should almost always be 2.
  */
  void set_annealing_schedule(double initial_spacing, 
                              int geometric_increase=2);

  /*!
    \brief Sets the annealing file and summary file names. If using the default for the annealing_summary_file
    the output will be the annealing_file+".summary".
  */
  void set_annealing_output(std::string annealing_file, std::string annealing_summary_file = "")
  {
    _anneal_file = annealing_file;
    if (annealing_summary_file == "")
      _anneal_summary_file = annealing_file+".summary";
    else
      _anneal_summary_file = annealing_summary_file;
  }


  /*!
    \brief Sets the distribution for the cpu.
    \param num_temperatures The number of temperatures for the sampler.
    \param num_exploration The number of processors to send each exploration sampler.
    \note If this isn't called we use the default distribution which set num_temperatures 
    equal to the number of MPI instances, and num_exploration to 1.
  */
  void set_cpu_distribution(int num_temperatures, int num_exploration);

  /*!
    \brief Reads in an annealing.dat file and set the initial ladder for the sampler to the ladder from the final round 
    in the annealing file.
    \param annealing_file is the file containing the annealing information for the run. Namely the round, ladder, and rejection rates.
  */
  void read_initial_ladder(std::string annealing_file);
  
  /*!
    \brief Cleans up the MPI instances of the sampler to gracefully exit;
  */
  void mpi_cleanup();
  
  
  /////////////////////////////////////////////////////////////////
  // Helper and access functions.

  void get_chain_state(std::vector<double>& chain, std::vector<double>& state){
    _sampler.get_chain_state(chain, state);
  }
  void set_chain_state(std::vector<double> chain, std::vector<double> state){
    _sampler.set_chain_state(chain, state);
  }

  std::vector<double> find_best_fit(std::string chain_file, std::string state_file) const
  {
    return _sampler.find_best_fit(chain_file, state_file);
  }
  
  std::vector<double> find_best_fit() const
  {
    return _sampler.find_best_fit();
  }

  /*<!
    \brief Writes the sampler checkpoint the ckpt_out stream. This must be implemented inherited 
    sampler.
  */
  void write_checkpoint(std::string file);
  void write_checkpoint(std::ostream& file);
  /*!
    \brief Loads the ckpt from a stream. This will set the sampler to the state given in the ckpt.
    \param ckpt_in A reference to the input stream. 
  */
  void read_checkpoint(std::istream& ckpt_in){};
  /*!
    \brief Reads the checkpoint file and restores the sampler to that state.
    \param ckpt_file The file name of the checkpoint file.
  */
  void read_checkpoint(std::string ckpt_file);
  void set_checkpoint(int ckpt_stride, std::string ckpt_file);

  /*!
    \brief Function to estimate the bayesian evidence using the Thermodynamic Integration method
    \param avg_lklhd[] Vector of likelihood expectations to compute the evidence, using trapezoid
    rule.
    \param beta[] Temperatures of a run.
   */
  double estimate_bayesian_evidence();




private:
  /*
    \brief The exploration sampler off of which the sampler is templated
  */
  S _sampler;
  //power tempered likelihood
  likelihood_power_tempered *_L_tmp;
  int _b;
  double _initial_spacing;
  int _TNum, _E_Num, _ckpt_stride_deo;
  bool _default_cpu_distribution, _no_initial_ladder, _write_all;
  std::vector<double> _initial;


  //<! DEO round run length params;
  int _current_round, _swap_stride, _current_length;
  bool _restart;
  std::vector<double> _beta, _R, _avg_lklhd;

  //<! Communicators
  MPI_Comm E_COMM, T_COMM;
  int T_size, T_rank, E_size, E_rank;
  MPI_Status Stat, Stat2;
  MPI_Request Req, Req2;

  //Annealing output stuff
  std::string _old_annealing_file;
  std::string _anneal_file, _anneal_summary_file;
  std::ofstream _anneal_out,_anneal_summary_out;


  //Initializes the mpi distribution for the sampler. This is called internally.
  void initialize_mpi();

  //Initializes the tempering ladder through beta.
  void initialize_ladder();

  // Algorithm 2 of Syed 2019 that uses the previous tempering schedule and average rejection rates
  // of the MH swapping statistic to find the new optimized schedule.
  // returns the estimate round trip rate for the previous round.
  double update_annealing_params();

  // Bisection method to find the optimal tempering ladder for for k using the monotone interpolator.
  // and solving equation (32) of Syed (2019).
  double find_beta(Interpolator1D& fLambda, int k, double Lambda, double eps=1e-12, int MAX_ITR=1e8);

  // Writes the annealing.dat and annealing.dat.summary files
  void write_annealing_headers();

  // Runs a deo round
  void run_deo_round(int round, int nlength, int thin, int refresh, 
                     int verbosity,
                     bool& even_swap
                    );
  // Helper function to generate transitions
  void generate_transitions(int niterations, int start, int finish, int thin, int refresh, int verbosity);

  // does a deo swap
  void deo_swap(const bool even_swap, const int round, const int swap);
  
  // finishes the deo run and resets the beta to the new optimal ladder.
  double finish_deo_round(int rank, int round);

  // write state does nothing since the writing is done by the individual sampler
  void write_state(){};

};//end sampler_deo_tempering_MCMC


 
/*********************************************************************************************************
 * *************************************  class implementation *******************************************/
template<class S>
sampler_deo_tempering_MCMC<S>::sampler_deo_tempering_MCMC(int seed, likelihood_power_tempered& Ltmp, 
                                                          std::vector<std::string> var_names,
                                                          size_t dimension)
  :sampler_MCMC_base(seed,Ltmp,var_names,dimension),_sampler(seed, Ltmp, var_names, dimension),
   _L_tmp(&Ltmp), _b(2), _initial_spacing(1.15),_default_cpu_distribution(true),
   _no_initial_ladder(true),_write_all(false), _current_round(0), _swap_stride(25), _current_length(10),_restart(false)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2){
    std::cerr << "DEO must be run with at least two tempering levels!\n" 
              << "Which means you need more than 1 MPI processes\n";
    std::exit(1);
  }
  MPI_Bcast(&_seed,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  _initial.resize(0);
  _chain_file = "chain.dat";
  _state_file = "state.dat";
  _anneal_file = "annealing.dat";
  _anneal_summary_file = "annealing.dat.summary";

  _output_precision = 8;
}

template<class S>
void sampler_deo_tempering_MCMC<S>::mpi_cleanup()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if ( rank == MASTER ){
    _anneal_out.close();
    _anneal_summary_out.close();  
  }
  _sampler.close_streams();
  MPI_Comm_free(&E_COMM);
  MPI_Comm_free(&T_COMM);

  //Unset the likelihood and sampler communicator
  _sampler.set_mpi_communicator(MPI_COMM_WORLD);
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_initial_location(std::vector<double> initial)
{
  if ( initial.size() != _dimension ){
    std::cerr << "initial location of sampler_deo_tempering_MCMC must have the same dimension as the problem\n";
    std::exit(1);
  }
  _initial = initial;
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_cpu_distribution(int num_temperatures, int num_exploration)
{
  if (num_temperatures < 2){
    std::cerr << "Must have at least 2 tempering levels! If you don't want to temper just use a regular sampler!\n";
    std::exit(1);
  }

  _TNum = num_temperatures;
  _E_Num = num_exploration;
  _default_cpu_distribution = false;

  if (_current_round != 0 && !_restart){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank at beginning: " << rank << std::endl;
    if ( rank == MASTER ){
      std::cout <<"Redistributing the sampler mid run\n"
                <<"The ladder will be recomputed using the results from the last round albeit\n"
                <<"with a different number of tempering levels.\n";
  
      //update the ladder using the optimal ladder from the last round.
      std::vector<double> betaRev(_beta.size(),0.0), Lambda(_beta.size(),0.0);
      for ( size_t i = 0; i < _beta.size(); ++i )
      {
        betaRev[i] = _beta[_beta.size()-1-i];
        for (size_t j = 0; j < i; ++j)
          Lambda[i] += _R[_beta.size()-1-j-1];
      }

      Interpolator1D fLambda(betaRev,Lambda, "mcubic");
      int old_tnum = _beta.size();
      _beta.resize(_TNum);
      _R.resize(_TNum);
      for ( int k = 1; k < _TNum-1; ++k){
        _beta[_TNum-1-k] = find_beta(fLambda, k, Lambda[old_tnum-1]);
        //std::cerr << "Found "<< k << " beta " << beta[TNum-1-k] << std::endl;
      }
      _beta[0] = 1.0;
      _beta[_TNum-1] = 0.0;
    } else{
      _beta.resize(_TNum);
      _R.resize(_TNum);
      for ( int i = 0; i < _TNum; ++i ){
        _beta[i] = 0.0;
        _R[i] = 0.0;
      }
    }
    double beta[_TNum];
    for ( int i = 0; i < _TNum; ++i )
      beta[i] = _beta[i];
    MPI_Barrier(MPI_COMM_WORLD);
    //Bcast the new ladder
    MPI_Bcast(&beta[0],_TNum, MPI_DOUBLE,0, MPI_COMM_WORLD);
    _beta.resize(0);
    for ( int i = 0; i < _TNum; ++i )
      _beta.push_back(beta[i]);
    //Redo the MPI initializaing
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&T_COMM);
    MPI_Comm_free(&E_COMM);
    //_sampler.set_mpi_communicator(MPI_COMM_WORLD);
    //this->initialize_mpi();
    _L_tmp->set_beta(_beta[T_rank]);
    this->initialize_mpi();
  
  }
  //this->initialize_mpi();
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_output_stream(std::string chain_file, std::string state_file, 
                         std::string sampler_file, int output_precision)
{
  _chain_file = chain_file;
  _state_file = state_file;
  _sampler_file = sampler_file;
  _output_precision = output_precision;
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_deo_round_params(int initial_length, int swap_stride)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == MASTER){
    std::cout << "Resetting the current number of swaps you want to " << initial_length
              << "\n with a swap stride of: " << swap_stride << std::endl;
    if (_restart){
      std::cout << "WARNING: A checkpoint file has been read in.\n" 
                << "We will ignore the change in initial_length\n";
    }
  }
  
  _swap_stride = swap_stride;
  if (!_restart)
    _current_length = initial_length;
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_annealing_schedule(double initial_spacing,
                                                           int geometric_increase)
{
  
  if (geometric_increase < 1)
  {
    std::cerr << "set_annealing_schedule: Number of round and geometric increase must be greater than 1\n";
    std::exit(1);
  }
  _initial_spacing = initial_spacing;
  _b = geometric_increase;
}

template <class S>
double sampler_deo_tempering_MCMC<S>::estimate_bayesian_evidence()
{
  double evidence = 0.0;
  std::vector<double> dZ;
  dZ.resize(0);
  //Now compute the thermodynamic integration using trapezoid rule.
  for ( int i = 0; i <  _TNum-1; ++i)
  {
    double delta = 0.5*(_avg_lklhd[i]+_avg_lklhd[i+1])*(_beta[i] - _beta[i+1]);
    evidence += delta;
  }
  double nlprior = (_L_tmp->get_lklhd_unit_beta())->priorlognorm();
  double nlref = _L_tmp->referencelognorm();
  return evidence+ nlprior - nlref;

}

template<class S>
void sampler_deo_tempering_MCMC<S>::read_initial_ladder(std::string annealing_file)
{
  _old_annealing_file = annealing_file;
  _no_initial_ladder = false;
}


template<class S>
void sampler_deo_tempering_MCMC<S>::initialize_ladder()
{
  _beta.resize(_TNum);
  _R.resize(_TNum,0.0);
  _avg_lklhd.resize(_TNum, 0.0);
  //Initialize the temperatures
  if(_no_initial_ladder){
    _beta[0] = 1.0;
    for(int it = 1; it < _TNum; ++it)
      _beta[it] = _beta[it-1]/_initial_spacing; 
  }else{
    std::ifstream ain;
    ain.open(_old_annealing_file.c_str(), std::ios::in);
    if(!ain.is_open()){
      std::cerr << "Annealing file " << _old_annealing_file << " doesn't exist!" << std::endl;
      std::cerr << "Cannot restart the sampler." << std::endl;
      std::exit(1);
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
    std::cout << "Reading initial tempering ladder from: " << _old_annealing_file << std::endl;
    if (ntemp != _TNum){
      std::cerr << "Annealing file must have the same number of temperatures as requested!\n"
                << "Expected ntemps = " << _TNum << std::endl
                << "File ntemps = " << ntemp << std::endl;
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
        _beta[irt] = beta_read;
        irt++;
      }else{
        roundd = newround;
        irt = 0;
        _beta[irt] = beta_read;
        irt = 1;
      }
    }
    ain.close();
  }
  _beta[_TNum-1] = 0.0;
}


template <class S>
void sampler_deo_tempering_MCMC<S>::run_sampler(int number_of_rounds, 
                                                int thin, int refresh,
                                                int verbosity)
{
  
  if (thin==0){
    thin = 1;
  }   

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (_default_cpu_distribution){
    _TNum = size;
    _E_Num = 1;
  }

  _E_Num = size/_TNum;
  if (_E_Num == 0)
    _E_Num = 1;


  
  //Test whether we have the right number of CPUs
  if((size%_TNum != 0))
  {
    if(rank == 0)
    {
      std::cerr << "###############################" << std::endl;
      std::cerr << "Error: invalid number of CPU's!" << std::endl;
	//std::cerr << "Default setup needs " << TNum * ChNum /2 << "  processes to run."<< std::endl;
      std::cerr << "Number of MPI processes should be a multiple of number of temperatures." << std::endl;
      std::cerr << "The minimum number of MPI processes for the current configuration is: " <<  _TNum << "." << std::endl;
      std::cerr << "###############################" << std::endl;
    }
    return;
  }

  // Initialize mpi
  if (_current_round == 0 )
    this->initialize_mpi();


  //Split the MPI_Processes
  if ( _current_round == 0 && !_restart){

    if (_initial.size() == 0){
      std::cerr << "ERROR: You must set the initial location of the sampler before you run it!";
      std::exit(1);
    }

    //this->initialize_mpi();

    _beta.resize(_TNum);
    _R.resize(_TNum,0.0);
    _avg_lklhd.resize(_TNum, 0.0);
    if (rank == MASTER){
      initialize_ladder();
      _anneal_out.open(_anneal_file, std::ios::out);
      _anneal_summary_out.open(_anneal_summary_file, std::ios::out);
      write_annealing_headers();
    }
  
    MPI_Bcast(&_beta[0], _TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::cout << "Done initializing the ladder " << rank << std::endl; 

    _L_tmp->set_beta(_beta[T_rank]);
    //initialize the chain.
    MPI_Barrier(MPI_COMM_WORLD);
    _sampler.set_initial_location(_initial);
    if (rank == MASTER){
      std::cout << "Finished Initialization\n";
    }
 
  }//end of if _current_round == 0

  //std::cerr << "Processor made it here: " << rank << std::endl;
  //open output streams
  if (rank == MASTER){
    std::stringstream header;
    header << "round" << std::setfill('0') << std::setw(3)
            << _current_round << "_";
    _sampler.set_output_stream(header.str()+_chain_file, 
                               header.str()+_state_file,
                               header.str()+_sampler_file,
                               _output_precision);
  }

  if (verbosity > 0){
    if ( (T_rank!=0) && (E_rank == 0) ){
      std::stringstream header; 
      header << "round" << std::setfill('0') << std::setw(3)
             << _current_round << "_";
      _sampler.set_output_stream(header.str()+_chain_file+"."+std::to_string(T_rank), 
                                 header.str()+_state_file+"."+std::to_string(T_rank),
                                 header.str()+_sampler_file+"."+std::to_string(T_rank),
                                 _output_precision);
      }
    }

  //If master rank we will output and otherwise it depends on verbosity.
  if (rank == MASTER){
    verbosity = 1;
  }else if (verbosity == 0){
    refresh = -1;
    verbosity = -1;
  }
  //All E_rank = 0 will do output if verbosity != 0.
  if (E_rank != 0){
    refresh = -1;
    verbosity = -1;
  }

  
 

  //Set the likelihood to use the new temperature.
  _L_tmp->set_beta(_beta[T_rank]);


  bool even_swap = true;
  //Main MCMC Loop
  for ( int ir = 0; ir < number_of_rounds; ++ir )
  {

    if ( !_restart ){
      //Zero the rejection rates and avg_lklhd
      for (int ti = 0; ti < _TNum; ++ti){
        _avg_lklhd[ti] = 0.0;
        _R[ti] = 0.0;
      }
    }

    if (rank == MASTER)
        std::cout << "On round " << _current_round << std::endl
                  << "Current number of swaps is " << _current_length << std::endl
                  << "which gives " << _current_length*_swap_stride << " steps"
                  << " for this round.\n";
    run_deo_round(_current_round, _current_length, thin, refresh, 
                  verbosity, even_swap);
    //average over the rejection rates.
    if (T_rank > 0)
      _R[T_rank - 1] /= double(_step_count+1e-10);
    
    MPI_Barrier(MPI_COMM_WORLD);

    double E = finish_deo_round(rank, _current_round);
    if ( rank == MASTER )
      std::cerr << "Done round "<< _current_round << " estimated round trip rate: " << 1.0/(2.0+2.0*E) << std::endl
                << "I estimate that you have completed: " << (_step_count)/(2.0+2.0*E) 
                << " round trips, hopefully this is greater than 1.\n";
    //Set the new tempature for the likelihood.
    _sampler.reset_sampler_step();
    _L_tmp->set_beta(_beta[T_rank]);
    
    
    _sampler.close_streams();
    //open output streams for next round
    if (rank == MASTER && ir < number_of_rounds - 1 ){

      std::stringstream header;
      header << "round" << std::setfill('0') << std::setw(3)
             << _current_round+1 << "_";
      _sampler.set_output_stream(header.str()+_chain_file, 
                                 header.str()+_state_file, 
                                 header.str()+_sampler_file,
                                 _output_precision);
    
    }
    if (verbosity > 0 && ir < number_of_rounds - 1){
      if ((T_rank!=0) && (E_rank == 0)){
        std::stringstream header;
        header << "round" << std::setfill('0') << std::setw(3)
               << _current_round+1 << "_";
        _sampler.set_output_stream(header.str()+_chain_file+"."+std::to_string(T_rank),
                                   header.str()+_state_file+"."+std::to_string(T_rank),
                                   header.str()+_sampler_file+"."+std::to_string(T_rank),
                                   _output_precision);  
     }
    }
    //Finished a round so move on.
    _step_count = 0;
    _current_round++;
    _current_length *= _b;
    MPI_Barrier(MPI_COMM_WORLD);
    //Write checkpoint at the end of the round.
    _restart = false;
    this->write_checkpoint(_ckpt_file);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

template<class S>
void sampler_deo_tempering_MCMC<S>::initialize_mpi()
{
  int color, flavour, rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == MASTER)
    std::cout << "Initializing DEO and splitting MPI processes\n";
  //color communicates between the samplers of the same temp.
  color = rank/(size/_TNum);
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &E_COMM);
  MPI_Comm_rank(E_COMM, &E_rank);
  MPI_Comm_size(E_COMM, &E_size);
 
  //Set the exploration sampler communicator.
  _sampler.set_mpi_communicator(E_COMM);
  

  //flavour communicates between the different temperatures.
  flavour = rank % (size/_TNum);
  MPI_Comm_split(MPI_COMM_WORLD, flavour, rank, &T_COMM);
  MPI_Comm_rank(T_COMM, &T_rank);
  MPI_Comm_size(T_COMM, &T_size);

  MPI_Bcast(&_seed,1,MPI_INT,0,MPI_COMM_WORLD);

  //Reset the seed to ensure that the likelihoods get the same random seed so they get the same position.
  if (E_rank == 0)
    _seed = _seed + T_rank;
  MPI_Bcast(&_seed,1,MPI_INT,0,E_COMM);
  _sampler.reset_rng_seed(_seed);
  MPI_Barrier(MPI_COMM_WORLD); 
  for(int k =-1; k< size; k++)
  {
    if (k<0 && rank==0)
      std::cout << "W  L  T  S" << std::endl;
    if (rank == k)
      std::cout << rank << "  " << E_rank << "  " << T_rank << " " 
                << _seed << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

template<class S>
void sampler_deo_tempering_MCMC<S>::run_deo_round(int round, int nswaps, int thin, int refresh,
                                          int verbosity, bool& even_swap
                                          )
{
  int total_steps = nswaps*_swap_stride;

  //Now do the adaptive steps
  for (int n = _step_count; n < nswaps; ++n){
    generate_transitions(_swap_stride, n*_swap_stride, total_steps, 
                         thin, refresh,
                         verbosity);
     
    this->_step_count++; 
    //MPI_Barrier(MPI_COMM_WORLD);
    this->deo_swap(even_swap, round, n);
    even_swap = !even_swap;

    if (n%_ckpt_stride_deo == 0 && _ckpt_stride_deo>0 && this->_step_count > 1){
      if ( T_rank == 0 && E_rank == 0 )
        std::cout << "Writing checkpoint\n";
      MPI_Barrier(MPI_COMM_WORLD);
      this->write_checkpoint(_ckpt_file);
    }
  }
}

template<class S>
void sampler_deo_tempering_MCMC<S>::write_checkpoint(std::ostream& out)
{
  std::cerr << "streamed write_checkpoint not implemented for DEO!\n";
}

template<class S>
void sampler_deo_tempering_MCMC<S>::write_checkpoint(std::string ckpt_file)
{
  if ( T_rank == MASTER && E_rank ==0 )
  {
    std::ifstream ickptback(ckpt_file);
    std::ofstream ockptback(ckpt_file+".bak");
    std::string bline;
    //Back up the checkpoint
    while(getline(ickptback,bline)){
      ockptback << bline << std::endl;
    }
    ickptback.close();
    ickptback.close();

    std::ofstream out(ckpt_file, std::ios::out);
    out.precision(16);
    //write the number of tempering levels and sampler processes
    out << _TNum << "  " << E_size << std::endl;
    //write the round and current length and step count
    out << _current_round
        << "  " << _current_length
        << "  " << _step_count
        << "  " << _b << std::endl;
    //Now write the temp, rej., rank 
    out << "#TT\n";
    out << T_rank << "  " << _beta[0] << "  " << _R[0] << std::endl;
    _sampler.write_checkpoint(out);
    out.close();
  }
  MPI_Barrier(T_COMM);
  //Now master collects the information from the other temperature levels and saves.
  for ( int i = 1; i < _TNum; ++i )
  {
    if ( T_rank == i && E_rank == 0 )
    {
      std::ofstream out(ckpt_file, std::ios::app);
      out << "#TT\n";
      out << T_rank << "  " << _beta[i] << "  " <<  _R[i-1] << std::endl;
      out.precision(16);
      _sampler.write_checkpoint(out);
      out.close();

    }
    MPI_Barrier(T_COMM);
  }
}

template<class S>
void sampler_deo_tempering_MCMC<S>::set_checkpoint(int ckpt_stride, std::string ckpt_file)
{
  sampler_MCMC_base::set_checkpoint(-1, ckpt_file);
  _sampler.set_checkpoint(-1,ckpt_file);
  _ckpt_stride_deo = ckpt_stride;
}

template<class S>
void sampler_deo_tempering_MCMC<S>::read_checkpoint(std::string ckpt_file)
{
  int rank, ssize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ssize);
  std::string word;
  std::ifstream in(ckpt_file);
  if ( !in.is_open() ){
    throw std::runtime_error("Could not open ckpt file: "+ckpt_file);
      //std::exit(1);
  }
  if (rank == MASTER)
  {
    in >> word;
    _TNum = std::stoi(word.c_str());
    in >> word;
    E_size = std::stoi(word.c_str());
    in >> word;
    _current_round = std::stoi(word.c_str());
    in >> word;
    _current_length = std::stoi(word.c_str());
    in >> word;
    _step_count = std::stoi(word.c_str());
    in >> word;
    _b = std::stoi(word.c_str());
    in.close();
  }
  MPI_Bcast(&_TNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&E_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&_current_round, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&_current_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&_step_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  //set _ENum here
  _E_Num = ssize/_TNum;
  if (_E_Num == 0)
    _E_Num = 1;
  

  if (rank==0)
    std::cerr << "Restart size info:"
	      << std::setw(15) << _TNum
	      << std::setw(15) << E_size
	      << std::setw(15) << _current_round
	      << std::setw(15) << _b
	      << '\n';

  std::cout << "Restarting DEO from " << _ckpt_file << std::endl;
  if ( _TNum*E_size != ssize){
    std::cerr << "The number of processes required for a restart in DEO differ\n"
              << "from those present in the checkpoint file!\n";
    std::exit(1);
  }
  
  // Now initialize mpi this is needed because of the sampler::read_checkpoint
  this->initialize_mpi();

  _beta.resize(_TNum,0.0);
  _R.resize(_TNum,0.0);
  _avg_lklhd.resize(_TNum, 0.0);
  for (int i = 0; i < _TNum; ++i)
    _R[i] = 0.0;

  // DEBUG
  /* std::cerr << "Before reading Trank: " << rank << ' ' << _beta.size() << ' ' << _R.size() << ' ' << _avg_lklhd.size() << '\n'; */
  /* std::cerr << "Ranks: " << rank << ' ' << T_rank << ' ' << E_rank << '\n'; */

  
  //Now read in the round information.
  for ( int i = 0; i < _TNum-1; ++i ){
    // if ( i == T_rank && E_rank ==0 ){
    if ( i == T_rank ){
      // DEBUG
      // std::cerr << "Started rank reads " << i << ' ' << rank << '\n';
      //load the file
      std::ifstream in(ckpt_file);
      std::string line;
      while (std::getline(in,line)){

	// DEBUG
	// std::cerr << "Line: " << rank << ' ' << line << '\n';
	
        if (line.rfind("#TT", 0) == 0) {
	  // std::cerr << "Found TT: " << rank << '\n';
	  
          std::string word;
          in >> word;
          int trank = std::stoi(word.c_str());
          in >> word;
          _beta[trank] = std::stod(word.c_str());
          in >> word;
          if (trank > 0){
            _R[trank-1] = std::stod(word.c_str());
          }
          if (trank == T_rank){
	    // DEBUG
	    // std::cerr << "Read trank before " << trank << ' ' << T_rank << ' ' << rank << '\n';

            _sampler.read_checkpoint(in);

	    // DEBUG
	    // std::cerr << "Read trank after " << trank << ' ' << T_rank << ' ' << rank << '\n';
          }
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

  }
  std::cerr << "Finished tranks " << T_rank  << ' ' << rank << '\n';
  
  MPI_Bcast(&_beta[0],_TNum, MPI_DOUBLE, 1, MPI_COMM_WORLD);
  MPI_Bcast(&_R[0],_TNum, MPI_DOUBLE, 1, MPI_COMM_WORLD);

  if (rank == MASTER){
    _R[0] = 0.0;
    _anneal_out.open(_anneal_file, std::ios::app);
    _anneal_summary_out.open(_anneal_summary_file, std::ios::app);
  }
  _restart = true;


  // Fix up the output streams
  //open output streams
  if (rank == MASTER){
    std::stringstream header;
    header << "round" << std::setfill('0') << std::setw(3)
            << _current_round << "_";
    _sampler.set_output_stream(header.str()+_chain_file, 
                               header.str()+_state_file,
                               header.str()+_sampler_file,
                               _output_precision);
  }
  //this->set_cpu_distribution(_TNum, E_size);
  //this->initialize_mpi();

}


template<class S>
double sampler_deo_tempering_MCMC<S>::finish_deo_round(int rank, int round)
{
    //Now average the likelihood
  if ( E_rank == 0 && T_rank < _TNum - 1){
    _avg_lklhd[T_rank] = _sampler.get_likelihood_sum()/_beta[T_rank];
    _sampler.reset_likelihood_sum();
  }
  //Reset the sampler step for the next round
  _sampler.reset_sampler_step();
  double E = 0.0;
  //Output the annealing round info
  double betaEnd, Rend;
  if ( rank == MASTER ){
    _anneal_out << std::setw(10) << round
                << std::setw(15) << _beta[0]
                << std::setw(15) << _avg_lklhd[0];
  }
  for(int n_t = 1; n_t < T_size; n_t++)
  {
    if ((T_rank == n_t) && (E_rank == 0)){
      MPI_Send(&_avg_lklhd[n_t], 1, MPI_DOUBLE, 0, n_t, T_COMM);
      MPI_Send(&_R[n_t-1],  1, MPI_DOUBLE, 0, T_size+n_t, T_COMM);
      MPI_Send(&_beta[n_t], 1, MPI_DOUBLE, 0, 2*T_size+n_t, T_COMM);
    }
    
    if(rank == MASTER){
      double avg_lklhd_end;
      MPI_Recv(&avg_lklhd_end, 1, MPI_DOUBLE, n_t, n_t, T_COMM, &Stat);
      MPI_Recv(&Rend, 1, MPI_DOUBLE, n_t, T_size+n_t, T_COMM, &Stat);
      MPI_Recv(&betaEnd, 1, MPI_DOUBLE, n_t, 2*T_size+n_t, T_COMM, &Stat);
      _avg_lklhd[n_t] = avg_lklhd_end;
      _beta[n_t] = betaEnd;
      _R[n_t-1] = Rend;
	
      _anneal_out << std::setw(15) << Rend << std::endl  
                  << std::setw(10) << round
	          << std::setw(15) << _beta[n_t]
                  << std::setw(15) << _avg_lklhd[n_t];
    }
  }
  if (rank==MASTER)
    _anneal_out << std::setw(15) << 0.0 << std::endl;
    //Now calculate the updated temperature ladder
    //Will do this on the master rank since we have already communicated with it for the output.
  if (rank==MASTER){
    std::cout << "Updating annealing schedule\n";
    double oroundtrip = update_annealing_params();
    double mean_rej = 0.0, std_rej = 0.0;
    E = 0.0;
    for ( int tt = 0; tt < _TNum-1; ++tt ){
      E += _R[tt]/(1.0-_R[tt]+1e-10);
      mean_rej += _R[tt];
    }
    double Lambda = mean_rej;
    mean_rej /= (_TNum-1.0);
    for ( int tt = 0; tt < _TNum-1; ++tt )
      std_rej = (_R[tt]-mean_rej)*(_R[tt]-mean_rej);
    std_rej = std::sqrt(std_rej/(_TNum-2.0));

    double evidence = estimate_bayesian_evidence();
    _anneal_summary_out << std::setw(5)  << round 
             << std::setw(15) << Lambda
             << std::setw(15) << oroundtrip
             << std::setw(15) << E
             << std::setw(15) << 1.0/(2.0+2.0*E)
             << std::setw(15) << mean_rej
             << std::setw(15) << std_rej  
             << std::setw(15) << evidence << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&_beta[0], _TNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return E;
}


template<class S>
void sampler_deo_tempering_MCMC<S>::generate_transitions(int niterations, int start, int finish, 
                                                 int thin, int refresh,
                                                 int verbosity)
{

  if (T_rank != _TNum-1){
    // DEBUG
    /* int world_rank; */
    /* MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); */
    /* std::cerr << world_rank << ' ' << world_rank << " Running exploration kernel " << '\n'; */
    _sampler.run_sampler(niterations, thin, refresh, verbosity);
  } else {
    for (int i = 0; i < niterations; ++i){
      //Draw from the reference distribution which right now is just a flat dist 
      std::vector<double> chain, state;
      _sampler.get_chain_state(chain, state);
      if (E_rank == 0){
        for (size_t n = 0; n < _dimension; ++n)
        { 
          double lower = _L->priors()[n]->lower_bound();
          double upper = _L->priors()[n]->upper_bound();
          if ( (std::isfinite(lower) && std::isfinite(upper)) ){
            chain[n] = lower + _rng.rand()*(upper - lower);
          }else if (std::isfinite(lower)){
            chain[n] = lower + _rng.rand()*1e20;
          }else if (std::isfinite(upper)){
            chain[n] = upper - _rng.rand()*1e20;
          }else{
          chain[n] = (_rng.rand()-0.5)*1e20;
          }
        }
      }
      MPI_Bcast(&chain[0], _dimension, MPI_DOUBLE, 0, E_COMM);
      //Check if L is indeed flat
      double L = _L_tmp->operator()(chain);
      double Lnbeta = _L_tmp->get_lklhd_no_temp();
      //std::cerr << "state size: " << state.size() << std::endl;
      //std::cerr << L << std::endl;
      state[0] = L;
      _avg_lklhd[T_rank] = (Lnbeta + _avg_lklhd[T_rank]*i)/(i+1);
      _sampler.set_chain_state(chain,state);
      
    }    
  }
}

template<class S>
void sampler_deo_tempering_MCMC<S>::deo_swap(const bool even_swap, const int round, const int swap_count)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool even_temp = (T_rank%2 == 0);
  //Get the chain state
  std::vector<double> preS, neiS, swapS, state;
  neiS.resize(_dimension,0.0);
  _sampler.get_chain_state(preS, state);
  //communicate likelihoods
  double preL=0.0, neiL=0.0, swapL;
  if (T_rank < _TNum - 1)
    preL = state[0]/_beta[T_rank]; //Find the likelihood without the power tempering
  else{
    preL = _L_tmp->get_lklhd_no_temp(); //Get the likelihood for the beta=0
  }
  if (T_rank < _TNum - 1)
    MPI_Send(&preL, 1, MPI_DOUBLE, T_rank+1, T_rank+1, T_COMM);
  
  if ( T_rank > 0 )
    MPI_Recv(&neiL, 1, MPI_DOUBLE, T_rank-1, T_rank, T_COMM, &Stat);
  
  /*
  std::ofstream error_out("error_"+std::to_string(T_rank)+"_r-"+std::to_string(round)+".dat", std::ios::app);
    error_out << "-------------------------------------------------------------------------------------------\n"
              << "swap count: " << swap_count << std::endl;
    error_out << "even swap: " << even_swap << std::endl;
    error_out << "even temp: " << even_temp << std::endl;;
    error_out << "pre swap:\n";
    error_out << std::setw(5) << "T_rank"
              << std::setw(15) << "_beta[T_rank]"
              << std::setw(15) << "_beta[T_rank-1]"
              << std::setw(15) << "preL"
              << std::setw(15) << "neiL" 
              << std::setw(15) << "mhratio" << std::endl
              << std::setw(5) << T_rank 
              << std::setw(15) << _beta[T_rank]
              << std::setw(15) << _beta[T_rank-1]
              << std::setw(15) << preL
              << std::setw(15) << neiL 
              << std::setw(15) 
              << std::exp(std::min(0.0 , (neiL - preL) * (_beta[T_rank] - _beta[T_rank-1]))) << std::endl;
  error_out << "state:\n";
  
  for (size_t p = 0; p < _dimension; ++p)
    error_out << std::setw(15) << preS[p];
  error_out << std::endl;
  */
  //Now decide if we need to swap
  int swap=0, nswap=0;
  //std::cerr << "Rank is here: " << rank << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (T_rank > 0 && E_rank==0){
    double alpha = _rng.rand();
    double mhRatio = std::exp(std::min(0.0 , (neiL - preL) * (_beta[T_rank] - _beta[T_rank-1])));
    _R[T_rank-1] += (1-mhRatio);
    //Swap is good!
    if(alpha < mhRatio && ((!even_swap&&even_temp)||(even_swap&&!even_temp)))
      swap = 1;
    
    std::cout << std::setw(10) << round
              << std::setw(10) << rank
              <<  std::setw(10)
              << swap_count
              << " / "
              << _current_length 
              << "  Tempering acceptance rate:  " 
              << mhRatio << "   beta:  " 
              << _beta[T_rank] << std::endl;
    
    MPI_Send(&swap, 1, MPI_INT, T_rank-1, T_rank, T_COMM);
  }
  
  if((T_rank < _TNum - 1) && (E_rank == 0)){
      MPI_Recv(&nswap, 1, MPI_INT, T_rank + 1, T_rank+1, T_COMM, &Stat); 
  }
  /*
  error_out << "swap: " << swap << std::endl;
  error_out << "nswap: " << nswap << std::endl;
  */
  //Now do the swapping
  if( (T_rank > 0) && (E_rank == 0)){
    if(swap == 1){
      //tempout << "For deo swap " << even_swap << " on temp rank " << T_rank-1 << "\n";
      MPI_Isend(&preS[0], _dimension, MPI_DOUBLE, T_rank - 1, 0, T_COMM, &Req);
      MPI_Recv(&neiS[0], _dimension, MPI_DOUBLE, T_rank - 1, 0, T_COMM, &Stat);
      MPI_Wait(&Req, &Stat);
    
      MPI_Isend(&preL, 1, MPI_DOUBLE, T_rank - 1, 1, T_COMM, &Req2);
      MPI_Recv(&swapL, 1, MPI_DOUBLE, T_rank - 1, 1, T_COMM, &Stat2);
      MPI_Wait(&Req2, &Stat2);
    
      /*
      tempout << '|' 
              << std::setw(15) << preL[m]
              << std::setw(15) << swapL;
      */
                                          
      preS = neiS;
      preL = swapL;
    }
  }
                                                                                         
  if( (T_rank < T_size - 1) && (E_rank == 0))
  {
    if((nswap) == 1)
    {
      MPI_Recv(&neiS[0], _dimension, MPI_DOUBLE, T_rank + 1, 0, T_COMM, &Stat);
      MPI_Send(&preS[0], _dimension, MPI_DOUBLE, T_rank + 1, 0, T_COMM);
      
      MPI_Recv(&swapL, 1, MPI_DOUBLE, T_rank + 1, 1, T_COMM, &Stat2);
      MPI_Send(&preL, 1, MPI_DOUBLE, T_rank + 1, 1, T_COMM);
      
      preS = neiS;
      preL = swapL;
    }
  }
  MPI_Bcast(&preS[0], _dimension, MPI_DOUBLE, 0, E_COMM);
  MPI_Bcast(&preL, 1, MPI_DOUBLE, 0, E_COMM);
  /*
  error_out << "post swap: " << std::endl;
  error_out << std::setw(5) << "T_rank"
            << std::setw(15) << "_beta[T_rank]"
            << std::setw(15) << "_beta[T_rank-1]"
            << std::setw(15) << "preL"
            << std::setw(15) << "neiL" 
            << std::setw(15) << "mhratio" << std::endl
            << std::setw(5) << T_rank 
            << std::setw(15) << _beta[T_rank]
            << std::setw(15) << _beta[T_rank-1]
            << std::setw(15) << preL
            << std::setw(15) << neiL 
            << std::setw(15) 
            << std::exp(std::min(0.0 , (neiL - preL) * (beta[T_rank] - beta[T_rank-1]))) << std::endl;
  error_out << "state:\n";
  for (size_t p = 0; p < _dimension; ++p)
    error_out << std::setw(15) << preS[p];
  error_out << std::endl;
  error_out.close();
  */
  state[0] = preL*_beta[T_rank];
  //std::cout << "End swap on: " << T_rank << std::endl;
  _sampler.set_chain_state(preS, state);
}                                                                                         


template<class S>
double sampler_deo_tempering_MCMC<S>::update_annealing_params()
{
  std::vector<double> Lambda(_TNum, 0.0);
  std::vector<double> betaRev(_TNum, 0.0);  
  for ( int i = 0; i < _TNum; ++i )
  {
    betaRev[i] = _beta[_TNum-1-i];
    for (int j = 0; j < i; ++j)
      Lambda[i] += _R[_TNum-1-j-1];
    //std::cerr << "adaption:    " << betaRev[i] << "    " << R[TNum-1-i] << "     " << Lambda[i] << std::endl;
  }


  Interpolator1D fLambda(betaRev,Lambda, "mcubic");
  for ( int k = 1; k < _TNum-1; ++k)
  {
    _beta[_TNum-1-k] = find_beta(fLambda, k, Lambda[_TNum-1]);  
    //std::cerr << "Found "<< k << " beta " << beta[TNum-1-k] << std::endl;
  }
  _beta[0] = 1.0;
  _beta[_TNum-1] = 0.0;
  return 1.0/(2.0*(1.0+Lambda[_TNum-1]));
}


template<class S>
double sampler_deo_tempering_MCMC<S>::find_beta(Interpolator1D& fLambda, int k, double Lambda, double eps, int MAX_ITR)
  {
    int i = 0;
    double N = _TNum-1;
    double betaMax = 1.0;
    double betaMin = 0.0;
    double c = 0.0;
    double fMax = fLambda(betaMax) - k/N*Lambda;
    double fMin = fLambda(betaMin) - k/N*Lambda;

    do{
      i++;
      double error = (betaMax-betaMin)/2;
      c = (betaMax+betaMin)/2.0;
      if (c/1000.0 < eps)
	eps /= 10.0;
      double fMid = fLambda(c) - k/N*Lambda;
      if (error < eps || fMid==0.0)
        return c;


      if (fMid*fMax > 0.0 )
      {
        betaMax = c;
        fMax = fMid;
      }
      else if (fMid*fMin>0.0)
      {
        betaMin = c;
        fMin = fMid; 
      }
    }while(i < MAX_ITR);

    std::cerr << "ERROR! sampler_deo_tempering_MCMC::Maximum iterations reached in find_temp, beta might be wrong\n";
    return c;
  
}


template<class S> 
void sampler_deo_tempering_MCMC<S>::write_annealing_headers()
{
  
  _anneal_out << "#" 
              << " nTemp = " << _TNum << std::endl
              << "#"
              << std::setw(10) << "round"
              << std::setw(15) << "Beta"
              << std::setw(15) << "avg_L"
              << std::setw(15) << "R" 
              << std::endl;


  _anneal_summary_out << "#"
                      << std::setw(5) << "round"
                      << std::setw(15) << "Lambda"
                      << std::setw(15) << "rt_opt"
                      << std::setw(15) << "E"
                      << std::setw(15) << "rt_est"
                      << std::setw(15) << "Avg(R)"
                      << std::setw(15) << "Std(R)" 
                      << std::setw(15) << "logZ" << std::endl;
  
}



}//end Themis

#endif
