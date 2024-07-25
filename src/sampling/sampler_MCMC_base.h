/*!
  \file sampler_MCMC_base.h
  \author Paul Tiede
  \brief Header file for base markov chain monte carlo sampler. 
  \details This forms the basis/interface for all samplers using mcmc. 
*/


#ifndef THEMIS_SAMPLER_MCMC_BASE_H
#define THEMIS_SAMPLER_MCMC_BASE_H

#include <vector>
#include <string>
#include <fstream>
#include "likelihood.h"
#include "random_number_generator.h"

namespace Themis{

/*
  \class sampler_MCMC_base
  \brief Base class for all mcmc based samplers using mcmc in Themis.

  \details This forms the abstract class for all Themis samplers using MCMC methods.
  This defines the interfaces that must be implemented in all future samplers.
*/

class sampler_MCMC_base
{
public:
  /*
     \brief Initializes the sampler.
     \param seed The seed for the random number generator.
     \param L A Themis likelihood to be be sampled from.
     \param var_names vector of variable names. If empty it will fill default to p{0..dimension}.
     \param dimension Dimension of the problem.
  */
  sampler_MCMC_base(int seed, likelihood& L, std::vector<std::string> var_names, size_t dimension);
  ~sampler_MCMC_base(){};


  /*
    \brief sets starting location of the sampler
    \params initial_parameters A vector containing the parameters for the starting point.
  */
  virtual void set_initial_location(std::vector<double> initial_parameters) = 0;
 

  /*
    \brief Main run function that will run the sampler.
    \param nsteps Number of steps to run the sampler.
    \param thin   Will save every thin steps.
    \param refresh Will save ouput every refresh steps. If =-1 will never save.
    \param verbosity How much progress you want to save. This differs across each implemented sampler! 
  */
  virtual void run_sampler(int nsteps,
                   int thin, int refresh,
                   int verbosity = 1) = 0;


  /*!
   \brief Sets the output names for the chains and sampler states. This must be implemented
    in any classes inheriting sampler_exploration_mcmc_base.
    \param chain_name File name for the chain during the run.
    \param state_name File name for the state of the sampler, i.e. likelihood value
                       and other sampler specific output.
    \param sampler_name File name that stores sampler information for a run, e.g. the adaptation parameters.
    \param output_precision Sets how many decimal points are saved for floats. Default is 8. 
  */
  virtual void set_output_stream(std::string chain_name, std::string state_name,
                         std::string sampler_name,
                         int output_precision = 8); 


  /*!
    \brief Finds the best fit within provided chain file and returns it.
    \param chain_file The file name of the chain.
    \param state_file The file name of the saved state.
    \note if no chain_file or state_file is provided it will just read in the one that was saved using
    set_output_stream. This also assumes that the first element of the state vector is the log-likelihood value.
  */
  virtual std::vector<double> find_best_fit(std::string chain_file, std::string state_file) const;
  /*!
    \brief Finds the best fit from the chain and state file saved in set_output_stream. 
    \note Only run this after calling run sampler.
  */
  virtual std::vector<double> find_best_fit() const;



  ///////////////////////////////////////////////////////////////
  //   Checkpoint tools

  /*!
    \brief Sets the checkpoint/restart functionality.
    \param ckpt_stride Number of steps between writing a new checkpoint.
    \param ckpt_file String variable holding the name of the output checkpoint file.
    \note If this isn't called no checkpointing will occur, since ckpt_stride=-1 by default.
   */
  void set_checkpoint(int ckpt_stride, std::string ckpt_file);
  /*<!
    \brief Writes the sampler checkpoint the ckpt_out stream. This must be implemented inherited 
    sampler.
    \param out The output stream for the checkpoint.
  */
  virtual void write_checkpoint(std::ostream& out) = 0;
  /*!
    \brief Restarts the sampler from the state specified in the ckpt_file.
    \param ckpt_file The file where the checkpoint information was saved.
  */
  virtual void read_checkpoint(std::string ckpt_file);
  /*
    \brief Loads the ckpt from a stream. This will set the sampler to the state given in the ckpt.
    \param ckpt_in A reference to the input stream. 
  */
  virtual void read_checkpoint(std::istream& ckpt_in) = 0;





  /////////////////////////////////////////////////////////////////////////////////
  //    These are helper functions and really only need to be used for implementations
  //
  //


  /*
    \brief Resets the sampler step count. Like starting a new run but with potentially
    previous adaptation taken into account.
  */
  virtual void reset_sampler_step(){_step_count = 0;}
  /*
    \brief Resets totaled likelihood to zero.
  */
  void reset_likelihood_sum(){_sum_lklhd = 0.0;};

  virtual int get_step_count(){return _step_count;};

  
  /*
    \brief gets the totaled likelihood.
  */
  double get_likelihood_sum(){return _sum_lklhd;};

  /*
    \brief gets a random number from the rng defined in the sampler.
  */
  virtual double get_rand(){return _rng.rand();};

  /*
    \brief reset the seed of the sampler. This is important for parallelizing the likelihood evaluation.
    \param seed The seed for the pseudo random number generator.
  */
  virtual void reset_rng_seed(int seed){_seed = seed;_rng.reset_seed(seed);};

  /*
    \brief Access function that grabs the current state of the sampler, i.e. the chain, 
    and state of the sampler, i.e. the likelihood value of the chain and other pertinent information.
    Every sampler that inherits sampler_exploration_mcmc_base must define this.
    about the sampler.
    \param chain_state vector containing the current parameter values of the chain.
    \param sampler_state vector containing the current state of the sampler, i.e. likelihood value, and other pertinent information.
  */
  virtual void get_chain_state(std::vector<double>& parameters, std::vector<double>& sample_state) = 0;

  /*
    \brief Access function that resets the current state of the sampler, i.e. the chain, 
    and state of the sampler, i.e. the likelihood value of the chain and other pertinent information.
    Every sampler that inherits sampler_exploration_mcmc_base must define this.
    about the sampler.
    \param chain_state vector containing the current parameter values of the chain.
    \param sampler_state vector containing the current state of the sampler, i.e. likelihood value, and other pertinent information.
  */
  virtual void set_chain_state(std::vector<double> parameters, std::vector<double> sample_state) = 0;

  /*!
    \brief Sets the MPI_Comm to be used for the likelihood evaluation.
    \param comm The MPI communicator for the likelihood evaluation.
  */
  virtual void set_mpi_communicator(MPI_Comm comm);

  /*!
    \brief Helper function that closes the streams from a sampler.
  */
  void close_streams(){_chain_out.close();_state_out.close();_sampler_out.close();};

protected: 
  /*
    \brief Utility function for writing the state of the chain to files from set_output_stream.
    \note Must be defined by child classes.
  */
  virtual void write_state() = 0;
  /*
    \brief Writes the header to the chain file. The default is to write var_names at the top
  */
  virtual void write_chain_header();
  /*
    \brief Write the header to the state file. The default is to just give the first column,
    which must be the likelihood value, and leave the rest blank
  */
  virtual void write_state_header();

  // Random number generator
  int _seed;
  Ran2RNG _rng;
  // Stores the variable names for the sampler
  std::vector<std::string> _var_names;
  // Stores the dimension of the problem
  size_t _dimension;
  size_t _step_count;
  
  // Stores a pointer to the likelihood to be used for computations
  likelihood* _L;
  // Streams to the output for the chains and state.  
  std::string _chain_file, _state_file, _sampler_file, _ckpt_file;
  std::ofstream _chain_out, _state_out, _sampler_out;
  int _output_precision, _ckpt_stride;
  // number of adaptation steps
  double _sum_lklhd;

  // Holds the MPI communicator.
  MPI_Comm _comm;
  int _rank;


};

}//end Themis
#endif
