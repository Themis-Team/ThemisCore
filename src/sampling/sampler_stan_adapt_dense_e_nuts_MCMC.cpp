/*!
  \file sampler_stan_adapt_dense_e_nuts_MCMC.cpp
  \author Paul Tiede
  \brief Implementation file for the sampler_stan_adapt_dense_e_nuts_MCMC.
*/

#include "sampler_stan_adapt_dense_e_nuts_MCMC.h"
#include "sampler_MCMC_base.h"
#include <iostream>
#include <string>
#include <cmath>

#include <stan/version.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/stan_csv_reader.hpp>
#include <stan/io/ends_with.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/services/diagnose/diagnose.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_dense_e_adapt.hpp>
#include <stan/services/sample/hmc_nuts_unit_e_adapt.hpp>
#include <stan/mcmc/chains.hpp>
#include <stan/services/util/validate_dense_inv_metric.hpp>

namespace Themis{

sampler_stan_adapt_dense_e_nuts_MCMC::sampler_stan_adapt_dense_e_nuts_MCMC(int seed, likelihood& L, std::vector<std::string> var_names, size_t dimension)
    : sampler_MCMC_base(seed, L, var_names, dimension),
      _stan_model(L, _var_names, &std::cout),
      _stan_rng(stan::services::util::create_rng(seed,1)),
      _stan_sampler(_stan_model, _stan_rng),
      _inverse_metric(dimension,dimension),
      _initial(dimension,0.0),
      _current_sample(Eigen::VectorXd(dimension),0.0,0.0),
      _logger(std::cout,std::cout,std::cout, std::cerr, std::cerr),
      _adaptation(false),
      _save_adaptation(true),
      _nadapt(1000),
      _restart(false)
{

  _stan_sampler.set_nominal_stepsize(1.0);
  _stan_sampler.set_stepsize_jitter(0.0);
  _stan_sampler.set_max_depth(10);

  _stan_sampler.get_stepsize_adaptation().set_mu(std::log(10.0));
  _stan_sampler.get_stepsize_adaptation().set_delta(0.65);
  _stan_sampler.get_stepsize_adaptation().set_gamma(0.05);
  _stan_sampler.get_stepsize_adaptation().set_kappa(0.75);
  _stan_sampler.get_stepsize_adaptation().set_t0(10.0);

  _init_buffer = 75;
  _term_buffer = 50;
  _window = 25;
  _inverse_metric = Eigen::MatrixXd::Identity(dimension,dimension);
  stan::services::util::validate_dense_inv_metric(_inverse_metric,_logger);
  _stan_sampler.set_metric(_inverse_metric);
  
}//end sampler_stan_adapt_dense_e_nuts_MCMC


void sampler_stan_adapt_dense_e_nuts_MCMC::set_initial_inverse_metric(Eigen::MatrixXd inv_metric)
{
  _inverse_metric = inv_metric;

  stan::services::util::validate_dense_inv_metric(inv_metric,_logger);
  _stan_sampler.set_metric(inv_metric); 
}


void sampler_stan_adapt_dense_e_nuts_MCMC::set_adaptation_parameters(int nadapts, bool save_adaptation)
{
  _nadapt = nadapts;
  _save_adaptation = save_adaptation;
}



void sampler_stan_adapt_dense_e_nuts_MCMC::set_initial_location(std::vector<double> initial_parameters)
{
  if (_dimension != initial_parameters.size())
  {
    std::cerr << "Dimension of the initial parameters does not equal dimension of the problem!";
    std::exit(1);
  }
  std::vector<std::vector<size_t> > dims(_dimension, std::vector<size_t> ());
  stan::io::array_var_context init_var(_var_names, initial_parameters, dims);
  std::ofstream out;
  stan::callbacks::stream_writer stan_writer(out, "#");
  std::shared_ptr<stan::io::var_context> init_context = std::make_shared<stan::io::array_var_context>(init_var); 
  std::vector<double> cont_vector = stan::services::util::initialize(
      _stan_model, init_var, _stan_rng, 0.0, true, _logger, stan_writer);
  //Create initial sample
  Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(), cont_vector.size());
  
  try{
    _stan_sampler.z().q = cont_params;
    _stan_sampler.init_stepsize(_logger);
  }catch (const::std::exception & e){
    _logger.info("Exception initializing step size.");
    _logger.info(e.what());
    return;
  }
  stan::mcmc::sample s(cont_params,0.0,0.0);
  _current_sample = s;
}

void sampler_stan_adapt_dense_e_nuts_MCMC::reset_rng_seed(int seed)
{
  _seed = seed;
  _stan_rng.seed(_seed);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_stepsize(double step_size)
{
  _stan_sampler.set_nominal_stepsize(step_size);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_delta(double delta)
{
  _stan_sampler.get_stepsize_adaptation().set_delta(delta);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_gamma(double gamma)
{
  _stan_sampler.get_stepsize_adaptation().set_gamma(gamma);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_kappa(double kappa)
{
  _stan_sampler.get_stepsize_adaptation().set_kappa(kappa);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_t0(double t0)
{
  _stan_sampler.get_stepsize_adaptation().set_t0(t0);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::set_window_parameters(unsigned int init_buffer, unsigned int term_buffer, unsigned int window)
{
  _init_buffer = init_buffer;
  _term_buffer = term_buffer;
  _window = window;
}


void sampler_stan_adapt_dense_e_nuts_MCMC::set_max_depth(int max_depth)
{
  _stan_sampler.set_max_depth(max_depth);
}

void sampler_stan_adapt_dense_e_nuts_MCMC::run_sampler(int nsteps, 
                                                      int thin, int refresh,
                                                      int verbosity)
{
  //Open the chains for output options
  MPI_Comm_rank(_comm, &_rank);
  if (_step_count == 0 && _rank == 0 && verbosity > -1){
    _chain_out.open(_chain_file);
    _state_out.open(_state_file);
    _sampler_out.open(_sampler_file);
    this->write_chain_header();
    this->write_state_header();
    this->write_sampler_header();
  } else if (_restart && _rank == 0 && verbosity > -1){
    _chain_out.open(_chain_file, std::ios::app);
    _state_out.open(_state_file, std::ios::app);
    _sampler_out.open(_sampler_file, std::ios::app);
  }

  
  bool save = verbosity > -1 ? true : false;
  //Now lets see if we should do some adaptation
  if ( (_step_count == 0 && _nadapt > 0) || 
       ( _nadapt>0 && _restart && _step_count < (size_t)_nadapt)){
    _adaptation = true;
    save = _save_adaptation;
    _stan_sampler.set_window_params(_nadapt, _init_buffer, _term_buffer, _window, _logger);
    _stan_sampler.engage_adaptation();
    if (verbosity > 0){
      _sampler_out << "Number of adaptation steps: "+std::to_string(_nadapt) << std::endl;
    }
  } else if ( _step_count < (size_t)_nadapt ){
    _adaptation = true;
    save = _save_adaptation&&save;
  }

  //Now sample!
  for ( int i = 0; i < nsteps; ++i ){
    if (_step_count == (size_t)_nadapt){
      save = verbosity > -1 ? true : false;
    }
    generate_transition(i, 0, nsteps, thin, refresh, save, _adaptation, verbosity);
    _step_count++;
  }
  //Turn off restart since sampler finished 
  _restart = false;
}

void sampler_stan_adapt_dense_e_nuts_MCMC::generate_transition( int step, int start, int finish,
                                                               int nthin, int refresh, bool save,
                                                               bool adaptation,
                                                               int verbosity)
{
    clock_t start_t = clock();
    _current_sample = _stan_sampler.transition(_current_sample, _logger);
    clock_t end_t  = clock();
    if (!adaptation){
      _sum_lklhd += _current_sample.log_prob();
    } 
  if ( verbosity > -1 ){
    if (refresh > 0
          && (start + step + 1 == finish || step == 0 || (step + 1) % refresh == 0)) {
        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        std::stringstream message;
        message << "Iteration: ";
        message << std::setw(it_print_width) << step + 1 + start << " / " << finish;
        message << " [" << std::setw(3)
                << static_cast<int>((100.0 * (start + step + 1)) / finish) << "%] ";
        message << (adaptation ? " (Warmup)" : " (Sampling)");
        message << "Last step took " << (end_t-start_t)/CLOCKS_PER_SEC << " s.";
        _logger.info(message);
    }
  }
  
  //If reached adaptation end
  if ( (_step_count == (size_t)_nadapt) && (_nadapt > 0)){
    _stan_sampler.disengage_adaptation();
    reset_likelihood_sum();
    stan::callbacks::stream_writer stan_writer(_sampler_out,"");
    _sampler_out << "Post adaptation tuning parameters\n";
    if (verbosity > 0)
      if (_rank == 0){
      _stan_sampler.write_sampler_state(stan_writer);    
    }
  }

  //Checkpoint 
  if (_ckpt_stride > 0 && ((start+_step_count)%_ckpt_stride == 0) && _rank ==0){
    std::ofstream ckpt_out(_ckpt_file);
    ckpt_out.precision(16); 
    write_checkpoint(ckpt_out);
    ckpt_out.close();
    
  }

  //Write the state of the sampler
  if (save && (((start+_step_count)%nthin) == 0) && _rank == 0){
    write_state();
  }
  
}

void sampler_stan_adapt_dense_e_nuts_MCMC::write_checkpoint(std::ostream& out)
{
  //First output the mass matrix
  out << _dimension << std::endl;
  out << _step_count << std::endl;
  out << _nadapt << std::endl;
  stan::callbacks::stream_writer metric_writer(out, " ");
  Eigen::MatrixXd metric = _stan_sampler.z().inv_e_metric_;
  out << metric << std::endl;
  //Now write the sampler adaptation parameters.
  out << std::setw(25) << _stan_sampler.get_stepsize_adaptation().get_gamma()
      << std::setw(25) << _stan_sampler.get_stepsize_adaptation().get_delta()
      << std::setw(25) << _stan_sampler.get_stepsize_adaptation().get_kappa()
      << std::setw(25) << _stan_sampler.get_stepsize_adaptation().get_t0()
      << std::setw(15) << _init_buffer
      << std::setw(15) << _term_buffer
      << std::setw(15) << _window
      << std::setw(15) << _stan_sampler.get_max_depth()
      << std::setw(25) << _stan_sampler.get_current_stepsize()
      << std::setw(25) << _stan_sampler.get_stepsize_jitter() << std::endl;
 
  //Now get the chain and state
  std::vector<double> chain, state;
  this->get_chain_state(chain,state);
  //Now output the chain and state
  for ( size_t i = 0; i < chain.size(); ++i )
    out << std::setw(25) << chain[i];
  out << std::endl;
  for ( size_t i = 0; i < state.size(); ++i )
    out << std::setw(25) << state[i];
  out << std::endl;
}

void sampler_stan_adapt_dense_e_nuts_MCMC::read_checkpoint(std::istream& ckpt_in)
{
  MPI_Comm_rank(_comm, &_rank);
  std::vector<double> chain(_dimension), state;
  std::vector<std::vector<double> > metric(_dimension,std::vector<double> (_dimension,0.0));
  double gamma, delta, kappa, t0, epsilon, epsilon_jitter;
  int max_depth;
  size_t dimension, step_count, nadapt;
  if (_rank == 0){
    std::string line, word, dummy;
    //Get the dimension
    ckpt_in >> word;
    dimension = std::stoi(word.c_str());
    if (dimension != _dimension){
      std::cerr << "Model dimension for sampler does not equal dimension in the checkpoint!\n"
                << "Are you sure you are restarting the same problem?\n";
      std::exit(1);
    }
    //Get the step count
    ckpt_in >> word;
    step_count = std::stoi(word.c_str());

    //Get the nadapt
    ckpt_in >> word;
    nadapt = std::stoi(word.c_str());
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //Now get the metric
    for ( size_t j = 0; j < _dimension; ++j ){
      for ( size_t i = 0; i < _dimension; ++i ){
        ckpt_in >> word;
        //std::cout << std::setw(20) << word;
        double gii = std::stod(word.c_str());
        metric[j][i] = gii;
      }
      //std::cout <<std::endl;
    }
    //std::cout << "done metric\n";
    //Now get the sampler adaptation information
    ckpt_in >> word;
    gamma = std::stod(word.c_str());
    ckpt_in >> word;
    delta = std::stod(word.c_str());
    ckpt_in >> word;
    kappa = std::stod(word.c_str());
    ckpt_in >> word;
    t0 = std::stod(word.c_str());
    ckpt_in >> word;
    _init_buffer = std::stoi(word.c_str());
    ckpt_in >> word;
    _term_buffer = std::stoi(word.c_str());
    ckpt_in >> word;
    _window = std::stoi(word.c_str());
    ckpt_in >> word;
    max_depth = std::stoi(word.c_str());
    ckpt_in >> word;
    epsilon = std::stod(word.c_str());
    ckpt_in >> word;
    epsilon_jitter = std::stod(word.c_str());

    //Now get the chain state
    std::cerr << line << std::endl;
    for ( size_t i = 0; i < _dimension; ++i ){
      ckpt_in >> word;
      //std::cout << word << std::endl;
      double chaini = std::stod(word.c_str());
      chain[i] = chaini;
    }
    ckpt_in.ignore(4096,'\n');
    //Now get the sampler state
    std::getline(ckpt_in, line);
    std::istringstream iss4(line);
    //std::cerr << line << std::endl;
    while (iss4 >> word){
      state.push_back(std::stod(word.c_str()));
    }

  }
  _restart = true;
  int state_size = state.size(); 
  //Now broadcast the results.
  MPI_Bcast(&dimension, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&step_count, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&nadapt, 1, MPI_INT, 0, _comm);
 
  MPI_Bcast(&metric[0], dimension*dimension, MPI_DOUBLE, 0, _comm);
  for ( size_t i = 0; i < _dimension; ++i )
    for (size_t j = 0; j < _dimension; ++j)
      _inverse_metric(i,j) = metric[i][j];
  MPI_Bcast(&gamma, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&_init_buffer, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&_term_buffer, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&_window, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&max_depth, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&epsilon_jitter, 1, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&chain[0], dimension, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&state_size, 1, MPI_INT, 0, _comm);
  if (_rank != 0){
    state.resize(state_size);
  }
  if (step_count < nadapt){
    if ( _rank == 0 ){
    std::cout << "sampler_stan_adapt_dense_e_nuts_MCMC::read_checkpoint Adaptation phase didn't finish last time!\n"
              << "Restarting the sampler in the adaptation state.\n"
              << "If you want a different number of adaptation parameters please " 
              << "set it using set_adaptation_parameters.\n";
    }
    _step_count = 1;
    _stan_sampler.set_window_params(nadapt, _init_buffer, _term_buffer, _window, _logger);
    //_stan_sampler.engage_adaptation();
    
  } else{
    _step_count = step_count;
    _nadapt = nadapt;
    //_stan_sampler.disengage_adaptation();
  }

  MPI_Bcast(&state[0], state_size, MPI_DOUBLE, 0, _comm);
  
  MPI_Barrier(_comm);
  this->set_initial_inverse_metric(_inverse_metric);
  _stan_sampler.set_nominal_stepsize(epsilon);
  _stan_sampler.set_stepsize_jitter(epsilon_jitter);
  _stan_sampler.set_max_depth(max_depth);

  _stan_sampler.get_stepsize_adaptation().set_mu(std::log(10.0)*epsilon);
  _stan_sampler.get_stepsize_adaptation().set_delta(delta);
  _stan_sampler.get_stepsize_adaptation().set_gamma(gamma);
  _stan_sampler.get_stepsize_adaptation().set_kappa(kappa);
  _stan_sampler.get_stepsize_adaptation().set_t0(t0);
  _stan_sampler.set_nominal_stepsize(epsilon);
  //Now update the stepsize and energy manually.
  _stan_sampler.sample_stepsize();
  _stan_sampler.energy_=state[state.size()-1];
  this->set_chain_state(chain,state);
  write_state();
}


void sampler_stan_adapt_dense_e_nuts_MCMC::get_chain_state(std::vector<double>& model_values, std::vector<double>& state_values)
{
  model_values.resize(_dimension);
  //get the parameters at the current sample
  std::vector<double> cont_params(
          _current_sample.cont_params().data(),
          _current_sample.cont_params().data() + _current_sample.cont_params().size());
  //transform the parameters to model space from continuous space
  double log_jacobian = 0.0;
  //change the model_values to live in the continuous space.
  for ( size_t i = 0; i < _dimension; ++i )
  {
    double jac = 0.0;
    //get the constrained initial position using the context.
    double lower = _L->priors()[i]->lower_bound();
    double upper = _L->priors()[i]->upper_bound();
    double param_scalar = cont_params[i];
    if ( (std::isfinite(lower)&&std::isfinite(upper)) )
      model_values[i] = stan::math::lub_constrain(param_scalar, lower, upper, jac);
    else if (std::isfinite(lower))
      model_values[i] = stan::math::lb_constrain(param_scalar, lower, jac);
    else if (std::isfinite(upper))
      model_values[i] = stan::math::ub_constrain(param_scalar, upper, jac);
    else
      model_values[i] = param_scalar;
    log_jacobian += jac;
  }
  //Get the state values, i.e. the likelihood and hmc specific stuff
  state_values.resize(0);
  _current_sample.get_sample_params(state_values);
  _stan_sampler.get_sampler_params(state_values);

  //Now I need to fix the likelihood so it is the likelihood for the actual parms not the continuous versions.
  state_values[0] -= log_jacobian;
}



void sampler_stan_adapt_dense_e_nuts_MCMC::set_chain_state(std::vector<double> model_values, std::vector<double> state_values)
{
  double log_jacobian = 0.0;
  //change the model_values to live in the continuous space.
  Eigen::VectorXd cont_params(_dimension);
  for ( size_t i = 0; i < _dimension; ++i )
  {
    double jac = 0.0;
    //get the constrained initial position using the context.
    double lower = _L->priors()[i]->lower_bound();
    double upper = _L->priors()[i]->upper_bound();
    double param_scalar = model_values[i];
    if ( (std::isfinite(lower)&&std::isfinite(upper)) ){
      cont_params(i) = stan::math::lub_free(param_scalar, lower, upper);
      double z = (param_scalar-lower)/(upper-lower);
      jac = -std::log(upper-lower) - std::log(z*(1-z)+1e-50);
    }
    else if (std::isfinite(lower)){
      cont_params(i) = stan::math::lb_free(param_scalar, lower);
      jac = -std::log(cont_params(i)+1e-50);
    }
    else if (std::isfinite(upper)){
      cont_params(i) = stan::math::ub_free(param_scalar, upper);
      jac = -std::log(cont_params(i)+1e-50);
    }
    else
      cont_params(i) = param_scalar;
    log_jacobian += jac;
  }
  //Now add the jacobian. 
  stan::mcmc::sample s(cont_params, state_values[0]-log_jacobian, state_values[1]);
  _stan_sampler.z().q = cont_params;
  _stan_sampler.energy_ -= -_current_sample.log_prob();
  _stan_sampler.energy_ += -state_values[0]+log_jacobian;
  
  _current_sample = s; 
}

void sampler_stan_adapt_dense_e_nuts_MCMC::write_state()
{
  std::vector<double> model_values, state_values;
  get_chain_state(model_values, state_values);
  //Now we output the values
  for ( size_t i = 0; i < _dimension; ++i )
    _chain_out << std::setw(15) << model_values[i];
  _chain_out << std::endl;

  for (size_t i = 0; i < state_values.size(); ++i)
    _state_out << std::setw(15) << state_values[i];
  _state_out << std::endl;
}

void sampler_stan_adapt_dense_e_nuts_MCMC::write_chain_header()
{
  _chain_out << "#chainfmt stan\n";
  sampler_MCMC_base::write_chain_header();
}

void sampler_stan_adapt_dense_e_nuts_MCMC::write_state_header()
{
  _state_out << "#lklhdfmt stan\n";
  std::vector<std::string> sampler_param_names;
  _current_sample.get_sample_param_names(sampler_param_names);
  _stan_sampler.get_sampler_param_names(sampler_param_names);
  _state_out << "#";
  for ( size_t i = 0; i < sampler_param_names.size(); ++i )
    _state_out << std::setw(15) << sampler_param_names[i];
  _state_out << std::endl;
}

void sampler_stan_adapt_dense_e_nuts_MCMC::write_sampler_header()
{
  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if (id == 0){
    std::stringstream ss;
    _sampler_out << "Metric: " << std::endl
                 << _inverse_metric << std::endl;
    _sampler_out << "sampler = densestan\n";
    _sampler_out << "Adaptation tuning parameters: \n";
    _sampler_out << "        nadapt = " << _nadapt 
                << std::endl
                << "        gamma = " + std::to_string(_stan_sampler.get_stepsize_adaptation().get_gamma()) 
                << std::endl
                << "        delta = " + std::to_string(_stan_sampler.get_stepsize_adaptation().get_delta()) 
                << std::endl
                << "        kappa = " + std::to_string(_stan_sampler.get_stepsize_adaptation().get_kappa()) 
                << std::endl
                << "        t0 = " + std::to_string(_stan_sampler.get_stepsize_adaptation().get_t0()) 
                << std::endl
                << "        init_buffer = " + std::to_string(_init_buffer) 
                << std::endl
                << "        term_buffer = " + std::to_string(_term_buffer)
                << std::endl
                << "        max_depth = " + std::to_string(_stan_sampler.get_max_depth())
                << std::endl
                << "        jitter = " + std::to_string(_stan_sampler.get_stepsize_jitter())
                << std::endl
                << "HMC Tuning Parameters before adaptation:\n"
                << "        stepsize = " + std::to_string(_stan_sampler.get_current_stepsize())
                << std::endl
                << "        dense cov = " << ss.str()
                << std::endl;
              
  }
}


}//end Themis
