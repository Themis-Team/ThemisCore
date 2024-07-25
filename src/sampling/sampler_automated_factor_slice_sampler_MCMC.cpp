/*!
  \file sampler_automated_factor_slice_sampler_MCMC.cpp
  \author Paul Tiede
  \brief Implementation file for the sampler_automated_factor_slice_sampler_MCMC.
*/

#include "sampler_automated_factor_slice_sampler_MCMC.h"
#include "sampler_MCMC_base.h"
#include "transform_base.h"
#include "transform_logit.h"
#include "transform_none.h"
#include <ctime>
#include <iomanip>
#include <string>
#include <cmath>
#include <sstream>
#include <algorithm>
namespace Themis{

sampler_automated_factor_slice_sampler_MCMC::sampler_automated_factor_slice_sampler_MCMC(int seed, likelihood& L, std::vector<std::string> var_names, size_t dimension)
  : sampler_MCMC_base(seed, L, var_names, dimension),
    _current_parameters(dimension, 0.0),
    _current_state(5, 0.0),
    _adaptation(false),
    _save_adaptation(true),
    _nadapt(10000),
    _width_tol(0.1),
    _cov_tol(1e-3),
    _width_adapt_stride(1),
    _adapt_width_count(0),
    _adapt_cov_count(1),
    _restart(false),
    _init_buffer(10),
    _window(dimension*dimension/2)
{
  _mean = Eigen::VectorXd::Zero(dimension);
  _covariance = Eigen::MatrixXd::Zero(dimension,dimension);
  _Gammak = Eigen::MatrixXd::Identity(dimension, dimension);
  _Dk = Eigen::VectorXd::Ones(_dimension);
  _sum_lklhd=0.0;


  //Initial widths will be determined by the prior range. 
  _transforms.resize(0);
  _initial_width.resize(dimension);
  _lbound.resize(dimension,0.0);_ubound.resize(dimension,0.0);
  for ( size_t i = 0; i < dimension; ++i )
  {
    double upper = L.priors()[i]->upper_bound();
    double lower = L.priors()[i]->lower_bound();
    _lbound[i] = lower;
    _ubound[i] = upper;

    if ( (std::isfinite(lower)) && std::isfinite(upper) ){
      _transforms.push_back(new transform_logit(lower, upper));
      //_initial_width[i] = (upper-lower)/4.0;
      //_transforms.push_back(new transform_none());//transform_logit(lower, upper));
      _initial_width[i] = 10.0;
    } else if ( (std::isfinite(lower)) ){
      _transforms.push_back(new transform_none());
      _initial_width[i] = lower*(1+1e2);
    } else if ( std::isfinite(upper) ){
      _transforms.push_back(new transform_none());
      _initial_width[i] = upper*(1+1e2);
    } else {
      _transforms.push_back(new transform_none());
      _initial_width[i] = 1000.0;
    }
  }
  _nexpansion.resize(0);
  _ncontraction.resize(0);
  _width_adapt.resize(0);
  _nexpansion.resize(dimension, 0.0);
  _ncontraction.resize(dimension, 0.0);
  _width_adapt.resize(dimension,true);
}

void sampler_automated_factor_slice_sampler_MCMC::set_window_parameters(int init_buffer, int window)
{
  _init_buffer = init_buffer;
  _window = window;
}

void sampler_automated_factor_slice_sampler_MCMC::set_intial_widths(std::vector<double> initial_widths)
{
  //We need to transform to our sampling space.
  //I will base this around the center of the prior to estimate the width.
  if ( initial_widths.size() != _dimension )
  {
    std::cerr << "sampler_automated_factor_slice_sampler_MCMC::set_intial_widths: initial_width must have same size as dimension of parameter space!\n";
    std::exit(1);
  }
  _initial_width.resize(_dimension);
  for ( size_t i = 0; i < _dimension; ++i ){
    double c = (_ubound[i] - _lbound[i])/2 + _lbound[i];
    _initial_width[i] = _transforms[i]->forward(c+initial_widths[i]/8.0)
                            -_transforms[i]->forward(c-initial_widths[i]/8.0);
  }
}

void sampler_automated_factor_slice_sampler_MCMC::set_initial_covariance(Eigen::MatrixXd covariance)
{
  if ( (covariance.rows() == covariance.cols()) && (covariance.rows() == (int)_dimension)){
    _covariance = covariance;
  } else{
    std::cerr << "Covariance matrix must be a square matrix\n" 
              << "with the same number of rows as the dimension of the problem.\n";
    std::exit(1);
  }
}

void sampler_automated_factor_slice_sampler_MCMC::set_adaptation_parameters(int nadapts, bool save_adaptation)
{
  _nadapt = nadapts;
  _save_adaptation = save_adaptation;
}

double sampler_automated_factor_slice_sampler_MCMC::loglklhd_transform(const std::vector<double> cont_params)
{
  
  double jacobian = 0.0;
  std::vector<double> model_params(_dimension,0.0);
  //std::cerr << "Model params: ";
  
  for ( size_t i = 0; i < _dimension; ++i ){
    model_params[i] = _transforms[i]->inverse(cont_params[i]);
    //std::cerr << std::setw(15) << model_params[i];
    jacobian += std::log(std::fabs(_transforms[i]->inverse_jacobian(cont_params[i])));
    //std::cerr << std::setw(15) << jacobian;
    if (!std::isfinite(jacobian))
    {
      return -std::numeric_limits<double>::infinity();
    }
  }
  
 // std::cerr << std::endl;
  
  double Lt = _L->operator()(model_params);
  if (std::isfinite(Lt))
      return Lt + jacobian;
  else
      return -std::numeric_limits<double>::infinity();
}

void sampler_automated_factor_slice_sampler_MCMC::set_initial_location(std::vector<double> initial_parameters)
{
  if (_dimension != initial_parameters.size())
  {
    std::cerr << "Dimension of the initial parameters does not equal dimension of the problem!";
    std::exit(1);
  }

  _current_parameters.resize(_dimension);
  _current_state.resize(5);
  _current_state[0] = 0.0;
  _current_state[1] = 0.0;
  _current_state[2] = 0.0;
  _current_state[3] = 0.0;
  _current_state[4] = 0.0;
  for ( size_t i = 0; i < _dimension; ++i ){
    if (_L->transform_state())
      _L->forward_transform(initial_parameters);
    _current_parameters[i] = _transforms[i]->forward(initial_parameters[i]);
    _mean(i) = _current_parameters[i];
  }
}

void sampler_automated_factor_slice_sampler_MCMC::set_adaptation_schedule()
{
  _number_slow_adapt = std::log2((double(_nadapt)-double(_init_buffer))/_window)-1;
  if (_number_slow_adapt <= 0){
    std::cerr << "Warning not enough adaptation steps for the factor adaptation\n"
              << "Going to just adapt the slice widths\n";
    _init_buffer = _nadapt; 
    _window = 0;
  }
  _width_adapt_stride = 1;
  _cov_adapt_stride = _window;
  _adapt_cov_count = 1;
  _covariance = Eigen::MatrixXd::Zero(_dimension,_dimension);
  _adapt_width_count = 0;
}

void sampler_automated_factor_slice_sampler_MCMC::reset_sampler_step()
{
  _step_count = 0;
}


void sampler_automated_factor_slice_sampler_MCMC::run_sampler(int nsteps, int thin, int refresh, int verbosity)
{
  MPI_Comm_rank(_comm, &_rank);
  if (_step_count == 0){
    double l = this->loglklhd_transform(_current_parameters);
    _current_state[0] = l;
  }
  //std::cout << "Entering sampler\n";
  //std::cout << nsteps << std::endl;

  // DEBUG
  // int world_rank;
  // MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
  // std::cerr << "Starting AFSS.run_sampler: " << world_rank << ' ' << _rank << " +" << _state_file  << "+ " << _restart << ' ' << _step_count << '\n';
  
  if (_step_count == 0 && _rank == 0 && verbosity > -1){
    _chain_out.open(_chain_file);
    _state_out.open(_state_file);
    _sampler_out.open(_sampler_file);
    this->write_chain_header();
    this->write_state_header();
    this->write_sampler_header();
    this->write_state();
  } else if (_restart && _rank == 0 && verbosity > -1){
    _chain_out.open(_chain_file, std::ios::app);
    _state_out.open(_state_file, std::ios::app);
    _sampler_out.open(_sampler_file, std::ios::app);
  }
  
  bool save = verbosity > -1 ? true : false;
  if (_step_count == 0 && _nadapt > 0){
    _adaptation = true;
    this->set_adaptation_schedule();
    save = _save_adaptation;
    if (verbosity > 0){
      _sampler_out << "Number of adaptation steps: "+std::to_string(_nadapt) << std::endl;
    }
  } else if ( _step_count < (size_t)_nadapt ){
    _adaptation = true;
    save = _save_adaptation&&save;
  }
  
  //std::cout << "Entering sampler step\n";
  for ( int i = 0; i < nsteps; ++i )
  {
    
    if (_step_count == (size_t)_nadapt){
      _adaptation = false;
      save = verbosity > -1 ? true : false;
      _sampler_out << "Adapted Gamma: \n"
                   << _Gammak
                   << std::endl;
      _sampler_out << "Eigenvalues: \n"
                   << _Dk << std::endl;
      _sampler_out << "Adapted Initial Widths\n";
      for ( size_t k = 0; k < _dimension; ++k )
        _sampler_out << std::setw(15) << _initial_width[k];
      _sampler_out << std::endl;
      _cov_adapt_stride = _window;
    }
    
    //std::cerr << "Entering sampler stepping\n";
    generate_transition(i, 0, nsteps, thin, refresh, save, verbosity);
    _step_count++;
    //std::cerr << "Done sampler stepping\n";
  }
  //Turn off restart since sampler finished 
  _restart = false;

  // DEBUG
  // std::cerr << "Finished AFSS.run_sampler: " << world_rank << ' ' << _rank << " +" << _state_file  << "+ " << _restart  << " " << nsteps << '\n';
}

void sampler_automated_factor_slice_sampler_MCMC::generate_transition(int step, int start, int finish, int nthin, int refresh, bool save, int verbosity)
{
  finish = std::max(finish, 1);
  refresh = std::max(refresh,1);
  //Find the log of the height we will sample across for the slice
  clock_t start_t = clock();
  //Now given the principal direction construct a slice
  for (size_t k = 0; k < _dimension; ++k){
    double logh;
    logh = _current_state[0] + std::log(_rng.rand());
    //std::cout << "factor baby " << k << std::endl;
    this->factor_slice_sample(k, logh);
    //std::cout << "done factor baby " << k << std::endl;
  }
  if (save && (((start+_step_count)%nthin) == 0) && _rank == 0){
    write_state();
  }
  //Reset the nexpand and ncontract to zero for the next transition
  _current_state[2] = 0;
  _current_state[3] = 0;
  //std::cout << "finish: " << finish << std::endl;
  clock_t end_t = clock();
  //if (!_adaptation){

  std::vector<double> model_values, state_values;
  get_chain_state(model_values, state_values);
  _sum_lklhd = (state_values[0] + (step)*_sum_lklhd)/(step+1);
  //}
  if (_adaptation && _step_count > (size_t)_init_buffer){
    this->update_covariance(_current_parameters);
    _adapt_width_count++;
  }

  if ( verbosity > -1 ){
    if (refresh > 0 && _rank ==0 
          && (start + step + 1 == finish || step == 0 || (step + 1) % refresh == 0)) {
        int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
        std::stringstream message;
        message << "Iteration: ";
        message << std::setw(it_print_width) << step + 1 + start << " / " << finish;
        message << " [" << std::setw(3)
                << static_cast<int>((100.0 * (start + step + 1.0)) / finish) << "%] ";
        message << (_adaptation ? " (Warmup)" : " (Sampling)");
        message << "Last step took " << (end_t-start_t)/CLOCKS_PER_SEC << " s.";
        std::cout << message.str() << std::endl;
    }
  }
  
  if (_ckpt_stride > 0 && ((start+_step_count)%_ckpt_stride == 0) && _rank ==0){
    std::ofstream ckpt_out(_ckpt_file);
    ckpt_out.precision(16); 
    write_checkpoint(ckpt_out);
    ckpt_out.close();
    
  }

  if ( _adaptation ){
    if (_step_count < (size_t)_init_buffer ){
      this->width_adapt();
      _width_adapt_stride = 1;
    }else if (_step_count < (size_t)(_nadapt-1)){
      if (_adapt_width_count == _width_adapt_stride){
        this->width_adapt();
      }
      if ( _adapt_cov_count == _cov_adapt_stride ){
        this->factor_adapt();
        _number_slow_adapt -= 1;
        _cov_adapt_stride *= 2;
        
      
        if (_number_slow_adapt > 1){
          _cov_adapt_stride *= 2;
        }
        else{
          _cov_adapt_stride = _nadapt-_step_count;
          std::cerr << "Final cov adaptation window length: " << _cov_adapt_stride << std::endl;
        }
      } 
    } 
      
  }
  //std::cout << "_adapt_cov_count: " << _adapt_cov_count << std::endl;
  //std::cout << "step count: " <<  _step_count << std::endl;
}

#define MAX_ITR 1e3
int sampler_automated_factor_slice_sampler_MCMC::step_out(const int k, double logh,
                                                           std::vector<double>& lower_ray, 
                                                           std::vector<double>& upper_ray, 
                                                           double& tmin, double& tmax)
{
  int nexpand = 0;
  double upper_L = this->loglklhd_transform(upper_ray);
  //Now step out if the upper limit is still in the slice
  int itr = 0;
  while ( (upper_L > logh) && itr < MAX_ITR+10 ){
    if (itr < MAX_ITR)
      tmax += 1.0;
    else{
      _initial_width[k] *= 10.0;
    }
    this->ray_position(k, tmax, upper_ray);
    upper_L = this->loglklhd_transform(upper_ray);
    if (!std::isfinite(upper_L)){
      upper_L = -std::numeric_limits<double>::infinity();
    }
    nexpand++;
    itr++;
  }
  if ( itr > MAX_ITR ){
    _current_state[4] = 1.0;
  }
  itr = 0;
  //Now step out if the lower limit is still in the slice
  double lower_L = this->loglklhd_transform(lower_ray);
  while (lower_L > logh && itr < MAX_ITR+10){
    if (itr < MAX_ITR)
      tmin -= 1.0;
    else{
      _initial_width[k] *= 10.0;
    }
    this->ray_position(k, tmin, lower_ray);
    //std::cout << "lower L " << lower_L << "height: " << logh <<  std::endl;
    lower_L = this->loglklhd_transform(lower_ray);
    if (!std::isfinite(lower_L))
      lower_L = -std::numeric_limits<double>::infinity();
    nexpand++;
    itr++;
  }
  if ( itr > MAX_ITR ){
    _current_state[4] = 1.0;
  }

  return nexpand;

}


void sampler_automated_factor_slice_sampler_MCMC::ray_position(int k, double t, std::vector<double>& ray)
{
  ray.resize(_dimension, 0.0);
  for ( size_t i = 0; i < _dimension; ++i ){
    ray[i] = _initial_width[k]*t*_Gammak.col(k)(i) + _current_parameters[i];
  }
}

void sampler_automated_factor_slice_sampler_MCMC::factor_slice_sample(const int k, const double logh)
{
  //Place initial interval around pos;
  std::vector<double> lower_ray(_dimension,0.0);
  std::vector<double> upper_ray(_dimension,0.0);
  int nshrink = 0, nexpand;
   
  double tmin = -_rng.rand();
  double tmax = tmin+1.0;
  //set up initial interval
  this->ray_position(k, tmin, lower_ray);
  this->ray_position(k, tmax, upper_ray);

  //Do the step out
  nexpand = this->step_out(k, logh, lower_ray, upper_ray, tmin, tmax);
  if ( nexpand > 15 && _adaptation){
    _initial_width[k] *= 2.0;
  }
  _current_state.resize(5);
  _current_state[4] = 0;

  //Do the shrink
  int itr = 0;
  //Ok so now we have a proper slice and we can sample from it.
  double tslice = (tmax-tmin)*_rng.rand() + tmin;
  std::vector<double> sliced_pos(_dimension, 0.0);
  this->ray_position(k,tslice,sliced_pos);
  double sliced_L = this->loglklhd_transform(sliced_pos);
  if (!std::isfinite(sliced_L))
    sliced_L = -std::numeric_limits<double>::infinity();
  //If the proposed value lies outside the slice contract the slice to the current value:
  while ( ((sliced_L < logh) && (itr < MAX_ITR))){
    itr++;
    if (tslice < 0.0 && std::fabs(tslice)>1e-10){
      tmin = tslice;  
    } else if (tslice > 0.0 && std::fabs(tslice)>1e-10){
      tmax = tslice;
    } else {
      _current_state[4] = 1;
      break;
    }

    tslice = (tmax - tmin)*_rng.rand() + tmin ;
    this->ray_position(k,tslice,sliced_pos);
    sliced_L = this->loglklhd_transform(sliced_pos);
    nshrink++;
  }
  if ( itr == MAX_ITR ){
    std::cerr << "Something is wrong the slice expander shouldn't be iterating this much\n";
    std::exit(1);
  }
  for ( size_t i = 0; i < _dimension; ++i ){
    _current_parameters[i] = sliced_pos[i];
  }
  if ( nshrink > 20 && _adaptation ){
    _initial_width[k] /= 1.5;
  }
  _current_state[0] = sliced_L;
  _current_state[1] = _adaptation;
  _current_state[2] += nexpand/(double)_dimension;
  _current_state[3] += nshrink/(double)_dimension;

  /*
  std::cout << _current_state[0] << std::endl
            << _current_state[1] << std::endl
            << _current_state[2] << std::endl
            << _current_state[3] << std::endl
            << _current_state[4] << std::endl;
  */
  if (_adaptation){
    _nexpansion[k] += nexpand;
    _ncontraction[k] += nshrink;
  }
}
#undef MAX_ITR


void sampler_automated_factor_slice_sampler_MCMC::update_covariance(std::vector<double> sample)
{
  Eigen::VectorXd delta(_dimension);
  //Update the mean
  _adapt_cov_count++;
  for (size_t i = 0; i < _dimension; ++i){
    delta(i) = (sample[i] - _mean(i));
    _mean(i) = (_adapt_cov_count*_mean(i) + _current_parameters[i])/(_adapt_cov_count+1.0);
  }
  //Now update the covariance matrix
  _covariance = (_adapt_cov_count-2.0)/(_adapt_cov_count-1.0)*_covariance 
                + (1.0/double(_adapt_cov_count))*delta*delta.transpose();
}

void sampler_automated_factor_slice_sampler_MCMC::width_adapt()
{

  for ( size_t k = 0; k < _dimension; ++k ){
    _nexpansion[k] = std::max(_nexpansion[k],1);

    _initial_width[k] = _initial_width[k]*(2.0*_nexpansion[k])/(double(_nexpansion[k])+double(_ncontraction[k])); 
    
    _nexpansion[k] = 0;
    _ncontraction[k] = 0;
  }
  _width_adapt_stride = std::max(int(1.2*_width_adapt_stride), _width_adapt_stride + 1);
  _adapt_width_count = 0;
}

double sampler_automated_factor_slice_sampler_MCMC::factor_adapt()
{
  std::cout << "Updating the covariance matrix on step " << _step_count << std::endl;
  //Find the eigenvectors and eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_covariance);

  //Now find the rotation matrix.
  Eigen::MatrixXd R = (_Gammak.transpose()*_Dk.asDiagonal().inverse()*_Gammak)*_covariance;
  double factor_error = (R-Eigen::MatrixXd::Identity(_dimension, _dimension)).sum();
  
  //Now find the new factors and their eigenvectors
  //Now use the eigenvalues as a guess for the initial slice size
  
  //Eigen by default sorts the eigen vectors in increasing order so
  //assuming that the width is related to the error in the principal 
  //direction I'll sort the widths in increasing order.
  //std::sort(_initial_width.begin(), _initial_width.end());
  //Additionally I will take the average of the initial width and the std dev
  Eigen::VectorXd widths(_dimension);
  for ( size_t i = 0; i < _dimension; ++i )
    widths(i) = _initial_width[i];
 
  widths = (es.eigenvectors().transpose()*_Gammak)*widths;
  _Gammak = es.eigenvectors();
  _Dk = es.eigenvalues();
  for ( size_t i = 0; i < _dimension; ++i )
    _initial_width[i] = std::fabs(widths(i))*10;

  //reset the covariance counter covariance and mean.
  _adapt_cov_count = 1;
  _covariance = Eigen::MatrixXd::Zero(_dimension,_dimension);
  _mean = Eigen::VectorXd::Zero(_dimension);
  for ( size_t i = 0; i < _dimension; ++i ){
    _mean(i) = _current_parameters[i];
    _width_adapt[i] = true;
  }


  _width_adapt_stride = 1;
  _adapt_width_count = 0;
  std::fill(_nexpansion.begin(), _nexpansion.end(), 0);
  std::fill(_ncontraction.begin(), _ncontraction.end(), 0);
  
  return factor_error;
}

void sampler_automated_factor_slice_sampler_MCMC::read_checkpoint(std::istream& ckpt_in)
{
  MPI_Comm_rank(_comm, &_rank);
  std::vector<double> chain(_dimension), state;
  std::vector<std::vector<double> > gammak(_dimension, std::vector<double>(_dimension,0.0)),
                                    cov(_dimension, std::vector<double>(_dimension, 0.0));
  std::vector<double> dk(_dimension,0.0), iw(_dimension, 0.0);
  std::vector<int> nex(_dimension,0), ncon(_dimension,0);
  size_t dimension;
  int step_count, nadapt, number_slow_adapt, was, awc, ib, cas, acc, w;
  
  //if ( _rank == 0 ){ // CREATES ALL KINDS OF PROBLEMS, RIGHT NOW WE WANT TO JUST GET IT WORKING.
  {
    std::string line, word, dummy;
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
    //Get n slow adapt
    ckpt_in >> word;
    number_slow_adapt = std::stoi(word.c_str());
    //Get width adaptation params
    ckpt_in >> word;
    was = std::stoi(word.c_str());
    ckpt_in >> word;
    awc = std::stoi(word.c_str());
    ckpt_in >> word;
    ib = std::stoi(word.c_str());
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //Get the widths
    //std::cerr << "Initial widths: \n";
    for ( size_t i = 0; i < _dimension; ++i ){
      ckpt_in >> word;
      double tmp = std::stod(word.c_str());
      iw[i] = tmp;
      //std::cerr << std::setw(25) << word;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //std::cerr << "nexpand: \n";
    for ( size_t i = 0; i < _dimension; ++i ){
      ckpt_in >> word;
      double tmp = std::stoi(word.c_str());
      nex[i] = tmp;
      //std::cerr << std::setw(25) << word;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //std::cerr << "ncontract: \n";
    for ( size_t i = 0; i < _dimension; ++i ){
      ckpt_in >> word;
      double tmp = std::stoi(word.c_str());
      ncon[i] = tmp;
      //std::cerr << std::setw(25) << word;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 

    
    //Get cov adaptation params
    ckpt_in >> word;
    cas = std::stoi(word.c_str());
    ckpt_in >> word;
    acc = std::stoi(word.c_str());
    ckpt_in >> word;
    w = std::stoi(word.c_str());
    //std::cerr << "adapt cov count: " << acc << std::endl;
    //Now read in the covariance stuff
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //std::cerr << "Dk: \n";
    for ( size_t i = 0; i < _dimension; ++i ){
      ckpt_in >> word;
      double tmp = std::stod(word.c_str());
      dk[i] = tmp;
      //std::cerr << std::setw(25) << word;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //std::cerr << "Gammak: \n";
    for ( size_t j = 0; j < _dimension; ++j ){
      for ( size_t i = 0; i < _dimension; ++i ){
        ckpt_in >> word;
        //std::cerr << std::setw(25) << word;
        double tmp = std::stod(word.c_str());
        gammak[j][i] = tmp;
      }
      //std::cerr <<std::endl;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    //std::cerr << "covariance: \n";
    for ( size_t j = 0; j < _dimension; ++j ){
      for ( size_t i = 0; i < _dimension; ++i ){
        ckpt_in >> word;
        //std::cerr << std::setw(25) << word;
        double tmp = std::stod(word.c_str());
        cov[j][i] = tmp;
      }
      //std::cerr <<std::endl;
    }
    //std::cerr<< std::endl;
    ckpt_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 

    //Now get the chain state
    //std::cerr << line << std::endl;
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

  // DEBUG
  // int world_rank, world_size;
  // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // std::cerr << "AFSS: Before Bcast " << _rank << ' '  << _comm << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';
  
  //Now broadcast the results
  /*
  MPI_Bcast(&dimension, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&step_count, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&nadapt, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&number_slow_adapt, 1, MPI_INT, 0, _comm);
  */
  
  // DEBUG
  // std::cerr << "AFSS: After Bcast 1 " << _rank << ' '  << _comm << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';
  // MPI_Barrier(_comm);

  /*
  MPI_Bcast(&was, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&awc, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&ib,  1, MPI_INT, 0, _comm);
  MPI_Bcast(&iw[0], _dimension, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&nex[0], _dimension, MPI_INT, 0, _comm);
  MPI_Bcast(&ncon[0], _dimension, MPI_INT, 0, _comm);
  */
  
  // DEBUG
  // std::cerr << "AFSS: After Bcast 2 " << _rank << ' '  << _comm << ' ' << _dimension << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';
  // MPI_Barrier(_comm);

  /*
  MPI_Bcast(&cas, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&acc, 1, MPI_INT, 0, _comm);
  MPI_Bcast(&w,  1, MPI_INT, 0, _comm);
  MPI_Bcast(&dk[0], _dimension, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&gammak[0], _dimension*_dimension, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&cov[0], _dimension*_dimension, MPI_DOUBLE, 0, _comm);
  */
  
  // DEBUG
  // std::cerr << "AFSS: After Bcast 3 " << _rank << ' '  << _comm << ' ' << _dimension << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';
  // std::cerr << "Chain size: " << chain.size() << ' ' << world_rank << '\n';
  // std::cerr << "State size: " << state.size() << ' ' << world_rank << '\n';
  // MPI_Barrier(_comm);

  /*
  MPI_Bcast(&chain[0], _dimension, MPI_DOUBLE, 0, _comm);
  MPI_Bcast(&state_size, 1, MPI_INT, 0, _comm);
  //std::cerr << "state_size: " << state_size << ' ' << world_rank << '\n';
  //state.resize(state_size); // Higher ranks don't have the state sized correctly.
  MPI_Bcast(&state[0], state_size, MPI_DOUBLE, 0, _comm);
  //std::cerr << state.size() << std::endl;

  // DEBUG
  // std::cerr << "AFSS: After Bcast 4 " << _rank << ' '  << _comm << ' ' << _dimension << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';

  MPI_Barrier(_comm);
  */

  // DEBUG
  // std::cerr << "AFSS: After Barrier " << world_rank << '\n';
  
  //Now we start assigning everything
  _dimension = dimension;
  _step_count = step_count;
  _nadapt = nadapt;
  _init_buffer = ib;
  _window = w;
  _number_slow_adapt = number_slow_adapt;

  _width_adapt_stride = was;
  _cov_adapt_stride = cas;
  _adapt_width_count = awc;
  _adapt_cov_count = acc;

  // DEBUG
  // std::cerr << "AFSS: Starting array assignments.  Are these all the right size? " << world_rank << '\n';

  for ( size_t i = 0; i < _dimension; ++i ){
    _initial_width[i] = iw[i];
    _nexpansion[i] = nex[i];
    _ncontraction[i] = ncon[i];
    _Dk(i) = dk[i];
    for ( size_t j = 0; j < _dimension; ++j ){
      _Gammak(i,j) = gammak[i][j];
      _covariance(i,j) = cov[i][j];
    }
  }

  this->set_chain_state(chain, state);
  write_state();
  //std::cerr << "Done reading in checkpoint\n";

  // DEBUG
  // std::cerr << "AFSS: Finished " << ' ' << world_rank << ' ' << MPI_COMM_WORLD << '\n';
}

void sampler_automated_factor_slice_sampler_MCMC::write_checkpoint(std::ostream& out)
{
  out << _dimension << std::endl;
  out << _step_count << std::endl;
  out << _nadapt << std::endl;
  out << _number_slow_adapt << std::endl;

  //Now write the width adaptation parameters
  out << _width_adapt_stride << std::endl;
  out << _adapt_width_count << std::endl;
  out << _init_buffer << std::endl;
  for ( size_t i = 0; i < _dimension; ++i )
    out << std::setw(25) << _initial_width[i];
  out << std::endl;
  for ( size_t i = 0; i < _dimension; ++i )
    out << std::setw(25) << _nexpansion[i];
  out << std::endl;
  for ( size_t i = 0; i < _dimension; ++i )
    out << std::setw(25) << _ncontraction[i];
  out << std::endl;

  //Now write the covariance adaptation stuff
  out << _cov_adapt_stride << std::endl;
  out << _adapt_cov_count << std::endl;
  out << _window << std::endl;
  out << _Dk << std::endl;
  out << _Gammak << std::endl;
  out << _covariance << std::endl;

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

void sampler_automated_factor_slice_sampler_MCMC::write_state()
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

void sampler_automated_factor_slice_sampler_MCMC::write_chain_header()
{
  _chain_out << "#chainfmt afss\n";
  sampler_MCMC_base::write_chain_header();
}

void sampler_automated_factor_slice_sampler_MCMC::write_state_header()
{
  _state_out << "#lklhdfmt afss\n";
  _state_out << "#";
  _state_out << std::setw(15) << "lklhd"
             << std::setw(15) << "adapt"
             << std::setw(15) << "nexpand"
             << std::setw(15) << "nshrink" 
             << std::setw(15) << "divergent" << std::endl;
}

void sampler_automated_factor_slice_sampler_MCMC::write_sampler_header()
{
  _sampler_out << "sampler = afss\n";
  _sampler_out << "Adaptation tuning parameters: \n"
               << "  nadapt = " << _nadapt << std::endl
               << "  initial width steps = " << _init_buffer << std::endl
               << "  initial cov steps = " << _window << std::endl
               << "Initial widths: " << std::endl;
  for ( size_t i = 0; i < _dimension; ++i )
    _sampler_out << std::setw(15) << _initial_width[i];
  _sampler_out << std::endl;

  _sampler_out << "Initial eigenvectors: \n"
               << _Gammak << std::endl;
}

void sampler_automated_factor_slice_sampler_MCMC::get_chain_state(std::vector<double>& params, std::vector<double>& state)
{
  state.resize(_current_state.size(),0.0);
  params.resize(_dimension);
  
  double jacobian=0.0;
  for ( size_t i = 0; i < _dimension; ++i ){
    params[i] = _transforms[i]->inverse(_current_parameters[i]);
    jacobian += std::log(std::fabs(_transforms[i]->inverse_jacobian(_current_parameters[i]))+1e-50);
  }
  
  for ( size_t i = 0; i < _current_state.size(); ++i )
    state[i] = _current_state[i];
  state[0] -= jacobian;
}

void sampler_automated_factor_slice_sampler_MCMC::set_chain_state(std::vector<double> params, std::vector<double> state)
{
  if (params.size() != _dimension ){
    std::cerr << "Number of parameters in set_chain_state does not equal number of dimensions\n";
    std::exit(1);
  }
  if (state.size() != 5){
    std::cerr << "Number of state parameters in set_chain_state does not equal number of dimensions\n";
    std::exit(1);
  }

  
  double jacobian=0.0;
  for ( size_t i = 0; i < _dimension; ++i ){
    _current_parameters[i] = _transforms[i]->forward(params[i]);
    jacobian += std::log(std::fabs(_transforms[i]->inverse_jacobian(_current_parameters[i]))+1e-50);
  }
  
  _current_state.resize(5,0.0);
  for ( size_t i = 0; i < _current_state.size(); ++i )
    _current_state[i] = state[i];
  _current_state[0] += jacobian;
  
  
}






}//end Themis
