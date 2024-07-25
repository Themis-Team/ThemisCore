#ifndef VRT2_PROGRESSINDICATOR_H
#define VRT2_PROGRESSINDICATOR_H

#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>


/*** Front End Declaration ***/
namespace VRT2 {
class ProgressIndicator
{
 public:
  virtual ~ProgressIndicator() {};

  // Counter initialization
  virtual void start()=0;
  // Counter incrementing
  virtual void increment(double f)=0;
  // Counter cleanup and finish
  virtual void finish()=0;
};


/*** Percentage Counter ***/
class ProgressCounter : public ProgressIndicator
{
 public:
  ProgressCounter(std::ostream& os=std::cout,
		  unsigned int precision=1,
		  std::string postfix="");

  // Counter bussiness
  virtual void start();
  virtual void increment(double f);
  virtual void finish();

 private:
  // Input
  std::ostream& os_; // Output stream
  unsigned int precision_; // Precision of output
  std::string postfix_; // Stuff to print after output
  // Local data
  unsigned int setw_size_;
  std::string backup_;
  // Ostream state stuff
  unsigned int old_precision_;
};


/*** Progress Bar ***/
class ProgressBar : public ProgressIndicator
{
 public:
  ProgressBar(std::ostream& os=std::cout);
  ProgressBar(std::ostream& os, double sstep, double mstep, double lstep);
  ProgressBar(std::ostream& os, std::string s, std::string m, std::string l);
  ProgressBar(std::ostream& os, std::string s, double sstep,
	      std::string m, double mstep, std::string l, double lstep);

  // Counter bussiness
  void start();
  void increment(double f);
  void finish();

 private:
  // Input data
  std::ostream& os_; // Output stream
  std::string s_, m_, l_;
  double sstep_, mstep_, lstep_;
  // Local data
  double oldf_;
};


/*** Percentage Counter AND Progress Bar ***/
class ProgressCounterBar : public ProgressIndicator
{
 public:
  ProgressCounterBar(std::ostream& os=std::cout,
		     unsigned int precision=1);
  ProgressCounterBar(std::ostream& os,
		     double sstep, double mstep, double lstep,
		     unsigned int precision=1);
  ProgressCounterBar(std::ostream& os, std::string s,
		     std::string m, std::string l,
		     unsigned int precision=1);
  ProgressCounterBar(std::ostream& os,
		     std::string s, double sstep,
		     std::string m, double mstep,
		     std::string l, double lstep,
		     unsigned int precision=1);

  // Counter bussiness
  void start();
  void increment(double f);
  void finish();

 private:
  void initialize();

  std::ostream& os_; // Output stream
  // Input data
  unsigned int precision_; // Precision of output
  std::string s_, m_, l_;
  double sstep_, mstep_, lstep_;
  //  Local data
  unsigned int setw_size_;
  std::string backup_;
  std::string bar_;
  unsigned int bar_size_;
  unsigned int bar_pos_;
  double oldf_;
  // Ostream state stuff
  unsigned int old_precision_;
};

};
#endif
