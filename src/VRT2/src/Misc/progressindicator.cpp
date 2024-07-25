#include "progressindicator.h"

/*** Percentage Counter ***/
namespace VRT2 {
ProgressCounter::ProgressCounter(std::ostream& os,
				 unsigned int precision, 
				 std::string postfix)
  : os_(os), precision_(precision), postfix_(postfix),
  setw_size_(precision_+4),
  backup_(setw_size_+postfix.size()+2,'\b')
{}

void ProgressCounter::start()
{
  // Save & Set precision
  os_.setf(std::ios::fixed);
  old_precision_ = os_.precision(precision_);
  // Output first line
  os_ << std::setw(setw_size_) << 0.0 << "% " << postfix_;
  os_.flush();
}

void ProgressCounter::increment(double f)
{
  os_ << backup_ << std::setw(setw_size_) << 100.0*f << "% " << postfix_;
  os_.flush();
}

void ProgressCounter::finish()
{
  increment(1.0);
  // Reset ostream properties
  os_.unsetf(std::ios::fixed);
  os_.precision(old_precision_);
}


/*** Progress Bar ***/
ProgressBar::ProgressBar(std::ostream& os)
  : os_(os),
  s_("."), m_(":"), l_("|"),
  sstep_(0.05), mstep_(0.25), lstep_(0.5)
{}

ProgressBar::ProgressBar(std::ostream& os,
			 double sstep,
			 double mstep,
			 double lstep)
  : os_(os),
  s_("."), m_(":"), l_("|"),
  sstep_(sstep), mstep_(mstep), lstep_(lstep)
{
  // Make sure that step sizes are monotonic
  if (!(sstep<mstep && mstep<=lstep))
    std::cerr << "Large steps are smaller than Small steps in ProgressBar!"
	      << std::endl;
}

ProgressBar::ProgressBar(std::ostream& os,
			 std::string s, std::string m, std::string l)
  : os_(os),
  s_(s), m_(m), l_(l),
  sstep_(0.05), mstep_(0.25), lstep_(0.5)
{}

ProgressBar::ProgressBar(std::ostream& os,
			 std::string s,
			 double sstep,
			 std::string m,
			 double mstep,
			 std::string l,
			 double lstep)
  : os_(os),
  s_(s), m_(m), l_(l),
  sstep_(sstep), mstep_(mstep), lstep_(lstep)
{
  // Make sure that steps are monotonic
  if (!(sstep<mstep && mstep<=lstep))
    std::cerr << "Large steps are smaller than Small steps in ProgressBar!"
	      << std::endl;
}

void ProgressBar::start()
{
  // Output first line
  os_ << l_;
  os_.flush();
  // Set old f value
  oldf_ = 0.0;
}

void ProgressBar::increment(double f)
{
  // Find number of small steps that have been taken from last increment
  //  and take steps in specified order
  int Ns = int(floor((f-oldf_)/sstep_));
  double f_last, f_new, df = (f-oldf_)/Ns;
  if (Ns) {
    for (int i=0; i<Ns; ++i) {
      f_last = oldf_+i*df;
      f_new = oldf_+(i+1)*df;
      // Assume that sstep > mstep > lstep
      //  and make | if should, else :, else . .
      if ( fmod(f_new,lstep_) < fmod(f_last,lstep_) )
	os_ << l_;
      else if ( fmod(f_new,mstep_) < fmod(f_last,mstep_) )
	os_ << m_;
      else
	os_ << s_; // Remember that we are stepping by pstep!
    }
    os_.flush();
    oldf_ = f;
  }
}

void ProgressBar::finish()
{
  // Make sure that progress bar is finished
  increment(1.0);
  if (oldf_<1)
    os_ << l_;
}


/*** Percentage Counter AND Progress Bar ***/
ProgressCounterBar::ProgressCounterBar(std::ostream& os,
				       unsigned int precision)
  : os_(os),
  precision_(precision),
  s_("."), m_(":"), l_("|"),
  sstep_(0.05), mstep_(0.25), lstep_(0.5),
  setw_size_(precision_+4)
{
  initialize();
}

ProgressCounterBar::ProgressCounterBar(std::ostream& os,
				       double sstep,
				       double mstep,
				       double lstep,
				       unsigned int precision)
  : os_(os),
  precision_(precision),
  s_("."), m_(":"), l_("|"),
  sstep_(sstep), mstep_(mstep), lstep_(lstep),
  setw_size_(precision_+4)
{
  // Make sure that steps are monotonic
  if (!(sstep<mstep && mstep<=lstep))
    std::cerr << "Large steps are smaller than Small steps in ProgressBar!"
	      << std::endl;
  initialize();
}

ProgressCounterBar::ProgressCounterBar(std::ostream& os,
				       std::string s,
				       std::string m,
				       std::string l,
				       unsigned int precision)
  : os_(os),
  precision_(precision),
  s_(s), m_(m), l_(l),
  sstep_(0.05), mstep_(0.25), lstep_(0.5),
  setw_size_(precision_+4)
{
  initialize();
}

ProgressCounterBar::ProgressCounterBar(std::ostream& os,
				       std::string s,
				       double sstep,
				       std::string m,
				       double mstep,
				       std::string l,
				       double lstep,
				       unsigned int precision)
  : os_(os),
  precision_(precision),
  s_(s), m_(m), l_(l),
  sstep_(sstep), mstep_(mstep), lstep_(lstep),
  setw_size_(precision_+4)
{
  // Make sure that steps are monotonic
  if (!(sstep<mstep && mstep<=lstep))
    std::cerr << "Large steps are smaller than Small steps in ProgressBar!"
	      << std::endl;
  initialize();
}

void ProgressCounterBar::initialize()
{
  // Determine size of the progress bar
  bar_size_ = 0;
  int Ns = int(1.0/sstep_);
  double f_last, f_new, df = 1.0/double(Ns);
  if (Ns) {
    for (int i=0; i<=Ns; ++i) {
      f_last = i*df;
      f_new = (i+1)*df;
      if ( fmod(f_new,lstep_) < fmod(f_last,lstep_) )
	bar_size_ += l_.size();
      else if ( fmod(f_new,mstep_) < fmod(f_last,mstep_) )
	bar_size_ += m_.size();
      else
	bar_size_ += s_.size();
    }
  }
  // Make backup string
  backup_.assign(setw_size_+bar_size_+2,'\b');
}

void ProgressCounterBar::start()
{
  // Set old f value to 0
  oldf_ = 0.0;
  // Reset bar to beginning values
  bar_pos_ = 0;
  bar_.assign(bar_size_,' ');
  bar_.replace(0,l_.size(),l_);
  bar_pos_ += l_.size();
  // Save & Set precision
  os_.setf(std::ios::fixed);
  old_precision_ = os_.precision(precision_);
  // Output first line
  os_ << std::setw(setw_size_) << 0.0 << "% " << bar_;
  os_.flush();
}

void ProgressCounterBar::increment(double f)
{
  // First Backup
  os_ << backup_;
  // Second, write progress counter
  os_ << std::setw(setw_size_) << 100.0*f << "% ";
  // Third, write bar to bar_
  int Ns = int(floor((f-oldf_)/sstep_));
  double f_last, f_new, df = (f-oldf_)/Ns;
  if (Ns) {
    for (int i=0; i<Ns; ++i) {
      f_last = oldf_+i*df;
      f_new = oldf_+(i+1)*df;
      // Assume that sstep > mstep > lstep
      //  and make | if should, else :, else . .
      if ( fmod(f_new,lstep_) < fmod(f_last,lstep_) ) {
	bar_.replace(bar_pos_,l_.size(),l_);
	bar_pos_ += l_.size();
      }
      else if ( fmod(f_new,mstep_) < fmod(f_last,mstep_) ) {
	bar_.replace(bar_pos_,m_.size(),m_);
	bar_pos_ += m_.size();
      }
      else {
	bar_.replace(bar_pos_,s_.size(),s_);
	bar_pos_ += s_.size(); // Remember that we are stepping by pstep!
      }
    }
    oldf_ = f;
  }
  // Fourth, write to bar
  os_ << bar_;
  os_.flush();
}

void ProgressCounterBar::finish()
{
  increment(1.0);
  if (oldf_<1)
    os_ << l_;

  // Reset ostream properties
  os_.unsetf(std::ios::fixed);
  os_.precision(old_precision_);
}
};
