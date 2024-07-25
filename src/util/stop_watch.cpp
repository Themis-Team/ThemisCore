/*!
  \file stop_watch.cpp
  \author Avery E. Broderick
  \date  June, 2017
  \brief Header file for a general purpose timer based on the C time function.
  \details To be added
*/

#include "stop_watch.h"
#include <iomanip>

namespace Themis
{
  StopWatch::StopWatch()
  {
    std::time(&start_time);
    lap_time=start_time;
  }

  StopWatch::~StopWatch()
  {
  }

  void StopWatch::print_lap(std::ostream& out, std::string prefix, std::string postfix)
  {
    out << prefix << std::setw(15) << lap() << postfix << std::endl;
  }

  void StopWatch::print_time(std::ostream& out, std::string prefix, std::string postfix)
  {
    out << prefix << std::setw(15) << time() << postfix << std::endl;
  }
}
