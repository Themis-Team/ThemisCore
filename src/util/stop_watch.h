/*!
  \file util/stop_watch.h
  \author Avery E. Broderick
  \date  June, 2017
  \brief Header file for a general purpose timer based on the C time function.
  \details To be added
*/

#ifndef THEMIS_STOP_WATCH_H
#define THEMIS_STOP_WATCH_H

#include <iostream>
#include <string>
#include <ctime>

namespace Themis
{

  /*! 
  \brief Defines a general purpose timer based on the C time function.

  \details This permits 1s precision dynamical performance reporting within Themis via a modular object that encapsulates the C time function details.  Implemented functions include time(), which reports total elapsed time since start(), lap() which reports elapsed time since either start or the previous lap(), and print functions for each of these.

  \warning This is subject to the same limitations as the C time function, and is therefore only appropriate for time measurements in excess of 1s, and is typically useful only for measuring events taking many seconds.
  */
  class StopWatch
  {
   public:

    //! Creates a StopWatch object and starts it at the current time.
    StopWatch();
    ~StopWatch();

    //! Restarts the StopWatch to the current time.
    inline void start()
    {
      std::time(&start_time);
      lap_time=start_time;
    };

    //! Reports the number of seconds since either the StopWatch was instantiated, start(), or lap() was last called, whichever is more recent.
    inline double lap()
    {
      std::time(&stop_time);
      double lapval=difftime(stop_time,lap_time);
      lap_time=stop_time;
      return (lapval);
    };

    //! Reports the number of seconds since either the StopWatch was instantiated or start() was last called, whichever is more recent.
    inline double time()
    {
      std::time(&stop_time);
      return (difftime(stop_time,start_time));
    };

    //! Prints the current lap time, prefixing with the string passed in prefix (defaults to empty) and postfixing with the string passed in postfix (defaults to empty), to the output stream passed as out.  If only one string is passed, it is assumed to be the prefix.
    void print_lap(std::ostream& out, std::string prefix="", std::string postfix="");

    //! Prints the current elapsed time, prefixing with the string passed in prefix (defaults to empty) and postfixing with the string passed in postfix (defaults to empty), to the output stream passed as out.  If only one string is passed, it is assumed to be the prefix.
    void print_time(std::ostream& out, std::string prefix="", std::string postfix="");
    
   private:
    time_t start_time;
    time_t stop_time;
    time_t lap_time;
  };
};

#endif
