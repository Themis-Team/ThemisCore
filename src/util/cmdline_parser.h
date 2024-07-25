/*!
  \file cmdline_parser.h
  \author Avery E. Broderick
  \date  March, 2023
  \brief Defines a lightweight command-line parser to facilitate adding options to drivers.
  \details To be added
*/

#ifndef Themis_CMDLINEPARSER_H
#define Themis_CMDLINEPARSER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <string>
#include <cstring>
#include <mpi.h>

namespace Themis {

/*! 
  \brief Defines a parser for managing command-line options.

  \details Provides an easily organized interface for specifying flag-delineated and positional command-line arguments.  The primary user interface is the parse_args function, that takes the standard C++ argc,argv[] pair, identifies specified options and their values, and returns the remaining non-flag (i.e., not beginning with '-') positional arguments.  Rudimentary quality control is performed, including redundant flags, unknown flags, and missing arguments.  Help is automatically generated.

  This should only be used at the beginning of a main() function and only one instance of the CMDLineParser need be constructed.  Command-line arguments may be added using the relevant child of CMDLineArg.  Finally, the command lines are parsed with the parse_args function.

  \warning 
*/  
class CMDLineParser
{
 public:

  /*! 
    \brief Defines a command-line argument base class for collecting value storage, command-line flags, help messages, etc.

    \details Provides basic functionality for a command-line option object, supporting multiple flags, required/optional arguments, and online help information.  Because it contains purely virtual functions, this class cannot be instantiated.  However, child classes that implement specific types, inherit the relevant functionality and interfaces.

    \warning This class contains multiple purely virtual functions, making it impossible to generate an explicit instantiation.
  */  
  class CMDLineArg
  {
  public:
    CMDLineArg();
    virtual ~CMDLineArg() {};

    //! Access to whether or not the value is defined
    virtual bool is_defined() const;

    //! Access to whether or not the value is required
    virtual bool is_required() const;

    //! Access to the help message to print
    std::string help_message() const;

    //! Access to the help message to print
    std::string type_message() const;

    //! Access to command line flags
    std::string cmdline_flag(size_t i) const;

    //! Access to command line flags
    size_t cmdline_flag_size() const;

    //! Access to value lists size
    size_t max_value_size() const;
  
    //! Utility function used to set the value, returns the number of arguments to skip forward, including flag (so a minimum of 1)
    virtual int set_value(int argc, char* argv[]) = 0;

    //! Utility function used to set the help message string
    void set_help_message(std::string help_message);

    //! Utility function used to set the type help message string
    void set_type_message(std::string type_message);

    //! Utility function used to add a command-line flag
    void add_cmdline_flag(std::string flag);

    //! Utility to set defined
    void set_defined();

    //! Utility to set required
    void set_required();

    //! Utility to set max value size
    void set_max_value_size(size_t size);
  
  protected:
    std::string _help_message;
    std::string _type_message;
    bool _is_defined;
    bool _is_required;
    std::vector<std::string> _cmdline_flag;
    size_t _max_value_size;
  };

  CMDLineParser(std::string name="PROGRAM", std::string description="");
  virtual ~CMDLineParser() {};

  virtual std::vector<std::string> parse_args(int &argc, char* argv[], bool dump_cmd=false);

  virtual void print_usage_message();


  virtual void add_cmdline_arg(CMDLineArg &arg);
  virtual void add_cmdline_arg(CMDLineArg *arg);
  
  virtual void set_program_name(std::string name, std::string description="");

 protected:
  std::string _program_name;
  std::string _program_description;
  std::vector< CMDLineArg* > _cmdline_args;

 public:

  class BoolArg : public CMDLineArg
  {
  public:
    BoolArg(CMDLineParser& cmd_parser, std::string flag, std::string help="", bool required=false);
    BoolArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool default_value, bool required=false);
    virtual ~BoolArg() {}
    
    //! Access to value
    virtual bool operator()() const;
    
    //! Utility function used to set the value
    virtual int set_value(int argc, char* argv[]);
    
  private:
    bool _default_value;
    bool _value;
    bool _has_default;
  };

 
 class FloatArg : public CMDLineArg
 {
 public:
   FloatArg(CMDLineParser& cmd_parser, std::string flag, std::string help="", bool required=false);
   FloatArg(CMDLineParser& cmd_parser, std::string flag, std::string help, float default_value, bool required);

   //! Access to value
   virtual float operator()() const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   float _value;
 };


 class IntArg : public CMDLineArg
 {
 public:
   IntArg(CMDLineParser& cmd_parser, std::string flag, std::string help="", bool required=false);
   IntArg(CMDLineParser& cmd_parser, std::string flag, std::string help, int default_value, bool required);

   //! Access to value
   virtual int operator()() const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   int _value;
 };

 
 class StringArg : public CMDLineArg
 {
 public:
   StringArg(CMDLineParser& cmd_parser, std::string flag, std::string help="", bool required=false);
   StringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, std::string default_value, bool required);

   //! Access to value
   virtual std::string operator()() const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   std::string _value;
 };

 
 class VectorFloatArg : public CMDLineArg
 {
 public:
   VectorFloatArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help="", bool required=false);
   VectorFloatArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<float> default_value, bool required=false);
   virtual ~VectorFloatArg() {}

   //! Access to vector value
   virtual std::vector<float> operator()() const;

   //! Access to vector element values
   virtual float operator()(size_t i) const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   std::vector<float> _value;
   size_t _count;
 };

 
 class VectorIntArg : public CMDLineArg
 {
 public:
   VectorIntArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help="", bool required=false);
   VectorIntArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<int> default_value, bool required=false);
   virtual ~VectorIntArg() {}

   //! Access to vector value
   virtual std::vector<int> operator()() const;

   //! Access to vector element values
   virtual int operator()(size_t i) const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   std::vector<int> _value;
   size_t _count;
 };

 
 class VectorStringArg : public CMDLineArg
 {
 public:
   VectorStringArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help="", bool required=false);
   VectorStringArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<std::string> default_value, bool required=false);
   virtual ~VectorStringArg() {}

   //! Access to vector value
   virtual std::vector<std::string> operator()() const;

   //! Access to vector element values
   virtual std::string operator()(size_t i) const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);
  
 private:
   std::vector<std::string> _value;
   size_t _count;
 };

 
 class VariableVectorStringArg : public CMDLineArg
 {
 public:
   VariableVectorStringArg(CMDLineParser& cmd_parser, std::string flag, std::string help="", bool required=false);
   VariableVectorStringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, std::vector<std::string> default_value, bool required=false);
   virtual ~VariableVectorStringArg() {}

   //! Access to vector value
   virtual std::vector<std::string> operator()() const;

   //! Access to vector element values
   virtual std::string operator()(size_t i) const;
  
   //! Utility function used to set the value
   virtual int set_value(int argc, char* argv[]);

   //! Access to current size
   virtual size_t size() const;
   
 private:
   std::vector<std::string> _value;
   size_t _count;
 };
};

//------ CMDLineArg Impelemenations ------
CMDLineParser::CMDLineArg::CMDLineArg()
: _help_message(""), _type_message("<Value>"), _is_defined(false), _is_required(false), _cmdline_flag(0), _max_value_size(0)
{
}

std::string CMDLineParser::CMDLineArg::help_message() const
{
  return _help_message;
}

std::string CMDLineParser::CMDLineArg::type_message() const
{
  return _type_message;
}

bool CMDLineParser::CMDLineArg::is_defined() const
{
  return _is_defined;
}

bool CMDLineParser::CMDLineArg::is_required() const
{
  return _is_required;
}

std::string CMDLineParser::CMDLineArg::cmdline_flag(size_t i) const
{
  if (i>=_cmdline_flag.size())
  {
    std::cerr << "ERROR: No command-line flags have been specified.\n";
    std::exit(1);
  }
  return _cmdline_flag[i];
}

size_t CMDLineParser::CMDLineArg::cmdline_flag_size() const
{
  return _cmdline_flag.size();
}

size_t CMDLineParser::CMDLineArg::max_value_size() const
{
  return _max_value_size;
}

void CMDLineParser::CMDLineArg::set_help_message(std::string help_message)
{
  _help_message = help_message;
}

void CMDLineParser::CMDLineArg::set_type_message(std::string type_message)
{
  _type_message = type_message;
}

void CMDLineParser::CMDLineArg::add_cmdline_flag(std::string flag)
{
  char *token = std::strtok(&flag[0], ",");
  while (token != NULL)
  {
    if (token[0]!='-')
      std::cerr << "WARNING: Adding a command-line flag that does not begin in -, this may behave badly.\n";
    _cmdline_flag.push_back(std::string(token));
    
    token = std::strtok(NULL, ",");
  }
  // Make sure we have at least one flag for each option.
  if (_cmdline_flag.size()==0)
  {
    std::cerr << "ERROR: At least one flag must be provided for each argument.\n";
    std::exit(1);
  }
}

void CMDLineParser::CMDLineArg::set_defined()
{
  _is_defined = true;
}

void CMDLineParser::CMDLineArg::set_required()
{
  _is_required = true;
}

void CMDLineParser::CMDLineArg::set_max_value_size(size_t size)
{
  _max_value_size = size;
}

//------ CMDLineParser Impelemenations ------
CMDLineParser::CMDLineParser(std::string name, std::string description)
: _program_name(name), _program_description(description), _cmdline_args(0)
{
}

std::vector<std::string> CMDLineParser::parse_args(int &argc, char* argv[], bool dump_cmd)
{
  if (dump_cmd)
  {
    // Dump command line for posterity
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank==0)
    {
      std::cout << "\n=============================================================" << std::endl;
      std::cout << "CMD$ <mpiexec>";
      for (int k=0; k<argc; k++)
	std::cout << " " << argv[k];
      std::cout << std::endl;
      std::cout << "=============================================================\n" << std::endl;
    }
  }

  std::vector<std::string> positional_args;
  
  // Check for flag consistency
  for (size_t j=0; j<_cmdline_args.size(); ++j)
    for (size_t i=0; i<_cmdline_args[j]->cmdline_flag_size(); ++i)
      for (size_t jj=j+1; jj<_cmdline_args.size(); ++jj)
	for (size_t ii=0; ii<_cmdline_args[jj]->cmdline_flag_size(); ++ii)
	  if ( _cmdline_args[j]->cmdline_flag(i) == _cmdline_args[jj]->cmdline_flag(ii) )
	  {
	    std::cerr << "ERROR: Duplicate command-line flag found. The flag " << _cmdline_args[j]->cmdline_flag(i) << " appears at least twice!\n";
	    std::exit(1);
	  }
  
  // Check for help
  for (int k=1; k<argc; ++k)
    if (std::string(argv[k])=="-h" || std::string(argv[k])=="--help")
	print_usage_message();

  // Parse command line
  int k=1;
  for (k=1; k<argc;)
  {
    if (k>=argc)
    {
      std::cerr << "ERROR: Processed too many command-line arguments!  Something fishy is happening ...\n";
      std::exit(1);
    }
    
    std::string opt = std::string(argv[k]);
    bool isopt = false;
    for (size_t j=0; j<_cmdline_args.size(); ++j)
    {
      for (size_t i=0; i<_cmdline_args[j]->cmdline_flag_size(); ++i)
	isopt = isopt || (opt==_cmdline_args[j]->cmdline_flag(i));
      if (isopt)
      {
	k += _cmdline_args[j]->set_value(argc-k,&argv[k]);
	break;
      }
    }
    
    if (isopt==false)
    {
      if (opt[0]=='-')
      {
	// Should only get here if we did not find an option.
	std::cerr << "ERROR: Unrecognized option " << opt << '\n'
		  << "Try -h or --help for options.\n";
	std::exit(1);
      }
      else
      {
	positional_args.push_back(opt);
	k += 1;
      }
    }
  }

  // Check that required parameters are defined
  for (size_t j=0; j<_cmdline_args.size(); ++j)
    if(_cmdline_args[j]->is_required()==true && _cmdline_args[j]->is_defined()==false)
    {	
      std::cerr << "ERROR: A required option is missing.\n"
		<< _cmdline_args[j]->cmdline_flag(0) << " must be provided.\n"
		<< "Try -h or --help for more information.\n";
      std::exit(1);
    }

  // Return the positional argumets (i.e., unflagged arguments)
  return positional_args;
}

void CMDLineParser::print_usage_message()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (world_rank==0)
  {
    // Print line usage
    std::cerr << "NAME\n\t" << _program_name << "\n" << _program_description << "\n\n";
    std::cerr << "SYNOPSIS\n\t" << _program_name << ' ';
    for (size_t j=0; j<_cmdline_args.size(); ++j)
      if (_cmdline_args[j]->is_required())
      {
	std::cerr << _cmdline_args[j]->cmdline_flag(0) << ' ' << _cmdline_args[j]->type_message();
	/*
	std::cerr << _cmdline_args[j]->cmdline_flag(0) << ' ';
	if (_cmdline_args[j]->max_value_size()>1)
	  std::cerr << "<Value(s)> ";
	else if (_cmdline_args[j]->max_value_size()>0)
	  std::cerr << "<Value> ";
	*/
      }
    std::cerr << "[Options ...]";
    
    // Required options
    std::cerr << "\n\nREQUIRED OPTIONS\n";
    for (size_t j=0; j<_cmdline_args.size(); ++j)
      if (_cmdline_args[j]->is_required())
      {
	for (size_t k=0; k<_cmdline_args[j]->cmdline_flag_size()-1; ++k)
	  std::cerr << _cmdline_args[j]->cmdline_flag(k) << ',';
	std::cerr << _cmdline_args[j]->cmdline_flag(_cmdline_args[j]->cmdline_flag_size()-1);
	for (size_t k=0; k<_cmdline_args[j]->max_value_size(); ++k)
	  std::cerr << ' ' << _cmdline_args[j]->type_message();
	std::string msg = _cmdline_args[j]->help_message();
	char *token = std::strtok(&msg[0], "\n");
	std::cerr << '\n';
	while (token != NULL)
	{
	  std::cerr << '\t' << std::string(token) << '\n';
	  token = std::strtok(NULL, "\n");	  
	}
      }

    // Optionial options
    std::cerr << "\nDESCRIPTION\n";
    for (size_t j=0; j<_cmdline_args.size(); ++j)
      if (_cmdline_args[j]->is_required()==false)
      {
	for (size_t k=0; k<_cmdline_args[j]->cmdline_flag_size()-1; ++k)
	  std::cerr << _cmdline_args[j]->cmdline_flag(k) << ',';
	std::cerr << _cmdline_args[j]->cmdline_flag(_cmdline_args[j]->cmdline_flag_size()-1);
	for (size_t k=0; k<_cmdline_args[j]->max_value_size(); ++k)
	  std::cerr << ' ' << _cmdline_args[j]->type_message();
	std::string msg = _cmdline_args[j]->help_message();
	char *token = std::strtok(&msg[0], "\n");
	std::cerr << '\n';
	while (token != NULL)
	{
	  std::cerr << '\t' << std::string(token) << '\n';
	  token = std::strtok(NULL, "\n");	  
	}
      }
    std::cerr << "-h,--help\n";
    std::cerr << "\tPrints this help message.\n";
    std::cerr << "\n";
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  std::exit(0);
}

void CMDLineParser::add_cmdline_arg(CMDLineArg& arg)
{
  _cmdline_args.push_back(&arg);
}

void CMDLineParser::add_cmdline_arg(CMDLineArg* arg)
{
  _cmdline_args.push_back(arg);
}

void CMDLineParser::set_program_name(std::string name, std::string description)
{
  _program_name = name;
  _program_description = description;
}


//------ BoolArg Impelemenations ------
CMDLineParser::BoolArg::BoolArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool required)
: _has_default(false)
{
  if (required)
    set_required();
  set_help_message(help);
  set_type_message("<bool>");
  add_cmdline_flag(flag);
  _max_value_size = 0;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::BoolArg::BoolArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool default_value, bool required)
: _default_value(default_value), _value(default_value), _has_default(true)
{
  set_defined();
  if (required)
    set_required();
  set_help_message(help+"\nDefault: "+(default_value ? "true" : "false"));
  set_type_message("<bool>");
  add_cmdline_flag(flag);
  _max_value_size = 0;
  cmd_parser.add_cmdline_arg(this);
}

bool CMDLineParser::BoolArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for BoolArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::BoolArg::set_value(int argc, char* argv[])
{
  if (argc<0)
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << defflag << " requires 0 arguments.\n";
    std::exit(1);
  }
  if (_has_default)
  {
    _value = !_default_value;
  }
  else
  {
    _value = true;
    set_defined();
  }
  return 1;
}


//------ FloatArg Impelemenations ------
CMDLineParser::FloatArg::FloatArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool required)
{
  if (required)
    set_required();
  set_help_message(help);
  set_type_message("<float>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::FloatArg::FloatArg(CMDLineParser& cmd_parser, std::string flag, std::string help, float default_value, bool required)
: _value(default_value)
{
  set_defined();
  if (required)
    set_required();
  set_help_message(help+"\nDefault: "+std::to_string(default_value));
  set_type_message("<float>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

float CMDLineParser::FloatArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for FloatArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::FloatArg::set_value(int argc, char* argv[])
{
  if (argc<2)
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires an argument.\n";
    std::exit(1);
  }
  _value = atof(argv[1]);
  set_defined();

  return 2;
}


//------ IntArg Impelemenations ------
CMDLineParser::IntArg::IntArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool required)
{
  if (required)
    set_required();
  set_help_message(help);
  set_type_message("<int>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::IntArg::IntArg(CMDLineParser& cmd_parser, std::string flag, std::string help, int default_value, bool required)
: _value(default_value)
{
  set_defined();
  if (required)
    set_required();
  set_help_message(help+"\nDefault: "+std::to_string(default_value));
  set_type_message("<int>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

int CMDLineParser::IntArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for IntArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::IntArg::set_value(int argc, char* argv[])
{
  if (argc<2)
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires an argument.\n";
    std::exit(1);
  }
  _value = atoi(argv[1]);
  set_defined();

  return 2;
}


//------ StringArg Impelemenations ------
CMDLineParser::StringArg::StringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool required)
{
  if (required)
    set_required();
  set_help_message(help);
  set_type_message("<string>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::StringArg::StringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, std::string default_value, bool required)
: _value(default_value)
{
  set_defined();
  if (required)
    set_required();
  set_help_message(help+"\nDefault: "+default_value);
  set_type_message("<string>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

std::string CMDLineParser::StringArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for StringArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::StringArg::set_value(int argc, char* argv[])
{
  if (argc<2)
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires an argument.\n";
    std::exit(1);
  }
  _value = std::string(argv[1]);
  set_defined();

  return 2;
}


//------ VectorFloatArg Impelemenations ------
CMDLineParser::VectorFloatArg::VectorFloatArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, bool required)
: _count(count)
{
  if (required)
    set_required();
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<float> ";
  type_msg += "<float>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::VectorFloatArg::VectorFloatArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<float> default_value, bool required)
: _value(default_value), _count(count)
{
  set_defined();
  if (required)
    set_required();
  if (_value.size()!=_count)
  {
    std::cerr << "ERROR: Invalid default value in VectorFloatArg.  Expected " << _count << " values, but received " << _value.size() << "\n";
    std::exit(1);
  }
  help = help+"\n\tDefault: [";
  for (size_t i=0; i<_count-1; ++i)
    help += std::to_string(_value[i]) + ", ";
  help += std::to_string(_value[_count-1])+"]";
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<float> ";
  type_msg += "<float>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

std::vector<float> CMDLineParser::VectorFloatArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for FloatArg.\n";
    std::exit(1);
  }
}

float CMDLineParser::VectorFloatArg::operator()(size_t i) const
{
  if (is_defined())
    if (i>=0 && i<_value.size())
      return _value[i];
    else
    {
      std::cerr << "ERROR: Out of bounds request in VectorFloatArg.\n";
      std::exit(1);
    }
  else
  {
    std::cerr << "ERROR: Value not set for VectorFloatArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::VectorFloatArg::set_value(int argc, char* argv[])
{
  if (argc<1+int(_count))
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires " << _count << " arguments.\n";
    std::exit(1);
  }
  _value.resize(_count);
  for (size_t i=0; i<_count; ++i)
    _value[i] = atof(argv[1+i]);
  set_defined();

  return 1+_count;
}


//------ VectorIntArg Impelemenations ------
CMDLineParser::VectorIntArg::VectorIntArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, bool required)
: _count(count)
{
  if (required)
    set_required();
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<int> ";
  type_msg += "<int>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::VectorIntArg::VectorIntArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<int> default_value, bool required)
: _value(default_value), _count(count)
{
  set_defined();
  if (required)
    set_required();
  if (_value.size()!=_count)
  {
    std::cerr << "ERROR: Invalid default value in VectorIntArg.  Expected " << _count << " values, but received " << _value.size() << "\n";
    std::exit(1);
  }
  help = help+"\nDefault: [";
  for (size_t i=0; i<_count-1; ++i)
    help += std::to_string(_value[i]) + ", ";
  help += std::to_string(_value[_count-1])+"]";
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<int> ";
  type_msg += "<int>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

std::vector<int> CMDLineParser::VectorIntArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for IntArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::VectorIntArg::operator()(size_t i) const
{
  if (is_defined())
    if (i>=0 && i<_value.size())
      return _value[i];
    else
    {
      std::cerr << "ERROR: Out of bounds request in VectorIntArg.\n";
      std::exit(1);
    }
  else
  {
    std::cerr << "ERROR: Value not set for VectorIntArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::VectorIntArg::set_value(int argc, char* argv[])
{
  if (argc<1+int(_count))
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires " << _count << " arguments.\n";
    std::exit(1);
  }
  _value.resize(_count);
  for (size_t i=0; i<_count; ++i)
    _value[i] = atoi(argv[1+i]);
  set_defined();

  return 1+_count;
}


//------ VectorStringArg Impelemenations ------
CMDLineParser::VectorStringArg::VectorStringArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, bool required)
: _count(count)
{
  if (required)
    set_required();
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<string> ";
  type_msg += "<string>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::VectorStringArg::VectorStringArg(CMDLineParser& cmd_parser, std::string flag, size_t count, std::string help, std::vector<std::string> default_value, bool required)
: _value(default_value), _count(count)
{
  set_defined();
  if (required)
    set_required();
  if (default_value.size()!=_count)
  {
    std::cerr << "ERROR: Invalid default value in VectorStringArg.  Expected " << _count << " values, but received " << default_value.size() << "\n";
    std::exit(1);
  }
  help = help+"\nDefault: [";
  for (size_t i=0; i<_count-1; ++i)
    help += _value[i] + ", ";
  help += _value[_count-1]+"]";
  set_help_message(help);
  std::string type_msg;
  for (int j=0; j<int(_count)-1; ++j)
    type_msg += "<string> ";
  type_msg += "<string>";
  set_type_message(type_msg);
  add_cmdline_flag(flag);
  _max_value_size = _count;
  cmd_parser.add_cmdline_arg(this);
}

std::vector<std::string> CMDLineParser::VectorStringArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for StringArg.\n";
    std::exit(1);
  }
}

std::string CMDLineParser::VectorStringArg::operator()(size_t i) const
{
  if (is_defined())
    if (i>=0 && i<_value.size())
      return _value[i];
    else
    {
      std::cerr << "ERROR: Out of bounds request in VectorStringArg.\n";
      std::exit(1);
    }
  else
  {
    std::cerr << "ERROR: Value not set for VectorStringArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::VectorStringArg::set_value(int argc, char* argv[])
{
  if (argc<1+int(_count))
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires " << _count << " arguments.\n";
    std::exit(1);
  }
  _value.resize(_count);
  for (size_t i=0; i<_count; ++i)
    _value[i] = std::string(argv[1+i]);
  set_defined();

  return 1+_count;
}


//------ VariableVectorStringArg Impelemenations ------
CMDLineParser::VariableVectorStringArg::VariableVectorStringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, bool required)
{
  if (required)
    set_required();
  set_help_message(help);
  set_type_message("<string>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

CMDLineParser::VariableVectorStringArg::VariableVectorStringArg(CMDLineParser& cmd_parser, std::string flag, std::string help, std::vector<std::string> default_value, bool required)
: _value(default_value)
{
  set_defined();
  if (required)
    set_required();
  help = help+"\nDefault: [";
  for (size_t i=0; i<_count-1; ++i)
    help += _value[i] + ", ";
  help += _value[_count-1]+"]";
  set_help_message(help);
  set_type_message("<string>");
  add_cmdline_flag(flag);
  _max_value_size = 1;
  cmd_parser.add_cmdline_arg(this);
}

std::vector<std::string> CMDLineParser::VariableVectorStringArg::operator()() const
{
  if (is_defined())
    return _value;
  else
  {
    std::cerr << "ERROR: Value not set for StringArg.\n";
    std::exit(1);
  }
}

std::string CMDLineParser::VariableVectorStringArg::operator()(size_t i) const
{
  if (is_defined())
    if (i>=0 && i<_value.size())
      return _value[i];
    else
    {
      std::cerr << "ERROR: Out of bounds request in VectorStringArg.\n";
      std::exit(1);
    }
  else
  {
    std::cerr << "ERROR: Value not set for VectorStringArg.\n";
    std::exit(1);
  }
}

int CMDLineParser::VariableVectorStringArg::set_value(int argc, char* argv[])
{
  if (argc<2)
  {
    std::string defflag = _cmdline_flag[0];
    std::cerr << "ERROR: " << argv[0] << " requires an argument.\n";
    std::exit(1);
  }
  _value.push_back(std::string(argv[1]));
  set_defined();

  return 2;
}

size_t CMDLineParser::VariableVectorStringArg::size() const
{
  return _value.size();
}

};
#endif
