Contribute {#Contribute}
============

\brief Welcome to the Themis development team!  Without you, Themis doesn't work!


Themis design philosophy
------------------------
Themis is a *framework*, and is therefore designed to be adaptable and easily applied.  Therefore, the key philosophical considerations in its design are that it must be:

**Modular**:
Users should only ever need to know the inner workings of their particular piece.  It should never be necessary for any user to be an expert on all parts of the code for it to work.

**Flexible**:
Parts should be interchangeable where possible.  For example, one model should be able to be swapped for another without needing to modify the data, likelihoods, or sampler.

**Extensible**:
Contributors should be able to add functionality to Themis easily, without understanding in detail large parts of the code.  This also means that it must be sufficiently well documented that everyone knows what each piece does without having to resort to reading the source code.

**Rigorous**:
Components of Themis must ultimately undergo validation testing.  This process is greatly simplified if each component is rigorously tested during creation.  While no component testing regime can eliminate all bugs, producing test routines concurrently with code generation can greatly reduce them.



Themis structure and coding guidelines
------------------------
Themis is written in C++, which naturally facilitates modularity and flexibility.  As with any programming language, C++ admits many styles choices.  Here we discuss some of the choices that have been made with Themis.

**Structure**

Themis itself has been organized into five directories, each of which hold different kinds of code elements:

- [data](files.html): These are types of data.  Each data type has a datum struct (e.g., Themis::datum_visibility_amplitude) and a data class (e.g., Themis::data_visibility_amplitude), where the latter has I/O functions that permit the reading and manipulation of many individual datum objects.  Anything that is directly measured should ultimately be a data object.

- [model](files.html): These are various models that produce theoretical predictions for the kinds of data found in *data*.  Note that not all models can produce all kinds of data.  All models are, however, based on a prototypical abstract base class associated with each data type, e.g., Themis::model_visibility_amplitude.  Models can, naturally, be based on more than one such class.  Anything that makes a prediction should be a model object.

- [likelihood](files.html): These provide a means to directly compare models with data to produce log-likelihoods and \f$\chi^2\f$'s.  These may be simply a sum of the squares (e.g., Themis::likelihood_visibility_amplitude) or may be more sophisticated expressions that analytically optimize over a subset of nuisance parameters (e.g., Themis::likelihood_marginalized_visibility_amplitude).  In addition, parameter priors and transforms are located here.

- [sampling](files.html): These are schemes for sampling likelihoods over a given parameter space.

- [util](files.html): This contains a collection of utility functions that are used by other elements of the code but do not properly belong in one of the foregoing categories, e.g., constants.h.

The remaining directories contain drivers, i.e., files that produce executables that use Themis, or analysis tools.

Within each source directory, we have followed a file naming scheme that makes clear what kind of object is defined within.  For example, a Themis::model_image_riaf is a RIAF model based upon the Themis::model_image class; whereas Themis::model_riaf is a RIAF model that is not based solely on the Themis::model_image class since it also provides a flux prediction.  While the documentation does provide an index of where each class definition may be found, having descriptive file names simplifies code maintenance.

In addition the Vacuum Ray Tracing and Radiative Transfer routine library (VRT2) is provided.  This is the code used to generate the RIAF image libraries utilized in [Broderick et al. (2016)](http://adsabs.harvard.edu/abs/2016ApJ...820..137B) and prior publications.


**Guidelines**

We recommend the [Google style guide](https://google.github.io/styleguide/cppguide.html) to novice and experienced coders alike, which discusses many C++ style choices.  With regard to Themis, we have a few guidelines of our own to maintain extensibility.

- *Always* use define guards on header files.  These should have the form Themis_<file name>_H_.  This prevents reading in the same header file multiple times (an error in C++ compilation).

- *Always* use the Themis namespace.  You can do this by encapsulating your code in "using namespace Themis { ... };".  *Never* use "using namespace ...", which tends to lead to namespace collisions in large codes.  

- *Always* separate declarations and implementations into header (.h) and source (.cpp) files, *except* when the function is "inline" or an abstract template.

- *Everywhere possible* use the "const" keyword to indicate things that *should* never change.

- *Everywhere possible* make class members private unless there is an overwhelming reason to permit it to be public.

- *Everywhere possible* create new classes from old classes to inherit their interface.  This is especially true for models, which should make use of the abstract base classes which define their interfaces.

- *Where possible* compile functions directly into Themis (i.e., make sparing use of "std::system").  Many codes are not implemented in C++ (or C) and thus compiling them directly into Themis is nontrivial.  We have made use of "std::system" to run executables generated via other programs (e.g., Themis::model_image_astroray).  While convenient and simple, this limits the extensibility by forcing the entire interface to run through a single external executable.

- *Usually* it is better to use longer, descriptive class names rather than short, obscure abbreviations.  Similarly, use descriptive variable names* to avoid confusion in the future.

- *Always* prepend an underscore (i.e., "_") to class variables to indicate that they are defined beyond a given function (and may change!).

- *Always* error on the side of too many comments!  That's a joke, you can never have too many comments!

- *Always* be transparent and verbose about choices that are made within the code.

- *Small and simple* code fragments are easier to understand and more likely to be correct than *large and complex* code soup.

- *Always* make sure that new code is MPI-safe, i.e., can be run on multiple processors without obvious failures.  Note that this does not mean it must be parallelized!  If you wish to implement MPI directives, *always* use C-style bindings.

- *Always* include header files only in the files where they are required.  *All* files should be header-complete, i.e., no files should anticipate some set of headers will be included upstream.

- *Never* link to proprietary commercial software if a widespread, free option exists.  Themis must work on a variety of platforms spread across many continents.  If possible, provide all functions within Themis itself.

- *Always* provide clear documentation of every public class function (see the documentation requirements below).  This includes a file-specific comment at the beginning of every file describing its purpose and contents.  *Always* pay attention to the Doxygen comment rules to avoid documentation errors.

- *Never* submit broken code or code that modifies pre-existing test performance.



Parallelization
------------------------
Themis currently supports parallelization with MPI.  This has been implemented flexibly, permitting independent parallel execution at multiple levels, including
- Within the sampler
- Within the likelihood computation
- Within the model generation

These have been organized via the distribution of MPI Communicators passed to each level of the computation (sampler to likelihood to model), permitting each to organize the set of processes the next will execute upon.  A practical example may be found in riaf_model_fitting.cpp, which splits the number of processes multiple times within the sampler (into tempering levels and then among individual walkers) and then passes collections of processors to the likelihoods which then use those to parallely generate images.

Support for OpenMP has not yet been tested.


Documentation requirements
------------------------
Good documentation is critical to facilitating the EHTC's use and development of Themis.  Thus, documentation should not be relegated to the status of necessary evil, but rather considered a critical programming task.  Themis uses [Doxygen](http://www.doxygen.org/) to automatically produce this documentation.  This has the great advantage of closely tying the code and documentation together.  However, it does place some specific requirements on the way in which the source code is documented.

The [Doxygen Quick Reference sheet](../Doxygen-QuickReference.pdf) provides a short primer on the key Doxygen documentation directives.  An example implementation of these rules can be found in model_visibility_amplitude.h and model_visibility_amplitude.cpp for C++ header and source files, and plots.py for python script files.

In summary:

- All files must have a Doxygen preamble that lists the file name, author, date created, and a brief description for what the file contains.

- Each class definition must have a preceding Doxygen comment that gives a brief and detailed description of what the class is.

- Each public class function must have a short description describing its purpose, inputs, and outputs.

Doxygen recognizes these comments via their specific format,
\verbatim
/*!
  <comment>
*/
\endverbatim
or
\verbatim
/// <comment>
\endverbatim
neither of which should be used otherwise.  In addition, within these comments there are a number of key words, e.g.,
\verbatim
\file, \author, \date, \brief, \details, \warning, etc.,
\endverbatim
that describe specific elements of the documentation.



Starting your Themis project
------------------------
Information about example Themis drivers, which step through the typical steps for setting up and running a parameter estimation analysis of EHT data, can be found on the [Examples and Tests page](@ref Tests).

Prior to beginning a new project, you should pull the latest develop branch from [github](https://github.com/PerimeterInstitute/Themis/tree/develop); should you need access to the repository contact Avery E. Broderick (abroderick@perimeterinstitute.ca).  You may create a new development subbranch for your project via the branch button on the left: simply enter your new branch name and you are ready to go.  When you are confident that your contribution works and is properly documented, you may make a pull request to have it incorporated back into the develop branch.  Note that a cursory review will be made at that time to ensure that all previous functionality remains unchanged, the coding guidelines are adhered to, and proper documentation has been supplied prior to inclusion.


Getting help
------------------------
The Themis Development Team is here to help! If you need assistance, please [contact us](@ref Developers).




