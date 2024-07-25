About {#About}
============

\brief All about what Themis is, why it's needed, and how you can participate.


What is Themis?
------------------------
Themis is a *framework* for performing analyses of the ground-breaking EHT observations that is flexible and extensible.  It is easier, therefore, to say what Themis is not:

- Themis is *not* simply a theoretical image generation code.
- Themis is *not* simply a way to compute visibilities.
- Themis is *not* simply an analysis suite for semi-analytic RIAF models.
- Themis is *not* simply an Markov chain Monte Carlo parameter estimation code.

Rather, Themis provides a natural way to facilitate *all* of these while maintaining the flexibility to easily extend to novel analysis schemes, incorporate new kinds of data, and assess new kinds of models.  That is, Themis is a collection of data, model, and likelihood-sampler interfaces, already with many specific implementations, that enables the rapid development of new comparisons.  Importantly, by design, Themis is easily extendable in all of these dimensions, leveraging individual contributions to generate across-the-board improvements in functionality.  In this way, the value of incremental additions (e.g., a new data type) becomes geometrically magnified (i.e., can instantly be used with all of the pre-existing models and samplers).

This creates opportunities for a number of additional key efficiencies.  For example, Themis provides a natural way to exploit modern high-performance computing without the attendant investment in code parallelization expertise.  That is, parallelization options have already been implemented at a variety of levels permitting Themis users to drop large analysis jobs on machines at state-of-the-art computing facilities without needing to implement parallelized models on their own.


Why is Themis necessary? 
------------------------
Themis is a response to the need of the EHTC for a uniform platform on which to develop and compare analyses.   Already at its inception, the EHT Collaboration has available to it a wide variety of analysis tools. However, most of these have been developed independently by various different groups, resulting in an array of siloed analysis schemes.  Themis seeks to fix this by creating a way to bring all of these efforts together thereby facilitating effective collaboration and minimizing discrepancies in results.

At the same time, the enormous investment in data analysis by individual groups is geometrically enhanced by making those advances available to the broad community in a fashion that is immediately usable.  That is, with Themis it becomes trivial to mix and match new analysis schemes, new model components, and new kinds of data, as they are individually implemented.

Equally important, Themis dramatically reduces the timescale of the validation cycle.  Samplers, data types, model infrastructure, and model types can be independently validated, and are then available for use broadly.  New analysis efforts that use pre-existing Themis components are well on their way to being validated at creation.  The need for code redevelopment for typical analysis steps is substantially reduced, or in many cases completely eliminated, with everyone having direct access to the latest and greatest implementations.


Where can I get Themis?
------------------------
The latest version of Themis is always available at [github](https://github.com/PerimeterInstitute/Themis/tree/develop).  All members of the EHTC are welcome (and encouraged) to download and play with Themis.  Contact Avery E. Broderick (abroderick@perimeterinstitute.ca) to request access to the git repository.

There are two official Themis branches: master and develop.  The master branch is infrequently updated and beginning with version 1.0 will contain only validated analysis tools.  The develop branch is much more frequently updated with additional functionality, but contains potentially unvalidated analysis tools.  As tools are validated they will be merged into the master branch.

Many more specific development branches may exist at any given time, and are expected to be hotbeds of development activity.


Can I contribute to Themis?
------------------------
Yes!  Please do!  Themis only works if many people participate in its development.


How do I contribute to Themis? 
------------------------
The simple answer is to obtain access to the git repository, pull the latest develop branch, create a new branch specific to your project, and start writing code.

To ensure that Themis continues to be broadly accessible, we do have standards for code structure and documentation that can be found in detail on the [Contributing page](@ref Contribute).  Contributor's can "make pull" requests to have their additions incorporated into the develop branch after a cursory review that they perform as advertised, do not break existing functionality, and adhere to the coding guidelines.

The core Themis development team is happy to provide assistance and guidance.


Who is the core Themis Development Team?
------------------------
Themis was created within the [EHT Initiative](https://www.perimeterinstitute.ca/research/research-initiatives/event-horizon-telescope-eht-initiative) at the [Perimeter Institute for Theoretical Physics](https://www.perimeterinstitute.ca).  The most up-to-date list of developers, including the features for which they are responsible, can be found on the [Development Team page](@ref Developers).





