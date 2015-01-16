
Vector field visualization in 3D using CUDA
=========
This project demonstrates visualization techniques like glyphs, stream lines, stream tubes, and stream surfaces — all in real time.
The key is RK4 integrator implemented in CUDA that is using very fast texture lookup functions to access a vector field.
The vector field itself is stored as a 3D texture which enables to use hardware accelerated trilinear interpolation lookup functions.
The project page contains more than 100 images and figures and commented code snippets.

**Author**: Marek Fiser &lt; code@marekfiser.cz &gt;

**Project page**: http://www.marekfiser.com/Projects/Real-time-visualization-of-3D-vector-field-with-CUDA


**License**: Public domain, see LICENSE.txt for details.

See other license files for inluded libraries: FreeGlut and Glew.


Features
--------

* Four 3D vector field visualization techniques.
  * Glyphs, stream lines, stream tubes, and stream surfaces.
  * Some techiniques asre using adaptive seeding.
* Two integrators of 3D vector fields written in CUDA.
  * Euler and Runge–Kutta 4
* Real-time visualization in OpenGL.
  * OpenGL-CUDA interoperability used for fast display (no GPU-CPU copy).
* Simple NRRD file loader.


Compiling and running
--------

In order to compile/run this application you probably need to have CUDA SDK installed and your NVIDIA graphics card needs to have CUDA Capability at least 2.0.
All other necessary DLLs are included in this package.

There is also compiled executable in the bin folder.

The program requires *.nrrd data file as an input.
Program was tested with delta-wind dataset and it was not ment to be used with anything else.
You can find thid dataset on the project page in the downloads secion.
However, it should be cappable to open other NRRD files with the same structure as well.