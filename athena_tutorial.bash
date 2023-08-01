# Athena++ Tutorial

# ------------------------------------------------------------------------------
# General setup
# ------------------------------------------------------------------------------

# 0. First, let's set up an environment and choose a location for the tutorial.
#
# For the environment, we will use the GCC compiler, MPI (for making a parallel
# code), and a parallel-aware HDF5 library (for writing some outputs).
#
# For the location, let's put this in ceph, which is Rusty's filesystem designed
# for data storage. Source code and scripts can go anywhere, but large I/O
# operations should be directed wherever the system administrators designate.
#
# On some systems, you may want to separate (1) the source code; (2) your
# working directory with job scripts, analysis scripts, etc.; and (3) the
# location of the outputs themselves. For this tutorial we'll do everything in
# one place.
#
# Do this in a "terminal" window through jupyterhub.

module load modules/2.2-alpha4 gcc/11.4.0 openmpi/4.0.7 hdf5/mpi-1.14.1-2
export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

cd ~/ceph/

# 1. Next, download the code. We'll be working entirely with the default branch,
# so there is no need to run any further git commands.

git clone https://github.com/PrincetonUniversity/athena athena_tutorial
cd athena_tutorial/

# 2. Take a look at the configure options. Some choices have to be made at
# compile time, and the configure.py script is what sets them. Any changes to
# these options requires compiling to a new executable.

nano configure.py  # options are described at the top
./configure.py -h

# ------------------------------------------------------------------------------
# Shock tube example
# ------------------------------------------------------------------------------

# 3. Let's compile the Sod shock tube problem. At this stage, we need to know
# the following:
#   - This is a hydrodynamics problem with an adiabatic equation of state, which
#     is set at compile time.
#   - This is simple enough to not need parallel computing, or even dedicated
#     resources.
#   - The default compiler (g++ here) is sufficient. Thus we do not need any
#     extra compiler flags.

./configure.py --prob shock_tube --coord cartesian --flux hllc --eos adiabatic \
  --nghost 3
make clean
make  # this creates bin/athena
mkdir tutorial_shock
mv bin/athena tutorial_shock/

# The last step moves the executable to keep it from being overwritten the next
# time we compile something. We'll place all our new files in this same place.

# 4. Grab a copy of the appropriate input file and modify it a bit. Take a look
# at this file; we'll only make a small modification at first, changing what
# name will be given to all the outputs from this simulation.

cp inputs/hydro/athinput.sod tutorial_shock/shock_base.athinput
nano tutorial_shock/shock_base.athinput
# Under the <job> section, change the "problem_id" from "Sod" to "shock_base".

# 5. Now, run the code. This is much less resource-intensive than compilation.
# Afterward, the outputs will be in the tutorial_shock/ directory.

tutorial_shock/athena -i tutorial_shock/shock_base.athinput -d tutorial_shock/

# The output to the terminal should end with something like:
#
# Terminating on time limit
# time=2.5000000000000000e-01 cycle=175
# tlim=2.5000000000000000e-01 nlim=-1
#
# zone-cycles = 44800
# cpu time used  = 2.8230999999999999e-02
# zone-cycles/cpu_second = 1.5869080089263576e+06

# 6. Let's visualize the output, starting with the initial conditions. Try doing
# this in a separate console, without the environment modifications we made in
# step 0.

module load python/3.10.8
cd ~/ceph/athena_tutorial/

vis/python/plot_lines.py tutorial_shock/shock_base.block0.out1.00000.tab x1v \
  rho,press,vel1 tutorial_shock/shock_base_init.png -l '$\rho$,$p$,$v^x$' \
  --x_min=-0.5 --x_max=0.5 --x_label '$x$'
# Open tutorial_shock/shock_base_init.png to see the plot.

# 7. Now, let's see the final state.

vis/python/plot_lines.py tutorial_shock/shock_base.block0.out1.00025.tab x1v \
  rho,press,vel1 tutorial_shock/shock_base_final.png -l '$\rho$,$p$,$v^x$' \
  --x_min=-0.5 --x_max=0.5 --x_label '$x$'
# Open tutorial_shock/shock_base_final.png to see the plot.

# ------------------------------------------------------------------------------
# Shock tube exercises
# ------------------------------------------------------------------------------

# 8. Try lower and higher resolutions of the shock tube.

cp tutorial_shock/shock_base.athinput tutorial_shock/shock_low.athinput
cp tutorial_shock/shock_base.athinput tutorial_shock/shock_high.athinput

# Edit the two new files. Try a low resolution of 64 cells, and a high
# resolution of 1024 cells. Be sure to rename the output ("problem_id") to
# something like "shock_low" or "shock_high", in order to not overwrite the
# existing files.

tutorial_shock/athena -i tutorial_shock/shock_low.athinput -d tutorial_shock/
# Similar for high resolution. Note the "cpu time used" statistics printed to
# the terminal. Is their ratio what you expect?

# After running, plot the results and see if there are differences.

vis/python/plot_lines.py ...

# 9. Try a new case. Copy and modify the shock_base.athinput file. This time,
# use a high-order time integrator ("rk3" instead of "vl2"), and a high-order
# reconstruction algorithm (xorder=3 instead of xorder=2).

# 10. In a jupyter notebook, try loading the raw data from one of the output
# files. You can use the "fluid" notebook set up yesterday.

%cd ~/ceph/athena_tutorial/
my_cwd = %pwd
import numpy as np
import sys
sys.path.insert(0, my_cwd + '/vis/python')
import athena_read

data = athena_read.tab('tutorial_shock/shock_base.block0.out1.00025.tab')

# "data" should be a dictionary containing arrays of all variables of interest.
# Try calculating the entropy, and maybe plotting it as a function of x ("x1v").

# ------------------------------------------------------------------------------
# Orszag-Tang vortex example
# ------------------------------------------------------------------------------

# 11. Let's compile the Orszag-Tang vortex problem, back in the original
# console. At this stage, we need to know the following:
#   - This is an MHD problem, so we need to compile with MHD enabled.
#   - We will want to distribute across multiple processes with MPI.
#     - This could run on a single process, but will take longer.
#   - We will want to use HDF5 (.athdf) outputs.
#     - Athena++ also supports ASCII (.tab) outputs, but these are extremely
#       clunky except in 1D, and this is a 2D problem.
#     - Athena++ also supports VTK (.vtk) outputs, but these are made per block
#       and are not recommended for domain-decomposed jobs.
#   - The default mpicxx already points to the desired C++ compiler (g++ here),
#     and the jupyter console node has the same architecture as our target
#     compute nodes (Intel here). Thus we do not need any extra compiler flags.

./configure.py --prob orszag_tang -b --coord cartesian --flux hlld \
  --eos adiabatic --nghost 3 -mpi -hdf5
make clean
make
mkdir tutorial_ot
mv bin/athena tutorial_ot

# 12. Again, grab the appropriate input file, and modify it to create our base
# case.

cp inputs/mhd/athinput.orszag-tang tutorial_ot/ot_base.athinput

nano tutorial_ot/ot_base.athinput
# Modify tutorial_ot/ot_base.athinput in the following ways:
#   - Under <job>:
#     - Change "problem_id" from "OrszagTang" to "ot_base".
#   - Under <output2>:
#     - Change "file_type" from "vtk" to "hdf5".
#     - Add the line "id = prim".
#     - (Optional) Add the line "xdmf = 0". This turns off helper files for
#       reading the outputs into a program like VisIt. If you want to play
#       around with using VisIt on the data, don't add this line.
#   - Under <mesh>:
#     - Change both "nx1" and "nx2" from "500" to "300".
#   - Under <meshblock>:
#     - Change both "nx1" and "nx2" from "500" to "150".

# 13. This time, we'll run the code via Slurm on multiple cores. Copy the
# separate ot_base.bash file into the tutorial_ot/ directory. Take a look at the
# file. This will run the code on 4 cores.

nano tutorial_ot/ot_base.bash

# 14. Run the code.

sbatch tutorial_ot/ot_base.bash

# Wait until the job finished (when it no longer appears when running
# "squeue --me". The file tutorial_ot/ot_base.out should contain the terminal
# output of the code, ending with a "Terminating on time limit" message and some
# summary statistics.

# 15. Visualize the outputs. Use the second console we opened for visualization.

vis/python/plot_slice.py tutorial_ot/ot_base.prim.00100.athdf rho \
  tutorial_ot/ot_base_rho.png
vis/python/plot_slice.py tutorial_ot/ot_base.prim.00100.athdf press \
  tutorial_ot/ot_base_press.png -c plasma

# Try adding "--stream Bcc --stream_density 2.0" to the above to add magnetic
# field stream lines.

# ------------------------------------------------------------------------------
# Orszag-Tang vortex exercises
# ------------------------------------------------------------------------------

# 16. Try running the code on more cores.

cp tutorial_ot/ot_base.athinput tutorial_ot/ot_many.athinput
cp tutorial_ot/ot_base.bash tutorial_ot/ot_many.bash

# In the input file, change the meshblock size from 150^2 to 100^2, and change
# the "problem_id" to "ot_many". In the submission file, change the number of
# tasks and number of tasks per node to 9, and change "base" to "many" in the
# "--output" and "input_file" lines.

sbatch tutorial_ot/ot_many.bash

# Wait for the job to finish. Is the cpu time used what you expect? Then on the
# visualization console:

vis/python/plot_slice.py ...

# Check that the solution is identical, no matter what the domain decomposition
# is.

# 17. Let's make a movie of the outputs, since we have 101 of them. Even a quick
# and dirty movie should avoid having axes jump around from frame to frame, so
# we need to fix the color scale. This will plot density for each output in
# sequence:

for n in 00{00{0..9},0{10..99},100}; do echo $n; vis/python/plot_slice.py \
  tutorial_ot/ot_base.prim.$n.athdf rho tutorial_ot/ot_base_rho.$n.png \
  --vmin 0.0 --vmax 0.35; done

# ffmpeg is a ubiquitous command-line tool for making and modifying movies.

module load ffmpeg
ffmpeg -framerate 12 -pattern_type glob -i 'tutorial_ot/ot_base_rho.*.png' \
  -r 12 -pix_fmt yuv420p -vf 'pad=962:958:0:0' -y tutorial_ot/ot_base_rho.mov

# You might have to copy the movie to your local machine to view it.

# Note: If using the yuv420p encoder like above, ffmpeg requires an even number
# of pixels in each dimension, which might not happen. Let's say your images are
# actually 961*958, as they are in this case. The -vf argument pads with a
# column of (white) pixels.

# 18. Finally, let's add adaptive mesh refinement to increase resolution only
# where needed.

# 18a. Make a copy of the input file:

cp tutorial_ot/ot_many.athinput tutorial_ot/ot_amr.athinput

# Modify it as follows:
#   - Give it a new name with problem_id, like ot_amr.
#   - We simultaneously want a somewhat coarse root grid, whose resolution is
#     given by the <mesh> parameters, and a fair number of domains at the root
#     level, whose size is given by the <meshblock> parameters. Try 180 for the
#     former and 30 for the latter.
#   - Under <mesh>, add "refinement = adaptive", "derefine_count = 5", and
#     "numlevel = 3".
#   - Make a new <problem> block. Add the parameters "refine_lim = 0.005" and
#     "derefine_lim = 0.0025".

# 18b. Make a copy of the submission file:

cp tutorial_ot/ot_many.bash tutorial_ot/ot_amr.bash

# Change the output file to "tutorial_ot/ot_amr.out". Modify "bin_name" to be
# "tutorial_ot/athena_amr" (we will use a new executable) and "input_file" to
# the file we just created and modified.

# 18c. Now we need to write code that decides whether a block of cells is too
# coarse, too fine, or at a good resolution. Make a copy of the pgen file for
# this problem, which is what we've been compiling all along:

cp src/pgen/orszag_tang.cpp src/pgen/orszag_tang_amr.cpp

# Toward the beginning of the file, before the first function definition, add
# the declarations:
#
# int RefinementCondition(MeshBlock *pmb);
# namespace {
#   Real refine_lim;
#   Real derefine_lim;
# }

# In Athena++, extra user functions must be enrolled, to tell the code to
# execute them at the appropriate time. This is done in the InitUserMeshData
# function, which doesn't yet exist. This same function can also read parameters
# from the input file and store them somewhere useful. Add the following to the
# .cpp file:
#
# void Mesh::InitUserMeshData(ParameterInput *pin) {
#   if (adaptive) {
#     EnrollUserRefinementCondition(RefinementCondition);
#     refine_lim = pin->GetReal("problem", "refine_lim");
#     derefine_lim = pin->GetReal("problem", "derefine_lim");
#   }
#   return;
# }
#
# This is also where custom boundary functions would be enrolled.

# We can now define the refinement criterion. Here well examine the largest
# value of the Laplacian of both density and pressure, with respect to cell
# widths rather than physical widths. For a given physical state, this quantity
# should decrease as the grid is refined, so we can set thresholds for it to
# trigger refinement (if too large) or derefinement (if sufficiently small). The
# code will run this function and (de)refine blocks of cells as necessary.
#
# int RefinementCondition(MeshBlock *pmb) {
#   AthenaArray<Real> &prim = pmb->phydro->w;
#   Real max_laplacian = 0.0;
#   int ks = pmb->ks;
#   int js = pmb->js;
#   int je = pmb->je;
#   int is = pmb->is;
#   int ie = pmb->ie;
#   for (int j = js; j <= je; j++) {
#     for (int i = is; i <= ie; i++) {
#       Real rho_x = prim(IDN,ks,j,i+1) - 2.0 * prim(IDN,ks,j,i) + prim(IDN,ks,j,i-1);
#       Real rho_y = prim(IDN,ks,j+1,i) - 2.0 * prim(IDN,ks,j,i) + prim(IDN,ks,j-1,i);
#       Real laplacian_rho = (std::abs(rho_x) + std::abs(rho_y)) / prim(IDN,ks,j,i);
#       Real press_x = prim(IPR,ks,j,i+1) - 2.0 * prim(IPR,ks,j,i) + prim(IPR,ks,j,i-1);
#       Real press_y = prim(IPR,ks,j+1,i) - 2.0 * prim(IPR,ks,j,i) + prim(IPR,ks,j-1,i);
#       Real laplacian_press = (std::abs(press_x) + std::abs(press_y)) / prim(IPR,ks,j,i);
#       max_laplacian = std::max(max_laplacian, std::max(laplacian_rho, laplacian_press));
#     }
#   }
#   if (max_laplacian > refine_lim) {
#     return 1;
#   }
#   if (max_laplacian < derefine_lim) {
#     return -1;
#   }
#   return 0;
# }

# 18d. Compile the code, move the executable to tutorial_ot/athena_amr, run the
# code and inspect the results:

./configure.py --prob orszag_tang_amr -b --coord cartesian --flux hlld \
  --eos adiabatic --nghost 4 -mpi -hdf5
make clean
make
mv bin/athena tutorial_ot/athena_amr
sbatch tutorial_ot/ot_amr.bash

vis/python/plot_slice.py tutorial_ot/ot_amr.prim.00000.athdf Levels \
  tutorial_ot/ot_amr_level.00000.png --vmin 0 --vmax 2
vis/python/plot_slice.py tutorial_ot/ot_amr.prim.00010.athdf Levels \
  tutorial_ot/ot_amr_level.00010.png --vmin 0 --vmax 2
vis/python/plot_slice.py tutorial_ot/ot_amr.prim.00100.athdf Levels \
  tutorial_ot/ot_amr_level.00100.png --vmin 0 --vmax 2

# AMR is as much an art as a science. Here, at early times, the grid was at
# different levels. Later, everything was refined, and there was no speedup
# anywhere. You can play with the refinement limits, the depth of refinement, or
# even the criterion itself (cell-to-cell gradients rather than Laplacians are
# also a common choice).

# ------------------------------------------------------------------------------
# Further directions
# ------------------------------------------------------------------------------

# If you have time today, or time throughout the school, there are many more
# advanced directions you can explore with Athena++. Some ideas:
#
# - Is there something coming up in your own research/use of Athena++?
# - Other common test problems, like Kelvin-Helmholtz, blast waves, or linear
#   waves.
# - Adding your own source terms.
# - Using other coordinate systems.
# - Static mesh refinement strategies for accretion problems.
# - Additional physics: relativity, general equations of state, radiation, etc.
# - Interfacing with other visualization/analysis tools, like VisIt.
#
# Feel free to bring them up to me (Chris White), or email me
# (chwhite@flatironinstitute.org).
