Hi,

1. First you should install tact. To do this clone my github repository:

git clone https://github.com/jls713/tact.git

2. then go into tact directory

cd tact

3. You need to change compiler (CCOMPILER). If using IoA system use

CCOMPILER=/opt/ioa/software/gcc/4.7.2/bin/g++

4. then run

make

5. Now copy and paste df directory into tact directory

6. go into df directory

cd df

7. and run

make

==============================================================================

To test, run

python williams_evans_hernquist_sph.py build
python williams_evans_hernquist_sph.py veldisp

This outputs files like 'data/*_it*.vis' that give 'x y z r rho' in the first five columns
and 'data/*.xdisp' that gives the dispersions

and

python D1_spherical_models.py build
python D1_spherical_models.py veldispstack beta_plot.eps

These should output a sequence of models with varying outer anisotropy.

========================================================================

To do:

1. See if you can work out how to alter williams_evans_hernquist.py to generate some of the other models from Williams & Evans (2015)
e.g. Jaffe, NFW, etc.

2. Adjust D1_spherical_models.py to create a sequence of models that have varying inner anisotropy and constant outer anisotropy.

3. Create a new distribution function class based on the williams evans models and add a net streaming motion.

4. Add functions to src/df.cpp and src/moments.cpp that compute the mean velocity moments.


