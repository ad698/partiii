### The series of commands to generate from scratch all plots in the
### self-consistent paper

OMP_NUM_THREADS=16 nice python2.7 williams_evans_hernquist.py build;
OMP_NUM_THREADS=16 nice python2.7 williams_evans_hernquist.py veldisp;
OMP_NUM_THREADS=16 nice python2.7 williams_evans_hernquist.py proj;
OMP_NUM_THREADS=16 nice python2.7 williams_evans_hernquist.py split;
nice python2.7 williams_evans_hernquist.py plot;
# Makes Fig 1, 2, 3, 4, 8


OMP_NUM_THREADS=16 nice python2.7 binney_isochrone.py build;
OMP_NUM_THREADS=16 nice python2.7 binney_isochrone.py veldisp;
OMP_NUM_THREADS=16 nice python2.7 binney_isochrone.py split;
nice python2.7 binney_isochrone.py plot;
# Makes Fig 5, 6, 7

OMP_NUM_THREADS=16 nice python2.7 D1_spherical_models.py build;
OMP_NUM_THREADS=16 nice python2.7 D1_triaxial_models.py build;
nice python2.7 D1_triaxial_models.py plot;
OMP_NUM_THREADS=16 nice python2.7 D1_spherical_models.py veldispstack;

# Then piss about with the shapes at r=0.1, r=1, r=10 --- currently not automated
# Makes Fig 9, 10
