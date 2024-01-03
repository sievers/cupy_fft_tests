Example script to test timing of lots of repeated FFTs of large matrices of a
variety of sizes (real life application is Fourier-filtering lots of data
chunks as part of an iterative PCG solver where the noise is defined
in Fourier space).

No build script, but you can compile as follows:
nvcc -o libpycufft.so pycufft.cu -shared -lcufft -Xcompiler -fPIC -lgomp

then, assuming you're in the same directory and have "." in your LD_LIBRARY_PATH,
you can run:

python3 cupy_vs_shared_plans.py

The version that caches the FFT plans first, then calls cufft directly with
pre-allocated storage runs nearly twice as fast on an RTX 3090 as the simple
cupy commands.
