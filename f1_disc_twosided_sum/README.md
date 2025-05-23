NOTE: this is a minimum working example of our code that may be used to reproduce our experimental results for discrete two-sided blotto with additive payoffs.

We have not put in much effort into making the code maintainable for this reason.

NOTE2: This code was an attempt at a faithful reimplementation of Farina et. al (2019) where the scaled extension was presented in the context of EFCEs. Hence, part of the C++ implementation follows their convention. We are merely adopting it here for our Blotto setting.

# Online Learning method
For online learning method, cd to ./fast and run make. 

Uncomment (and comment) the relevant lines, and the executable generated should be blotto_basic or blotto_alt. 
The former is for the basic (slow) version, and blotto_alt should be the faster version.

To change the size of the game and/or the method used, modify blotto_basic.cpp or blotto_basic_speedup.cpp respectively.

# LP
Run ./unit_tests/test_lp_solver.py (with the appropriate lines commented).