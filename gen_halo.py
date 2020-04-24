from halo import Hernquist
import sys

#try:
M = float(sys.argv[1])
a = float(sys.argv[2])
N = int(float(sys.argv[3]))
fname = sys.argv[4]
#except:
#    sys.exit("USAGE: python3 gen_halo.py M a N fname")

pot = Hernquist(M, a)
pot.gen_ics(N, fname)

print("successfully generated ics at:", fname, "with M=", str(M), "a=", str(a), "and N=", str(N))

