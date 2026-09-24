"""Run AS_Full_FT through the common reproducible launcher."""
import sys
from aspect.train import main

if __name__ == "__main__":
    main(["--method", "AS_Full_FT", *sys.argv[1:]])

