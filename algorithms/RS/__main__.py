"""Run RS through the common reproducible launcher."""
import sys
from aspect.train import main

if __name__ == "__main__":
    main(["--method", "RS", *sys.argv[1:]])

