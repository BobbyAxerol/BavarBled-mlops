import sys
import os

# Align python path to the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate.evaluate_model import main

if __name__ == '__main__':
    main()
