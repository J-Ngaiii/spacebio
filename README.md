# ML Dev Template
This repository is a basic template for a machine learning model developement repo. 

# Setup
For Python version, this repo requires at least Python 3.9 and strictly less than Python 3.12. You can do this by creating a conda enviornment:
    - run: `conda create -n mldev python=3.11.8`
    - then run: `conda activate mldev`

Move to the root directory via `cd` then run `pip install -r requirements.txt`. 

If you're running this repo on a Macbook and you run into installation issues install some of the dependencies via conda then try to install using requirements.txt again:
    - run: `conda install scipy pyarrow scikit-learn`
    - then run: `pip install -r requirements.txt`

Finally run `pip install -e .` to run files in the `src` folder of this repo as a package locally.
