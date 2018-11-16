#!/usr/bin/env bash

# Decision tree with data set 1 (English letters)
python3 model.py -te 'ds1/ds1Test.csv' -r 'ds1/ds1Test-dt.csv' -m 'ds1/ds1Model-dt.pkl'

# Decision tree with data set 2 (Greek letters)
python3 model.py -te 'ds2/ds2Test.csv' -r 'ds2/ds2Test-dt.csv' -m 'ds2/ds2Model-dt.pkl'

# Naive bayes with data set 1 (English letters)
python3 model.py -te 'ds1/ds1Test.csv' -r 'ds1/ds1Test-nb.csv' -m 'ds1/ds1Model-nb.pkl'

# Naive bayes with data set 2 (Greek letters)
python3 model.py -te 'ds2/ds2Test.csv' -r 'ds2/ds2Test-nb.csv' -m 'ds2/ds2Model-nb.pkl'

# Neural network with data set 1 (English letters)
python3 model.py -te 'ds1/ds1Test.csv' -r 'ds1/ds1Test-3.csv' -m 'ds1/ds1Model-3.pkl'

# Neural network with data set 2 (Greek letters)
python3 model.py -te 'ds2/ds2Test.csv' -r 'ds2/ds2Test-3.csv' -m 'ds2/ds2Model-3.pkl'