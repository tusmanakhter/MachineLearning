#!/usr/bin/env bash

# Decision tree with data set 1 (English letters)
python3 model.py -d  -tr 'ds1/ds1Train.csv' -v 'ds1/ds1Val.csv' -r 'ds1/ds1Results-dt.csv' -s 'ds1/ds1Model-dt.pkl'

# Decision tree with data set 2 (Greek letters)
python3 model.py -d  -tr 'ds2/ds2Train.csv' -v 'ds2/ds2Val.csv' -r 'ds2/ds2Results-dt.csv' -s 'ds2/ds2Model-dt.pkl'

# Naive bayes with data set 1 (English letters)
python3 model.py -n  -tr 'ds1/ds1Train.csv' -v 'ds1/ds1Val.csv' -r 'ds1/ds1Results-nb.csv' -s 'ds1/ds1Model-nb.pkl'

# Naive bayes with data set 2 (Greek letters)
python3 model.py -n  -tr 'ds2/ds2Train.csv' -v 'ds2/ds2Val.csv' -r 'ds2/ds2Results-nb.csv' -s 'ds2/ds2Model-nb.pkl'

# Neural network with data set 1 (English letters)
python3 model.py -nn  -tr 'ds1/ds1Train.csv' -v 'ds1/ds1Val.csv' -r 'ds1/ds1Results-nb.csv' -s 'ds1/ds1Model-nb.pkl'

# Neural network with data set 2 (Greek letters)
python3 model.py -nn  -tr 'ds2/ds2Train.csv' -v 'ds2/ds2Val.csv' -r 'ds2/ds2Results-nb.csv' -s 'ds2/ds2Model-nb.pkl'