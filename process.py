import numpy as np
import pandas as pd


from sent_lexicon import get_sent_score

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
#https://github.com/nalepae/pandarallel/blob/master/docs/examples_mac_linux.ipynb


# import df with pandas

df['sent'] = df['text'].parallel_apply(lambda x: get_sent_score(x))
