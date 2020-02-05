import pandas as pd

# L1.Classification
house_votes_columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador', 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
house_votes = pd.read_csv('data/house-votes-84.csv', true_values=['y'], false_values=['n'], na_values=['?'], names=house_votes_columns)
house_votes.fillna(False, inplace=True)
house_votes.iloc[:,1:] = house_votes.iloc[:,1:].astype('int8')
