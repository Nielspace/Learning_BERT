from requirements import *
from sklearn.model_selection import train_test_split

data = pd.read_csv('asap-aes/training_set_rel3.tsv',sep='\t', encoding='ISO-8859-1')

essay = data.rename(columns={0:'essay_id',1:'essay_set',2:'essay',
                                 6:'domain1_score'})

# print(essay)
scores = essay.domain1_score
dataset = essay.loc[:,['essay_id', 'essay_set', 'essay', 'domain1_score']]


dataset.domain1_score = pd.to_numeric(dataset.domain1_score, errors='coerce')
dataset = dataset[dataset['domain1_score'] < 61]
dataset = dataset.drop('essay_id', 1)

essay_sets = np.unique(dataset['essay_set'])


for set_no in essay_sets:
    indices = dataset[dataset['essay_set'] == set_no].index.tolist()
    grade_max = np.max(dataset.loc[indices, 'domain1_score'])
    grade_min = np.min(dataset.loc[indices, 'domain1_score'])
    dataset.loc[indices, 'domain1_score'] = 12*(dataset.loc[indices, 'domain1_score'] - grade_min)/(grade_max - grade_min)


X_train, X_test, y_train, y_test = train_test_split(dataset['essay'], dataset['domain1_score'], test_size=0.10, random_state = 23)



print('All the preprocessing data has been imported. Please make sure that you have also imported requirements')