import pandas as pd
import pickle
from raw_data_to_pickle import HealthClassifier

def review_predict(filename):
    with open('data_files/model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(filename, 'r') as myfile:
        new_review=myfile.read().replace('\n', '')

    series = pd.Series(new_review)
    print('Predicted Health Score: {}'.format(model.predict(series)))


if __name__ == '__main__':
    review_predict('ichiban_test.txt')
