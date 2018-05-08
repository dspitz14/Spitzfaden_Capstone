import mysql.connector
import pandas as pd
import numpy as np
import string
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class HealthClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit RandomForestClassifier model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english', ngram_range=(2,2))
        self._classifier = RandomForestClassifier(class_weight="balanced", max_depth=6)

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        vec_X = self._vectorizer.fit_transform(X)
        self._classifier.fit(vec_X, y)

        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        vec_X = self._vectorizer.transform(X)
        return self._classifier.predict_proba(vec_X)

    def predict(self, X):
        """Make predictions on new data."""
        vec_X = self._vectorizer.transform(X)
        return self._classifier.predict(vec_X)


def transform_data_for_model(df):
    '''Splits into target and features ()'''

    df['A'] = np.where(df['inspection_grade']== 'A', 1, 0) #1 is A, 0 is B or C


    grade_df= df[['yelp_business_id', 'inspection_date', 'last_inspection','inspection_grade', 'A']].reset_index()
    grade_df= grade_df.sort_values(['yelp_business_id', 'inspection_date', 'last_inspection'])
    grade_df= grade_df.drop_duplicates(['yelp_business_id', 'inspection_date', 'last_inspection'])


    grouped_text= df.groupby(['yelp_business_id', 'inspection_date', 'last_inspection'])['yelp_review_text'].apply(lambda x: "{%s}" % '~~ '.join(x)).reset_index().sort_values(['yelp_business_id', 'inspection_date', 'last_inspection'])
    
    return grouped_text['yelp_review_text'].values, grade_df['A'].values




def match_reviews_and_inspections(mapper, inspections_df, reviews_df):
    '''Uses the mapper to combine inspections and reviews based on Yelp business id and county permit_number
       Returns a df that has a row for every yelp review and specifies which health period
       (should be jup 4)'''

    inspections_with_yelp = inspections_df.merge(mapper, left_on='permit_number', right_on='permit_number')
    inspections_with_yelp.sort_values('fuzz_ratio', ascending=False)
    reviews_df['yelp_review_date'] = pd.to_datetime(reviews_df['yelp_review_date'], errors='coerce')

    inspections_with_yelp['inspection_date'] = pd.to_datetime(inspections_with_yelp['inspection_date'], errors='coerce')
    inspections_with_yelp['last_inspection']= pd.to_datetime(inspections_with_yelp['last_inspection'], errors='coerce')

    merged= reviews_df.merge(inspections_with_yelp, left_on='yelp_business_id', right_on='id')
    merged = merged[(merged['yelp_review_date']>merged['last_inspection']) & (merged['yelp_review_date']<=merged['inspection_date'])]
    merged = merged.drop_duplicates(subset= ['review_id', 'yelp_business_id'])
    merged['A'] = np.where(merged['inspection_grade']== 'A', 1, 0) #1 is A, 0 is B or C
    merged= merged[['yelp_business_id','facility_id_x','permit_number','inspection_date', 'yelp_review_date','last_inspection', 'yelp_review_text','yelp_review_stars', 'inspection_grade', 'A']]
    return merged

def make_mapper(df1, df2, address1_column_name, address2_column_name, name1, name2, fuzz_thresh=60):
    '''Matches Yelp Business IDs with County Business IDs based on adresses and names
       Can set own threshold for fuzz ratio
       Outputs df with yelp_id, permit_id, etc.
       jup 3'''

    stripped_df1= strip_address(df1, address1_column_name, 'stripped1')
    stripped_df2= strip_address(df2, address2_column_name, 'stripped2')
    nonnull_df1 = stripped_df1[stripped_df1['stripped1']!= '']
    nonnull_df2 = stripped_df2[stripped_df2['stripped2']!= '']


    merge_df = nonnull_df1.merge(nonnull_df2, left_on=['stripped1'], right_on = ['stripped2'])
    merge_df = merge_df[pd.notnull(merge_df['stripped1'])]
    merge_df= get_fuzz(merge_df, merge_df['name'], merge_df['restaurant_name'])
    # merge_df[['id','facility_id','permit_number','name', 'restaurant_name', 'address_x', 'address_y', 'fuzz_ratio']].to_csv('data_files/full_mapper.csv')

    step_mapper = merge_df[(merge_df['fuzz_ratio']>= fuzz_thresh)]
    mapper = step_mapper[['id','facility_id','permit_number','name', 'restaurant_name', 'address_x', 'address_y', 'fuzz_ratio']].sort_values('fuzz_ratio', ascending=True)
    mapper= mapper[(mapper['facility_id']!='FA0082600')|(mapper['facility_id']!='FA0008423')|(mapper['facility_id']!='FA0004910')|(mapper['facility_id']!='FA0004969')|(mapper['facility_id']!='FA0005087')|(mapper['facility_id']!='FA0031638')|(mapper['facility_id']!='FA0031679')|(mapper['facility_id']!='FA0006336')|(mapper['facility_id']!='FA0060862')|(mapper['facility_id']!='FA0003331')|(mapper['facility_id']!='FA0008867')|(mapper['facility_id']!='FA0009125')]
    mapper= mapper.drop_duplicates('id')

    return mapper


def get_fuzz(df, col1, col2):
    '''Calculates fuzz scores for each row of given df on given columns and
       makes new column titled 'fuzz_ratio'
       Jup 3'''

    fuzz_scores=[]
    for i in range(len(df)): #improve to not loop like this
        fuzz_scores.append(fuzz.ratio(str(col1.iloc[i]), str(col2.iloc[i])))
    df['fuzz_ratio'] = fuzz_scores
    return df


def strip_address(df, column, new_column_name):
    '''Strips non-digit characters out of a column and appends values to df as new column.
       Input should look like strip_address(df, 'column', 'new_column_name')
       Jup 3'''

    non_dig= '''!()-[]{}abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ;:'"\,<>./?@#$%^&*_~'''
    stripped_address=[]

    #this next chunk needs to be improved :|
    for i in range(len(df)):
        name = str(df.iloc[i][column])
        no_punct = ""
        for char in name:
            if char not in non_dig:
                no_punct = no_punct + char
        stripped_address.append(no_punct.rstrip().lower())
    df[new_column_name] = stripped_address
    return df


def get_yelp_data():
    '''Gets Yelp data out of SQL database, returns 3 DFs (business, reviews, and categories)
    Jup 1'''
    cnx = mysql.connector.connect(user='root', password='root',
                                  host='localhost', port='8889',
                                  database='yelp_db', unix_socket='/Applications/MAMP/tmp/mysql/mysql.sock',
                                  connection_timeout=5)
    cursor = cnx.cursor()

    #Gets business info that are in Nevada
    bizquery = (
    '''
    SELECT * FROM business
    WHERE state = 'NV'
    ;''')

    cursor.execute(bizquery)
    cursor.fetchall()
    biz_df = pd.read_sql(bizquery, con=cnx)


    reviewquery = ('''
    SELECT
    business.id AS yelp_business_id, review.id AS review_id, review.date AS yelp_review_date,
    review.stars AS yelp_review_stars, review.text AS yelp_review_text, review.useful AS yelp_review_useful,
    review.funny AS yelp_review_funny, review.cool AS yelp_review_cool
    FROM review
    LEFT JOIN business ON review.business_id = business.id
    WHERE business.state = 'NV'
    ;''')

    cursor.execute(reviewquery)
    cursor.fetchall()
    review_df = pd.read_sql(reviewquery, con=cnx)


    # Gets categories for businesses that are in Nevada and are matched in dataset
    business_ids_for_cats_df = pd.read_csv('business_ids_for_cats.csv')
    business_ids_for_cats= ','.join(business_ids_for_cats_df['ids'].tolist())

    catquery=(
    '''
    SELECT business.id, category.category
    FROM business
    LEFT JOIN category ON business.id = category.business_id
    WHERE category.business_id IN ({})
    ;
    ''').format(business_ids_for_cats)

    cursor.execute(catquery)
    cursor.fetchall()

    # df is duplicated id with each category-- next chunk of code makes df with one row for a business_id
    cat_df = pd.read_sql(catquery, con=cnx)

    dumdums= pd.get_dummies(cat_df['category'])

    cat_cat= pd.concat([cat_df, dumdums], axis=1)
    cat_cat= cat_cat.drop('category', axis=1)
    groupped_cat= cat_cat.groupby('id').agg(np.sum).reset_index()

    cnx.close()

    # biz_df.to_csv('/Users/dspitzfaden/Galvanize/capstone/data_files/all_nevada_biz.csv')
    # review_df.to_csv('/Users/dspitzfaden/Galvanize/capstone/data_files/all_nevada_reviews.csv')
    # groupped_cat.to_csv('/Users/dspitzfaden/Galvanize/capstone/data_files/categories_from_sql.csv')
    return biz_df, review_df #, groupped_cat


if __name__ == '__main__':
    inspections_establishments_df = pd.read_csv('/Users/dspitzfaden/Galvanize/capstone/data/restaurant_establishments.csv', sep=';')
    health_inspections_df = pd.read_csv('/Users/dspitzfaden/Galvanize/capstone/data/health_inspections.csv')
    yelp_biz_df, yelp_review_df = get_yelp_data()
    print('Yelp Buisinesses: {}'.format(len(yelp_biz_df)))

    mapper = make_mapper(yelp_biz_df, inspections_establishments_df, 'address', 'address', 'name', 'restaurant_name')
    print('Matched Yelp Buisinesses: {}'.format(len(mapper)))
    # mapper.to_csv('data_files/mapper.csv')

    yelp_reviews_with_period = match_reviews_and_inspections(mapper, health_inspections_df, yelp_review_df)
    print('Number of Matched Yelp Review: {}'.format(len(yelp_reviews_with_period)))
    # yelp_reviews_with_period.to_csv('data_files/yelp_reviews_with_period.csv')

    # reviews_X, grades_y = transform_data_for_model(yelp_reviews_with_period)
    reviews_X, grades_y = transform_data_for_model(yelp_reviews_with_period)
    # print('Number of Health Inspection Periods: {}'.format(len(grades_y)))

    model= HealthClassifier()
    model.fit(reviews_X, grades_y)

    # rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(rf_pred, y_test).ravel()
    # precision, recall = (rf_tp / (rf_tp+rf_fp)), (rf_tp/(rf_tp+rf_fn))

    with open('data_files/model.pkl', 'wb') as f:
        pickle.dump(model, f)
