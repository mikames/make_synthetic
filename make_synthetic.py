from __future__ import print_function, absolute_import, division
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import numpy as np
import random 
import argparse
import json
import re
import os
import sys
import csv
import shutil
import s3fs
import boto3
s3 = boto3.resource('s3')

# --- Faker ---
from faker import Faker
from fake_useragent import UserAgent
fake = Faker()
# raker.seed(123)
from faker.providers import internet, profile, ssn, date_time, phone_number, credit_card
fake = Faker()
fake.add_provider(internet)
ua = UserAgent()

# --- catboost --- 
from catboost import CatBoostClassifier, Pool

# --- sklearn stuff 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score,  auc, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction import FeatureHasher

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, Imputer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample


# --- functions ---
def get_config(config):
    """ convert json config file into a dict  """
    print("load config...")
    with open(config, 'r') as f:
        config_dict = json.load(f)[0]
    return config_dict

def get_dataset(config):
    """ creates a predictive dataset based on the config"""
    print("generate base dataset...")
    config = get_config(config)

    n_features = len(config['categorical_features'])  + len(config['numeric_features'] )
    
    X, y = make_classification(n_samples= config['n_samples'],
                               n_features= n_features, 
                               n_informative = n_features,
                               n_redundant= 0,
                               n_repeated=0,
                               weights=config['weights'],
                               n_classes=config['n_classes'],
                               flip_y=config['flip_y'],
                               class_sep = config['class_sep'],
                               random_state=42)
    # -- main processing --
    df = pd.DataFrame(X)
    df = set_columns(df,config)
    df = set_nulls(df, config)
    
    df[config['target_feature']] = y
    df["EVENT_LABEL"] = df[config['target_feature']].map({0:"legit", 1:"fraud"})
    df_stats = summary_stats(df)
    
    if config["predict_baseline"] == "True":
        m = model_rpt(df, config)
    # -- write dataset -- 
    df = df.drop([config['target_feature']], axis=1)
    df.to_csv(config['output_path'] + config['output_file'] +".csv",  index=False)
    #df.to_csv(config['output_path'] + config['output_file'] +".csv",  index=False, quoting=csv.QUOTE_NONNUMERIC)
    #print("--- writing to s3 ---")
    
    if config["s3_upload"] == "True":
        ret = upload_df_s3(df, config)
    
    
    return df

def upload_df_s3(df, config):
    #df.to_csv("s3://" + config['s3_bucket'] + "/" + config['s3_path'] + "/" + config['output_file'] +".csv",  index=False, quoting=csv.QUOTE_NONNUMERIC)
    df.to_csv("s3://" + config['s3_bucket'] + "/" + config['s3_path'] + "/" + config['output_file'] +".csv",  index=False)
    return 0

def upload_file_s3(txt, config):
    s3.Bucket(config["s3_bucket"]).upload_file(config['output_path'] + config['output_file']+'.txt', config['s3_path'] + "/" +config['output_file']+'.txt')
    return 0

def rename_columns(df):
    """
    Rename the columns of a dataframe to have X in front of them
    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    df = df.copy()
    df.columns = ['x' + str(i) for i in df.columns]
    return df

def set_category (df, key, val):
    df = df.copy()
    # use cut to convert that column to categorical
    df[key] = pd.cut(df[key], bins=len(val), labels=val)
    return df 

def set_numeric(df):
     # --- numeric features --- 
    for col in num_features:
        print("\t - " + col)
        df = df.rename(columns={'x' + str(i) : col})
        val = list(range(int(num_features[col][0]), int(num_features[col][1])))
        df = set_category(df,col,set(val))
        df[col] = df[col].astype('float64')
        i += 1

def get_fakeid():
    p1 = random.randint(10, 900)  
    p2 = random.randint(1000, 90000)
    return str(p1) + "-" + str(p2) 

def set_columns(df,config):
    print("building columns :")
    trg_feature  = config['target_feature']
    cat_features = config['categorical_features']
    num_features = config['numeric_features']
    id_features  = config['identity_features']
    i = 0
    df = rename_columns(df)
    
    # --- identity features ---
    for col in id_features:
        print("\t - " + col)
        df[col] = df.apply(lambda x: eval(id_features[col]), axis=1)

    # --- numeric features --- 
    for col in num_features:
        print("\t - " + col)
        df = df.rename(columns={'x' + str(i) : col})
        val = list(range(int(num_features[col][0]), int(num_features[col][1])))
        df = set_category(df,col,set(val))
        df[col] = df[col].astype('float64')
        i += 1
        
    # --- category features ---
    for col in cat_features:
        print("\t - " + col)
        df = df.rename(columns={'x' + str(i) : col})
        val = []
        for x in range(0,int(config['n_categories'])):
            val.append(eval(cat_features[col]))
        df = set_category(df,col,set(val))

        i += 1
 
    return df 

def summary_stats(df):
    """ Generate summary statsitics for a panda's data frame 
        
        Args:
            df (DataFrame): panda's dataframe to create summary statisitcs for.
    
        Returns:
            DataFrame of summary statistics 
    """
    df = df.copy()
    rowcnt = len(df)
    df_s1  = df.agg(['count', 'nunique']).transpose().reset_index().rename(columns={"index":"_column"})
    df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
    df_s1["not_null"] = rowcnt - df_s1["null"]
    df_s1["null_pct"] = df_s1["null"] / rowcnt
    df_s1["nunique_pct"] = df_s1['nunique']/ rowcnt
    dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"_column", 0:"_dtype"})
    df_stats = pd.merge(dt, df_s1, on='_column', how='inner').round(4)
    df_stats['nunique'] = df_stats['nunique'].astype('int64')
    df_stats['count'] = df_stats['count'].astype('int64')
    
    # -- null check 
    df_stats['null_check'] =  df_stats['null_pct'].apply(lambda x: 'Pass' if x <= 0.5 else '-- Fail --')
    # -- unique check 
    df_stats['nunique_check'] =  df_stats['nunique_pct'].apply(lambda x: 'Pass' if x <= 0.5 else '-- Fail --')
    # -- target check 
    

    print("--- summary stats ---")
    print(df_stats)
    print("\n")
    return df_stats

def set_nulls(df, config):
    # -1 because last col will always be y
    def _insert_random_null(x):
        x[random.randint(0, len(x) - 1)] = np.nan
        return x
    df = df.copy()
    pct_missing = config["pct_missing"]
    sample_index = df.sample(frac=pct_missing).index 
    df.loc[sample_index] = df.loc[sample_index].apply(_insert_random_null, axis=1)

    return df 

def safe_email():
    email = 'synth_' + fake.safe_email()
    return email

def safe_name():
    name = "synth_" + fake.first_name() + " " + fake.last_name()
    return name

def safe_phone():
    phone = "(555)" + str(random.randint(100,999)) + ' - ' + str(random.randint(1000,9999))
    return phone 

def get_null():
    return ""

def safe_address1():
    addr1 = str(fake.building_number()) + " " + fake.street_name() + " Fake St."
    return addr1

# --- model builder --- 

def prep_df(df, numeric_features, categorical_features):
    df = df.copy()
    df[numeric_features]=df[numeric_features].fillna(-1)
    df[categorical_features]=df[categorical_features].astype('str').fillna('<UNK>')
    
    return df 

def part_df(df,pct):
    df = df.copy()
    return train_test_split(df, test_size=pct)

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def model_rpt(df, config):
    print("training model...")
    numeric_features = list(config['numeric_features'].keys())
    categorical_features = list(config['categorical_features'].keys())
    target =  config['target_feature']

    #rpt = open('samplefile.txt', 'w')
    rpt = open(config['output_path'] + config['output_file'] +".txt", 'w')
    rpt.write("--- dataset summary --- \n")
    rpt.write("output_file          = " + config['output_file'] + ".csv\n")
    rpt.write("n_samples            = " + str(config['n_samples']) + "\n")
    rpt.write("n_features           = " + str(len(numeric_features)+ len(categorical_features)) + "\n")
    rpt.write("pct_missing          = " + "{:.2%}".format(config['pct_missing'])  + "\n")
    rpt.write("\n")
    rpt.write("--- model features --- \n")
    rpt.write("numeric_features     = " + str(numeric_features) +"\n")
    rpt.write("categorical_features = " + str(categorical_features) + "\n")
    rpt.write("\n")

    """ df_stats = summary_stats(df)
    rpt.write("--- data summary --- \n")
    rpt.write("Column \t Dtype \t Count\t N_unique\t N_null\t N_notnull\t PCT_null\t PCT_unique\t CHK_null\t CHK_unique\n")
    for index, row in fprtbl.iterrows():
        rpt.write( "%1.2f\t %1.2f\t %1.2f\n" % (row['threshold'],row['fpr'], row['tpr'] ))
    rpt.write("\n") """

    df  = prep_df(df , numeric_features, categorical_features)
    X_train, X_eval = part_df(df[numeric_features + categorical_features + [target]], 0.2)
    X_eval, X_test  = part_df(X_eval, 0.5)

    categorical_features_pos = column_index(X_train[numeric_features + categorical_features], categorical_features )

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(100)
    # Fit model
    model.fit(X_train[numeric_features + categorical_features], 
                X_train[target].values,
                plot=False,
                verbose=False,
                cat_features=categorical_features_pos ,
                eval_set=(X_eval[numeric_features + categorical_features], X_eval[target].values))
    # --- model predict  --- 
    prob_cat_train = model.predict_proba(X_train[numeric_features + categorical_features ])[:,1]

    prob_cat_eval = model.predict_proba(X_eval[numeric_features + categorical_features ])[:,1]

    prob_cat_test = model.predict_proba(X_test[numeric_features + categorical_features])[:,1]

    print ("(Train)")
    print ("AUC Score        : %f" % roc_auc_score( X_train[target].values, prob_cat_train))
    print ("\n")
    print ("(eval)")
    print ("AUC Score        : %f" % roc_auc_score( X_eval[target].values, prob_cat_eval))
    print ("\n")
    print ("(Test)")
    print ("AUC Score        : %f" % roc_auc_score(X_test[target], prob_cat_test))
    print ("\n")
    
    rpt.write ("--- dataset performance ---\n")
    rpt.write ("Train AUC Score        : %f" % roc_auc_score( X_train[target].values, prob_cat_train))
    rpt.write ("\n")
    rpt.write ("Eval  AUC Score        : %f" % roc_auc_score( X_eval[target].values, prob_cat_eval))
    rpt.write ("\n")
    rpt.write ("Test  AUC Score        : %f" % roc_auc_score(X_test[target], prob_cat_test))
    rpt.write ("\n\n")


    fpr, tpr, thr = roc_curve(X_test[target], prob_cat_test)
    model_stat = pd.concat([
        pd.DataFrame(fpr).rename(columns={0:'fpr'}),
        pd.DataFrame(tpr).rename(columns={0:'tpr'}),
        pd.DataFrame(thr).rename(columns={0:'threshold'})
        ],axis=1
        ).round(decimals=2)
   
    # m = model_stat.loc[model_stat['fpr'] <= 0.1] 
    m = model_stat.loc[model_stat.groupby(["fpr"])["threshold"].idxmax()]    
    
    # m1 = m.loc[model_stat['threshold'].idxmax()]
    print("--- score thresholds ---")
    print(m.loc[(m['fpr'] > 0.0 ) & (m['fpr'] <= 0.1)].reset_index(drop=True))
    print("\n")
    fprtbl = m.loc[(m['fpr'] > 0.0 ) & (m['fpr'] <= 0.1)].reset_index(drop=True)
    rpt.write ("--- score thresholds ---\n")
    rpt.write("THR \t FPR \t TPR\t \n")
    for index, row in fprtbl.iterrows():
        rpt.write( "%1.2f\t %1.2f\t %1.2f\n" % (row['threshold'],row['fpr'], row['tpr'] ))
    rpt.write("\n")
    shap_values = model.get_feature_importance(Pool(X_test[numeric_features + categorical_features], 
                                                label=X_test[target],
                                                cat_features=categorical_features_pos), type="ShapValues")
    
    shap_values = shap_values[:,:-1]

    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['feature_name','feature_importance'])
    feature_importance.sort_values(by=['feature_importance'],ascending=False,inplace=True)
    print("--- feature importance  ---")
    print(feature_importance.reset_index(drop=True))
    rpt.write ("--- feature importance  ---\n")
    for index, row in feature_importance.iterrows():
        rpt.write( "%-30s\t %1.4f\n" % (row['feature_name'],row['feature_importance'] ))
    
    rpt.write("\n")
    rpt.close() 
    if config["s3_upload"] == "True":
        ret = upload_file_s3('txt', config)
        
    
    return model 

    
if __name__ == "__main__":
    config = sys.argv[1:][0]
    print('processing config file: ' + config)
    df = get_dataset(config)
    print("\n")
  


    
