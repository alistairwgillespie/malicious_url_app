# Imports
import pandas as pd
import tldextract
from datetime import datetime
from socket import gethostbyname, gaierror
import whois
import warnings
import re
import joblib
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder


def creation_date(domain_name):
    """
    Gets creation date of domain
    """
    
    # Get creation date of Domain 
    domain_name = whois.whois(domain_name).creation_date
    
    # Handling exceptions
    if type(domain_name) is list:
        return domain_name[0]
    elif str(domain_name).find('Aug'):
        domain_name = "1996-07-01 00:00:01"
        return domain_name
    elif domain_name == np.nan:
        currentDT = datetime.now()
        domain_name = currentDT.strftime("%Y-%m-%d %H:%M:%S")
        return domain_name
    else:
        return domain_name


def countSpecial(string):
    """
    Counts number of special characters in a string
    """
    new = re.sub('[\w]+' ,'', x)
    return len(new)


def entropy(string):
    """
    Calculates the Shannon entropy of a string
    """

    # Get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]

    # Calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])

    return entropy


def host_ip(domain):
    """
    Gets Host IP of Domain
    """

    # Get HOST IP     
    try:
        host = gethostbyname(domain)
        return host
    except gaierror:
        return 'missing'


def get_domain_parts(df, feature_col):
    """
    Extract domain components
    """
    
    # Extract domain
    df['domain'] = df[feature_col].apply(lambda x: tldextract.extract(x).domain)
    
    # Extract suffix
    df['suffix'] = df[feature_col].apply(lambda x: tldextract.extract(x).suffix)
    
    # Extract suffix
    df['domain_name'] = df[feature_col].apply(lambda x: tldextract.extract(x).registered_domain)
    
    return df


def get_host_ip(df, domain_col):
    """
    Gets host IP address associated with domain
    """
    
    # Extract Host IP 
    df['host_ip'] = df[domain_col].apply(lambda x: host_ip(x))
    

    return df


def get_prefix(df, host_col):
    """
    Gets first octet of IP
    """
    
    # Extract prefix, first octet
    df['prefix'] = df[host_col].str.extract('(\d+)\.').astype(int, errors='ignore').astype(str)
    df['prefix'] = df['prefix'].fillna('missing')
    df['prefix'] = df['prefix'].replace('nan', 'missing')
    
    return df


def get_creation_date(df, feature_col):
    """
    Gets creation date of domain
    """
    
    df['domain_creation'] = df[feature_col].apply(lambda x: creation_date(x))

    return df


def get_domain_age(df, creation_col):
    """
    Calculates the age of the domain in days
    """
      
    # Cast domain age columnt to datetime
    df[creation_col] = pd.to_datetime(df[creation_col], format='%Y-%m-%d %H:%M:%S')
    
    # Calculate the age of the domain
    df['domain_age'] = pd.datetime.today().date() - df[creation_col].dt.date
    
    # Cast age to an integer
    df['domain_age'] = df['domain_age'].astype(str).str.extract("(\d+)").astype(float)

    return df


def get_domain_entropy(df, feature_col):
    """
    Calculates entropy of a feature for a given data set
    """
    
    # Calculate entropy 
    df['entropy'] = df[feature_col].apply(lambda x: entropy(str(x)))

    return df


def get_number_suffix(df, feature_col):
    """
    Calculates number of suffix in the URL
    """
    
    # Calculates number of suffix in the URL
    df['number_suffix'] = df[feature_col].str.count('\.')
    
    return df


def get_number_digits(df, feature_col):
    """
    Calculates number of numerical characters in a string
    """
    
    # Calculates number of digits
    df['number_digits'] = df[feature_col].str.count('[0-9]')
    
    return df


def get_percent_digits(df):
    """
    Calculates percentage of string is a digit
    """
    
    # Calculate percentage
    df['digits_percentage'] = (df['number_digits']/df['string_length'])*100
    
    return df


def get_string_length(df, feature_col):
    """
    Calculates length of string
    """

    # Calculates length of string
    df['string_length'] = df[feature_col].str.len()
    
    return df


def get_specials(df, feature_col):
    """
    Calculates number of special characters in string
    """
    
    # Count of special characters
    df['specials'] = df[feature_col].apply(lambda x: countSpecial(str(x)))
   
    return df


def get_iana_designations(df, iana, prefix_col):
    """
    Merges data sets on the prefix i.e. first octect of the IPv4 address
    """

    # Enrich sample with IPv4 Registry data
    df = df.merge(iana, on=prefix_col, how='left')

    # Clean prefix and drop unneeded columns
    df['prefix'] = df['prefix'].astype(str)
    df['designation'] = df['designation'].fillna('missing')
    df.rename(columns={ 'status [1]': 'status'}, inplace=True)
    df.drop(['note'], axis=1, inplace=True)

    return df


def feature_extraction_train(data, iana_data):
    """
    Pipeline utility for extracting all candidate functions
    """
    
    # Assumes string or dataframe input
    
    # Handle string or dataframe input
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data=[data], columns=['url']) # Create dataframe from string  
    else:
        df = data
    
    # Conduct extraction
    print('  * Loading features')
    df = get_domain_entropy(df, 'domain_name') # Extract domain entropy
    df[df['domain_creation'] == str]
    df = get_domain_age(df, 'domain_creation') # Extract domain age
    df = get_number_suffix(df, 'domain_name') # Extract number of suffix
    df = get_number_digits(df, 'domain_name') # Extract number of digits
    df = get_string_length(df, 'domain_name') # Extract string length
    df = get_percent_digits(df) # Extract percentage digits
    df = get_specials(df, 'domain_name') # Extract number of specials
    df = get_iana_designations(df, iana_data,'prefix') # Extract designation
    
    print('  * Number of features extracted: ' + str(len(df.columns.tolist())))
    
    return df


def feature_extraction_prod(data, iana_data):
    """
    Pipeline utility for extracting all candidate functions
    """
    
    # Assumes string or dataframe input
    
    # Handle string or dataframe input
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data=[data], columns=['url']) # Create dataframe from string  
    else:
        df = data
    
    # Conduct extraction
    print('  * Loading features')
    df = get_domain_parts(df, 'url') # Extract domain parts
    df = get_creation_date(df, 'domain_name') # Extract domain creation date
    df = get_domain_age(df, 'domain_creation') # Extract domain age
    df = get_domain_entropy(df, 'domain_name') # Extract domain entropy
    df = get_number_suffix(df, 'domain_name') # Extract number of suffix
    df = get_number_digits(df, 'domain_name') # Extract number of digits
    df = get_string_length(df, 'domain_name') # Extract string length
    df = get_percent_digits(df) # Extract percentage digits
    df = get_specials(df, 'domain_name') # Extract number of specials
    df = get_host_ip(df, 'domain_name') # Extract host IP
    df = get_prefix(df, 'host_ip') # Extract prefix
    df = get_iana_designations(df, iana_data,'prefix') # Extract designation
    
    print('  * Number of features extracted: ' + str(len(df.columns.tolist())))
    
    return df


def feature_engineering_prod(df): 
    """
    Conduct encoding, normalisation and standardisation of features
    """
    
    # Setup Label Encoders
    suffix_le = LabelEncoder()
    designation_le = LabelEncoder()
    prefix_le = LabelEncoder()
    
    # Load encodings
    suffix_le.classes_ = np.load('suffix.npy', allow_pickle=True)
    designation_le.classes_ = np.load('designation.npy', allow_pickle=True)
    prefix_le.classes_ = np.load('prefix.npy', allow_pickle=True)
    
    # Get integer mappings
    integer_mapping = {l: i for i, l in enumerate(prefix_le.classes_)}
    
    # If prefix exists already, then transform
    if df['prefix'].isin(integer_mapping).any():
        # If value is in mapping list, transform
        prefix_le.transform(df.prefix)
        print('  * Value successfully transformed.') 
    
    # Else value is not in mapping list, add to mapping list and transform
    else:
        prefix_le.classes_ = np.append(prefix_le.classes_, df['prefix'].iloc[0:1])
        prefix_le.transform(df.prefix)
        print('  * New unseen value added to label encoder.')
    
    # Transform categorical variables
    df['suffix'] = suffix_le.transform(df.suffix)
    df['designation'] = designation_le.transform(df.designation)
    
    # Load MinMax models and transform features
    entropy_mms = joblib.load('entropy_model.pkl')
    domain_age_mms = joblib.load('domain_age_model.pkl')
    df['entropy'] = entropy_mms.transform(df[['entropy']])
    df['domain_age'] = domain_age_mms.transform(df[['domain_age']])
    
    # Load Max Abs models and transform features
    dnd_mas =joblib.load('dnd_model.pkl')
    dnl_mas = joblib.load('dnl_model.pkl')
    da_mas = joblib.load('da_model.pkl')
    s_mas = joblib.load('s_model.pkl')
    ns_mas = joblib.load('ns_model.pkl')
    df['number_digits'] = dnd_mas.transform(df[['number_digits']])
    df['string_length'] = dnl_mas.transform(df[['string_length']])
    df['digits_percentage'] = da_mas.transform(df[['digits_percentage']])
    df['specials'] = s_mas.transform(df[['specials']])
    df['number_suffix'] = ns_mas.transform(df[['number_suffix']])
    
    return df


def predict_maliciousness(url, features):
    """
    Predict maliciousness of URL
    """
    
    # Ingest IANA dataset
    iana = pd.read_csv("https://www.iana.org/assignments/ipv4-address-space/ipv4-address-space.csv", sep=",")
    iana.columns = iana.columns.str.strip().str.lower()
    iana.rename(columns={'Prefix': 'prefix'}, inplace=True)

    # Clean up prefix since it uses old/BSD formatting
    iana['prefix']= iana['prefix'].apply(lambda x: re.sub('^(00|0)','',x))
    iana['prefix'] = iana['prefix'].apply(lambda x: re.sub('/8$','',x))
    iana['prefix'] = pd.to_numeric(iana['prefix'], downcast='float',errors='ignore').astype(str)

    # Load model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = joblib.load('malicious_url_model.pkl')
    
    # Extract features
    url_features = feature_extraction_prod(url, iana)
    
    # Engineer features
    url_features = feature_engineering_prod(url_features)
    
    # Produce features and score
    score = str(model.predict_proba(url_features[features]).tolist()[0][1])
    return [url_features, score]
