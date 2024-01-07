import pandas                  as     pd
import numpy                   as     np
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
import                                pickle
import                                warnings


from   sklearn.preprocessing   import StandardScaler, OneHotEncoder, PolynomialFeatures
from   sklearn.impute          import SimpleImputer
from   sklearn.model_selection import train_test_split, GridSearchCV, cross_validate                         
from   sklearn.pipeline        import Pipeline
from   sklearn.metrics         import precision_score, recall_score, accuracy_score, f1_score, make_scorer
from   sklearn.metrics         import confusion_matrix, euclidean_distances
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.compose         import ColumnTransformer
from   sklearn.tree            import DecisionTreeClassifier, plot_tree
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.linear_model    import LogisticRegression
from   sklearn.cluster         import DBSCAN
from   sklearn.decomposition   import PCA
from   sklearn.exceptions      import ConvergenceWarning
from   sklearn.cluster         import KMeans
from   sklearn.preprocessing   import RobustScaler, Normalizer

cat_cols = ['Auction', 'VehicleAge', 'Make'    , 'WheelType', 'VehYear',
            'Model'  , 'Trim'      , 'SubModel', 'Color'    , 'Transmission',  
            'Nationality', 'Size'  , 'TopThreeAmericanName' , 'IsOnlineSale',
            'PRIMEUNIT'  , 'AUCGUART', 'BYRNO' , 'VNZIP1'   , 'VNST', 
            'PurchDate_year','PurchDate_month' , 'PurchDate_dayofweek', 
           ]     
#       ------------------------------------------------------------------------
num_cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice' , 'MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice'    , 'MMRCurrentAuctionCleanPrice',
            'MMRCurrentRetailAveragePrice'     , 'MMRCurrentRetailCleanPrice',
            'VehBCost', 'VehOdo', 'WarrantyCost']

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ====== this function decomposes time columns in df to its Year, month, day of week components 

def decompose_time(df):
    """  """
    df_copy = df.copy()                                                 # Create a copy of the DataFrame
    for col in df_copy.columns:
        if df_copy[col].dtype          == 'datetime64[ns]':
            df_copy[f'{col}_year']      = df_copy[col].dt.year
            df_copy[f'{col}_month']     = df_copy[col].dt.month
            df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
            df_copy.drop(col, axis=1, inplace=True)                      # Drop the original datetime column
    return df_copy   

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ======= This functions converts Object and Numerical Nomical features to category, it does not need a return =======

def convert_to_category(df, columns):
    """   """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"Column '{col}' not found in DataFrame.")
            
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ====== this function decomposes time columns in df to its Year, month, day of week components 

def impute_car_missing_data(df):
    """
    Fills NaN values and makes updates for specific conditions.
    
    """
    df['PRIMEUNIT'] = df['PRIMEUNIT'].fillna('NO')
    df['AUCGUART']  = df['AUCGUART'].fillna('YELLOW')
    # -----------------------------------------------
    df.loc[(df['WheelTypeID'] == 0.0) & (df['WheelType'].isna()), 'WheelType'] = 'Type_0'
    # -----------------------------------------------------------------------------------
    df.loc[df['Transmission'] == 'Manual', 'Transmission'] = 'MANUAL'
    # ------------------------------------------------------------------------------------
    # Define masks
    mask_jeep    = (df['Make']  == 'JEEP')        & (df['Model'] == 'PATRIOT 2WD 4C')
    mask_hyundai = (df['Make']  == 'HYUNDAI')     & (df['Model'] == 'ELANTRA 2.0L I4 MPI')
    mask_Sierra  = (df['Model'] == 'SIERRA 1500') & (df['Trim']  == 'SL')

    # Operations for mask_jeep
    df.loc[mask_jeep, ['Nationality', 'Size', 'TopThreeAmericanName']] = \
    df.loc[mask_jeep, ['Nationality', 'Size', 'TopThreeAmericanName']] \
    .fillna({'Nationality'         : 'AMERICAN', 
             'Size'                : 'SMALL SUV', 
             'TopThreeAmericanName': 'CHRYSLER'})

    # Operations for mask_hyundai
    df.loc[mask_hyundai, ['Nationality', 'Size', 'TopThreeAmericanName']] = \
    df.loc[mask_hyundai, ['Nationality', 'Size', 'TopThreeAmericanName']] \
    .fillna({'Nationality'         : 'OTHER ASIAN', 
             'Size'                : 'MEDIUM', 
             'TopThreeAmericanName': 'OTHER'})

    # Operations for mask_Sierra
    df.loc[mask_Sierra, 'Model'] = '1500 SIERRA PICKUP 2'
    df.loc[mask_Sierra, ['Nationality', 'Size', 'TopThreeAmericanName']] = \
    df.loc[mask_Sierra, ['Nationality', 'Size', 'TopThreeAmericanName']] \
    .fillna({'Nationality'         : 'AMERICAN', 
             'Size'                : 'LARGE TRUCK', 
             'TopThreeAmericanName': 'GM'})
    
    return df
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def CLEAN_All_DATA(df, X_aim):
    """
    Returns cleaned DataFrame.
    
    Transform datatypes:
        - transform Date and features to correct format
        - Imputes Numerical and categorical columns
       
    Returns:
        df  - Clean 
        
    """
    cat_cols = ['Auction', 'VehicleAge', 'Make'    , 'WheelType', 'VehYear',
            'Model'  , 'Trim'      , 'SubModel', 'Color'    , 'Transmission',  
            'Nationality', 'Size'  , 'TopThreeAmericanName' , 'IsOnlineSale',
            'PRIMEUNIT'  , 'AUCGUART', 'BYRNO' , 'VNZIP1'   , 'VNST', 
            'PurchDate_year','PurchDate_month' , 'PurchDate_dayofweek', 
           ]     
#       ------------------------------------------------------------------------
    num_cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice' , 'MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice'    , 'MMRCurrentAuctionCleanPrice',
            'MMRCurrentRetailAveragePrice'     , 'MMRCurrentRetailCleanPrice',
            'VehBCost', 'VehOdo', 'WarrantyCost']
    

    # -----------------------------------------
    df          = impute_car_missing_data(df)                   # fill in specific columns NaN data
    X_aim       = impute_car_missing_data(X_aim) 
    # ------------------------------------------
   
    df.drop('WheelTypeID', axis=1, inplace=True)                # Dropping redundant WheelTypeID
    X_aim.drop('WheelTypeID', axis=1, inplace=True)             # Dropping redundant WheelTypeID
    # -------------------------------------------
    
    df_features = df.drop('IsBadBuy', axis = 1)                 # define features 
    df_target   = df['IsBadBuy']                                # define target
    # -----------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(df_features,                     # train_test_split    
                                                        df_target,
                                                        test_size    = 0.1,
                                                        shuffle      = True,
                                                        random_state = 42)
    
    #to datetime
    X_train.loc[:, 'PurchDate'] = pd.to_datetime(X_train.loc[:, 'PurchDate'], unit='s')
    X_test.loc[: , 'PurchDate'] = pd.to_datetime(X_test.loc[: , 'PurchDate'], unit='s')
    X_aim.loc[:  , 'PurchDate'] = pd.to_datetime(X_aim.loc[:  , 'PurchDate'], unit='s')
    #-----------------------------------------------------------------------------------
    X_train = decompose_time(X_train)                             # decomposing time to year, month, day of the week             # converting column types
    X_test  = decompose_time(X_test)
    X_aim   = decompose_time(X_aim)
    convert_to_category(X_train, cat_cols)
    convert_to_category(X_test, cat_cols)
    convert_to_category(X_aim, cat_cols)
    #-------------------------------------------

    category_imputer         = SimpleImputer(strategy   = 'constant', 
                                             fill_value = 'UNKNOWN')                         # categorical imputer
    
    category_imputer.fit(X_train[cat_cols])                                                  # fit to X_train    
    X_train.loc[:, cat_cols] = category_imputer.transform(X_train.loc[:, cat_cols])          # Transform X_train
    X_test.loc[: , cat_cols] = category_imputer.transform(X_test.loc[: , cat_cols])          # Transform X_test
    X_aim.loc[:  , cat_cols] = category_imputer.transform(X_aim.loc[:  , cat_cols])          # Transform X_aim
    #----------------------------------------------------------------------------------

    # Handle missing values in numerical columns
    median_imputer           = SimpleImputer(strategy = 'median')                             # Numerical imputer
    median_imputer.fit(X_train[num_cols])
    X_train.loc[:, num_cols] = median_imputer.transform(X_train.loc[:, num_cols])
    X_test.loc[: , num_cols] = median_imputer.transform(X_test.loc[: , num_cols])
    X_aim.loc[:  , num_cols] = median_imputer.transform(X_aim.loc[:  , num_cols])
    #-----------------------------------------------
    
    # The data type of a DataFrame column can be lost after using a SimpleImputer
    # SimpleImputer returns a NumPy array, and the data type of the array elements 
    # may not always match the original data type

    X_train = decompose_time(X_train)                             # decomposing time to year, month, day of the week             # converting column types
    X_test  = decompose_time(X_test)
    X_aim   = decompose_time(X_aim)
    convert_to_category(X_train, cat_cols)
    convert_to_category(X_test, cat_cols)
    convert_to_category(X_aim, cat_cols)
    #------------------------------------------------
   
    return X_train, X_test, y_train, y_test, X_aim

# ----------------------------------------------------------------------------------------------------------------------------------

def Confusion_Matrix_Func(y_test, y_pred, model_name):
    """   """
    fig, ax = plt.subplots(figsize=[10, 6])

# -------------------------------------------------------------------------------------------------
    cf_matrix     = confusion_matrix(y_test, y_pred)
    group_names   = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts  = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percent = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percent)]
    labels = np.asarray(labels).reshape(2, 2)

    # ----------------------------------------------------------------------------------------------
    sns.heatmap(cf_matrix,
                annot     = labels,
                ax        = ax,
                annot_kws = {'size': 13},
                cmap      = 'Blues',
                fmt       = ''
                )

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Truth')
    ax.xaxis.set_ticklabels(['G-Buy', 'Lemon'])
    ax.yaxis.set_ticklabels(['G-Buy', 'Lemon'])
    ax.set_title(f'Confusion Matrix of {model_name}\n')
    
    cf_matrix, fig

