import pandas                  as     pd
import numpy                   as     np
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
import                                pickle
import                                warnings


from   sklearn.preprocessing   import StandardScaler, OneHotEncoder, PolynomialFeatures
from   sklearn.svm             import SVC
from   sklearn.impute          import SimpleImputer
from   sklearn.model_selection import train_test_split, GridSearchCV, cross_validate                         
from   sklearn.pipeline        import Pipeline
from   sklearn.metrics         import precision_score, recall_score, accuracy_score, f1_score, make_scorer
from   sklearn.metrics         import confusion_matrix, euclidean_distances
from   sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from   sklearn.compose         import ColumnTransformer
from   sklearn.tree            import DecisionTreeClassifier, plot_tree
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.linear_model    import LogisticRegression
from   sklearn.cluster         import DBSCAN
from   sklearn.decomposition   import PCA
from   sklearn.exceptions      import ConvergenceWarning
from   sklearn.cluster         import KMeans
from   sklearn.preprocessing   import RobustScaler, Normalizer

from   imblearn.over_sampling  import SMOTE, RandomOverSampler
from   imblearn.under_sampling import RandomUnderSampler
from   imblearn.pipeline       import Pipeline as imbPipeline   