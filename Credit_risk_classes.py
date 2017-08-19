#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.model_selection as split
import sklearn as sk
import sklearn.linear_model as lm





class Data_Splitter(object):
    def __init__(self,Data,Target,fraction_training,Group_variable,criteria_column=None):
            assert isinstance(Data, pd.core.frame.DataFrame) ,"data should be a dataframe "
            if Group_variable != None :
                Group_names=Data[Group_variable].unique()

                if criteria_column == None:
                    # distribute the different groups according to the fraction of training/testing

                    Training_set,Validation_set=sk.model_selection.train_test_split(Group_names,train_size=fraction_training)
                    self.Training_names = Training_set
                    self.Testing_names = Validation_set
                    Training_frame=Data[Data[Group_variable].isin(Training_set)]
                    Validation_frame=Data[Data[Group_variable].isin(Validation_set)]
                    self.Y_Training = Training_frame[Target]
                    self.X_Training = Training_frame.drop([Target],axis=1)

                    self.Y_Testing = Validation_frame[Target]
                    self.X_Testing = Validation_frame.drop([Target], axis=1)

                else:
                    # for example within proportion count criteria column should be equivalent to fraction upon which is split
                    raise NotImplementedError




def rank_nan(data):
    assert isinstance(data, pd.core.frame.DataFrame) ,"data should be a dataframe "
    return data.apply(lambda x: x.isnull().sum()/len(data)).sort_values(ascending=False)

#%%
def impute_mean(data,target_column,Group_variable=None):

    if  Group_variable !=None :
        Target=data.groupby(Group_variable)[[target_column]]
        Imputation=Target.apply(lambda x:x.fillna(x.mean()))

        if Imputation.isnull().any().any():
            "this means we have a loan with only NAN"
            Median=Imputation.median()
            Imputation = Imputation.fillna(Median).reset_index(drop="True")
        return data.assign(**{target_column: Imputation})


    else:
        Target=data[[target_column]]
        mean=Target.mean()
        Imputation=Target.fillna(mean)
        return data.assign(**{target_column: Imputation})

def impute_median(data,target_column,Group_variable=None):

    if  Group_variable != None:

        Target=data.groupby(Group_variable)[[target_column]]
        Imputation=Target.apply(lambda x:x.fillna(x.median()))

        if Imputation.isnull().any().any():
            "this means we have a loan with only NAN"
            Median    = Imputation.median()
            Imputation= Imputation.fillna(Median).reset_index(drop="True")
        return data.assign(**{target_column: Imputation})

    else:
        Target=data[[target_column]]
        mean=Target.median()
        Imputation=Target.fillna(median)
        return data.assign(**{target_column:Imputation})

class Imputer(object):

    def __init__(self,data,missing_values="NaN"):
        self.missing_values = "NaN"
        self.Frame = data



    def Transform(self, X, y=None):
        raise NotImplementedError
    def fit_Transform(self,X,y=None):
        raise NotImplementedError

    def Grouper_target(self,target_column,Group_variable):
        #assert isinstance(Data,pd.core.frame.DataFrame) ,"data should be a dataframe "
        # Imputation per Group takes more time
        # set Group to False if you want to impute mean while ignoring groups
        # If one wants to set a custom group column one needs to

        Grouped=self.Frame.groupby(Group_variable)[[target_column]]
        return Grouped






    def impute_linear_model(self,target_column,training_frame,independent_variable,intercept=False,drop_outliers=False):
            self.training_frame=training_frame
            self.independent_variable=independent_variable

            # FRAME ON WHICH WE TRAIN THE DATA  ---> Only training frame can be used
            Model_frame=(training_frame
                        [[target_column,independent_variable]]
                        .dropna()
                        .drop_duplicates())

            Model= LinearRegressionModel().fit(training_X=Model_frame[[independent_variable]],
                                               training_Y=Model_frame[[target_column]],
                                               intercept=intercept)

            if drop_outliers==False:
                pass
            #1 Both available --> Do Nothing
            subset_both_available=self.Frame[~(self.Frame[target_column].isnull())& ~(self.Frame[self.independent_variable].isnull())]

            #2. ################## Both  NAN --> use median, CALCULATED FROM training_frame ########################
            #Median from training frame
            Median=self.training_frame[target_column].median()
            #Subset we need to full in full frame
            subset_both_nan=self.Frame[self.Frame[target_column].isnull() & self.Frame[self.independent_variable].isnull()]
            subset_both_nan=subset_both_nan.assign(**{target_column: Median})

            #3 ###################### Target available but independent_variable not available--> Do nothing ################
            subset_indepvar_nan=self.Frame[~(self.Frame[target_column].isnull())& (self.Frame[self.independent_variable].isnull())]

            #4 ##################Target nan ,independent variable is available-> use regression with Beta's from training_frame! ####
            subset_Target_nan=self.Frame[(self.Frame[target_column].isnull())& ~(self.Frame[self.independent_variable].isnull())]
            if subset_Target_nan[target_column].any() != True:
                pass
            else:
                # Predict imputation on missing data
                Imputation = Model.predict(subset_Target_nan[[independent_variable]])

                # Fill in the missing data
                subset_Target_nan=subset_Target_nan.assign(**{target_column: Imputation})




            # PUTTING ALL THE SUBFRAMES TOGETHER
            subframes=[subset_both_available,subset_both_nan,subset_indepvar_nan,subset_Target_nan]
            imputed_frame=pd.concat(subframes).sort_index()
            return imputed_frame




class Model(object):
    def __init__(self):
        self.models={}
    def drop_outliers(self):
        raise NotImplemented


    def fit(X, y):
        raise NotImplemented("Need to subclass Model")

    def predict(X):
        raise NotImplemented("Need to subclass Model")


class RandomModel(Model, Imputer):
    '''
    Very Stupid Model, randomly predicts 0 or 1
    '''

    def __init__(self):
        self.labels = None

    def impute(X):
        '''
        No need to impute, we hardly care about inputdata anyway
        '''
        return X

    def fit(X, y):
        self.labels = y.copy()

    def predict(X):
        if self.labels is None:
            raise Exception('Need to call fit() first you dumbo')
        else:
            return np.random.choice(self.labels)

class LinearRegressionModel(Model):
    def __init__(self,Groups=False):
        self.Groups=Groups
        self.linear_models={}

    def fit(self,training_X,training_Y,intercept=False):
        if self.Groups==False:
            return lm.LinearRegression(fit_intercept=intercept).fit(X=training_X,y=training_Y)

    def predict(self,X):
        return self.model.predict(x)
#%%


class LogisticRegressionModel(Model):
    def __init__(self,Response):
        self.logistic_models = {}


    def Sigmoid(Z):
        return 1/(1+np.e**(-Z))


    def fit(self,X, y,Group=None):
        if Group==None:
            return lm.LogisticRegression(penalty='l2').fit(X,y)
    def predict(self,X,probability=False):
        # Probability returns a number between 0 and 1
        # if probability is false will return 0 or 1
        if probability == False:
            return self.model.predict(X)

        else:
            return self.model.predict_proba(X)
