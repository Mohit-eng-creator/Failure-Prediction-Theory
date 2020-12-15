import flask
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import pickle
import csv
import json 
from flask import send_file
from sklearn.metrics import fbeta_score

app = Flask(__name__,template_folder='templates')
app.debug=True


@app.route('/',methods=['GET','POST'])
def first_page():
    return flask.render_template('first_page.html')

    
@app.route('/final_function',methods=['GET','POST'])
def final(test_original=None,y_original=None):

    """
    This function takes in either test data or both the test data and it's original y values.
    If only test data is given, then it predicts the y values.
    If both the test data and y_values are given then it returns the predicted values and the 
    metric by which we measure the model. In this case the metric is F2 Score.
    """


    file = request.files['test_original']
    
    if not file:
        return "Enter the test data. It is a must. Go Back."
        
    print("type(file) : ",type(file))
                
    test_original = pd.read_csv(file)
    print("test_original : ", test_original)


    file = request.files['y_original']
    
    if file:
        y_original = pd.read_csv(file)
    else :
        y_original=None
        
    print("y_original : ", y_original)

    #Remove the id if present
    try:
        test_original.drop("id",axis=1,inplace=True)
    except KeyError:
        pass


    # ignoring SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    #Replace the string that is "na" in the features by np.NaN 
    for column in list(test_original.columns.values):
        test_original[column]=test_original[column].replace('na', np.NaN)
    
    #sort index
    test_original.sort_index(inplace=True)

    #Convert the rest of the non numeric elements in the format of a string to a numerical feature.
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
    test_original=test_original.apply(pd.to_numeric,downcast='float')


    #If y is given and is not None
    if y_original is not None:
        
        #Remove the id if present
        try:
            y_original.drop("id",axis=1,inplace=True)
        except KeyError:
            pass

        y_original.sort_index(inplace=True)
        
        #As input is being taken from a csv , the each value has become numpy array instead of numeric values
        #So we convert it into numpy array and then to pandas Series
        y_original=pd.Series(np.array(y_original).flatten())

        #impute the datapoints with their class values in the train data only. 
        test_original,y_original=impute_median_given_y(test_original,y_original)
        
        
        #Standardize the data and only include those columns that were selected during smotetomek.
        y_predicted=standardize_featurize_n_predict(test_original)


        #Calculate the F2_score
        f2_score=fbeta_score(y_original, y_predicted, pos_label=1.0, beta=2)
        
        index=test_original.reset_index()["index"]

        
        #JSON stands for JavaScript Object Notation. 
        #JSON is a lightweight format for storing and transporting data. 
        #JSON is often used when data is sent from a server to a web page. 
        #JSON is "self-describing" and easy to understand.
        #index=json.dumps(index.tolist())
        
        index=index.tolist()
        y_original = y_original.tolist()
        y_predicted = y_predicted.tolist()
        f2_score = f2_score
        
	#To display on the ipython notebook
        #return y_predicted,f2_score
        
        
	#return the prediction and the F2_Score
        return render_template('output.html',f2=f2_score,index_y_original_y_predicted=zip(index,y_original,y_predicted))
        


    #else
    else :

        #Load what median values to impute in this data if np.NaNs are there.
        #These values are taken previously from the train data.
        train_dataframe_median_values_smotetomek_data=\
        pickle.load(open("train_dataframe_median_values_smotetomek_data.pickle","rb"))

        #Fill the np.NaN in a particular feature with the median values for that particular feature.
        test_original.fillna(train_dataframe_median_values_smotetomek_data,inplace=True)

        #Standardize the data and only include those columns that were selected during smotetomek.
        y_predicted=standardize_featurize_n_predict(test_original)


        index=test_original.reset_index()["index"]
        
        
        #JSON stands for JavaScript Object Notation. 
        #JSON is a lightweight format for storing and transporting data. 
        #JSON is often used when data is sent from a server to a web page. 
        #JSON is "self-describing" and easy to understand.
        index=index.tolist()
        y_predicted = y_predicted.tolist()
        
        #return the prediction and the F2_Score
        #To display on the ipython notebook
        #return y_predicted
        
        #return the prediction and the F2_Score
        return render_template('output.html',index_y_predicted=zip(index,y_predicted))

def standardize_featurize_n_predict(dataframe):


    """
    This function to be used to standardize the particular columns of the test data as per the
    Scalar model that was fitted on the train data columns.There is a list of fitted scalar models.
    One for each column. Followed by the selection of only those features that we found after
    svd/rfe and spearman corelation coefficient.Then we load the best performing model make the 
    model predict the y values.
    """
    

    #Load the StandardScalar dictionary that contains the trained StandardScalar model for each feature to use
    #on that particular feature.
    scalar_dict=pickle.load(open("scalar_dict_smotetomek_data.pkl","rb"))

    #create an empty dataframe
    test_dataframe=pd.DataFrame()

    #Feature wise load the StandardScalar model for that particular feature and standardize the values for that
    #particular feature with the StandardScalar model for that particular feature.
    for column in list(scalar_dict.keys()):
        #Convert into column vector that is many rows but only one column.
        #After standardizing , put the feature into another dataframe.
        test_dataframe[column] = scalar_dict[column].transform(dataframe[column].values.reshape(-1,1)).flatten()

    #After all the process of smotetomek/SVD/RFE/Spearman correlation coefficient,
    #load the name of the reduced features that remain.
    new_x_smotetomek_df_columns=np.load('new_x_smotetomek_df_columns.npy',allow_pickle=True)

    #Take only those features in your test dataset.
    test_dataframe=test_dataframe[new_x_smotetomek_df_columns]

    #load the machine learning XGB model that performed the best, with all it's parameters.
    Model= pickle.load(open("Model.pkl",'rb'))

    #Make the model predict the y_values for our test dataset.
    y_predicted=Model.predict(test_dataframe)

    return y_predicted
    
def impute_median_given_y(dataframe,target):


    """
    In this function I impute the median values of the 0 class as well as the 1 class 
    to those data points whose class I know to be 0 or 1 . I will impute the values accordingly from the train data.
    """

    train_1_dataframe_median_values_smotetomek_data=\
    pickle.load(open("train_1_dataframe_median_values_smotetomek_data.pickle","rb"))
    train_0_dataframe_median_values_smotetomek_data=\
    pickle.load(open("train_0_dataframe_median_values_smotetomek_data.pickle","rb"))


    dataframe_1=dataframe.loc[target[target==1].reset_index()["index"]]
    dataframe_0=dataframe.loc[target[target==0].reset_index()["index"]]


    y_1=target[target==1]
    y_0=target[target==0]


    dataframe_1.fillna(train_1_dataframe_median_values_smotetomek_data,inplace=True)
    dataframe_0.fillna(train_0_dataframe_median_values_smotetomek_data,inplace=True)

    dataframe_new = pd.concat([dataframe_0,dataframe_1])
    dataframe_new.sort_index(inplace=True)

    y_new=pd.concat([y_0,y_1])
    y_new.sort_index(inplace=True)
    
    return dataframe_new,y_new
    
    
@app.route('/download_x_test')
def download_x ():
    path = "x_test_sample.csv"
    return send_file(path, as_attachment=True)
    
    
@app.route('/download_y_test')
def download_y ():
    path = "y_test_sample.csv"
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run()
