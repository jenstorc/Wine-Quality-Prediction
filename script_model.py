# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
import json 
import os
from sys import exit
import joblib # save model

# import libraries
from sklearn.preprocessing import StandardScaler  # importing module for feature scaling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from __init__ import Wine, WineFull
#from api/endpoints/__init__ import Wine, WineFull
from pydantic import BaseModel
class Wine(BaseModel):
    fixed_acidity:  float
    volatile_acidity:  float
    citric_acid:  float
    residual_sugar:  float
    chlorides:  float
    free_sulfur_dioxide:  float
    total_sulfur_dioxide:  float
    density:  float
    pH:  float
    sulphates:  float
    alcohol:  float

class WineFull(Wine):
    quality: float
    id: int 
# GENERAL FUNCTION
def check_file(path_name : str):
    """ Arrêt du programme si le nom du fichier donné en paramètre n'existe pas ou n'est pas lisible

    Args:
        path_name (str): nom (dont chemin) du fichier qu'on souhaite lire

    Raises:
        Exception: Le fichier n'existe pas
        Exception: Le fichier n'est pas lisible
    """
    try :
        if not(os.path.isfile(path_name)):
            raise Exception("Erreur : Le fichier \"{}\" n'existe pas.".format(path_name))
        
        if not(os.access(path_name, os.R_OK)):
            raise Exception("Erreur : Le fichier \"{}\" n'est pas accessible (problème de droit de lecture).".format(path_name))

    except Exception as e:
        print(e)
        exit()

def open_json_file(path_json : str) : # TODO : trouver comment renvoyer tuple 
    """ Ouvre le fichier json donné en paramètre et retourne une liste des variables explicatives et la variable cible

    Args:
        path_json (str): nom (et chemin) du fichier json qu'on souhaite lire

    Raises:
        Exception: Si la clé "input" n'est pas dans le dictionnaire json
        Exception: Si la clé "output" n'est pas dans le dictionnaire json
        Exception: Si la valeur de la clé input n'est pas une liste de chaînes de caractères
        Exception: Si la valeur de la clé output n'est pas une chaîne de caractères

    Returns:
        lst_input_variables (list) : liste des variables explicatives
        output_variable (str) : variable cible
    """
    try:
        # Si le fichier json n'existe pas ou n'est pas lisible
        check_file(path_json)

        # Ouverture du fichier json qui indique quelles sont les variables explicatives et la variable cible
        input_file = open(path_json) # Ouverture du fichier json      
        dict_input_output_variable = json.load(input_file) # Stocker l'objet json dans un dictionnaire
        input_file.close # Fermer le fichier

        # Si la clé input est bien dans le dictionnaire, on ajoute l'objet associé à la clé dans la liste des variables explicatives
        if 'input' in dict_input_output_variable.keys(): 
            lst_input_variables = dict_input_output_variable['input'] # Variables explicatives
        else : 
            raise Exception("Erreur : Il manque la clé \"input\" dans le dictionnaire.")

        # Si la clé output est bien dans le dictionnaire, on ajoute l'objet associé à la clé dans la variable cible
        if 'output' in dict_input_output_variable.keys(): 
            output_variable = dict_input_output_variable['output'] # Variable cible
        else : 
            raise Exception("Erreur : Il manque la clé \"output\" dans le dictionnaire.")

        # Si lst_input_variables n'est pas une liste de str des variables explicatives
        if not(isinstance(lst_input_variables, list) and all([isinstance(input_variable, str) for input_variable in lst_input_variables])):
            raise Exception("Erreur : L'objet de la clé \"input\" doit être de type list(str).")
       
        # Si output_variable n'est pas de type str
        if not(isinstance(output_variable, str)):
            raise Exception("Erreur : L'objet de la clé \"output\" doit être de type str.")
       
        return(lst_input_variables, output_variable)

    except Exception as e:
        print(e)
        exit()

def load_dataset(filename : str, lst_input_variables : list, output_variable : str) : #TODO : comment renvoyer plusieurs variables
    """ Charge les données csv du fichier donné en paramètre et renvoie des dataframes du fichier

    Args:
        filename (str): Nom et chemin du fichier csv des données
        lst_input_variables (list): liste des noms des variables explicatives
        output_variable (str): variable cible

    Raises:
        Exception: Un des noms de variable explicative de lst_input_variables ne figure pas dans les noms de colonnes du dataframe du jeu de données
        Exception: Le nom de la variable cible de output_variable ne figure pas dans les noms de colonnes du dataframe du jeu de données

    Returns:
        dataframe_wine (pd.DataFrame) : dataframe du jeu de données
        input_data (pd.DataFrame) : dataframe du jeu de données avec seulement les données correspondants aux variables explicatives
        output_data (pd.DataFrame) : dataframe du jeu de données avec seulement les données correspondants à la variable cible
    """
    try:
        # Si le fichier json n'existe pas ou n'est pas lisible
        check_file(filename)

        # load the dataset as a pandas DataFrame
        dataframe_wine = pd.read_csv(filename)

        # Vérifier si toutes les variables explicatives et la variable cible existent bien dans le dataframe du jeu de données
        for input_variable in lst_input_variables:
            if input_variable not in dataframe_wine.keys():
                raise Exception("Erreur : Le nom de variable explicative \"{}\" n'existe pas dans le jeu de données.".format(input_variable))
        
        if output_variable not in dataframe_wine.keys():
            raise Exception("Erreur : Le nom de la variable cible \"{}\" n'existe pas dans le jeu de données.".format(output_variable))
        
        input_data = dataframe_wine[lst_input_variables]
        output_data = dataframe_wine[output_variable]

        # format all fields as float for input variables, and as int for output variables
        input_data = input_data.astype(float)
        output_data = output_data.astype(int)

        return dataframe_wine, input_data, output_data

    except Exception as e:
        print(e)
        exit()

def save_split_data():
    """ 
        Splits train and test data and saves them in datasource repertory
    """
    # Chargement du dataset et séparation les variables explicatives et la variable cible
    input_variable, output_variable = open_json_file('./datasource/variable.json')
    _, input_data, output_data = load_dataset('./datasource/Wines.csv', input_variable, output_variable)

    # Séparation du jeu de données en base d'apprentissage (75%) et base de test (25%)
    input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size=0.25, random_state=1)
        
    input_data_train.to_csv('./datasource/input_data_train.csv', index=False)
    input_data_test.to_csv('./datasource/input_data_test.csv', index=False)
    output_data_train.to_csv('./datasource/output_data_train.csv', index=False)
    output_data_test.to_csv('./datasource/output_data_test.csv', index=False)

def open_saved_split_data() -> tuple :
    """
        Checks that split datas (train / test) exist and returns them as DataFrame

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: _description_
    """
    check_file('./datasource/input_data_train.csv')
    input_data_train = pd.read_csv('./datasource/input_data_train.csv')
    check_file('./datasource/input_data_test.csv')
    input_data_test = pd.read_csv('./datasource/input_data_test.csv')
    check_file('./datasource/output_data_train.csv')
    output_data_train = pd.read_csv('./datasource/output_data_train.csv')
    check_file('./datasource/output_data_test.csv')
    output_data_test = pd.read_csv('./datasource/output_data_test.csv')

    return(input_data_train, input_data_test, output_data_train, output_data_test)

def load_model() -> RandomForestClassifier:
    """ Load the random forest classifier model

    Returns:
        RandomForestClassifier: the random forest classifier model
    """
    path_name = "./random_forest.joblib"

    # check if the path and the file exists.
    check_file(path_name)

    # load, no need to initialize the loaded_rf
    loaded_model_rfc = joblib.load(path_name)

    return loaded_model_rfc

def save_model(model : RandomForestClassifier):
    """ Save the current random forest classifier model

    Args:
        model (RandomForestClassifier): current model (random forest classifier)
    """
    # save
    joblib.dump(model, "./random_forest.joblib")

# DATA INFORMATIONS
def data_exploration(dataframe : pd.DataFrame) :
    """ Displays dataframe informations and describes it

    Args:
        dataframe (pd.DataFrame): the dataframe we want to explore
    """
    # check basic features and dataframe types
    print("\n\n----- DATA EXPLORATION -----")
    # Description du df
    print("\n DESCRIPTION :")
    print(dataframe.describe())

    # Information du df
    print("\n INFORMATIONS :")
    print(dataframe.info())

def data_cleaning(dataframe : pd.DataFrame):
    """ Data cleaning : drop duplicate data or data with null value

    Args:
        dataframe (pd.DataFrame): the dataframe we want to clean
    """
    # check for doublications
    print("\n\n----- DATA CLEANING -----")
    print(" Duplicated ?", dataframe.duplicated().any())
    print("\n\n\nNb val manquante : ", dataframe.isnull().sum())

    # supprimer les valeurs manquantes ou celles avec un mauvais typage
    if dataframe.duplicated().any():
        dataframe.dropna(inplace = True)

    print(" is null ?", dataframe.isnull().any())
    print("\n\n\nNb doublon : ", dataframe.duplicated().sum())

def data_visualisation(dataframe : pd.DataFrame, output_variable : str):
    """ Data visualisation

    Args:
        dataframe (pd.DataFrame): the dataframe we want to visualize
        output_variable (str): name of output variable
    """
    print("\n\n----- DATA VISUALISATION -----")
    print(dataframe.head())

    # ratings distribution
    plt.figure()
    sns.kdeplot(dataframe[output_variable], fill = True)
    plt.title('Rating Distribution\n')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

    # number of books per rating
    data_val_count = dataframe[output_variable].value_counts()
    plt.figure()
    sns.barplot(y = data_val_count, x = data_val_count.index)
    plt.title('Fréquence des notes\n')
    plt.xlabel('Note')
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    plt.show()

def data_preprocessing(dataframe : pd.DataFrame, input_variable : list, output_variable : str):

    print("\n\n----- DATA PREPROCESSING -----")
    print('Before Removing Outliers')
    plt.figure()
    sns.boxplot(dataframe[input_variable])
    plt.show()

    # remove outliers from no. of pages 
    #dataframe = dataframe.drop(dataframe.index[dataframe['# num_pages'] >= 1000])
    # TODO : créer fonction qui remove outliers


    ## iterating through Column_Names using try and except for distinguishing between numerical and categorical columns
    for x_column_name in input_variable:
        try:
            print('Before Removing Outliers')

            ##visualisation of outliers
            """plt.figure()
            a = sns.boxplot(dataframe = dataframe, x = dataframe[x_column_name])
            plt.tight_layout() 
            plt.show()""" 

            xy = dataframe[x_column_name]    
            mydata = pd.DataFrame()

            updated=[]
            Q1,Q3=np.percentile(xy,[25,75])
            IQR=Q3-Q1
            minimum=Q1-1.5*IQR
            maximum=Q3+1.5*IQR

            ## using the maximum and minimum values obtained from quartiles and inter-quartile range
            ## any outliers greater than maximum are updated to be equal to maximum
            ## any outliers lesser than minimum are updated to be equal to minimum
            ## here, no outliers have been removed to prevent loss of dataframe

            for i in xy:
                if(i>maximum):
                    i=maximum
                    updated.append(i)
                elif(i<minimum):
                    i=minimum
                    updated.append(i)
                else:
                    updated.append(i)

            dataframe[x_column_name] = updated
            print('After Removing Outliers')

            ## visualising after removing outliers
            """plt.figure()
            b= sns.boxplot(dataframe = dataframe, x = dataframe[x_column_name])
            plt.tight_layout() 
            plt.show()"""

        except:
            continue

    print('After Removing Outliers')
    plt.figure()
    sns.boxplot(dataframe[input_variable])
    plt.tight_layout()
    plt.show()

    return dataframe

# RANDOM FOREST CLASSIFIER
# TODO : non fini
def random_forest_main(dataframe : pd.DataFrame, input_var : list, output_var : str):
    # divide the dataframe into attributes and labels
    X = dataframe[input_var]
    y = dataframe[output_var]

    # split 80% of the dataframe to the training set and 20% of the dataframe to test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

    print("Avant choix var expl : ")

    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(X_train)  # fitting standardization on feature dataframe
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = input_var) # transforming feature dataframe into standardized feature dataframe
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = input_var)  # transforming new dataframe into standardized form

    random_forest_classifier_model(X_train_scaled, X_test_scaled, y_train, y_test, input_var)

# TODO : non fini
def random_forest_classifier_model(X_train : pd.DataFrame, X_test : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, input_var):
    # instantiate the classifier 
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)

    # fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    return random_forest_classifier_model

# TODO : non fini
def backward_selected_forest(X_train : pd.DataFrame, X_test : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, input_var : list, alpha : float) -> list :
    # instantiate the classifier 
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)

    # fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    pred = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': y_pred})
    print(pred.head(10))

    model = RandomForestClassifier(n_jobs=-1)
    # Try different numbers of n_estimators - this will take a minute or so
    estimators = np.arange(10, 200, 10)
    scores = {"accuracy" : [], "matrix_erreur" : []}
    for n in estimators:
        model.set_params(n_estimators=n)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    plt.figure(figsize=(7, 5))
    plt.title("Effect of Estimators")
    plt.xlabel("no. estimator")
    plt.ylabel("score")
    plt.plot(estimators, scores)
    plt.show()

    # Check accuracy score 
    from sklearn.metrics import accuracy_score
    print('Model accuracy score with 10 decision-trees : {0:0.4f}%'. format(accuracy_score(y_test, y_pred)*100))

    # Print the Confusion Matrix and slice it into four pieces
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n\n', cm)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

# TODO : non fini      
def main():
    
    # Chargement du dataset et séparation les variables explicatives et la variable cible
    input_variable, output_variable = open_json_file('./datasource/variable.json')
    dataframe_wine, input_data, output_data = load_dataset('./datasource/Wines.csv', input_variable, output_variable)

    # Séparation du jeu de données en base d'apprentissage (75%) et base de test (25%)
    input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size=0.25, random_state=1)
    
    input_data_train.to_csv('./datasource/input_data_train', index=False)
    input_data_test.to_csv('./datasource/input_data_test', index=False)
    output_data_train.to_csv('./datasource/output_data_train', index=False)
    input_data_train.to_csv('./datasource/output_data_test', index=False)

    print("Frequency distribution of categorical variables")
    categorical = [i for i in range(10)]
    print(categorical)

    print("Percentage of frequency distribution of values")

    print(dataframe_wine[output_variable].value_counts()/np.float(len(dataframe_wine)))

    # visualize frequency distribution of income variable
    f, ax = plt.subplots(figsize=(8, 6))
    ax = sns.countplot(y = output_variable, dataframe = dataframe_wine, palette="Set1")
    ax.set_title("Frequency distribution of output variable")
    plt.show()

    # DATA EXPLORATION
    data_exploration(dataframe_wine)

    # DATA CLEANING
    data_cleaning(dataframe_wine)

    # DATA VISUALISATION
    data_visualisation(dataframe_wine, output_variable)
    
    # DATA PREPROCESSING
    #dataframe = data_preprocessing(dataframe, input_variable, output_variable)

    # Modele linear regression
    #regression_model(dataframe, input_variable, output_variable)
    #general_regre(dataframe, input_variable, output_variable)

    # Random forest model
    print("\n\n\n\n\ncoucou")
    random_forest_main(dataframe_wine, input_variable, output_variable)

# TODO : tout tester

# TODO : faire le cas où les fichiers split n'existent pas

# MAIN FUNCTIONS
# POST /api/predict
def prediction(new_wine : Wine) -> int :
    """
    Realizes a prediction by giving in body the necessary data of the wine to this one
    The prediction should be given by a score out of 10 of the wine entered.

    Args:
        input_var (dict): wine characteristics

    Returns:
        int: score of the wine
    """
    # Open the last Random Forest Classifier model saved
    model = load_model()

    # Open X_train data
    X_train, _, _, _ = open_saved_split_data()

    # Récupération des variables explicatives
    input_variable, _ = open_json_file('./datasource/variable.json')

    # Scale 
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(X_train)  # fitting standardization on feature dataframe

    array_wine = np.array([
        new_wine.fixed_acidity, 
        new_wine.volatile_acidity,
        new_wine.citric_acid,
        new_wine.residual_citric,
        new_wine.chlorides,
        new_wine.free_sulfur_dioxide,
        new_wine.total_sulfur_dioxide,
        new_wine.density,
        new_wine.pH,
        new_wine.sulphates,
        new_wine.alcohol
        ])

    print(array_wine)
    X_to_predict_scaled = pd.DataFrame(scaler.transform(array_wine), columns = input_variable)
    print("\n\n 3 \n\n")

    # Prediction
    prediction = model.predict(X_to_predict_scaled)

    return prediction

# TODO : à faire :'(
# GET /api/predict
def find_perfect_wine() -> dict :
    """ 
    Generates a combination of value to identify the "perfect wine" (probably non-existent but statistically possible)
    The prediction should provide the characteristics of the "perfect wine".

    Returns:
        dict: the characteristics of the "perfect wine"
    """
    return None

# GET /api/model permet d’obtenir le modèle sérialisé
def get_model() :
    """ Gets serialised model

    Returns:
        _type_: _description_
    """
    model = load_model()
    return model

# GET /api/model/description permet d’obtenir des informations sur le modèle
def get_model_information():
    """ Gets informations about the model such as:
        - parameters
        - performance metrics over the test data
        - other (depends on the algorithme)

    """
    # Get model
    model = load_model()

    # Open split datas
    X_train, X_test, y_train, y_test = open_saved_split_data()

    # Récupération des variables explicatives
    input_variable, output_variable = open_json_file('./datasource/variable.json')

    # Scale 
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(X_train)  # fitting standardization on feature dataframe
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = input_variable)

    # Prediction
    y_prediction = model.predict(X_test_scaled)

    # Accuracy
    print('Model accuracy score with 10 decision-trees : {0:0.4f}%\n'. format(accuracy_score(y_test, y_prediction)*100))

    # Prediction VS True value
    pred = pd.DataFrame({'Actual': y_test["quality"].tolist(), 'Predicted': y_prediction})
    print('Some actual values VS Predictions :\n', pred.head(10))

    # Print the Confusion Matrix and slice it into four pieces
    cm = confusion_matrix(y_test, y_prediction)
    print('\n\nConfusion matrix :\n', cm)

    y = list(y_train)
    y.extend(y_test)
    y = set(y)

    # show the main metrics
    print("\n\nMain metrics of the model :\n", classification_report(y_test, y_prediction, zero_division = 1))

    #model.summary
    #return None

# PUT /api/model
def add_wine(dict_wine_to_add : dict):
    """ 
    Enriches the model with an additional data entry (one more wine).
    An additional data must have the same format as the rest of the data

    Args:
        dict_wine_to_add (dict): dictionnary of the new wine to add

    :raise Exception: at least one necessary column is missing
    """
    try :
        path_csv = './datasource/Wines.csv'

        # Vérifier que le fichier json existe et qu'il est lisible
        check_file(path_csv)

        # Stocke le fichier csv dans un dataframe
        dataframe_wine = pd.read_csv(path_csv)

        # Nom des colonnes nécessaires
        list_column_names = list(dataframe_wine.columns)

        # Vérifier que toutes les données nécessaires soient renseignées
        for column_name in list_column_names:
            if column_name not in dict_wine_to_add.key():
                raise Exception("Erreur: Il manque la colonne >{}< dans la donnée que vous voulez insérer.".format(column_name))

        # Ajouter le nouveau vin dans le dataframe
        for column_name in list_column_names:
            dataframe_wine[column_name].append(dict_wine_to_add[column_name])

        # Enregistrer le nouveau dataframe (on écrase l'ancier)
        dataframe_wine.to_csv(path_csv, index=False)

    except Exception as e:
        print(e)

# POST /api/model/retrain 
def model_train():
    """
        re-train the model taking into account the data added afterwards and save the new model    
    """
    # Chargement du dataset et séparation les variables explicatives et la variable cible
    input_variable, output_variable = open_json_file('./datasource/variable.json')
    dataframe_wine, input_data, output_data = load_dataset('./datasource/Wines.csv', input_variable, output_variable)

    save_split_data()

    # Open the data
    (input_data_train, input_data_test, output_data_train, output_data_test) = open_saved_split_data()

    print("X_train :", input_data_train.head(10))
    print("X_train :", input_data_train.shape)
    print("y_train : ", output_data_train.head(10))
    print("y_train : ", output_data_train.shape)

    # Scale the model
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(input_data_train)  # fitting standardization on feature dataframe
    X_train_scaled = pd.DataFrame(scaler.transform(input_data_train), columns = input_variable) # transforming feature dataframe into standardized feature dataframe
    
    
    print("X_train_scaled", X_train_scaled.head(10))
    # Reentraine le modèle
    #model_rfc = train_random_forest_classifier((input_data_train, input_data_test, output_data_train, output_data_test, input_variable, output_variable)
        
    # instantiate the classifier 
    model_rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    model_rfc.fit(X_train_scaled, output_data_train.values.ravel())

    print("fin")
    # Sauvegarde le modèle
    save_model(model_rfc)


def test_fct():
    #model_train()
    #get_model_information()
    
    new_wine = {
        "fixed acidity" : 11.2, 
        "volatile acidity" : 0.7, 
        "citric acid" : 0.0, 
        "residual sugar": 1.9, 
        "chlorides" : 0.076, 
        "free sulfur dioxide" : 11.0, 
        "total sulfur dioxide" : 34.0, 
        "density": 0.9978, 
        "pH": 3.51, 
        "sulphates": 0.56, 
        "alcohol": 9.4
        }
        
    prediction(new_wine)

test_fct()