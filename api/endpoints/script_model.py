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

from __init__ import Wine, WineFull

# GENERAL FUNCTION
def __check_file(path_name : str):
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

def __open_json_file(path_json : str) : # TODO : trouver comment renvoyer tuple 
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
        __check_file(path_json)

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

def __load_dataset(filename : str, lst_input_variables : list, output_variable : str) : #TODO : comment renvoyer plusieurs variables
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
        __check_file(filename)

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

def __save_split_data():
    """ 
        Splits train and test data and saves them in datasource repertory
    """
    # Chargement du dataset et séparation les variables explicatives et la variable cible
    input_variable, output_variable = __open_json_file('../../datasource/variable.json')
    _, input_data, output_data = __load_dataset('../../datasource/Wines.csv', input_variable, output_variable)

    # Data cleaning (gère les valeurs dupliquées ou manquantes)
    input_data = __data_cleaning(input_data)
    output_data = __data_cleaning(output_data)

    # Séparation du jeu de données en base d'apprentissage (75%) et base de test (25%)
    input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size=0.25, random_state=1)
        
    input_data_train.to_csv('../../model/input_data_train.csv', index=False)
    input_data_test.to_csv('../../model/input_data_test.csv', index=False)
    output_data_train.to_csv('../../model/output_data_train.csv', index=False)
    output_data_test.to_csv('../../model/output_data_test.csv', index=False)

def __open_saved_split_data() -> tuple :
    """
        Checks that split datas (train / test) exist and returns them as DataFrame

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: _description_
    """
    __check_file('../../model/input_data_train.csv')
    input_data_train = pd.read_csv('../../model/input_data_train.csv')
    __check_file('../../model/input_data_test.csv')
    input_data_test = pd.read_csv('../../model/input_data_test.csv')
    __check_file('../../model/output_data_train.csv')
    output_data_train = pd.read_csv('../../model/output_data_train.csv')
    __check_file('../../model/output_data_test.csv')
    output_data_test = pd.read_csv('../../model/output_data_test.csv')

    return(input_data_train, input_data_test, output_data_train, output_data_test)

def __load_model() -> RandomForestClassifier:
    """ Load the random forest classifier model

    Returns:
        RandomForestClassifier: the random forest classifier model
    """
    path_name = "../../model/random_forest.joblib"

    # check if the path and the file exists.
    __check_file(path_name)

    # load, no need to initialize the loaded_rf
    loaded_model_rfc = joblib.load(path_name)

    return loaded_model_rfc

def __save_model(model : RandomForestClassifier):
    """ Save the current random forest classifier model

    Args:
        model (RandomForestClassifier): current model (random forest classifier)
    """
    # save
    joblib.dump(model, "../../model/random_forest.joblib")

# DATA INFORMATIONS
def __data_exploration(dataframe : pd.DataFrame) :
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

def __data_cleaning(dataframe : pd.DataFrame):
    """ Data cleaning : drop duplicate data or fill NaNs with column median

    Args:
        dataframe (pd.DataFrame): the dataframe we want to clean
    """
    if dataframe.duplicated().any():
        print("Il existe des données dupliquées : elles seront supprimées.")
        dataframe.dropna(inplace = True)

    # Gestion des valeurs manquantes
    if dataframe.isnull().values.any():
        print("Certaines valeurs sont manquantes : elles seront remplacées par la médiane de la colonne.")
        # Remplir les valeurs NaNs (Not a Number) avec les médianes de chaque colonne pour chaque colonne 
        dataframe = dataframe.fillna(dataframe.median())

    return dataframe

def __data_visualisation(dataframe : pd.DataFrame, output_variable : str):
    """ Data visualisation

    Args:
        dataframe (pd.DataFrame): the dataframe we want to visualize
        output_variable (str): name of output variable
    """
    print("\n\n----- DATA VISUALISATION -----")
    print(dataframe.head())

    # Fréquence des notes
    data_val_count = dataframe[output_variable].value_counts()
    plt.figure()
    sns.barplot(y = data_val_count, x = data_val_count.index)
    plt.title('Fréquence des notes\n')
    plt.xlabel('Note')
    plt.ylabel('Nombre')
    plt.show()

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
    model = __load_model()

    # Open X_train data
    X_train, _, _, _ = __open_saved_split_data()

    # Récupération des variables explicatives
    input_variable, _ = __open_json_file('../../datasource/variable.json')

    # Scale 
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(X_train)  # fitting standardization on feature dataframe

    # Récupérer les informations du premier élément
    array_new_wine = np.array([[
        new_wine.fixed_acidity, 
        new_wine.volatile_acidity,
        new_wine.citric_acid,
        new_wine.residual_sugar,
        new_wine.chlorides,
        new_wine.free_sulfur_dioxide,
        new_wine.total_sulfur_dioxide,
        new_wine.density,
        new_wine.pH,
        new_wine.sulphates,
        new_wine.alcohol
        ]])

    # Les transformer en dataframe
    df_new_wine = pd.DataFrame(array_new_wine, columns = input_variable)

    # Scale les valeurs
    df_new_wine_scaled = pd.DataFrame(scaler.transform(df_new_wine), columns = input_variable)

    # Prédiction
    prediction = model.predict(df_new_wine_scaled)
    return {"prediction" : prediction[0]}

# TODO : à faire :'(
# GET /api/predict
def find_perfect_wine() -> dict :
    """ 
    Generates a combination of value to identify the "perfect wine" (probably non-existent but statistically possible)
    The prediction should provide the characteristics of the "perfect wine".

    Returns:
        dict: the characteristics of the "perfect wine"
    """
    # Open the last Random Forest Classifier model saved
    model = __load_model()

    return {}

# GET /api/model permet d’obtenir le modèle sérialisé
def get_model() :
    """ Gets serialised model

    Returns:
        _type_: _description_
    """
    model = __load_model()
    print("Modèle chargé")
    return model

# GET /api/model/description permet d’obtenir des informations sur le modèle
def get_model_information() -> dict :
    """Gets informations about the model such as:
        - parameters
        - performance metrics over the test data
        - other (depends on the algorithme)

    Returns:
        dict: return informations of the model
    """
    # Get model
    model = __load_model()

    # Open split datas
    X_train, X_test, y_train, y_test = __open_saved_split_data()

    # Récupération des variables explicatives
    input_variable, output_variable = __open_json_file('../../datasource/variable.json')

    # Scale 
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(X_train)  # fitting standardization on feature dataframe
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = input_variable)

    # Prediction
    y_prediction = model.predict(X_test_scaled)

    # Print the Confusion Matrix and slice it into four pieces
    cm = confusion_matrix(y_test, y_prediction)
    
    dict_result = {
        "param_model": model.get_params(),
        "main_metrics": classification_report(y_test, y_prediction, zero_division = 1),
        "confusion_matrix" : cm
    }

    return dict_result

# PUT /api/model
def add_wine(new_wine : WineFull):
    """ 
    Enriches the model with an additional data entry (one more wine).
    An additional data must have the same format as the rest of the data

    Args:
        dict_wine_to_add (dict): dictionnary of the new wine to add

    :raise Exception: at least one necessary column is missing
    """
    try :
        path_csv = '../../datasource/Wines.csv'

        # Vérifier que le fichier json existe et qu'il est lisible
        __check_file(path_csv)

        # Stocke le fichier csv dans un dataframe
        dataframe_wine = pd.read_csv(path_csv)

        # Nom des colonnes nécessaires
        list_column_names = list(dataframe_wine.columns)

        # nouveau vin
        new_wine = [
            new_wine.fixed_acidity, 
            new_wine.volatile_acidity,
            new_wine.citric_acid,
            new_wine.residual_sugar,
            new_wine.chlorides,
            new_wine.free_sulfur_dioxide,
            new_wine.total_sulfur_dioxide,
            new_wine.density,
            new_wine.pH,
            new_wine.sulphates,
            new_wine.alcohol, 
            new_wine.quality,
            new_wine.id
        ]

        # Ajouter le nouveau vin dans le dataframe
        item = 0
        for column_name in list_column_names:
            dataframe_wine[column_name].append(new_wine[item])
            item += 1

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
    input_variable, output_variable = __open_json_file('../../datasource/variable.json')

    # Charger le nouveau jeu de données
    __save_split_data()

    # Open the data
    (input_data_train, input_data_test, output_data_train, output_data_test) = __open_saved_split_data()

    # Scale the model
    scaler = StandardScaler()  # instantiating StandardScaler class
    scaler.fit(input_data_train)  # fitting standardization on feature dataframe
    X_train_scaled = pd.DataFrame(scaler.transform(input_data_train), columns = input_variable) # transforming feature dataframe into standardized feature dataframe
    
    # Reentraine le modèle
    #model_rfc = train_random_forest_classifier((input_data_train, input_data_test, output_data_train, output_data_test, input_variable, output_variable)
        
    # instantiate the classifier 
    model_rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    model_rfc.fit(X_train_scaled, output_data_train.values.ravel())

    # Sauvegarde le modèle
    __save_model(model_rfc)

    return {"post" : "Nouveau modèle entraîné avec succès !"}

# TODO : à supprimer à la fin
def test_fct():
    #model_train()
    get_model_information()
    
    """new_wine = Wine()

    new_wine.fixed_acidity = 11.2
    new_wine.volatile_acidity = 0.7
    new_wine.citric_acid = 0.0
    new_wine.residual_sugar = 1.9
    new_wine.chlorides = 0.076
    new_wine.free_sulfur_dioxide = 11.0
    new_wine.total_sulfur_dioxide = 34.0
    new_wine.density = 0.9978
    new_wine.pH = 3.51
    new_wine.sulphates = 0.56
    new_wine.alcohol = 9.4"""

    prediction()

test_fct()