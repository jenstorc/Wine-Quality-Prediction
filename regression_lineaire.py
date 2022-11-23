# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
import json 
import os
from sys import exit
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

## TODO : A partir d'ici c'est brouillon
def check_dataframe(dataframe : pd.DataFrame) :
    # Description du df
    print("\n\n\n-------------- DESCRIPTION --------------")
    print(dataframe.describe())

    # Information du df
    print("\n\n\n-------------- INFORMATIONS --------------")
    print(dataframe.info())

    # TODO : vérifier les data (pas de val nulles)
    # Pour chaque colonne vérifier si il y a des valeurs manquantes. Par exemple avec .isnull() de pandas. 
    print("\n\n\nNb val manquante : ", dataframe.isnull().sum())
    #for i in df.columns:
    #    print(i," : ",df[i].isnull().sum())

    #Puis vérifier si il y a des doublons. Vous pouvez utiliser .duplicated()
    print("\n\n\nNb doublon : ", dataframe.duplicated().sum())

def check_model(model):

    #view model summary
    print(model.summary())

    print("\n\n\nParamètres = \n{}".format(model.params))
    print("\n\n\nR² = {}".format(model.rsquared))
    # c'est nul, mauvais modèle : La qualité du vin est expliquée à 35,9% par ce modèle
    # Modèle pas bien ajusté!!!!

def ascendant_AIC(output_data, input_data):
    input_data_model = sm.add_constant(input_data)
    model = sm.OLS(output_data, input_data_model).fit()
    modele_ameliorable = True

    i = 0 # TODO = suppr
    while modele_ameliorable:
        meilleur_modele = model
        meilleur_modele_aic = meilleur_modele.aic
        
        # On sélectionne la variable la moins significative
        input_variable_aic = list(model.pvalues.index) # liste des variables 
        var_name_max_pval_input_ = input_variable_aic[model.pvalues.argmax()] # récupérer la variable la moins significative
        input_variable_aic.remove(var_name_max_pval_input_) # la supprimer de la liste des variables explicative
        input_variable_aic = input_variable_aic[1:len(input_variable_aic)+1] # suppression de la constante
        
        #On crée un nouveau modèle
        X_train = input_data[input_variable_aic] # nouveau input_data_train
        X_train_model = sm.add_constant(X_train)
        print("\n\n{} : {}".format(i, list(X_train_model.columns))) # TODO = suppr
        i += 1
        model = sm.OLS(output_data, X_train_model).fit() # nouveau modele
        aic = model.aic # calcul aic pour le nouveau modele

        print("\n\n\nnouveau aic = {}, ancien aic = {}".format(aic, meilleur_modele_aic))
        print("\n\n\nR² = {}".format(model.rsquared))
        
        modele_ameliorable = (aic < meilleur_modele_aic) #and any([pval > 0.05 for pval in model.pvalues])
    
    #check_model(meilleur_modele)

    return meilleur_modele

"""def ascendant_BIC(output_data, input_data):
    input_data_model = sm.add_constant(input_data)
    model = sm.OLS(output_data, input_data_model).fit()
    modele_ameliorable = True

    while modele_ameliorable:
        meilleur_modele = model
        bic = meilleur_modele.bic
        
        # On sélectionne la variable la moins significative
        input_variable_bic = list(model.pvalues.index)
        print(input_variable_bic)
        print("max = {}, argmax = {}".format(model.pvalues.argmax(), model.pvalues.max()))

        var_name_max_pval_input_ = input_variable_bic[model.pvalues.argmax()]
        input_variable_bic = input_variable_bic[1:len(input_variable_bic)+1]

        # On enlève la variable la moins significative de la liste des variables input
        input_variable_bic.remove(var_name_max_pval_input_)

        #On crée un modèle sans ça, on recalcule bic
        X_train = input_data[input_variable_bic]
        X_train_model = sm.add_constant(X_train)
        model = sm.OLS(output_data, X_train_model).fit()
        bic_tmp = model.bic

        print("\n\n\nnouveau bic = {}, ancien bic = {}".format(bic_tmp, bic))

        modele_ameliorable = (bic_tmp < bic) and any([pval > 0.05 for pval in model.pvalues])
    
    check_model(meilleur_modele)

    return meilleur_modele
"""

def main():
    # Chargement du dataset et séparation les variables explicatives et la variable cible
    input_variable, output_variable = open_json_file('./datasource/variable.json')
    dataframe_wine, input_data, output_data = load_dataset('./datasource/Wines.csv', input_variable, output_variable)

    # Séparation du jeu de données en base d'apprentissage (75%) et base de test (25%)
    input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data, output_data, test_size=0.25, random_state=1)

    # Check du dataframe
    #check_dataframe(dataframe_wine)

    #add constant to predictor variables
    input_data_train_model = sm.add_constant(input_data_train)
    model = sm.OLS(output_data_train, input_data_train_model).fit()

    check_model(model)

    # r² nul
    # p-value = 3.43e-74 << 5% donc globalement significatif
    # volatile acidity, chlorides, total sulfur dioxide, pH, sulphates et alcohol ont une p-valeur < 5% -> individuellement signficatif
    # const, fixed acidity, citric acid, residual sugar, free sulfur dioxide et density p-val > 5%

    model_best_aic = ascendant_AIC(output_data_train, input_data_train)
    #model_best = ascendant_BIC(output_data_train, input_data_train)

main()