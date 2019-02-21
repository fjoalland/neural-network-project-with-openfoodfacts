import csv
import pandas as pd
import random

#***************************************************
#******CHARGEMENT DES DONNEES DU FICHIER CSV********
#***************************************************



def chargement_donnees_entrainement():
	dataCount = 0
	colonnes = []
	donnees = []
	
	#Chargement du fichier CSV avec toutes les données
	with open("openfoodfacts-auchan.csv",  encoding="utf8", newline='') as csvFichier:
		#On récupère les données dans une liste
		csvFichierDonnees = csv.reader(csvFichier, delimiter='\t')
		for index, ligne in enumerate(csvFichierDonnees):
			if(index == 0):
				colonnes  = ligne
			else:
				nutrition_grade_fr = ligne[43]
				fat_100g = ligne[59]
				saturated_fat_100g = ligne[60]
				salt_100g = ligne[110]
				if(len(ligne) == len(colonnes ) 
					and nutrition_grade_fr != ""
					and fat_100g != ""
					and salt_100g!= ""
					and saturated_fat_100g != ""):
					ligne[59] = float(ligne[59])
					donnees.append(ligne)
					dataCount += 1


	openfoodfacts = pd.DataFrame(donnees, columns=colonnes ) 
	return openfoodfacts, dataCount - 1
	
def chargement_donnees_evaluer():
	dataCount = 0
	colonnes = []
	donnees = []
	
	#Chargement du fichier CSV avec toutes les données
	with open("repas_touristes.csv",  encoding="utf8", newline='') as csvFichier:
		#On récupère les données dans une liste
		csvFichierDonnees = csv.reader(csvFichier, delimiter=';')
		for index, ligne in enumerate(csvFichierDonnees):
			if(index == 0):
				colonnes  = ligne
			else:
				donnees.append(ligne)
				dataCount += 1


	repasTouristes = pd.DataFrame(donnees, columns=colonnes ) 
	return repasTouristes, dataCount - 1