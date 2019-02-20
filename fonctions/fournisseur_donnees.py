import csv
import pandas as pd
import random

#***************************************************
#******CHARGEMENT DES DONNEES DU FICHIER CSV********
#***************************************************

colonnes = []
donnees = []

def chargement_donnees():
	dataCount = 0

	#Chargement du fichier CSV avec toutes les données
	with open("openfoodfacts.csv",  encoding="utf8", newline='') as csvFichier:
		#On récupère les données dans une liste
		csvFichierDonnees = csv.reader(csvFichier, delimiter='\t')
		for index, ligne in enumerate(csvFichierDonnees):
			if(index == 0):
				ligne[43] = "healthy"
				colonnes  = ligne
			else:
				nutrition_grade_fr = ligne[43]
				fat_100g = ligne[59]
				saturated_fat_100g = ligne[60]
				salt_100g = ligne[110]
				if(len(ligne) == len(colonnes ) 
					and nutrition_grade_fr != ""
					and fat_100g != ""
					and salt_100g!= ""):
					if(nutrition_grade_fr in ['a', 'b', 'c']):
						ligne[43] = 1
					else:
						ligne[43] = 0
					donnees.append(ligne)
					dataCount += 1


	openfoodfacts = pd.DataFrame(donnees, columns=colonnes ) 
	return openfoodfacts, dataCount - 1