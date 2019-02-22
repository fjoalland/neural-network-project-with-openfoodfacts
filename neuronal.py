import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import fonctions.fournisseur_donnees as fournisseur
import matplotlib.pyplot as plt

#Le nombre d'elements maximum qu'on va afficher sur un graphique.
#Pour eviter de detruire l'ordinateur
graphiqueLimitation = 20

#Nombre d'époques permettant aux neurones de s'entrainer 
epoqueTotal = 2000

#Fonction permettant de dessigner le grapg
#nourritureListe = liste des ingredients
#solutionsGraph = Liste si l'ingrédient est sain ou malsain (0 ou 1). Ce parametre peut être null
#limit = limites de ce que on va afficher
#title = titre du graphique
def dessinnerGraphique(nourritureListe, solutionsGraph, limit, title):

	#Graphique affichant les graisses par rapport aux graisses saturées
	fig, axs = plt.subplots()

	#On parcourt la liste des ingredients
	for index in range(limit):

		#Si on ne sait pas si l'ingredient est sain ou malsain, on affiche en bleu
		if(len(solutionsGraph) == 0):
			axs.plot(['fat_100g', 'saturated-fat_100g', 'salt_100g', 'sugars_100g'], nourritureListe[index],'o-', color='b')
		else:

			#Si l'ingredient est malsain
			if(solutionsGraph[index] == 0):
				axs.plot(['fat_100g', 'saturated-fat_100g', 'salt_100g', 'sugars_100g'], nourritureListe[index], 'o-',color='r')
			#Si l'ingredient est sain
			if(solutionsGraph[index] == 1):
				axs.plot(['fat_100g', 'saturated-fat_100g', 'salt_100g', 'sugars_100g'], nourritureListe[index],'o-', color='g')

	axs.legend()
	axs.set_title(title)
	axs.set_xlabel('Ingredients')
	axs.set_ylabel('Grams')
	plt.show()	



def get_dataset():

	#Liste de 1 ou de 0 indiquant si l'aliment est sain ou malsain (1 = sain, 2= malsain)
	#Variable utilisé uniquement dans la conception d'un graphique
	solutionsGraph = []
	#Liste de array de 1 ou de 0 indiquant si l'aliment est sain ou malsain.
	#exemple= [[1], [0], ..., [1]]
	#La difference avec solutionGraph est que c'est un Array de Array
	#Cela permet a Tensorflow de l'attribuer avec un tenseur par le biai d'un placeholder
	nourriture_objectif = [] 
	#C'est la liste de la nourriture qui va etre teste au reseau de neurone afin de l'entrainer.
	nourriture_entrainement = []

	#Recuperation des donnees du fichier CSV
	openfoodfacts, donneesTotal = fournisseur.chargement_donnees_entrainement()

	#Pour chaque aliment de la liste
	for index in range(donneesTotal):

		#On verifie si l'ingredient / l'aliment parcouru a pour nutrition a, b ou c
		#Nutri-Score est un code permettant de connaitre la qualité d'un aliment
		#(a, b fait reference a une bonne qualité, la lettre c a une qualite moyenne)
		if(openfoodfacts['nutrition_grade_fr'][index] in ['a', 'b', 'c']):

			#On ajoute les differents composants de l'aliment dans le tableau
			nourriture_entrainement.append([float(openfoodfacts['fat_100g'][index]),
											float(openfoodfacts['saturated-fat_100g'][index]),
											float(openfoodfacts['salt_100g'][index]),
											float(openfoodfacts['sugars_100g'][index])])

			#On indique que cet aliment est sain
			#1 = malsain
			nourriture_objectif.append([1])
			solutionsGraph.append(1)
		else:
			nourriture_entrainement.append([float(openfoodfacts['fat_100g'][index]),
											float(openfoodfacts['saturated-fat_100g'][index]),
											float(openfoodfacts['salt_100g'][index]),
											float(openfoodfacts['sugars_100g'][index])])

			#On indique que cet aliment est malsain
			#0 = malsain
			nourriture_objectif.append([0])
			solutionsGraph.append(0)

	#Empilement des tableaux en séquence verticalement
	parametres = np.vstack([nourriture_entrainement])

	title = 'Display of ingredients by their nutritional characteristics'
	#On affiche le grapgique avec des ingredients et leurs composants (fat, etc)
	dessinnerGraphique(nourriture_entrainement, solutionsGraph, graphiqueLimitation, title)

	return parametres, nourriture_objectif, openfoodfacts

if __name__ == '__main__':
	parametres, targets, openfoodfacts = get_dataset()
	
	#On insere un espace réservé pour un tenseur qui sera toujours alimenté.
	#"shape" permet de lui définir la forme du tenseur.
	#Un espace réservé avec la forme [None, 4] prend un tableau à 4 dimensions (fat_100g, saturated-fat_100g, salt_100g, sugars_100g)
	tf_parametres = tf.placeholder(tf.float32, shape=[None, 4])
	#Un espace réservé avec la forme [None, 4] prend un tableau à 1 dimension (1 ou 0 = Sain ou Malsain)
	tf_cibles = tf.placeholder(tf.float32, shape=[None, 1])


	#Definission du premier neurone
	#Définission du poid pour chacune des caracteristiques d'un aliment (ici on en a 4)
	poid1 = tf.Variable(tf.random_normal([4, 3]))
	b1 = tf.Variable(tf.zeros([3]))


	# Operations PRE ACTIATION
	# wi*pi + ... + bias
	z1 = tf.matmul(tf_parametres, poid1) + b1


	#ACTIVATION
	#La fonction sigmoid retourne toujours un résultat entre 0 et 1
	#Ce resultat est la probabilité
	a1 = tf.nn.sigmoid(z1)


	# La sortie du neurone
	#Définission du poid et du bias
	poid2 = tf.Variable(tf.random_normal([3, 1]))
	b2 = tf.Variable(tf.zeros([1]))


	# Operations PRE ACTIATION
	# wi*pi + ... + b
	z2 = tf.matmul(a1, poid2) + b2


	#ACTIVATION
	#La fonction sigmoid retourne toujours un résultat entre 0 et 1
	#Ce resultat est la probabilité
	probabilite = tf.nn.sigmoid(z2)

	#Permet de définir l'erreur des predictions
	#Opération réalisée: différence entre la prediction et la valeur attendu au carré
	erreur = tf.reduce_mean(tf.square(probabilite - tf_cibles))

	#Definir si l'algorithme a fait une bonne prediction
	bonnePrediction = tf.equal(tf.round(probabilite), tf_cibles)
	#Calculer la fréquence a laquelles les predictions correspondent a la realite
	precision = tf.reduce_mean(tf.cast(bonnePrediction, tf.float32))

	#Minimise les erreurs avec la methode de descente de gradient
	optimiseur = tf.train.GradientDescentOptimizer(learning_rate=15)
	#Entrainement 
	entrainement = optimiseur.minimize(erreur)

	#Definission de la Session contenant le graph
	session = tf.Session()
	#Initialisations de toutes les variables du graph
	session.run(tf.global_variables_initializer())

	for e in range(epoqueTotal):

		#Entrainement du neurone
		session.run(entrainement, feed_dict={
			tf_parametres: parametres,
			tf_cibles: targets
		})

		#Affichage de la predection de l'époque actuelle. 
		#Plus la prediction est proche de 1, plus le neurone fait peu d'erreurs
		print('Frequency of correct accuracy = ',session.run(precision, feed_dict={
			tf_parametres: parametres,
			tf_cibles: targets
		}))


	#Chargement des repas des touristess
	repasTouristes, donneesTotal = fournisseur.chargement_donnees_evaluer()

	#Converti les donnes en liste de liste pour les afficher dans le grapgique
	repasTouritesListeGrapgique = repasTouristes[['fat_100g', 'saturated-fat_100g','salt_100g', 'sugars_100g']].values.tolist()

	title = 'Before train - Display of ingredients by their nutritional characteristics'

	#Afficher le graphique avec les ingredients
	dessinnerGraphique(repasTouritesListeGrapgique, [], len(repasTouritesListeGrapgique),title)

	#liste des predictions
	predictions = []

	print("\n")
	print("****Showing predictions****")
	print("\n")

	for index in range(donneesTotal  + 1):

		#Liste qui va contenir les ingredients du repas à evaluer
		repas_evaluer = []

		#Recuperation du nom du repas
		product_name = repasTouristes['product_name'][index]
		#Taux de matieres grasses sur 100g
		fat_100g = repasTouristes['fat_100g'][index]
		#taux de matieres grasses saturées sur 100g
		saturated_fat_100g = repasTouristes['saturated-fat_100g'][index]
		#taux de sel sur 100g
		salt_100g = repasTouristes['salt_100g'][index]
		#taux de sucre sur 100g
		sugars_100g = repasTouristes['sugars_100g'][index]

		#Ajout des différents composants 
		repas_evaluer.append([fat_100g, saturated_fat_100g, salt_100g, sugars_100g])

		#Empilement des tableaux en séquence verticalement
		parametres = np.vstack(repas_evaluer)

		#Calcul de la probabilité que l'element soit sain (1) ou malsain (0)
		result = session.run(probabilite, feed_dict={
			tf_parametres: parametres
		})

		#On a joute la prediction de ce neurone dans la liste des predictions
		predictions.append(np.around(result))

		#Si le repas est sain
		if(np.around(result) == 1):
			print("Healthy: " + product_name)
		#Si le repas est malsain
		else: 
			print("Unhealthy: " + product_name)

	title = 'After train - Display of ingredients by their nutritional characteristics'
	dessinnerGraphique(repasTouritesListeGrapgique, predictions, len(repasTouritesListeGrapgique),title)
