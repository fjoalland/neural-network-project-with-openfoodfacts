import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import fonctions.fournisseur_donnees as fournisseur

#1 bonne sante
#0 mauvaise sante

def get_dataset():
	nourriture_entrainement = []
	nourriture_objectif = [] 
	openfoodfacts, dataCount = fournisseur.chargement_donnees()

	for index in range(dataCount):
		if(int(openfoodfacts['healthy'][index]) == 1):
			nourriture_entrainement.append([float(openfoodfacts['fat_100g'][index]),
											float(openfoodfacts['saturated-fat_100g'][index]),
											float(openfoodfacts['salt_100g'][index]),
											float(openfoodfacts['sugars_100g'][index])])
			nourriture_objectif.append([1])
		else:
			nourriture_entrainement.append([float(openfoodfacts['fat_100g'][index]),
											float(openfoodfacts['saturated-fat_100g'][index]),
											float(openfoodfacts['salt_100g'][index]),
											float(openfoodfacts['sugars_100g'][index])])
			nourriture_objectif.append([0])

	parametres = np.vstack([nourriture_entrainement])

	return parametres, nourriture_objectif, openfoodfacts

if __name__ == '__main__':
	parametres, targets, openfoodfacts = get_dataset()
	
	tf_features = tf.placeholder(tf.float32, shape=[None, 4])
	tf_targets = tf.placeholder(tf.float32, shape=[None, 1])

	# First
	w1 = tf.Variable(tf.random_normal([4, 3]))
	b1 = tf.Variable(tf.zeros([3]))

	# Operations
	z1 = tf.matmul(tf_features, w1) + b1
	a1 = tf.nn.sigmoid(z1)

	# Output neuron
	w2 = tf.Variable(tf.random_normal([3, 1]))
	b2 = tf.Variable(tf.zeros([1]))

	# Operations
	z2 = tf.matmul(a1, w2) + b2
	py = tf.nn.sigmoid(z2)

	erreur = tf.reduce_mean(tf.square(py - tf_targets))

	correct_prediction = tf.equal(tf.round(py), tf_targets)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.10)
	train = optimizer.minimize(erreur)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	print(openfoodfacts['product_name'][0])
	print(openfoodfacts['healthy'][0])
	print("py=", sess.run(py, feed_dict={
		tf_features: [parametres[0]]
	}))

	print("correct_prediction=", sess.run(correct_prediction, feed_dict={
		tf_features: [parametres[0]],
		tf_targets: [targets[0]]
	}))


	print("erreur=", sess.run(erreur, feed_dict={
		tf_features: [parametres[0]],
		tf_targets: [targets[0]]
	}))

	for e in range(10000):
		sess.run(train, feed_dict={
			tf_features: parametres,
			tf_targets: targets
		})

		print('sess = ',sess.run(accuracy, feed_dict={
			tf_features: parametres,
			tf_targets: targets
		}))
		
	print(openfoodfacts['product_name'][0])
	print(openfoodfacts['healthy'][0])
	print("py=", sess.run(py, feed_dict={
		tf_features: [parametres[0]]
	}))

	print("correct_prediction=", sess.run(correct_prediction, feed_dict={
		tf_features: [parametres[0]],
		tf_targets: [targets[0]]
	}))


	print("erreur=", sess.run(erreur, feed_dict={
		tf_features: [parametres[0]],
		tf_targets: [targets[0]]
	}))

	print(parametres[0])
	print(targets[0])


	print('*********')

	for e in range(100):

		tabl = []
		tabl.append([float(32), float(23), float(3.7), float(0.5)])

		test = np.vstack(tabl)
		print(test)
		print("Il doit etre mauvais=", sess.run(py, feed_dict={
			tf_features: test
		}))

		tabl2 = []

		tabl2.append([float(0.1), float(0), float(0.1), float(10.6)])

		test2 = np.vstack(tabl2)
		print(test)
		print("Il doit etre sain=", sess.run(py, feed_dict={
			tf_features: test2
		}))

		

