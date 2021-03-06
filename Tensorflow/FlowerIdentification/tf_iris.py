from sklearn import metrics,cross_validation
import tensorflow as tf
from tensorflow.contrib import learn

def main(unused_argv):
	iris = learn.datasets.load_dataset('iris')
	x_train,x_test,y_train,y_test = cross_validation.train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)

	feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]
	classifier = learn.DNNClassifier(feature_columns = feature_columns,hidden_units=[10,20,10],n_classes=3)

	classifier.fit(x_train,y_train,steps=200)
	score = classifier.evaluate(x=x_test,y=y_test)["accuracy"]
	print 'Accuracy: {0:f}'.format(score)

if __name__ == "__main__":
	tf.app.run()
