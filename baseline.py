import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def initialize_params(dim):
	# Create our weights vector
	w = np.zeros((dim, 1))
	b = 0

	return w, b

def propagate(w, b, X, Y):
	# Forward propagate
	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)

	# Number of train examples
	m = X.shape[1]
	# Compute the cost
	cost = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	# Copmute gradients
	dZ = A - Y
	dw = 1/m * np.dot(X, dZ.T)
	db = 1 / m * np.sum(dZ)

	grads = {"dw": dw,
			 "db": db}

	return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	costs = []

	for i in range(num_iterations):
        
		# Compute cost and gradient
		grads, cost = propagate(w, b, X, Y)

		# Retrieve derivatives from grads
		dw = grads["dw"]
		db = grads["db"]

		# Update weights
		w = w - learning_rate * dw
		b = b - learning_rate * db

		# Record the costs
		if i % 100 == 0:
		    costs.append(cost)

		# Print the cost every 100 training iterations
		if print_cost and i % 100 == 0:
		    print ("Cost after iteration %i: %f" %(i, cost))

	params = {"w": w, "b": b}

	grads = {"dw": dw, "db": db}

	return params, grads, costs

def predict(w, b, X):
	# Do the prediction
	Y_predictions = np.zeros((1, X.shape[1]))
	w = w.reshape(X.shape[0], 1)

	y_hat = sigmoid(np.dot(w.T, X) + b)

	Y_predictions = np.around(y_hat)

	return Y_predictions


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

	# Initialize the training parameters
	w, b = initialize_params(X_train.shape[0])

	parameters, gradients, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

	# Make predictions
	Train_predictions = predict(w, b, X_train)
	Test_predictions = predict(w, b, X_test)

	# Print train/test Errors
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Train_predictions - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Test_predictions - Y_test)) * 100))

	d = {"costs": costs, "Test_predictions": Test_predictions, "Train_predictions" : Train_predictions, "w" : w, "b" : b, "learning_rate" : learning_rate, "num_iterations": num_iterations}
    
	return d




def read_data():
	arr = []
	ans = []
	with open('diabetic_data.csv', 'rb') as f:
		reader = csv.reader(f)
		i = 0
		for row in reader:
			if i == 0:
				i += 1
				continue
			new = []
			row18 = row[18]
			if row18.isdigit():
				row18 = float(row18)
			else:
				row18 = 0

			row19 = row[19]
			if row19.isdigit():
				row19 = float(row19)
			else:
				row19 = 0

			row20 = row[20]
			if row20.isdigit():
				row20 = float(row20)
			else:
				row20 = 0

			new.append(row18)
			new.append(row19)
			new.append(row20)
			arr.append(new)

			result = row[len(row) - 1]
			if result == 'NO':
				result = 0
			else:
				result = 1
			ans.append(result)

	return arr, ans


# We need to read in the training data
def main():

	X_data, Y_data = read_data()
	X_data = np.array(X_data).T
	Y_data = np.array(Y_data)
	Y_data = Y_data.reshape(1, Y_data.shape[0])

	x_train = X_data[ : , : 10000]
	y_train = Y_data[ : , : 10000]

	x_test = X_data[ : , 10000: ]
	y_test = Y_data[ : , 10000: ]
	#print Y_data.shape

	#x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1)

	d = model(x_train, y_train, x_test, y_test)

	costs = np.squeeze(d['costs'])
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(d["learning_rate"]))
	plt.show()


if __name__ == '__main__':
	main()