####################################################################
# Data Eng 414 - Python Practice                                   #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2023                                 #
#                                                                  #
# This skeleton aims to help get familiar with python, loading     #
# data into memory, making predictions with models, and            #
# calculating accuracy                                             #
####################################################################
####################################################################
#               (1) Import numpy and matplotlib                    #
####################################################################

#************************* YOUR CODE HERE *************************#

import numpy as np
import matplotlib.pyplot as plt

#******************************************************************#

####################################################################
#   (2) Read in the MNIST dataset from files data/train.csv and    #
#       data/test.csv and store as numpy variables with shapes:    #                                         
#                   train : (60000,784)                            #
#                   test : (10000,784)                             #
#                   train targets : (60000,1)                      #
#                   test targets : (10000,1)                       #
#                                                                  #
#    Note: * open data/train.csv to see how the data is stored:    #
#          target,pixel_0_0,pixel_0_1,...,pixel_28_27,pixel_28_28  #
#                                                                  #
#          * The MNIST data set consists of 28 x 28 pixel          #
#          handwritten digits where each pixel has a value         #
#          between 0 and 255                                       #
####################################################################

#************************* YOUR CODE HERE *************************#

train = np.genfromtxt('data/train.csv', delimiter=',')
train_targets = train[:, 0]
train = train[:, 1:]

test = np.genfromtxt('data/test.csv', delimiter=',')
test_targets = test[:, 0]
test = test[:, 1:]

#******************************************************************#

####################################################################
#       (3) Print the shapes of the training and test sets         #
####################################################################

#************************* YOUR CODE HERE *************************#

print("train shape: ", train.shape)
print("test shape: ", test.shape)
print("train targets shape: ", train_targets.shape)
print("test targets shape: ", test_targets.shape)

#******************************************************************#

####################################################################
#           (4) Normalise the dataset to the range 0,1             #
####################################################################

#************************* YOUR CODE HERE *************************#

min_train = np.min(np.min(train))
max_train = np.max(np.max(train))
print("Min train: ", min_train)
print("Max train: ", max_train)

min_test = np.min(np.min(test, axis=1))
max_test = np.max(np.max(test))
print("Min test: ", min_test)
print("Max test: ", max_test)

train = (train - min_train) / (max_train - min_train)
test = (test - min_test) / (max_test - min_test)

print("Normalised train min: ", np.min(np.min(train, axis=1)))
print("Normalised train max: ", np.max(np.max(train, axis=1)))
print("Normalised test shape: ", test.shape)
print("Normalised train shape: ", train.shape)

#******************************************************************#

####################################################################
#            (5) Plot an example from the loaded datset            #
####################################################################

#************************* YOUR CODE HERE *************************#

example = train[0, :]
example = np.reshape(example, (28, 28))
plt.imshow(example, cmap = 'gray')
plt.title('Example at index 0 from Training data')
plt.axis('off')
plt.show()

#******************************************************************#

####################################################################
#      (6) Load in weights from a logistic regression model from   #
#           the file data/weights.txt                              #
#                                                                  #
#          * Store the weights as a numpy array called weights     #
#            the shape of the array should be (785, 10)            #
####################################################################

#************************* YOUR CODE HERE *************************#

weights = np.genfromtxt('data/weights.csv', delimiter=',')
print("weights shape:", weights.shape)

#******************************************************************#

####################################################################
#  (7) Use the loaded weights to create a model by uncommenting    #
#      the following lines, then use the model to make a           #
#      prediction on an example from the test set                  #
#      * For an interesting result look at index 7                 #
#                                                                  #
#       Note: When feeding input to the model make sure the shape  #
#             of the input is a row vector (1,784)                 #    
#              i.e. taking a row from the X matrix whose shape     #
#                   is (N,D)                                       #
#                                                                  #
#                                                                  #
#       * Load the model weights by using the class function       #
#       model.load() - for more information call help(model.load)  #
####################################################################

#*********************** UNCOMMENT THIS  **************************#

from models import SoftmaxRegression
model = SoftmaxRegression()

#help(model)

#******************************************************************#


#************************* YOUR CODE HERE *************************#

model.load(weights)
input = np.reshape(test[7], (1,784))
y = model(input).reshape(-1)
print("Prediction: ")
x = np.arange(0,10,1)

for num in x:
    element = y[num]
    print(f"Prob({num}): {element}")

#******************************************************************#


####################################################################
#   (8) Plot a bar graph of the model probabilities                #
#                                                                  #
#       * Also plot which number you are trying predict            #
####################################################################

#************************* YOUR CODE HERE *************************#

plt.bar(x, y)
plt.title("Probabilities for index 7")
plt.show()

#******************************************************************#

####################################################################
#        (9) Define a function to calculate accuracy               #
#                                                                  #
#       * Then calculate the average for the test set              #
#                                                                  #
#     * Note: You can retrieve the model prediction by finding     #
#       the index whose probability is the maximum in the array    #
#      this can be accomplished using numpys argmax() function     #
#          prediction_id = np.argmax([0,0.1,0.2,0.5,0.2])          #
#          Results in prediction_id = 3                            #
#                                                                  #
#                                                                  #
#     * Additional note: If you have two arrays with equal shapes  #
#         you can find the elements are equal by using numpys      #
#               equal function.                                    #
#          tar = [1,2,3,4]                                         #
#          pred = [4,3,3,4]                                        #
#          eq = np.equal(pred,tar)                                 #
#           Results in:                                            #
#           eq = [False,False,True,True]                           #
#                                                                  #
#              Hint: np.sum(eq) is a quick way to count the        #
#                     number of correct predictions                #
#                                                                  #
####################################################################

#************************* YOUR CODE HERE *************************#
def accuracy(true_data, pred_data):
    """
    The function will read in true data and predicted data from a model
    and compare them to test the model's accuray.
    :param true_data: True outputs provided in the data
    :param pred_data: Outputs predicted by the model
    :return: Accuracy of the model
    """
    pred_out = np.argmax(pred_data, axis=1)
    eq = np.equal(pred_out, true_data)
    correct = np.sum(eq)
    return correct/len(true_data)

acc = accuracy(test_targets, model.predict(test))
print(f"Accuracy of model: {acc*100}%")

#******************************************************************#