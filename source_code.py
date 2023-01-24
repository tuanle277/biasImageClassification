from matplotlib.image import imread
import os 
import numpy as np
import pandas as pd
import tensorflow as tf
import timeit

from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_source = "UTKFace"

image_links = os.listdir(image_source)

gender = {1: "female", 0: "male"}
races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}


age_list = [int(x.split("_")[0]) for x in image_links if len(x) > 3]
gender_list = [gender[int(x.split("_")[1])] for x in image_links if len(x) > 3]
race_list = [races[int(x.split("_")[1])] for x in image_links if len(x) > 3]

aa =  np.array([[[[0] * 3] * 200] * 200])


for i in image_links:
    a = Image.open("UTKFace/" + i)
    a = np.asarray(a)
    if len(a.shape) == 3:
        aa = np.concatenate((aa, [a]), axis=0)
    print(aa.shape)

np.save("image_data", aa)    
# loaded_data = np.load("image_data.npy")

# Definig path
# cat_source_dir = "dog_cat_dataset/PetImages/Cat"
# dog_source_dir = "dog_cat_dataset/PetImages/Dog"

# cat_images = os.listdir(cat_source_dir)
# dog_images = os.listdir(dog_source_dir)

# cat_images.remove("Thumbs.db")
# dog_images.remove("Thumbs.db")
# cat_images.remove(".DS_Store")
# dog_images.remove(".DS_Store")

# image_size = 256
# data_size = len(cat_images)

# cat_images = cat_images[:data_size]
# dog_images = dog_images[:data_size]

# cat_data, dog_data = np.array([[[[0] * 3] * image_size] * image_size]), np.array([[[[0] * 3] * image_size] * image_size])
# print(cat_data.shape)

# for a in cat_images:
#     cat = Image.open("dog_cat_dataset/PetImages/Cat/" + a)
#     cat = cat.resize((image_size, image_size))
#     cat = np.asarray(cat)
#     if len(cat.shape) == 3:
#         cat_data = np.concatenate((cat_data, [cat]), axis=0)
#     print(a, cat_data.shape, cat.shape)

# for a in dog_images:
#     dog = Image.open("dog_cat_dataset/PetImages/Dog/" + a)
#     dog = dog.resize((image_size, image_size))	
#     dog = np.asarray(dog)
#     print(a, dog_data.shape, dog.shape)
#     if len(dog.shape) == 3 and dog.shape == (image_size, image_size, 3):
#         dog_data = np.concatenate((dog_data, [dog]), axis=0)

# cat_data = np.delete(cat_data, 0, 0)
# dog_data = np.delete(dog_data, 0, 0)

# print(dog_data.shape, dog_data.shape)

# y = np.concatenate((np.array([x ** 0 for x in range(cat_data.shape[0])]), np.array([x * 0 for x in range(dog_data.shape[0])])))
# x = np.concatenate((cat_data, dog_data))

# print("to Numpy array done")
# print("# ================================================== #")

# print(x.shape, y.shape)

# df_cat = pd.DataFrame(data = {"image": cat_data, "class": [x * 0 for x in range(data_size)]})
# df_dog = pd.DataFrame(data = {"image": dog_data, "class": [x ** 0 for x in range(data_size)]})
# print("to Dataframe done")
# print("# ================================================== #")

# df = pd.concat([df_cat, df_dog])
# df.to_csv("dorc.csv")

# print("to csv file done")

# # for a in cat_images:
#     k = Image.open("dog_cat_dataset/PetImages/Cat/" + a)
#     cat = k.resize((image_size, image_size))
#     cat = np.asarray(cat)
#     cat_data = np.append(cat_data, cat, )

# for a in dog_images:
#     k = Image.open("dog_cat_dataset/PetImages/Dog/" + a)
#     dog = k.resize((image_size, image_size))  
#     dog = np.asarray(dog)
#     dog_data = np.append(dog_data, dog, axis=3)


# for i in range(x.shape[0] // 2, x.shape[0]):
#     x[i] = Image.open("dog_cat_dataset/PetImages/Dog/" + x[i])
#     x[i] = x[i].resize((image_size, image_size))
#     x[i] = np.asarray(x[i])

# # for i in x:
# #     print(i)
# #     print(np.array(i))

# # print(x)
# # print(x.shape, y.shape)


# cat_data, dog_data = np.asarray(cat_data), np.asarray(dog_data)


def get10ClassData(x, y, hParams, flatten=True, proportion=1.0, test_prop = 0.25):
    # == get the data set == #
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train, y_train = x[:int((1 - test_prop) * x.shape[0]):], y[:int((1 - test_prop) * y.shape[0]):]
    x_test, y_test = x[:int(test_prop * x.shape[0]):], y[:int(test_prop * y.shape[0]):]

    # == slice the dataset == #
    x_train = x_train[:int(proportion * x_train.shape[0]):]
    y_train = y_train[:int(proportion * y_train.shape[0]):]
    x_test = x_test[:int(proportion * x_test.shape[0]):]
    y_test = y_test[:int(proportion * y_test.shape[0]):]

    # == print the shape and structure == #
    print(x_train.shape, x_train)
    print(y_train.shape, y_train)
    print(x_test.shape, x_test)
    print(y_test.shape, y_test)

    # == convert to float == #
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255 

    # == flatten == #
    # pass False when called 
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # == Slice for validation data == #
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

    if hParams['valProportion'] != 0.0:
        return x_train, y_train, x_test, y_test, x_val, y_val

    return x_train, y_train, x_test, y_test


def correspondingShuffle(x, y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)

    return shuffled_x, shuffled_y


def writeExperimentalResults(hParams, trainResults, testResults):
    # == open file == #
    f = open("results/" + hParams["experimentName"] + ".txt", 'w')

    # == write in file == #
    f.write(str(hParams) + '\n\n')
    f.write(str(trainResults) + '\n\n')
    f.write(str(testResults))

    # == close file == #
    f.close()

def readExperimentalResults(fileName):
    f = open("results/" + fileName + ".txt",'r')

    # == read in file == #
    data = f.read().split('\n\n')

    # == process data to json-convertible == #
    data[0] = data[0].replace("\'", "\"")
    data[1] = data[1].replace("\'", "\"")
    data[2] = data[2].replace("\'", "\"")

    # == convert to json == #
    hParams = json.loads(data[0])
    trainResults = json.loads(data[1])
    testResults = json.loads(data[2])

    return hParams, trainResults, testResults

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "__" + "__".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)


def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)

def buildValAccuracyPlot(fileNames, title):
    # == get hParams == #
    hParams = readExperimentalResults(fileNames[0])[0]

    # == plot curves with yList being the validation accuracies == #
    plotCurves(x=np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name)[1]['val_accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= title)



def cnnGrayHardcode(dataSubsets, hParams):
    x_train, y_train, x_test, y_test, x_val, y_val = dataSubsets

    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_val = tf.reshape(x_val, (-1, 28, 28, 1))
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))

    # == Shuffle data == #
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)

    # == Sequential Constructor == #
    startTime = timeit.default_timer()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    # == Loss function == #
    lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(True)

    # == fitting == #
    model.compile(loss=lossFunc, metrics=['accuracy'], optimizer=hParams['optimizer'])
    hist = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val) 
                                                if hParams['valProportion']!=0.0 
                                                else None, 
                    epochs=hParams['numEpochs'],
                    verbose=1)  
    trainingTime = timeit.default_timer() - startTime

    # == Evaluation == #
    print('============ one unit 10 class, training set size:', x_train.shape[0], ' =============')
    print(model.summary())
    print('Training time:', trainingTime)
    print(model.evaluate(x_test, y_test))
    hParams['paramCount'] = model.count_params()

    return hist.history, model.evaluate(x_test, y_test)


def getHParams(expName=None):
    # Set up what's the same for each experiment
    hParams = {
        'experimentName': expName,
        'datasetProportion': 1.0,
        'valProportion': 0.1,
        'numEpochs': 20
    }
    shortTest = False # hardcode to True to run a quick debugging test
    if shortTest:
        print("+++++++++++++++++ WARNING: SHORT TEST +++++++++++++++++")
        hParams['datasetProportion'] = 0.0001
        hParams['numEpochs'] = 2

    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams

    if (expName == 'C32_64__d0.0__D128_10__rms'):
        dropProp = 0.0
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.0__D128_10__adam'):
        dropProp = 0.0
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.2__D128_10__rms'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },

    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.2__D128_10__rms_test'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 4, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 4, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.2__D128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.02__D128_10__adam'):
        dropProp = 0.02
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.05__D128_10__adam'):
        dropProp = 0.05
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.1__D128_10__adam'):
        dropProp = 0.1
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.3__D128_10__adam'):
        dropProp = 0.3
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.2__D256_128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [256, 128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.2__D512_256_128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [512, 256, 128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32__d0.2__D128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64_128__d0.2__D128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'


    elif (expName == 'C32_64_128_256__d0.2__D128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 256, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32__d0.2__D128_10__rms'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64_128__d0.2__D128_10__rms'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64_128_256__d0.2__D128_10__rms'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 256, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'test'):
        dropProp = 0.05
        hParams['convLayers'] = [
        {
            'conv_numFilters': 36, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 3, 
            'pool_s': 3,
            'drop_prop': dropProp
        },
        # {
        #     'conv_numFilters': 76, 
        #     'conv_f': 3, 
        #     'conv_p': 'same',
        #     'conv_act': 'relu', 
        #     'pool_f': 3, 
        #     'pool_s': 3,
        #     'drop_prop': dropProp
        # },
    ]
        hParams['denseLayers'] = [1024]
        hParams['optimizer'] = 'adam'

    return hParams

def cnnGray(dataSubsets, hParams):
    x_train, y_train, x_test, y_test, x_val, y_val = dataSubsets

    print(x_train.shape)

    x_train = tf.reshape(x_train, (-1, 32, 32, 3))
    x_val = tf.reshape(x_val, (-1, 32, 32, 3))
    x_test = tf.reshape(x_test, (-1, 32, 32, 3))

    # == Shuffle data == #
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)

    # == Sequential Constructor == #
    startTime = timeit.default_timer()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Resizing(64, 64))
    for layer in hParams['convLayers']:
        model.add(tf.keras.layers.Conv2D(layer['conv_numFilters'], layer['conv_f'], activation=layer['conv_act'], input_shape=(32, 32, 3), padding=layer['conv_p']))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(layer['pool_f'], layer['pool_s'])))
        model.add(tf.keras.layers.Dropout(layer['drop_prop']))

    model.add(tf.keras.layers.Flatten())
    for layer in range(len(hParams['denseLayers'])):
        model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer], activation='relu'))

    model.add(tf.keras.layers.Dense(4, activation="softmax"))

    # == Loss function == #
    lossFunc = tf.keras.losses.SparseCategoricalCrossentropy()

    # == fitting == #
    model.compile(loss=lossFunc, metrics=['accuracy'], optimizer=hParams['optimizer'])
    hist = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val) 
                                                if hParams['valProportion']!=0.0 
                                                else None, 
                    epochs=hParams['numEpochs'],
                    verbose=1)  
    trainingTime = timeit.default_timer() - startTime

    # == Evaluation == #
    print('============ one unit 10 class, training set size:', x_train.shape[0], ' =============')
    print(model.summary())
    print('Training time:', trainingTime)
    print(model.evaluate(x_test, y_test))
    hParams['paramCount'] = model.count_params()

    return hist.history, model.evaluate(x_test, y_test)


def main():
    theSeed = 50
    np.random.seed(theSeed)
    tf.random.set_seed(theSeed)

    hParams = {
        "datasetProportion": 1.0,
        "numEpochs": 20,
        "denseLayers": [128, 10],
        "valProportion": 0.1,
        "experimentName": "128_10e20",
        "optimizer": "adam"
    }

    image_source = "UTKFace"

    image_links = os.listdir(image_source)

    gender = {1: "female", 0: "male"}
    races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}


    age_list = np.array([int(x.split("_")[0]) for x in image_links if len(x) > 3])
    gender_list = np.array([int(x.split("_")[1]) for x in image_links if len(x) > 3])
    race_list = np.array([int(x.split("_")[1]) for x in image_links if len(x) > 3])

    loaded_data = np.load("image_data.npy")

    expNames = [
    
        # 'C32_64__d0.0__D128_10__rms', 
        # 'C32_64__d0.2__D128_10__rms', 
        # 'C32_64__d0.0__D128_10__adam',
        # 'C32_64__d0.2__D128_10__adam',

        # 'C32_64__d0.02__D128_10__adam',
        # 'C32_64__d0.05__D128_10__adam',
        # 'C32_64__d0.1__D128_10__adam',
        # 'C32_64__d0.3__D128_10__adam',

        # 'C32__d0.2__D128_10__adam',
        # 'C32_64_128__d0.2__D128_10__adam', 
        # 'C32_64_128_256__d0.2__D128_10__adam',

        # 'C32_64__d0.2__D256_128_10__adam', 
        # 'C32_64__d0.2__D512_256_128_10__adam'

        # 'C32__d0.2__D128_10__rms',
        # 'C32_64_128__d0.2__D128_10__rms', 
        # 'C32_64_128_256__d0.2__D128_10__rms'
        # 'C32_64__d0.2__D128_10__rms_test'
        "test"

    ]

    dataSubsets = get10ClassData(loaded_data, race_list, hParams, False)
    for currExp in expNames:
        hParams = getHParams(currExp)
        trainResults, testResults = cnnGray(dataSubsets, hParams)
        writeExperimentalResults(hParams, trainResults, testResults)


# main()
fileNames = [
            [
            'C32_64__d0.0__D128_10__adam',
            'C32_64__d0.2__D128_10__adam',
            'C32_64__d0.02__D128_10__adam',
            'C32_64__d0.05__D128_10__adam',
            'C32_64__d0.1__D128_10__adam',
            'C32_64__d0.3__D128_10__adam'
            ],
            # [
            # 'C32_64__d0.0__D128_10__rms', 
            # 'C32_64__d0.2__D128_10__rms', 
            # 'C32_64__d0.0__D128_10__adam',
            # 'C32_64__d0.2__D128_10__adam'
            # ],
            [     
            'C32__d0.2__D128_10__adam',
            'C32_64__d0.2__D128_10__adam', 
            'C32_64_128__d0.2__D128_10__adam', 
            'C32_64_128_256__d0.2__D128_10__adam'
            ], 
            [     
            'C32_64__d0.2__D128_10__adam', 
            'C32_64__d0.2__D256_128_10__adam', 
            'C32_64__d0.2__D512_256_128_10__adam'
            ],
            # [       
            # 'C32__d0.2__D128_10__rms',
            # 'C32_64__d0.2__D128_10__rms', 
            # 'C32_64_128__d0.2__D128_10__rms', 
            # 'C32_64_128_256__d0.2__D128_10__rms'
            # ]
        ]

# for index in range(len(fileNames)):
#   buildValAccuracyPlot(fileNames[index], "val_accuracy_" + str(index))
# buildValAccuracyPlot(fileNames[0], "rms_val_accuracy")


