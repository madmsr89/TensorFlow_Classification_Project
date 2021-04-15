from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


#LABEL DATA AND MINING
CSV_COLUMN_NAMES = ['length', 'breadth', 'height','wheelbase','cartype']
CAR_TYPE = ['Hatchback', 'PremHatch-Sedan', 'SUV']


train_path = "/Users/msr89/Documents/Python_datascience/TensorFlow/Data_inputs/CarsData_Train.csv"
test_path = "/Users/msr89/Documents/Python_datascience/TensorFlow/Data_inputs/CarsData_Test.csv"



train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('cartype')
test_y = test.pop('cartype')
train.shape


#BUILDING AND TRAINING MODEL
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=5)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=2000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


#PREDICT MODEL BASED ON CLASSIFICATION 
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['length', 'breadth', 'height','wheelbase']
predict = {}

print("Please type numeric values as prompted.")

for feature in features:
    print(feature)
    val = input(feature + ": ")
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        CAR_TYPE[class_id], 100 * probability))
    