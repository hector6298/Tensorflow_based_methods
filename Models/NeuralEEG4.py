from __future__ import print_function
# ALL INFO, WARNINGS and  GPU debbuging information ARE OFF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
import csv
from sklearn.model_selection import KFold
import KARA


#gpu memory alloc 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

word_matrix = KARA.all_wrd_matrix('EEG_feat','prompts.mat')
dataset,feat_names = KARA.make_table(word_matrix)

def preprocess_features (EEG_featureSet):
    
    selected_features = dataset[feat_names]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(EEG_featureSet):
    output_targets = EEG_featureSet[['class']]
    return output_targets

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features ])


# input pipe
def input_function(features,targets, batch_size =1, shuffle= True, num_epochs= None):
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = tf.data.Dataset.from_tensor_slices((features,targets)) #2gb limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)
    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels

def nn_classification_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

    periods = 10
    steps_per_period = steps/periods

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    classifier = tf.estimator.DNNClassifier(
        feature_columns =construct_feature_columns(training_examples),
        n_classes = 11,
        hidden_units = hidden_units,
        optimizer = optimizer,
        
        
    )
    training_input_fn = lambda: input_function(training_examples, 
                                          training_targets["class"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: input_function(training_examples, 
                                                  training_targets["class"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: input_function(validation_examples, 
                                                    validation_targets["class"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range (0, periods):
        classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period)
  
        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,11)
        
        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,11)    
    
        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
     #   Remove event files to save disk space.
   
  
    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
  
  
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    #  Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()
  
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    print([final_predictions, validation_targets])
    
   
    return classifier, accuracy


#K-fold operation area

examples = preprocess_features(dataset)
targets = preprocess_targets(dataset)
acc_arr = np.array([])
accuracy = 0

kf = KFold(n_splits=10) # Define the split - into 2 folds 
for train_index, test_index in kf.split(examples):
    
    training_examples, validation_examples = examples.iloc[train_index], examples.iloc[test_index]
    training_targets, validation_targets = targets.iloc[train_index].astype(int), targets.iloc[test_index].astype(int)
    #np.savetxt('dummy_folder\dummy.csv',training_examples,delimiter = ',')
    dnn_classifier, accuracy = nn_classification_model(
    learning_rate=0.05,
    steps=2000,
    batch_size=10,
    hidden_units=[10, 10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
    acc_arr = np.append(acc_arr,accuracy )
print (acc_arr)  


