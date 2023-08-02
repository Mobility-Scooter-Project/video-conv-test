import time
import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from vutils import load_settings
from keras import models, Input, Model
from keras.callbacks import EarlyStopping


settings = load_settings()
labels2int = {b: a for a, b in enumerate(settings["labels"])}

landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    

def get_filenames(folder_path):
    return list(map(lambda y: os.path.join(folder_path, y), filter(lambda x: x[-4:]=='.csv', os.listdir(folder_path))))


#=======================================
########################################
# Description: A function that splits training/testing data appropriately for the time series data
# Input: 
    # df_normal: dataframe of non-augmented coordinate data
    # test_size: the percentage of data to be allocated for test data (decimal number between 0-1)
    # valid_size: the percentage of data to be allocated for validation data (decimal number between 0-1)
    # (OPTIONAL) use_aug: boolean for whether augmented data should be included or not
    # (OPTIONAL) df_aug: dataframe of augmented coordinate data
# Output: 
    # x_train: Input data for model training
    # y_train: Corresponding labels for model training
    # x_valid: Input data for model validation
    # y_valid: Corresponding labels for model validation
    # x_test: Input data for model testing
    # y_test: Corresponding labels for model testing
########################################
#=======================================
def get_train_test_data(df_normal, test_size, valid_size, use_aug=False, df_aug=None):
    ############
    ### Initialization
    #############
    df_input = df_normal.copy()                     # Copy the normal dataframe
    df_target = df_input.pop("label")               # Make a dataframe of just the labels
    groups = {}                                     # Dictionary for each label group (from normal dataframe)
    current_group_label = None                      # Keeps track of the current label group
    current_group = []
    n_aug = {}                                      # Dictionary that keeps track of how many rows of augmented data exist for a corresponding label
    x_train, x_valid, x_test = [], [], []           # Arrays that store training/testing/validation data
    y_train, y_valid, y_test = [], [], []           # Arrays that store corresponding labels to ^^^
    
    
    ############
    ### Iterate through each row in the dataframe + correesponding index "i" for the NORMAL dataframe, and add it to the dictionary
    ############
    for i, row in enumerate(df_input.itertuples(index=False)):
        
        # If there is no label assigned to the current_group_label (first iteration):
        if current_group_label is None:
            # Assign to the label corresponding with ith row
            current_group_label = df_target[i]
        
        # If the current label corresponds to the label of the ith row:
        if current_group_label == df_target[i]:
            # Append row to appropriate group
            current_group.append(row)
            
        
        else:
            groups[current_group_label] = groups.get(current_group_label, [])
            groups[current_group_label].append(current_group)
            current_group_label = df_target[i]
            current_group = []
            
    if len(current_group):
        groups[current_group_label] = groups.get(current_group_label, [])
        groups[current_group_label].append(current_group)


    ###########
    ### (OPTIONAL) Make another dictionary similar to the one above, but of purely augmented data. Include in training dataset
    ###########
    if use_aug:
        
        # ---------- Setup for augmneted data ---------------
        df_input_aug = df_aug.copy()                 # Copy the augmented dataframe
        df_target_aug = df_input_aug.pop("label")        # Make a dataframe of just the labels
        groups_aug = {}                              # Dictionary for each label group (from normal dataframe)
        current_group_label = None                   # Keeps track of the current label group
        current_group = []
    
        # ---------- Same iteration for augmented data -----------------
        for i, row in enumerate(df_input_aug.itertuples(index=False)):
            
            
            # If there is no label assigned to the current_group_label (first iteration):
            if current_group_label is None:
                # Assign to the label corresponding with ith row
                current_group_label = df_target_aug[i]
            
            # If the current label corresponds to the label of the ith row:
            if current_group_label == df_target_aug[i]:
                # Append row to appropriate group
                current_group.append(row)
                
            # Otherwise if the label changes (when moving to next row)
            else:
                
                # Append to the number of rows for the corresponding label
                n_aug[current_group_label] = n_aug.get(current_group_label, 0)
                n_aug[current_group_label] += len(current_group) + 1
                
                # Append the checked rows to the appropriate label group (+ check if there is no elements yet)
                groups_aug[current_group_label] = groups_aug.get(current_group_label, [])
                groups_aug[current_group_label].append(current_group)
                current_group_label = df_target[i]
                current_group = []
                
        # ----------- Handle remaining rows -------------------
        
        # If there are remaining elements to add, add them into the corresponding dictionary
        if len(current_group):
            # Also add remaining rows to the count as needed
            n_aug[current_group_label] += len(current_group)
            
            # Add the rows to the corresponding label, make new empty array if there is none yet
            groups_aug[current_group_label] = groups_aug.get(current_group_label, [])
            groups_aug[current_group_label].append(current_group)
            
            
        # ---------- Add the rows to the training dataset -------------
        for label, group in groups_aug.items():
            combined = [j for i in group for j in i]                        # Create array of all rows of a specific group
            
            # Append the rows of augmented data to the training set
            for i in range(len(combined)):
                (x_train).append(combined[i])
                (y_train).append(label)
        
        print(y_train)
        print("-----------")
        print(len(y_train))
        print("-----------")
        print(n_aug)
        print("-------")
    
    ############
    ### Split the training/testing/validation data
    #############
    for label, group in groups.items():
        # random.shuffle(group)
        combined = [j for i in group for j in i]                        # Create array of all rows of a specific group
        
        n_aug[label] = n_aug.get(label, 0)                              # Double check for labels that exist in normal dataset but not in augmented dataset
        
        n_test = int(len(combined) * test_size)                         # Calculate amount of test data needed
        n_valid = int(len(combined) * valid_size)                       # Calculate amount of validation data needed
        n_train = len(combined) - n_test - n_valid - n_aug[label]       # Calculate amount of training data needed (OPTIONAL: FACTOR IN AUGMENTED DATA) (*NOTE: CALCULATION IS PROBABLY WRONG BUT IT WORKS FOR NOW)
        
        # If there are more augmented data samples than "real" ones
        if n_train < 0:
            n_train = 0                                                     # Reset to zero
            rem = len(combined) - n_test - n_valid                          # Store remaining amount of data
            valid_rem = valid_size/(valid_size+test_size)                   # % of remaining data to include for valid set
            test_rem = test_size/(valid_size+test_size)                     # % of remaining data to include for test set
            
            # Add remaining number of rows
            n_valid += int(rem * valid_rem)
            n_test += int(rem * test_rem)

        # Add rows
        for i in range(len(combined)):
            (
                x_train if i < n_train else x_valid if i < n_train + n_valid else x_test
            ).append(combined[i])
            
            (
                y_train if i < n_train else y_valid if i < n_train + n_valid else y_test
            ).append(label)
            
    return (
        np.array(x_train),
        np.array(y_train),
        np.array(x_valid),
        np.array(y_valid),
        np.array(x_test),
        np.array(y_test),
    )
    
    
    
    

#=======================================
########################################
# Description: A function that gets the total number of rows with their corresponding label, and saves it to a CSV
# Input: 
    # folder_path: path to get data from
    # save_path: path to save the data to
    # csv_name: name of the saved csv file **DON'T INCLUDE ".csv"
# Output: None
########################################
#=======================================
def get_num_labels_in_folder(folder_path, save_path, csv_name):
    ############
    ### Initialization
    #############
    # Create a dictionary containing label name and count corresponding to the label name
    label_dict = {}                                 # Initially empty, no labels have been read yet
    
    # Define directory + file list (follows: https://www.youtube.com/watch?v=_TFtG-lHNHI)
    file_list = get_filenames(folder_path)          # List of all CSV file names
    # numFiles = 0  # Variable to keep track of the number of files
    
    #############
    ### Iteration
    #############
    # Iterate through each csv file (outer loop)
    for path in file_list:
        # numFiles += 1  # Increment to the total # of files
        
        df = pd.read_csv(path)                      # Dataframe from Nth csv file
        
        # Iterate through each element of Nth csv file (inner loop)
        for i in df.index:
            if df["label"][i] in label_dict:            # If a label DOES exists in the dictionary...
                label_dict[df["label"][i]] += 1             # Increment to the corresponding count
            else:                                       # Otherwise (if label does NOT exist)...
                label_dict[df["label"][i]] = 1              # Initialize the count as "1"

    #############
    ### Create dataframe to save
    #############
    final_dict = {"label_name": [], "num_rows": []}     # Dictionary to convert to dataframe
    for key in label_dict:
        final_dict["label_name"].append(key)            # ith label name in ith position of "label_name" array
        final_dict["num_rows"].append(label_dict[key])  # ^ corresponding number of rows

    #############
    ### Save CSV
    #############
    new_df = pd.DataFrame(final_dict)                   # Convert the label dictionary into a dataframe
    os.makedirs(save_path, exist_ok=True)               # Make directory for where to save the file
    new_df.to_csv(save_path + "\\" + csv_name + ".csv", index=False)     # Save the file


# convert landmarks to only selected landmarks
def convert(landmarks):
    result = []
    for index in landmark_indices:
        landmark = landmarks[index]
        """without visibility"""
        result.extend([landmark.x, landmark.y, landmark.z])
        """with visibility"""
        # result.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return result


# offset according to previous frame
def offset(curr, prev):
    """without visibility"""
    result = [a - b for a, b in zip(curr, prev)]
    """with visibility"""
    # result = [v[0] - v[1] if i%4!=3 else v[0] for i, v in enumerate(zip(curr, prev))]
    return result


def convert_df_labels(df1, labels2int):
    df = df1.copy()
    for i in range(len(df)):
        label = df["label"][i]
        df.at[i, "label"] = labels2int[label]
    return df


def split_data_with_label(df, valid_size, test_size):
    df_input = df.copy()
    df_target = df_input.pop("label")
    groups = {}
    current_group_label = None
    current_group = []
    for i, row in enumerate(df_input.itertuples(index=False)):
        if current_group_label is None:
            current_group_label = df_target[i]
        if current_group_label == df_target[i]:
            current_group.append(row)
        else:
            groups[current_group_label] = groups.get(current_group_label, [])
            groups[current_group_label].append(current_group)
            current_group_label = df_target[i]
            current_group = []
    if len(current_group):
        groups[current_group_label] = groups.get(current_group_label, [])
        groups[current_group_label].append(current_group)

    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    for label, group in groups.items():
        # random.shuffle(group)
        combined = [j for i in group for j in i]
        n_test = int(len(combined) * test_size)
        n_valid = int(len(combined) * valid_size)
        n_train = len(combined) - n_test - n_valid
        for i in range(len(combined)):
            (
                x_train if i < n_train else x_valid if i < n_train + n_valid else x_test
            ).append(combined[i])
            (
                y_train if i < n_train else y_valid if i < n_train + n_valid else y_test
            ).append(label)
    return (
        np.array(x_train),
        np.array(y_train),
        np.array(x_valid),
        np.array(y_valid),
        np.array(x_test),
        np.array(y_test),
    )


def split_data_without_label(df, valid_size, test_size):
    df_input = df.copy()
    df_target = df_input.pop("label")
    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    n_test = int(len(df_input) * test_size)
    n_valid = int(len(df_input) * valid_size)
    n_train = len(df_input) - n_test - n_valid
    for i, row in enumerate(df_input.itertuples(index=False)):
        (
            x_train if i < n_train else x_valid if i < n_train + n_valid else x_test
        ).append(row)
        (
            y_train if i < n_train else y_valid if i < n_train + n_valid else y_test
        ).append(df_target[i])
    return [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]


def split_data(DATA, VALID_RATIO, TEST_RATIO):
    DBs = [
        pd.read_csv(name, index_col=0) for name in DATA
    ]
    DB = pd.concat(DBs, axis=0, ignore_index=True, sort=False)
    DB = convert_df_labels(DB, labels2int)

    return split_data_without_label(DB, VALID_RATIO, TEST_RATIO)


def group_data(data, group_size, target_function):
    x, y = data
    x_result = []
    y_result = []
    x_temp = []
    y_temp = []
    for i in x:
        x_temp.append(i)
        if len(x_temp) == group_size:
            x_result.append(x_temp)
            x_temp = []
    for i in y:
        y_temp.append(i)
        if len(y_temp) == group_size:
            # result.append(sum(y_temp) / group_size / 2)
            y_result.append(target_function(y_temp))
            y_temp = []

    return np.array(x_result), np.array(y_result)




class ModelOperation:
    def __init__(
        self,
        model_class,
        data,
        max_epochs=100,
        valid_ratio=0.1,
        test_ratio=0.1,
        early_stop_valid_patience=10,
        early_stop_train_patience=5,
        num_train_per_config=10,
        loss='mse',
        metrics=['mse'],
        verbose=0,
        test_data=None
    ):
        self.max_epochs = max_epochs
        self.early_stop_valid_patience = early_stop_valid_patience
        self.early_stop_train_patience = early_stop_train_patience
        self.num_train_per_config = num_train_per_config
        self.loss= loss
        self.metrics = metrics
        self.verbose = verbose

        self.counter = 0
        self.model_class = model_class
        self.base_model = model_class.model
        self.preprocess = False
        self.preprocessor = None
        self.layer_options = [None] * len(self.base_model.layers)

        # x_train, y_train, x_valid, y_valid, x_test, y_test
        self.raw_data = split_data(data, valid_ratio, test_ratio)
        self.test_data = test_data


        self.defalut_params = {
            "batchsize": 16,
            "timestamp": 32,
            "optimizer": "adam",
            "preprocess": None,
        }

        self.model = None
        self.final_data = None
        self.params = self.defalut_params
        self.history = None

    def run(self):
        raise Exception("<run> method must be defined for ModelOperation")

    def build(self):
        # Reconstruct model
        layers = self.base_model.layers
        input_shape = self.final_data[0][0].shape[1:]
        print(self.final_data[0][0].shape)
        input_layer = Input(shape=input_shape)
        current_layer = input_layer
        for i, option in enumerate(self.layer_options[1:]):
            layer = layers[i + 1]
            config = layer.get_config()
            if option is not None:
                for k, v in option.items():
                    config[k] = v
            current_layer = layer.__class__(**config)(current_layer)
        model = Model(inputs=input_layer, outputs=current_layer)
        if self.verbose:
            model.summary()
        return model

    def train(self, clean_model):
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = self.final_data
        model = models.clone_model(clean_model)
        model.compile(
            optimizer=self.params.get("optimizer"), loss=self.loss, metrics=self.metrics
        )
        batchsize = self.params.get("batchsize")
        history = model.fit(
            x_train,
            y_train,
            epochs=self.max_epochs,
            validation_data=(x_valid, y_valid),
            batch_size=batchsize,
            callbacks=[
                EarlyStopping(
                    monitor="loss",
                    patience=self.early_stop_train_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                    # start_from_epoch=8,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stop_valid_patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                    # start_from_epoch=8,
                ),
            ],
            verbose=self.verbose,
            shuffle=False,
        )
        self.epochs_record = history.history
        epochs = len(history.history["loss"])
        loss = model.evaluate(x_train, y_train, batch_size=batchsize, verbose=0)[0]
        val_loss = model.evaluate(x_valid, y_valid, batch_size=batchsize, verbose=0)[0]
        test_loss = []
        if self.test_data is not None:
            timestamp = self.params.get("timestamp")
            for test in self.test_data:
                x_test, y_test = group_data(test, timestamp, self.model_class.target_function)
                test_loss.append(model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0])
        elif len(x_test)>0:
            test_loss.append(model.evaluate(x_test, y_test, batch_size=batchsize, verbose=0)[0])
        self.model = model
        return epochs, loss, val_loss, test_loss


class ModelTest(ModelOperation):
    def __init__(self, model_class, data, options, *args, **kwargs):
        super().__init__(model_class=model_class, data=data, *args, **kwargs)
        self.final_options = [
            (k, (v if isinstance(v, list) else [v]))
            for k, v in options.items()
            if not (isinstance(v, list) and len(v) == 0)
        ]
        for name1, param1 in self.defalut_params.items():
            found = False
            for name2, param2 in self.final_options:
                if name1 == name2:
                    found = True
            if not found:
                self.final_options.append((name1, [param1]))
        self.current_options = [None] * len(self.final_options)

    def process_options(self):
        self.final_data = list(self.raw_data)
        self.params = {}
        for i in range(len(self.layer_options)):
            self.layer_options[i] = None
        for i, (name, options) in enumerate(self.final_options):
            option_idx = self.current_options[i]
            option = options[option_idx]
            if name == "preprocess" and option is not None:
                for i in range(3):
                    if self.test_data and i==2: continue
                    self.final_data[i] = option.transform(self.final_data[i])
            if name[:5] == "layer":
                layer_number = int(name[5:])
                self.layer_options[layer_number] = option
            self.params[name] = option
        timestamp = self.params.get("timestamp")
        for i in range(3):
            self.final_data[i] = group_data(self.final_data[i], timestamp, self.model_class.target_function)

    def run(self):
        self.history = []
        self.test(0)
        output_path = os.path.join("test_results", str(int(time.time())) + ".csv")
        pd.DataFrame(
            data=self.history,
            columns=list(next(zip(*self.final_options)))
            + ["avg_epochs", "avg_loss", "avg_valid_loss"]+ [f"avg_test_loss_{i}" for i in range(len(self.test_data))],
        ).to_csv(output_path)

    def test(self, option_idx):
        if option_idx == len(self.final_options):
            return self.build_and_train()
        name, options = self.final_options[option_idx]
        for i, v in enumerate(options):
            self.current_options[option_idx] = i
            self.test(option_idx + 1)

    def build_and_train(self):
        self.process_options()
        print("=================================================================")
        [
            print(f"{name:12}: {self.params.get(name) or 'No Change'}")
            for name in self.params.keys()
        ]
        print()
        model = self.build()
        # model.summary()
        train_results = []
        # labels = ["round", "epochs", "train", "valid", "test"]
        # print("{:>8} {:>8} {:>8} {:>8} {:>8}".format(*labels))
        for i in range(self.num_train_per_config):
            record = self.train(model)
            record = list(record[:-1]) + list(record[-1])
            train_results.append(record)
            # print("{:8} {:8.0f} {:8.4f} {:8.4f} {:8.4f}".format(i, *record))
        record = [sum(i) / len(i) for i in zip(*train_results)]
        print(("{:>8} {:8.0f}"+" {:8.4f}"*(len(record)-1)).format("avg", *record))
        self.history.append(
            [self.params.get(name) or "No Change" for name in self.params.keys()]
            + record
        )
        print("-----------------------------------------------------------------\n")


class ModelTrain(ModelOperation):
    def __init__(self, model_class, data, options, *args, **kwargs):
        super().__init__(model_class=model_class, data=data, *args, **kwargs)
        for name, param in options.items():
            self.params[name] = param

        option = self.params.get("preprocess")
        self.final_data = list(self.raw_data)
        if option is not None:
            for i in range(3):
                if self.test_data and i==2: continue
                self.final_data[i] = option.transform(self.final_data[i]) 
        for i in range(3):
            self.final_data[i] = group_data(
                self.final_data[i], self.params.get("timestamp"), self.model_class.target_function
            )

    def run(self):
        print("=================================================================")
        [
            print(f"{name:12}: {self.params.get(name) or 'No Change'}")
            for name in self.params.keys()
        ]
        print()
        model = self.build()
        train_results = []
        # labels = ["round", "epochs", "train", "valid", "test"]
        # print("{:>8} {:>8} {:>8} {:>8} {:>8}".format(*labels))
        models = []
        # for i in range(self.num_train_per_config):
        record = self.train(model)
        train_results.append(record)
        print(record)
            # print("{:8} {:8.0f} {:8.4f} {:8.4f} {:8.4f}".format(i, *record))
            # models.append(self.model)
        try:
            # number = int(input("Enter the round number to save model: "))
            # self.save_model(models[number], self.epochs_record)
            self.save_model(self.model, self.epochs_record)
        except Exception as e:
            print(e)
            print("Model not saved.")
        print("-----------------------------------------------------------------\n")


    def save_model(self, model, record):
        join = os.path.join
        model_path = join("model", str(int(time.time())))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        models.save_model(model, join(model_path, 'model.h5'))
        output_path = join(model_path, 'history.csv')
        pd.DataFrame(data=record).to_csv(output_path)
        with open(join(model_path, 'info.txt'), 'w') as f:
            # labels = ["epochs", "train", "valid", "test"]
            # f.write("{:>8} {:>8} {:>8} {:>8}\n".format(*labels))
            # f.write("{:8.0f} {:8.4f} {:8.4f} {:8.4f}\n\n".format(*record))
            [f.write(f'{str(k)}: {str(v)}\n') for k, v in self.params.items()]
        print(f"Model saved to <{model_path}>.")

