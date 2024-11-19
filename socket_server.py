import socket
import json
import os
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading

# Placeholder for  test functions and XGBoost model
from features.feature_2 import feature_2
from features.feature_5 import feature_5
from features.feature_3 import feature_3
from features.feature_7 import feature_7
from features.feature_8 import feature_8
from features.feature_9 import feature_9
from features.feature_12 import feature_12
from features.feature_6 import feature_6
from features.feature_1 import feature_1
from features.feature_10 import feature_10
from features.feature_4 import feature_4


# Color codes for terminal print
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"

MAX_WORKERS = 20

# Define the ranges for each class as specified
ranges = {
    1: (0.95, 1.0),
    2: (0.90, 0.95),
    3: (0.85, 0.90),
    4: (0.80, 0.85),
    5: (0.75, 0.80),
    6: (0.70, 0.75),
    7: (0.00, 0.70)
}


# remove .bin
def remove_bin_extension(file_name):
    return file_name.rsplit(".", 1)[0]


model_lock = threading.Lock()
model_cache = {"model": None, "last_loaded_time": None}

# return always new model
def load_model():
    """
    Load the latest XGBoost model. If the model is already loaded and up-to-date, reuse it.
    """
    global model_cache

    model_path = "xgboost_model.model"
    model_load_time = time.ctime(os.path.getmtime(model_path))  # Get model file modification time
    
    with model_lock:
        # Check if the model needs to be reloaded
        if model_cache["model"] is None or model_cache["last_loaded_time"] != model_load_time:
            print(f"{CYAN}Loading updated model from disk...{RESET}")
            model_cache["model"] = xgb.Booster()
            model_cache["model"].load_model(model_path)
            model_cache["last_loaded_time"] = model_load_time
        else:
            print(f"{GREEN}Using cached model loaded at {model_cache['last_loaded_time']}.{RESET}")

        return model_cache["model"]

# features
def features_extraction(binary_data):
    features = {
        "feature_1": feature_2.monobit(binary_data),
        "feature_2": feature_2.block_feature_2(binary_data),
        "feature_3": feature_5.run(binary_data),
        "feature_4": feature_5.longest_one_block(binary_data),
        "feature_5": feature_3.binary_matrix_rank(binary_data),
        "feature_6": feature_7.feature_7(binary_data),
        "feature_7": feature_8.non_overlapping(binary_data),
        "feature_8": feature_8.overlapping_patterns(binary_data),
        "feature_9": feature_9.statistical(binary_data),
        "feature_10": feature_12.linear_feature_12(binary_data),
        "feature_11": feature_6.feature_6(binary_data),
        "feature_12": feature_1.approximate_entropy(binary_data),
        "feature_13": feature_10.cumulative_sums(binary_data, 0),
        "feature_14": feature_10.cumulative_sums(binary_data, 1),
        "feature_15": feature_4.random_excursions(binary_data),
        "feature_16": feature_4.variant(binary_data),
    }
    return features

# for p values
def compute_final_p_value(result):
    """Extracts and aggregates p-values from the test result according to the NIST suite logic."""
    if isinstance(result, list) and isinstance(result[0], tuple):
        # Select the p-value for the state '+1.0'
        selected_state = '+1' 
        for r in result:
            if r[0] == selected_state:
                #print(f"Debug: Selected State: {selected_state}, p-value: {r[3]}")  # Debug print
                return r[3]  # Return the p-value for the selected state
        
        # If the selected state is not found, return the p-value at index 0
        #print(f"Debug: Selected State not found, returning p-value from index 0: {result[0][3]}")
        return result[0][3]
    
    elif isinstance(result, (float, np.float64)):
        return result
    elif isinstance(result, tuple):
        return result[0] if isinstance(result[0], (float, np.float64)) else 0.0

    return 0.0


def classify_data_based_on_ranges_list(results, minimum_passing=6):
    """
    Classify data based on the number of passing tests for each class range.

    Args:
    - results (list): A list of computed p-values, each corresponding to a feature.
    - minimum_passing (int): Minimum number of passing tests required to consider a class.

    Returns:
    - int: Predicted class based on passing test counts.
    """
    # Initialize a counter for each class based on the ranges
    class_passing_counts = {i: 0 for i in ranges.keys()}
    
    # Tally the count of features that fall into each class range
    for p_value in results:
        for class_label, (low, high) in ranges.items():
            if low <= p_value <= high:
                class_passing_counts[class_label] += 1
                break  # Stop once the range is matched

    # Print count for each class
    print("Class counts based on p-value ranges:")
    for class_label, count in class_passing_counts.items():
        print(f"Class {class_label}: {count} features")

    # Calculate total counts for classes 1-6 and for class 7
    classes_1_to_6_count = sum(class_passing_counts[i] for i in range(1, 7))
    class_7_count = class_passing_counts[7]
    
    print("Total counts:")
    print(f"Classes 1 to 6: {classes_1_to_6_count}")
    print(f"Class 7: {class_7_count}")

    if classes_1_to_6_count >= minimum_passing and classes_1_to_6_count > class_7_count:
        # Identify the highest numbered class with the maximum count within classes 1-6
        max_count = max(class_passing_counts[i] for i in range(1, 7))
        highest_class_with_max_count = max(i for i in range(1, 7) if class_passing_counts[i] == max_count)
        predicted_class = highest_class_with_max_count
    elif class_7_count > classes_1_to_6_count and class_7_count >= minimum_passing:
        predicted_class = 7
    elif classes_1_to_6_count == class_7_count and classes_1_to_6_count >= minimum_passing:
        # If tie, choose highest numbered class among classes 1-6
        max_count = max(class_passing_counts[i] for i in range(1, 7))
        highest_class_with_max_count = max(i for i in range(1, 7) if class_passing_counts[i] == max_count)
        predicted_class = highest_class_with_max_count
    else:
        predicted_class = None  # Return None if no class meets the minimum passing criteria

    return predicted_class


def predict_class(binary_data, file_name):
    start_time = time.time()  # Start timing
    real_start_time = datetime.now().strftime("%H:%M")
    print(f"Process started at: {real_start_time}")
    
    # Load the pre-trained XGBoost model
    loaded_model = load_model()
    
    # Extract features and compute p-values
    test_start_time = time.time()
    test_results = features_extraction(binary_data)
    result = [compute_final_p_value(value) for key, value in test_results.items()]
    print(f"{MAGENTA}Features completed in {time.time() - test_start_time:.2f} seconds for file: {file_name}.{RESET}")
    print(np.array(result))
    
    # Create test data
    X_test = pd.DataFrame([result], columns=list(test_results.keys()))
    dtest = xgb.DMatrix(X_test)
    
    # AI Prediction
    y_prob = loaded_model.predict(dtest)
    predicted_class_ai = np.argmax(y_prob[0]) + 1
    confidence_score = y_prob[0][np.argmax(y_prob[0])]
    
    # Helper function prediction
    predicted_class_helper = classify_data_based_on_ranges_list(result)
    
    print(f"{BLUE}Predicted class (AI): {predicted_class_ai}{RESET}")
    print(f"{CYAN}Confidence score: {confidence_score * 100:.2f}%{RESET}")
    print(f"{GREEN}Predicted class (Helper Function): {predicted_class_helper}{RESET}")
    
    # Compare predictions
    if predicted_class_ai != predicted_class_helper:
        print(f"{YELLOW}Mismatch detected: AI predicted {predicted_class_ai}, Helper predicted {predicted_class_helper}. Retraining...{RESET}")
        
        # Retrain the model with the helper function's class
        retrain_data = result + [predicted_class_helper]  # Add the helper class label
        retrain_model(retrain_data)
        retrain_data.append(file_name)
        result_string = ",".join(map(str, retrain_data))
        append_to_csv("combined_file.csv", result_string)
        print(f"{GREEN}Model retrained successfully with the helper's prediction.{RESET}")
    else:
        print(f"{GREEN}Predictions matched: No retraining needed.{RESET}")
    
    real_end_time = datetime.now().strftime("%H:%M")
    print(f"Process ended at: {real_end_time}")

# how to train the model using new .bin file
def train_data(binary_data, file_name):
    train_result = features_extraction(binary_data)
    result = [compute_final_p_value(value) for key, value in train_result.items()]
    result_copy = result.copy()
    loaded_model = xgb.Booster()
    loaded_model.load_model("xgboost_model.model")
    X_test = pd.DataFrame([result], columns=list(train_result.keys()))
    dtest = xgb.DMatrix(X_test)
    y_pred = loaded_model.predict(dtest)
    y_pred = np.argmax(y_pred[0])
    predicted_class = (y_pred) + 1
    print(f"{GREEN}Predicted class: {predicted_class}{RESET}")
    result_copy.append(predicted_class)
    file_name = remove_bin_extension(file_name)
    result_copy.append(file_name)
    result_string = ",".join(map(str, result_copy))
    append_to_csv("combined_file.csv", result_string)
    retrain_model(result_copy)
    

def retrain_model(new_data):
    """
    Train the XGBoost model with the new provided data.

    Args:
    - new_data (list): A list containing features and class for training.
                       The last element in the list is the class label.

    Returns:
    - None
    """

    # Load the existing model to get the feature importance
    loaded_model = xgb.Booster()
    loaded_model.load_model("xgboost_model.model")  # Use forward slash here

    test_results = new_data[:16]  # First 16 elements are the test results
    class_label = (
        int(new_data[16]) - 1
    )  # The 17th value is the class label (adjust to be 0-indexed)

    print(
        f"{CYAN}Training on new data with features: {test_results} and class: {class_label + 1}{RESET}"
    )

    # Get feature importance from the model (for calculating the weighted sum)
    importance_dict = loaded_model.get_score(importance_type="gain")
    test_names = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
    "feature_10",
    "feature_11",
    "feature_12",
    "feature_13",
    "feature_14",
    "feature_15",
    "feature_16"
]

    X_train = pd.DataFrame([test_results], columns=test_names)

    y_train = pd.Series([class_label])  # Target class

    # Convert to DMatrix for XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set parameters (use the same ones used during initial training)
    params = {
        "max_depth": 4,
        "eta": 0.1,
        "objective": "multi:softprob",
        "num_class": 7,
        'eval_metric': 'mlogloss',
    }

    # Incrementally train the model with the new data
    print(f"{YELLOW}Updating XGBoost model with new data...{RESET}")
    loaded_model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=loaded_model)

    # Save the updated model
    loaded_model.save_model("xgboost_model.model")  # Use forward slash here
    print(f"{GREEN}Model updated and saved successfully with the new data.{RESET}")


# update the csv
def append_to_csv(file_name, result_string):
    """
    Append the result string to the CSV file. If the file does not exist, create it and add the header.

    Args:
    - file_name (str): Path to the CSV file.
    - result_string (str): The result data string to append (should match the column count in the header).

    Returns:
    - None
    """
    # Open the file in append mode
    with open(file_name, "a") as f:
        # Write the result string as a new line
        f.write(result_string + "\n")


def handle_client(client_socket, client_address):
    try:
        print(f"{GREEN}Connection from {client_address} established.{RESET}")
        client_socket.settimeout(10)  # Set timeout for the client socket

        # Read data from the client
        binary_data = b""
        while True:
            chunk = client_socket.recv(1024)
            if not chunk:
                break
            binary_data += chunk

        if binary_data:
            # Decode the JSON-encoded message
            message = json.loads(binary_data.decode("utf-8"))
            mode = message.get("mode", "default")

            binary_data_content = message.get("data", "")
            file_name = message.get("file_name", "unknown_file")

            print(
                f"Received mode: {mode}, data size: {len(binary_data_content)}, file: {file_name}"
            )

            if mode == "test":
                predict_class(binary_data_content, file_name)
            elif mode == "train":
                train_data(binary_data_content, file_name)
        else:
            print(f"{RED}No data received from client.{RESET}")

    except socket.timeout:
        print(f"{RED}Connection from {client_address} timed out.{RESET}")
    except Exception as e:
        print(f"{RED}Error occurred: {e}{RESET}")
    finally:
        client_socket.close()
        print(f"{GREEN}Connection with {client_address} closed.{RESET}")


# Function to set up the server
def start_server(host, port):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"{BLUE}Server started on {host}:{port}. Waiting for clients...{RESET}")

        while True:
            try:
                client_socket, client_address = server_socket.accept()
                print(f"{GREEN}Client {client_address} connected.{RESET}")
                executor.submit(handle_client, client_socket, client_address)

            except Exception as e:
                print(f"{RED}Error occurred: {e}{RESET}")
                break

        server_socket.close()
        print(f"{MAGENTA}Server closed.{RESET}")


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 6548
    start_server(HOST, PORT)
