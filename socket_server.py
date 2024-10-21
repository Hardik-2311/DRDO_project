import socket
import json
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Placeholder for NIST test functions and XGBoost model
from tests.FrequencyTest import FrequencyTest
from tests.RunTest import RunTest
from tests.Matrix import Matrix
from tests.Spectral import SpectralTest
from tests.TemplateMatching import TemplateMatching
from tests.Universal import Universal
from tests.Complexity import ComplexityTest
from tests.Serial import Serial
from tests.ApproximateEntropy import ApproximateEntropy
from tests.CumulativeSum import CumulativeSums
from tests.RandomExcursions import RandomExcursions


# Color codes for terminal print
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"

MAX_WORKERS = 20


# remove .bin
def remove_bin_extension(file_name):
    return file_name.rsplit(".", 1)[0]


# tests
def features_extraction(binary_data):
    tests = {
        "Frequency (Monobit) Test": FrequencyTest.monobit_test(binary_data),
        "Block Frequency Test": FrequencyTest.block_frequency(binary_data),
        "Run Test": RunTest.run_test(binary_data),
        "Longest Run of Ones in a Block Test": RunTest.longest_one_block_test(
            binary_data
        ),
        "Binary Matrix Rank Test": Matrix.binary_matrix_rank_text(binary_data),
        "Discrete Fourier Transform (Spectral) Test": SpectralTest.spectral_test(
            binary_data
        ),
        "Non-Overlapping Template Matching Test": TemplateMatching.non_overlapping_test(
            binary_data
        ),
        "Overlapping Template Matching Test": TemplateMatching.overlapping_patterns(
            binary_data
        ),
        "Universal Statistical Test": Universal.statistical_test(binary_data),
        "Linear Complexity Test": ComplexityTest.linear_complexity_test(binary_data),
        "Serial Test": Serial.serial_test(binary_data),
        "Approximate Entropy Test": ApproximateEntropy.approximate_entropy_test(
            binary_data
        ),
        "Cumulative Sums (Forward) Test": CumulativeSums.cumulative_sums_test(
            binary_data, 0
        ),
        "Cumulative Sums (Backward) Test": CumulativeSums.cumulative_sums_test(
            binary_data, 1
        ),
        "Random Excursions Test": RandomExcursions.random_excursions_test(binary_data),
        "Random Excursions Variant Test": RandomExcursions.variant_test(binary_data),
    }
    return tests


# for p values
def compute_final_p_value(result):
    """Extracts and aggregates p-values from the test result."""
    if isinstance(result, list):
        if isinstance(result[0], tuple):
            p_values = [
                r[3]
                for r in result
                if len(r) > 3 and isinstance(r[3], (float, np.float64))
            ]
            return sum(p_values) / len(p_values) if p_values else 0.0

        return sum(result) / len(result) if result else 0.0

    if isinstance(result, tuple):
        return result[0] if isinstance(result[0], (float, np.float64)) else 0.0

    return result if isinstance(result, (float, np.float64)) else 0.0


# Function to process binary data and predict the class
def predict_class(binary_data, file_name):
    start_time = time.time()  # Start timing

    # Load the pre-trained XGBoost model
    loaded_model = xgb.Booster()
    loaded_model.load_model("xgboost_model.model")

    print(f"{GREEN}XGBoost model loaded for file: {file_name}.{RESET}")

    # Perform NIST tests
    print(f"{CYAN}Running NIST tests for file: {file_name}...{RESET}")
    test_start_time = time.time()  # Start timing NIST tests
    test_results = features_extraction(binary_data)
    result = [compute_final_p_value(value) for key, value in test_results.items()]
    print(
        f"{MAGENTA}NIST tests completed in {time.time() - test_start_time:.2f} seconds for file: {file_name}.{RESET}"
    )
    
    print(np.array(result)) 
    
    if np.sum(np.array(result)<0.01)>=3:
        print("Your bin sequence is Non random")
        return
    X_test = pd.DataFrame(
        [result], columns=list(test_results.keys()) 
    )
    dtest = xgb.DMatrix(X_test)
    y_pred = loaded_model.predict(dtest)
    y_prob = loaded_model.predict(dtest)
    print(y_pred)

    # Print and log the predicted class with the file name
    # predicted_class = int(y_pred[0]) + 1
    predicted_class = np.argmax(y_prob[0]) + 1

    # Confidence of the prediction (highest probability value)
    confidence_score = y_prob[0][np.argmax(y_prob[0])]

    print(f"{BLUE}Predicted class for file {file_name}: {predicted_class}{RESET}")
    print(
        f"{MAGENTA}Class prediction completed in {time.time() - start_time:.2f} seconds for file: {file_name}.{RESET}"
    )
    print(f"{CYAN}Confidence score: {confidence_score * 100:.2f}%{RESET}")


# how to train the model using new .bin file
def train_data(binary_data, file_name):
    train_result = features_extraction(binary_data)
    result = [compute_final_p_value(value) for key, value in train_result.items()]
    result_copy = result.copy()
    loaded_model = xgb.Booster()
    loaded_model.load_model("xgboost_model.model")

    # Predict the class using XGBoost
    X_test = pd.DataFrame([result], columns=list(train_result.keys()))
    dtest = xgb.DMatrix(X_test)
    y_pred = loaded_model.predict(dtest)

    y_pred = np.argmax(y_pred[0])

    # Print and log the predicted class with the file name
    predicted_class = (y_pred) + 1


    print(f"{GREEN}Predicted class: {predicted_class}{RESET}")

    result_copy.append(predicted_class)
    file_name = remove_bin_extension(file_name)
    result_copy.append(file_name)
    result_string = ",".join(map(str, result_copy))
    
    # append the newly added data
    append_to_csv("final.csv", result_string)

    # retrain your model
    retrain_model(result_copy)


# retrain the model with updated csv
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
    loaded_model.load_model("xgboost_model.model")

    test_results = new_data[:16]  # First 16 elements are the test results
    class_label = int(new_data[16]) - 1  # The 17th value is the class label (adjust to be 0-indexed)

    print(f"{CYAN}Training on new data with features: {test_results} and class: {class_label + 1}{RESET}")

    # Get feature importance from the model (for calculating the weighted sum)
    importance_dict = loaded_model.get_score(importance_type="gain")
    test_names = [
        "Frequency (Monobit) Test", "Block Frequency Test", "Run Test", "Longest Run of Ones in a Block Test",
        "Binary Matrix Rank Test", "Discrete Fourier Transform (Spectral) Test",
        "Non-Overlapping Template Matching Test", "Overlapping Template Matching Test",
        "Universal Statistical Test", "Linear Complexity Test", "Serial Test", 
        "Approximate Entropy Test", "Cumulative Sums (Forward) Test", 
        "Cumulative Sums (Backward) Test", "Random Excursions Test", "Random Excursions Variant Test"
    ]


    X_train = pd.DataFrame([test_results], columns=test_names)
    
    y_train = pd.Series([class_label])  # Target class

    # Convert to DMatrix for XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set parameters (use the same ones used during initial training)
    params = {
        "max_depth": 2,
        "eta": 0.3,
        "objective": "multi:softprob",
        "num_class": 7,  # Adjust based on the number of classes
    }

    # Incrementally train the model with the new data
    print(f"{YELLOW}Updating XGBoost model with new data...{RESET}")
    loaded_model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=loaded_model)

    # Save the updated model
    loaded_model.save_model("xgboost_model.model")
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

            print(f"Received mode: {mode}, data size: {len(binary_data_content)}, file: {file_name}")

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
    PORT = 65432
    start_server(HOST, PORT)
