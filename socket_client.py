import socket
import json
import argparse
import os
import threading

# Function to send data to the server
def send_data_to_server(mode, bin_file_path):
    # Server IP and port hardcoded (localhost and port 65432)
    server_ip = '127.0.0.1'
    server_port = 65432
    
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        client_socket.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port} for file {bin_file_path}")

        # Read the binary (text) file in 'r' mode (as a string)
        with open(bin_file_path, 'r') as bin_file:
            text_data = bin_file.read()

        # Extract the file name
        bin_file_name = os.path.basename(bin_file_path)

        # Prepare the message with mode, file name, and file data
        message = {
            'mode': mode,
            'file_name': bin_file_name,  # Include the file name in the message
            'data': text_data
        }

        # Convert the message to JSON format
        message_json = json.dumps(message)

        # Send the JSON-encoded data to the server
        client_socket.sendall(message_json.encode('utf-8'))
        print(f"Sent data from {bin_file_path} to the server in {mode} mode.")

    except Exception as e:
        print(f"Error occurred while sending {bin_file_path}: {e}")

    finally:
        # Close the connection
        client_socket.close()

# Function to handle multiple files with multithreading
def send_all_files_in_directory(mode, directory):
    print(f"Scanning directory: {directory}")
    
    # List to hold all threads
    threads = []

    # Use os.walk to traverse the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Filter for .bin files
        bin_files = [f for f in files if f.endswith('.bin')]
        
        for bin_file in bin_files:
            # Get the full path to the .bin file
            file_path = os.path.join(root, bin_file)
            
            # Create a thread for each .bin file
            thread = threading.Thread(target=send_data_to_server, args=(mode, file_path))
            threads.append(thread)

            # Start the thread
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All files from the directory and subdirectories have been sent to the server.")
# Parse CLI arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send multiple .bin files to the server concurrently.")

    # Required arguments
    parser.add_argument('mode', type=str, choices=['test', 'train'], help="Mode of operation: test or train")
    parser.add_argument('directory', type=str, help="Directory containing the .bin files")

    args = parser.parse_args()

    # Send all .bin files from the specified directory to the server using multithreading
    send_all_files_in_directory(args.mode, args.directory)
