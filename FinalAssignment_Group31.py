import argparse
import numpy as np
from scipy.sparse import csr_matrix

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="LSH implementation for Netflix dataset.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the user_movie_rating.npy file.")
    return parser.parse_args()

# Set the random seed for reproducibility
def set_random_seed(seed):
    np.random.seed(seed)

# Load the dataset from the provided path and create a sparse matrix
def load_data_sparse(file_path):
    try:
        # Load raw data
        data = np.load(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {data.shape}")

        # Extract user IDs and movie IDs
        user_ids = data[:, 0]
        movie_ids = data[:, 1]

        # Binary representation (1 for rated, 0 for not rated)
        binary_ratings = np.ones(len(user_ids), dtype=np.int8)

        # Create a sparse matrix
        num_users = user_ids.max()
        num_movies = movie_ids.max()
        sparse_matrix = csr_matrix((binary_ratings, (movie_ids - 1, user_ids - 1)), shape=(num_movies, num_users))

        return sparse_matrix
    except Exception as e:
        print(f"Error loading the data or creating sparse matrix: {e}")
        return None
    
def LSH_Imp (sparse_matrix):
    try:
        # Create Permutation Matrix of 
        
        PS = 100 #Permutation Size 
        mov_size = sparse_matrix.shape[0]
        usr_size = sparse_matrix.shape[1]

        #Create the Permutation Matrix with size 100 * Movies Number
        Perm_matrix = np.array([np.random.permutation(mov_size) + 1 for _ in range(PS)]).T
        print(f"Perm_matrix shape: {Perm_matrix.shape}")


        # Initialize the signature matrix with zeros values
        signature_matrix = np.zeros((PS, usr_size))

        # Compute MinHash signatures
        for perm_idx in range(PS):  # Iterate over range(PS Values)
            
            for user_idx in range(usr_size):
                # Extract the permutation vector for the current permutation index
                perm_vector = Perm_matrix[:, perm_idx]

                # Extract the sparse vector for the current user (convert to dense for element-wise multiplication)
                user_vector = sparse_matrix[:, user_idx].toarray().flatten()

                # Perform element-wise multiplication
                Pem_Spars = perm_vector * user_vector

                # Replace zeros with a high number (at least as high as the number of movies)
                Pem_Spars[Pem_Spars == 0] = mov_size + 1

                # Get the minimum value for the current user
                min_value = np.min(Pem_Spars)
                signature_matrix[perm_idx, user_idx] = min_value

            print(f"Current cycle for permutation: {perm_idx + 1} of {PS}")

        print("MinHash signatures computed.")
        return signature_matrix

    except Exception as e:
        print(f"Error loading the data or creating sparse matrix: {e}")
        return None


def main():
    # Parse arguments
    args = parse_arguments()
    set_random_seed(args.seed)

    # Load the user_movie_rating.npy file and convert to a sparse binary matrix
    sparse_matrix = load_data_sparse(args.input_file)
    if sparse_matrix is None:
        return


    print(f"MinHash signature matrix shape: {sparse_matrix.shape}")

    print(f"Random seed set to: {args.seed}")

    # Compute MinHash signatures
    minhash_signatures = LSH_Imp(sparse_matrix)

    print(f"MinHash signature matrix shape: {minhash_signatures.shape}")

if __name__ == "__main__":
    main()
