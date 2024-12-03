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

        
        # Filter out zero rows
        non_zero_row_indices = sparse_matrix.getnnz(axis=1) > 0
        sparse_matrix = sparse_matrix[non_zero_row_indices]


        print(f"Sparse matrix created with shape {sparse_matrix.shape}")
        return sparse_matrix
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

    print(f"Random seed set to: {args.seed}")

    # Placeholder for LSH implementation
    print("LSH implementation goes here...")

if __name__ == "__main__":
    main()
