import argparse
import numpy as np
from scipy.sparse import csr_matrix

# Parse command-line arguments
# EXAMPLE: python3 FinalAssignment_Group31.py --seed 42 --input_file /Users/amrshata/Desktop/RES/user_movie_rating.npy  --output_file result.txt
def parse_arguments():
    parser = argparse.ArgumentParser(description="LSH implementation for Netflix dataset.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the user_movie_rating.npy file.")
    parser.add_argument('--output_file', type=str, default="result.txt", help="Path to the output file.")
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
    
def LSH_Imp (sparse_matrix, PS = 120):
    try:
        # Create Permutation Matrix of 
        

        mov_size = sparse_matrix.shape[0]
        usr_size = sparse_matrix.shape[1]

        #Create the Permutation Matrix with size 100 * Movies Number
        Perm_matrix = np.array([np.random.permutation(mov_size) + 1 for _ in range(PS)]).T
        print(f"Perm_matrix shape: {Perm_matrix.shape}")




        # Initialize the signature matrix with zeros values
        signature_matrix = np.zeros((PS, usr_size))

        # Compute MinHash signatures
        for perm_idx in range(PS):  # Iterate over permutations
            # Current permutation of movie indices
            perm_vector = Perm_matrix[:, perm_idx]  # 1-based indices

            # Convert to 0-based indices
            perm_vector = perm_vector - 1

            # Reorder the sparse matrix rows based on the permutation
            permuted_matrix = sparse_matrix[perm_vector, :]

            # Get the index of the first non zero 
            first_nonzero_indices = np.argmax(permuted_matrix != 0, axis=0)+1

            signature_matrix[perm_idx,] = first_nonzero_indices
        

            print(f"Processed permutation {perm_idx + 1}/{PS}")

        print("MinHash signatures computed.")
        return signature_matrix
    
    except Exception as e:
        print(f"Error loading the data or creating sparse matrix: {e}")
        return None
    
    
def apply_banding(signature_matrix, bands=10, rows_per_band=12):

    try:
        num_permutations, num_users = signature_matrix.shape  # Rows = num_permutations, Columns = num_users
        candidate_pairs = set()  # Set to store unique candidate pairs (user1, user2)

        # Iterate over each band
        for band in range(bands):
            start_row = band * rows_per_band  # Start index of the current band
            end_row = start_row + rows_per_band  # End index of the current band

            # Dictionary to store hashed buckets for the current band
            band_hashes = {}

            # Iterate over all users (columns in the signature matrix)
            for user in range(num_users):
                # Extract the mini-signature (rows belonging to this band) for the current user
                band_signature = tuple(signature_matrix[start_row:end_row, user])
                
                # Hash the mini-signature to determine the bucket
                bucket = hash(band_signature)

                # Add the user to the corresponding bucket
                if bucket in band_hashes:
                    # If other users are already in the bucket, add pairs (user, candidate)
                    for candidate in band_hashes[bucket]:
                        # Ensure pairs are ordered as (min, max) to avoid duplicates
                        candidate_pairs.add((min(user, candidate), max(user, candidate)))
                    band_hashes[bucket].append(user)  # Add current user to the bucket
                else:
                    # Create a new bucket with the current user
                    band_hashes[bucket] = [user]

        # Print the total number of candidate pairs found
        print(f"Number of candidate pairs: {len(candidate_pairs)}")
        return candidate_pairs  # Return the set of candidate pairs
    except Exception as e:
        # Handle errors gracefully
        print(f"Error in banding technique: {e}")
        return set()

 # Calculate Jaccard Similarity
def calculate_jaccard_similarity(sparse_matrix, candidate_pairs, threshold=0.5):

    try:
        similar_pairs = []  # List to store pairs with similarity above the threshold


        sparse_matrix_csc = sparse_matrix.tocsc()

        # Iterate over all candidate pairs
        for u1, u2 in candidate_pairs:
            # Retrieve the set of movies rated by user u1 and user u2
            movies_u1 = set(sparse_matrix_csc[:, u1].indices)  # Indices of non-zero entries for user u1
            movies_u2 = set(sparse_matrix_csc[:, u2].indices)  # Indices of non-zero entries for user u2

            # Compute the intersection and union of the movie sets
            intersection = len(movies_u1 & movies_u2)  # Number of common movies rated by both users
            union = len(movies_u1 | movies_u2)         # Total number of unique movies rated by either user

            # Calculate Jaccard Similarity
            jaccard_similarity = intersection / union

            # If similarity exceeds the threshold, add the pair to the result
            if jaccard_similarity > threshold:
                similar_pairs.append((u1 + 1, u2 + 1, jaccard_similarity))   # Convert to 1-based indexing as required
            
            # Sort the pairs by similarity score in descending order
            similar_pairs.sort(key=lambda x: x[2], reverse=True)

        # Print the total number of similar pairs found
        print(f"Number of similar pairs: {len(similar_pairs)}")
        return similar_pairs  # Return the list of similar pairs
    except Exception as e:
        # Handle errors gracefully and return an empty list
        print(f"Error in Jaccard Similarity calculation: {e}")
        return []   
    
# Write results to a file
def write_results(output_file, similar_pairs):
    try:
        with open(output_file, 'w') as f:
            f.write(f"U1,U2,Jaccard Similarity\n")
            for u1, u2, score in similar_pairs:
                f.write(f"{u1},{u2},{score:.4f}\n")  # Write user1, user2, and similarity score
        print(f"Results written to {output_file}")
    except Exception as e:
        print(f"Error writing results to file: {e}")


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
    minhash_signatures = LSH_Imp(sparse_matrix, 120)

    print(f"MinHash signature matrix shape: {minhash_signatures.shape}")

    candidate_pairs = apply_banding(minhash_signatures,14,8)
    
    # Compute Jaccard Similarity for candidate pairs
    similar_pairs = calculate_jaccard_similarity(sparse_matrix, candidate_pairs)

    write_results(args.output_file, similar_pairs)

if __name__ == "__main__":
    main()
