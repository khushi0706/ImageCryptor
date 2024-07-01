from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from scipy.integrate import solve_ivp

app = Flask(__name__)

# Directories for uploads and processed images
UPLOAD_FOLDER = 'uploads'
ENCRYPTED_FOLDER = 'uploads/encrypt'
DECRYPTED_FOLDER = 'uploads/decrypt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENCRYPTED_FOLDER'] = ENCRYPTED_FOLDER
app.config['DECRYPTED_FOLDER'] = DECRYPTED_FOLDER

for folder in [UPLOAD_FOLDER, ENCRYPTED_FOLDER, DECRYPTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def logistic_map(r, x0, size):
    sequence = [x0]
    for _ in range(1, size):
        x_next = r * sequence[-1] * (1 - sequence[-1])
        sequence.append(x_next)
    return sequence

def shuffle_pixels(image_array, sequence):
    flat_image = image_array.flatten()
    indices = sorted(range(len(sequence)), key=lambda i: sequence[i])
    shuffled = flat_image[indices]
    return shuffled.reshape(image_array.shape)

def unshuffle_pixels(shuffled_array, sequence):
    flat_image = shuffled_array.flatten()
    indices = sorted(range(len(sequence)), key=lambda i: sequence[i])
    unshuffled = [0] * len(flat_image)
    for i, idx in enumerate(indices):
        unshuffled[idx] = flat_image[i]
    return np.array(unshuffled).reshape(shuffled_array.shape)

def load_image(image_path):
    with Image.open(image_path) as img:
        r, g, b = img.split()
        return np.array(r), np.array(g), np.array(b)

def generate_vectors(initial_state, parameters, transient_steps, total_steps, step_size):
    a, b, c, d, k = parameters
    def system_equations(t, state):
        x, y, z, w = state
        return [a * (y - x), -x*z + d*x + c*y - w, x*y - b*z, x + k]
    t_transient = np.arange(0, transient_steps * step_size, step_size)
    transient_result = solve_ivp(system_equations, [t_transient[0], t_transient[-1]], initial_state, t_eval=t_transient, method='RK45')
    new_initial_state = transient_result.y[:, -1]
    t_vector_gen = np.arange(0, total_steps * step_size, step_size)
    vector_result = solve_ivp(system_equations, [t_vector_gen[0], t_vector_gen[-1]], new_initial_state, t_eval=t_vector_gen, method='RK45')
    return vector_result.y[0], vector_result.y[1], vector_result.y[2], vector_result.y[3]

def create_matrices(x_vector, y_vector, z_vector, w_vector, M, N, K):
    X = np.zeros((M, N), dtype=np.uint8)
    Y = np.zeros((M, N), dtype=np.uint8)
    Z = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            index = i * N + j
            X[i, j] = np.floor((x_vector[index] * 1013 + (z_vector[index] * 1013) % 1) * 10 * 13) % 256
            Y[i, j] = np.floor((y_vector[index] * 1013 + (z_vector[index] * 1013) % 1) * 10 * 13) % K
            Z[i, j] = np.floor((x_vector[index] * 1013 + w_vector[index] * 1013 + (w_vector[index] * 1013) % 1) * 10 * 13) % 256
    return X, Y, Z

# Confusion functions
def generate_sbox(sequence, size=256):
    sbox = sorted(range(size), key=lambda i: sequence[i % len(sequence)])
    return sbox

def confusion(matrix, sbox):
    substituted_matrix = np.vectorize(lambda x: sbox[x])(matrix)
    return substituted_matrix

def reverse_confusion(matrix, sbox):
    reverse_sbox = np.argsort(sbox)
    reversed_matrix = np.vectorize(lambda x: reverse_sbox[x])(matrix)
    return reversed_matrix

def diffuse_image(P, X, Y, Z, r1, r2, r3):
    M, N = P.shape
    A = np.zeros_like(P)
    A[0, 0] = (P[0, 0] + X[0, 0] + r1 + r2 * r3) % 256
    for j in range(1, N):
        A[0, j] = (P[0, j] + A[0, j-1] + X[0, j] + (r2 + j) % r3) % 256
    for i in range(1, M):
        A[i, 0] = (P[i, 0] + A[i-1, 0] + X[i, 0] + (r3 + i) % r2) % 256
    for i in range(1, M):
        for j in range(1, N):
            A[i, j] = (P[i, j] + A[i, j-1] + A[i-1, j] + X[i, j] + (i*j) % (r2 + r3)) % 256
    return A

def reverse_diffusion(A, X, r1, r2, r3, M, N):
    P = np.zeros_like(A)
    P[0, 0] = (A[0, 0] - X[0, 0] - r1 - r2 * r3) % 256
    for j in range(1, N):
        P[0, j] = (A[0, j] - A[0, j-1] - X[0, j] - (r2 + j) % r3) % 256
    for i in range(1, M):
        P[i, 0] = (A[i, 0] - A[i-1, 0] - X[i, 0] - (r3 + i) % r2) % 256
    for i in range(1, M):
        for j in range(1, N):
            P[i, j] = (A[i, j] - A[i, j-1] - A[i-1, j] - X[i, j] - (i*j) % (r2 + r3)) % 256
    return P


# Define the second_diffusion function as provided
def second_diffusion(B, Z, r3, M, N):
    C = np.zeros_like(B)  # Initialize matrix C for the ciphered image
    C[M-1, N-1] = (B[M-1, N-1] + Z[M-1, N-1] + r3) % 256
    for j in range(N-2, -1, -1):
        C[M-1, j] = (B[M-1, j] + C[M-1, j+1] + Z[M-1, j]) % 256
    for i in range(M-2, -1, -1):
        C[i, N-1] = (B[i, N-1] + C[i+1, N-1] + Z[i, N-1]) % 256
    for i in range(M-2, -1, -1):
        for j in range(N-2, -1, -1):
            C[i, j] = (B[i, j] + C[i, j+1] + C[i+1, j] + Z[i, j]) % 256
    return C

# Define the inverse_diffusion_step_two function as provided
def inverse_diffusion_step_two(C, Z, r3, M, N):
    B = np.zeros_like(C)
    B[M-1, N-1] = (C[M-1, N-1] - Z[M-1, N-1] - r3 + 256) % 256
    for j in range(N-2, -1, -1):
        B[M-1, j] = (C[M-1, j] - C[M-1, j+1] - Z[M-1, j] + 256) % 256
    for i in range(M-2, -1, -1):
        B[i, N-1] = (C[i, N-1] - C[i+1, N-1] - Z[i, N-1] + 256) % 256
    for i in range(M-2, -1, -1):
        for j in range(N-2, -1, -1):
            B[i, j] = (C[i, j] - C[i, j+1] - C[i+1, j] - Z[i, j] + 256 * 2) % 256
    return B



def process_component(component, r1, r2, r3, M, N, K, operation='encrypt'):
    logistic_sequence = logistic_map(3.99, 0.5, M * N)
    sbox = generate_sbox(logistic_sequence)
    initial_state = [0.3838, 0.9876, 32.1234, 0.6565]
    parameters = (36, 3, 28, 16, 0.2)
    transient_steps = 10000
    total_steps = M * N
    step_size = 0.0001
    x_vector, y_vector, z_vector, w_vector = generate_vectors(initial_state, parameters, transient_steps, total_steps, step_size)
    X, Y, Z = create_matrices(x_vector, y_vector, z_vector, w_vector, M, N, K)
    
    if operation == 'encrypt':
        shuffled_component = shuffle_pixels(component, logistic_sequence)
        confused_component = confusion(shuffled_component, sbox)
        diffused1_component = diffuse_image(confused_component, X, Y, Z, r1, r2, r3)
        encrypted_component = second_diffusion(diffused1_component, Z, r3, M, N)
        return encrypted_component
    else:  # 'decrypt'
        pre_deconfused_component = inverse_diffusion_step_two(component, Z, r3, M, N)
        decrypted_diffusion1_component = reverse_diffusion(pre_deconfused_component, X, r1, r2, r3, M, N)
        deconfused_component = reverse_confusion(decrypted_diffusion1_component, sbox)
        unshuffled_component = unshuffle_pixels(deconfused_component, logistic_sequence)
        return unshuffled_component


def save_image_color(R, G, B, file_path):
    color_image_array = np.stack((R, G, B), axis=-1)
    color_image = Image.fromarray(color_image_array.astype('uint8'), 'RGB')
    color_image.save(file_path, format='PNG')

# Encryption and decryption functions remain unchanged
# Encrypt image function
def encrypt_image_function(input_path, encrypted_output_path, r1, r2, r3):
    r_matrix, g_matrix, b_matrix = load_image(input_path)
    M, N = r_matrix.shape
    K = max(M, N)
    A_r = process_component(r_matrix, r1, r2, r3, M, N, K, 'encrypt')
    A_g = process_component(g_matrix, r1, r2, r3, M, N, K, 'encrypt')
    A_b = process_component(b_matrix, r1, r2, r3, M, N, K, 'encrypt')
    save_image_color(A_r, A_g, A_b, encrypted_output_path)

# Decrypt image function
def decrypt_image_function(input_path, decrypted_output_path, r1, r2, r3):
    A_r, A_g, A_b = load_image(input_path)  # Assume this loads the encrypted image properly
    M, N = A_r.shape  # Assuming all channels have the same shape
    K = max(M, N)  # This is used in process_component but not needed for reverse_diffusion directly
    P_r = process_component(A_r, r1, r2, r3, M, N, K, 'decrypt')
    P_g = process_component(A_g, r1, r2, r3, M, N, K, 'decrypt')
    P_b = process_component(A_b, r1, r2, r3, M, N, K, 'decrypt')
    save_image_color(P_r, P_g, P_b, decrypted_output_path)
# Flask routes for uploading and processing images
@app.route('/')
def index():
    #return render_template('index.html')
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/encrypt', methods=['POST'])
def encrypt_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        encrypted_output_path = os.path.join(app.config['ENCRYPTED_FOLDER'], filename)
        file.save(input_path)

        # Read secret keys from the form
        r1 = int(request.form['r1'])
        r2 = int(request.form['r2'])
        r3 = int(request.form['r3'])

        # Encrypt the image
        encrypt_image_function(input_path, encrypted_output_path, r1, r2, r3)

        # Return or display the encrypted image
        return send_from_directory(app.config['ENCRYPTED_FOLDER'], filename)

@app.route('/decrypt', methods=['POST'])
def decrypt_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        decrypted_output_path = os.path.join(app.config['DECRYPTED_FOLDER'], filename)
        file.save(input_path)

        # Read secret keys from the form
        r1 = int(request.form['r1'])
        r2 = int(request.form['r2'])
        r3 = int(request.form['r3'])

        # Decrypt the image
        decrypt_image_function(input_path, decrypted_output_path, r1, r2, r3)

        # Return or display the decrypted image
        return send_from_directory(app.config['DECRYPTED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
