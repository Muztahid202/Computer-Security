from BitVector import *
import hashlib
import importlib.util
import sys
import os
import time

file_path = os.path.join(os.path.dirname(__file__), "2005067_bitvector-demo.py")


module_name = "bitvectordemo"


spec = importlib.util.spec_from_file_location(module_name, file_path)
bitvectordemo = importlib.util.module_from_spec(spec)
sys.modules[module_name] = bitvectordemo
spec.loader.exec_module(bitvectordemo)

# print(bitvectordemo.Sbox)
# print(bitvectordemo.InvSbox)

def format_hex_bytes(data):
    """Format bytes as space-separated hex values"""
    return ' '.join(f"{b:02x}" for b in data)

#function for counting the no of bytes given a hex value
def count_bytes(hex_value):
    """
    Counts the number of bytes in a given hex value.
    """
    return len(bytes.fromhex(hex_value))  # Convert hex to bytes and count the length

def print_state_matrix(state):
    """
    Print state matrix in hex format
    Args:
        state: 4x4 state matrix
    """
    for row in state:
        print(' '.join(f"{byte:02X}" for byte in row))

#print the key in this format : print(f"Round {round_num}:", ' '.join(f"{b:02X}" for b in round_key))
def print_key(key):
    """
    Print the key in hex format
    Args:
        key: The key (bytes)
    """
    print(' '.join(f"{b:02X}" for b in key))


def get_plaintext():
    """Get plaintext of any length (will be padded automatically)"""
    plaintext = input("Enter your plaintext (any length): ")
    return plaintext.encode('utf-8')

def bytes_to_state_matrix(key_bytes):
    """
    Convert 16-byte key/state to a 4x4 column-major matrix
    Args:
        key_bytes: 16-byte input
    Returns:
        4x4 list of lists (state matrix)
    """
    state = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            state[j][i] = key_bytes[i*4 + j]
    return state

def state_matrix_to_bytes(state):
    """
    Convert 4x4 state matrix back to bytes
    Args:
        state: 4x4 list of lists
    Returns:
        16-byte output
    """
    bytes_out = bytearray(16)
    for i in range(4):
        for j in range(4):
            bytes_out[i*4 + j] = state[j][i]
    return bytes(bytes_out)    

def select_aes_mode():
    """Prompt user to select AES mode (128, 192, or 256)"""
    while True:
        mode = input("Select AES mode (128, 192, or 256): ")
        if mode in ['128', '192', '256']:
            return int(mode)
        print("Invalid mode. Please enter 128, 192, or 256.")


def get_user_key_input(mode):
    """
    Prompts user for key input
    Args:
        mode: 128, 192, or 256 (key size in bits)
    Returns:
        User input as string
    """
    while True:
        key = input(f"Enter your AES-{mode} key (any length): ").strip()
        if key:
            return key
        print("Error: Key cannot be empty")

def process_aes_key_with_hash(key_str, mode):
    """
    Processes key using cryptographic hashing
    Args:
        key_str: Raw key input
        mode: 128, 192, or 256
    Returns:
        Properly sized key bytes derived via SHA-256
    """
    required_lengths = {128: 16, 192: 24, 256: 32}
    req_len = required_lengths[mode]
    
    # Hash the key to get fixed length key
    hashed = hashlib.sha256(key_str.encode('utf-8')).digest()
    return hashed[:req_len]
    
def pad_data(data):
    """Pad data to be multiple of 16 bytes using PKCS#7 padding"""
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)

def unpad_data(data):
    """Remove PKCS#7 padding"""
    pad_len = data[-1]
    return data[:-pad_len]

def generate_iv():
    """Generate a random 16-byte initialization vector"""
    return os.urandom(16)

def generate_round_constants(mode):
    """
    Generates AES round constants based on mode
    Returns a list of 32-bit BitVector words in the format [rc_i, 0x00, 0x00, 0x00]
    """
    rounds = {128: 10, 192: 12, 256: 14}
    num_rounds = rounds[mode]
    
    # Initializing rcon array (rcon[0] is unused as in the round 0 we have to xor with the original key)
    rcon = [None] * (num_rounds + 1)
    
    # First round constant (i=1)
    rcon[1] = 0x01
    
    # Generate subsequent round constants
    for i in range(2, num_rounds + 1):
        if rcon[i-1] < 0x80:
            rcon[i] = (2 * rcon[i-1]) & 0xFF
        else:
            rcon[i] = ((2 * rcon[i-1]) ^ 0x11B) % 0x100
    
    # Convert to BitVector format
    round_constants = []
    for i in range(num_rounds + 1):
        if i == 0:
            round_constants.append(BitVector(intVal=0, size=32))
        else:
            bv = BitVector(size=32)
            bv[0:8] = BitVector(intVal=rcon[i], size=8)
            bv[8:16] = BitVector(intVal=0, size=8)
            bv[16:24] = BitVector(intVal=0, size=8)
            bv[24:32] = BitVector(intVal=0, size=8)
            round_constants.append(bv)
    
    return round_constants

def key_expansion(key, mode):
    """
    Perform AES key expansion for the specified mode
    Args:
        key: The original key (bytes)
        mode: 128, 192, or 256
    Returns:
        List of 32-bit words
    """

    if mode == 128:
        N, R = 4, 11
    elif mode == 192:
        N, R = 6, 13
    elif mode == 256:
        N, R = 8, 15
    
    # Initialize W[0..4R-1]
    W = [None] * (4 * R)
    
    # First N words are just the key
    for i in range(N):
        word = BitVector(size=32)
        for j in range(4):
            word[8*j:8*(j+1)] = BitVector(intVal=key[4*i + j], size=8)
        W[i] = word

    # Generate round constants
    rcon = generate_round_constants(mode)
    
    # Expand the key
    for i in range(N, 4*R):
        if i < N:
            pass
        elif i >= N and i % N == 0:
            # RotWord + SubWord + Rcon
            bytes_list = [W[i-1][8*j:8*(j+1)] for j in range(4)]
            rotated_bytes = bytes_list[1:] + bytes_list[:1]
            
            rotated = BitVector(size=32)
            for j in range(4):
                rotated[8*j:8*(j+1)] = rotated_bytes[j]
            
            substituted = BitVector(size=32)
            for j in range(4):
                byte = rotated[8*j:8*(j+1)].intValue()
                substituted[8*j:8*(j+1)] = BitVector(intVal=bitvectordemo.Sbox[byte], size=8)
            
            W[i] = W[i-N] ^ substituted ^ rcon[i//N]
        elif i >= N and N > 6 and i % N == 4:
            # AES-256 specific case
            substituted = BitVector(size=32)
            for j in range(4):
                byte = W[i-1][8*j:8*(j+1)].intValue()
                substituted[8*j:8*(j+1)] = BitVector(intVal=bitvectordemo.Sbox[byte], size=8)
            W[i] = W[i-N] ^ substituted
        else:
            # Default case
            W[i] = W[i-N] ^ W[i-1]
    
    return W

def get_round_key(expanded_key, round_num):
    """
    Get the round key for a specific round
    Args:
        expanded_key: The expanded key (list of 32-bit words)
        round_num: Round number (0-10 for AES-128)
    Returns:
        16-byte round key
    """
    round_key = BitVector(size=0)
    # The words need to be concatenated in order and so we have to do it
    for i in range(4*round_num, 4*round_num + 4):
        round_key += expanded_key[i]
    
    # Convert to bytes while maintaining proper byte order
    round_key_bytes = []
    for word in expanded_key[4*round_num:4*round_num+4]:
        for j in range(0, 32, 8):
            round_key_bytes.append(word[j:j+8].intValue())
    
    return bytes(round_key_bytes)

def add_round_key(state, round_key):
    """
    Perform AddRoundKey operation
    Args:
        state: Current state (128-bit block) as a BitVector
        round_key: Round key (128 bits) as bytes
    Returns:
        New state after XOR with round key
    """
    # Here we are converting round key to bitvector
    round_key_bv = BitVector(size=0)
    for byte in round_key:
        round_key_bv += BitVector(intVal=byte, size=8)
    
    # XOR state with round key
    return state ^ round_key_bv

def sub_bytes(state_matrix):
    """
    Perform SubBytes transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after SubBytes transformation
    """
    new_state = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            byte = state_matrix[i][j]
            new_state[i][j] = bitvectordemo.Sbox[byte]
    return new_state

def shift_rows(state_matrix):
    """
    Perform ShiftRows transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after ShiftRows transformation
    """
    new_state = [row.copy() for row in state_matrix] 
    
    # Row 0: No shift
    # Row 1: Shift left by 1
    new_state[1] = [state_matrix[1][1], state_matrix[1][2], state_matrix[1][3], state_matrix[1][0]]
    
    # Row 2: Shift left by 2
    new_state[2] = [state_matrix[2][2], state_matrix[2][3], state_matrix[2][0], state_matrix[2][1]]
    
    # Row 3: Shift left by 3 
    new_state[3] = [state_matrix[3][3], state_matrix[3][0], state_matrix[3][1], state_matrix[3][2]]
    
    return new_state

def mix_columns(state_matrix):
    """
    Perform MixColumns transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after MixColumns transformation
    """
    new_state = [[0 for _ in range(4)] for _ in range(4)]
    AES_modulus = BitVector(bitstring='100011011')  # AES irreducible polynomial x^8 + x^4 + x^3 + x + 1(GF(2^8) tend to forget:'))
    
    for col in range(4):
        # Get the current column as a list of BitVectors
        column = [BitVector(intVal=state_matrix[row][col], size=8) for row in range(4)]
        
        # Multiply with the Mixer matrix
        for row in range(4):
            # Initialize result for this cell
            result = BitVector(intVal=0, size=8)
            
            # Multiply and accumulate each element
            for i in range(4):
                term = bitvectordemo.Mixer[row][i].gf_multiply_modular(
                    column[i], 
                    AES_modulus, 
                    8
                )
                result = result ^ term
            
            new_state[row][col] = result.intValue()
    
    return new_state
    

def aes_encrypt(plaintext, expanded_key, mode):
    """
    Perform AES encryption on a single 128-bit block
    Args:
        plaintext: 16-byte plaintext block
        expanded_key: The expanded key
        mode: 128, 192, or 256
    Returns:
        16-byte ciphertext block
    """
    rounds = {128: 10, 192: 12, 256: 14}
    num_rounds = rounds[mode]
    
    state = bytes_to_state_matrix(plaintext)
    
    # Initial round
    round_key = get_round_key(expanded_key, 0)
    state_bv = BitVector(rawbytes=plaintext)
    state_bv = add_round_key(state_bv, round_key)
    state = bytes_to_state_matrix(state_bv.get_bitvector_in_ascii().encode('latin1'))
    
    # Main rounds
    for round_num in range(1, num_rounds):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        round_key = get_round_key(expanded_key, round_num)
        state_bytes = state_matrix_to_bytes(state)
        state_bv = BitVector(rawbytes=state_bytes)
        state_bv = add_round_key(state_bv, round_key)
        state = bytes_to_state_matrix(state_bv.get_bitvector_in_ascii().encode('latin1'))
    
    # Final round
    state = sub_bytes(state)
    state = shift_rows(state)
    round_key = get_round_key(expanded_key, num_rounds)
    state_bytes = state_matrix_to_bytes(state)
    state_bv = BitVector(rawbytes=state_bytes)
    state_bv = add_round_key(state_bv, round_key)
    
    return state_bv.get_bitvector_in_ascii().encode('latin1')

def inv_shift_rows(state_matrix):
    """
    Perform Inverse ShiftRows transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after Inverse ShiftRows transformation
    """
    new_state = [row.copy() for row in state_matrix]  # Create a copy of the state
    
    # Row 0: No shift
    # Row 1: Shift right by 1 
    new_state[1] = [state_matrix[1][3], state_matrix[1][0], state_matrix[1][1], state_matrix[1][2]]
    
    # Row 2: Shift right by 2
    new_state[2] = [state_matrix[2][2], state_matrix[2][3], state_matrix[2][0], state_matrix[2][1]]
    
    # Row 3: Shift right by 3
    new_state[3] = [state_matrix[3][1], state_matrix[3][2], state_matrix[3][3], state_matrix[3][0]]
    
    return new_state

def inv_sub_bytes(state_matrix):
    """
    Perform Inverse SubBytes transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after Inverse SubBytes transformation
    """
    new_state = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            byte = state_matrix[i][j]
            new_state[i][j] = bitvectordemo.InvSbox[byte]
    return new_state

def inv_mix_columns(state_matrix):
    """
    Perform Inverse MixColumns transformation on the state matrix
    Args:
        state_matrix: 4x4 list of lists containing bytes
    Returns:
        New state matrix after Inverse MixColumns transformation
    """
    new_state = [[0 for _ in range(4)] for _ in range(4)]
    AES_modulus = BitVector(bitstring='100011011')  # AES irreducible polynomial x^8 + x^4 + x^3 + x + 1
    
    for col in range(4):
        # Get the current column as a list of BitVectors
        column = [BitVector(intVal=state_matrix[row][col], size=8) for row in range(4)]
        
        # Multiply with the InvMixer matrix
        for row in range(4):
            # Initialize result for this cell
            result = BitVector(intVal=0, size=8)
            
            # Multiply and accumulate each element
            for i in range(4):
                term = bitvectordemo.InvMixer[row][i].gf_multiply_modular(
                    column[i], 
                    AES_modulus, 
                    8
                )
                result = result ^ term
            
            new_state[row][col] = result.intValue()
    
    return new_state

def aes_decrypt(ciphertext, expanded_key, mode):
    """
    Perform AES decryption on a single 128-bit block
    Args:
        ciphertext: 16-byte ciphertext block
        expanded_key: The expanded key
        mode: 128, 192, or 256
    Returns:
        16-byte plaintext block
    """
    rounds = {128: 10, 192: 12, 256: 14}
    num_rounds = rounds[mode]
    
    state = bytes_to_state_matrix(ciphertext)
    
    # Initial round
    round_key = get_round_key(expanded_key, num_rounds)
    state_bv = BitVector(rawbytes=ciphertext)
    state_bv = add_round_key(state_bv, round_key)
    state = bytes_to_state_matrix(state_bv.get_bitvector_in_ascii().encode('latin1'))
    
    # Main rounds
    for round_num in range(num_rounds-1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        round_key = get_round_key(expanded_key, round_num)
        state_bytes = state_matrix_to_bytes(state)
        state_bv = BitVector(rawbytes=state_bytes)
        state_bv = add_round_key(state_bv, round_key)
        state = bytes_to_state_matrix(state_bv.get_bitvector_in_ascii().encode('latin1'))
        state = inv_mix_columns(state)
    
    # Final round
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    round_key = get_round_key(expanded_key, 0)
    state_bytes = state_matrix_to_bytes(state)
    state_bv = BitVector(rawbytes=state_bytes)
    state_bv = add_round_key(state_bv, round_key)
    
    return state_bv.get_bitvector_in_ascii().encode('latin1')

def aes_cbc_encrypt(plaintext, key, mode):
    """
    Encrypt using AES in CBC mode
    Args:
        plaintext: Plaintext bytes (any length)
        key: Encryption key
        mode: 128, 192, or 256
    Returns:
        Tuple of (IV, ciphertext)
    """
    # Pad the plaintext
    padded_plaintext = pad_data(plaintext)
    
    # Generate random IV
    iv = generate_iv()
    
    # Expand key
    expanded_key = key_expansion(key, mode)
    
    # Split into 16-byte blocks
    blocks = [padded_plaintext[i:i+16] for i in range(0, len(padded_plaintext), 16)]
    
    ciphertext = bytearray()
    prev_block = iv
    
    for block in blocks:
        # XOR with previous ciphertext block (or IV for first block)
        xored = bytes(a ^ b for a, b in zip(block, prev_block))
        
        # Encrypt the result
        encrypted = aes_encrypt(xored, expanded_key, mode)
        ciphertext.extend(encrypted)
        prev_block = encrypted
    
    return iv, bytes(ciphertext)

def aes_cbc_decrypt(ciphertext, key, iv, mode):
    """
    Decrypt using AES in CBC mode
    Args:
        ciphertext: Ciphertext bytes
        key: Encryption key
        iv: Initialization vector
        mode: 128, 192, or 256
    Returns:
        Plaintext bytes
    """
    # Expand key
    expanded_key = key_expansion(key, mode)
    
    # Split into 16-byte blocks
    blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
    
    plaintext = bytearray()
    prev_block = iv
    
    for block in blocks:
        # Decrypt the block
        decrypted = aes_decrypt(block, expanded_key, mode)
        
        # XOR with previous ciphertext block (or IV for first block)
        xored = bytes(a ^ b for a, b in zip(decrypted, prev_block))
        plaintext.extend(xored)
        prev_block = block
    

    #return both the padded and unpadded plaintext
    return plaintext, unpad_data(bytes(plaintext))

#----for testing task1 we need to uncomment it----

# if __name__ == "__main__":
#     print("Sample I/O:\n")
    
#     mode = select_aes_mode()
    
#     input_key = get_user_key_input(mode)
#     plaintext = get_plaintext()
    
#     print("\nKey:")
#     print(f"In ASCII: {input_key}")
#     print(f"In HEX: {format_hex_bytes(input_key.encode('latin1'))}\n")

#     key = process_aes_key_with_hash(input_key, mode)
    

#     print("Plain Text:")
#     print(f"In ASCII: {plaintext.decode('latin1')}")
#     print(f"In HEX: {format_hex_bytes(plaintext)}\n")
    

#     padded_plaintext = pad_data(plaintext)
#     print(f"In ASCII (After Padding): {padded_plaintext.decode('latin1')}")
#     print(f"In HEX (After Padding): {format_hex_bytes(padded_plaintext)}\n")
    

#     start_key = time.perf_counter()
#     expanded_key = key_expansion(key, mode)
#     key_time = (time.perf_counter() - start_key) * 1000  # Convert to milliseconds
    
#     start_enc = time.time()
#     iv, ciphertext = aes_cbc_encrypt(plaintext, key, mode)
#     enc_time = (time.time() - start_enc) * 1000
    
#     start_dec = time.time()
#     padded_decrypted_text, decrypted_text = aes_cbc_decrypt(ciphertext, key, iv, mode)
#     dec_time = (time.time() - start_dec) * 1000
    
#     print("Ciphered Text:")
#     print(f"In HEX: {format_hex_bytes(ciphertext)}")
#     print(f"In ASCII: {ciphertext.decode('latin1')}\n")
    
#     print("Deciphered Text:")
#     print("Before Unpadding:")
#     print(f"In HEX: {format_hex_bytes(padded_decrypted_text)}")
#     print(f"In ASCII: {padded_decrypted_text.decode('latin1')}")
#     print("After Unpadding:")
#     print(f"In ASCII: {decrypted_text.decode('latin1')}")
#     print(f"In HEX: {format_hex_bytes(decrypted_text)}\n")
    
#     print("Execution Time Details:")
#     print(f"Key Schedule Time: {key_time} ms")
#     print(f"Encryption Time: {enc_time} ms")
#     print(f"Decryption Time: {dec_time} ms")

#     # Verify
#     print("\nOriginal plaintext matches decrypted text:", plaintext == decrypted_text)

#------------------------------------------------------------------------------------------------------