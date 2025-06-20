import socket
import sys
import random
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

ecc = import_module_from_file("ecc", "./2005067_ecc.py")
aes = import_module_from_file("aes", "./2005067_aes.py")

class CryptoSocket:
    def __init__(self, host='localhost', port=5000, key_size=128):
        self.host = host
        self.port = port
        self.key_size = key_size  # 128, 192, or 256
        self.shared_key = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def start_server(self):
        """Run as Bob (receiver)"""
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Bob ({self.key_size}-bit) listening on {self.host}:{self.port}")
        
        conn, addr = self.sock.accept()
        print(f"Connection from Alice: {addr}")
        
        # ECDH Key Exchange
        self._bob_key_exchange(conn)
        
        # Receive encrypted message
        iv_ct = conn.recv(1024)
        iv = iv_ct[:16]
        ct = iv_ct[16:]

        # #print the received ciphertext
        # print("Received Ciphertext:", ct.hex())
        
        # Decrypt using the negotiated key size
        _, pt = aes.aes_cbc_decrypt(ct, self.shared_key, iv, self.key_size)
        print("\nDecrypted Message:", pt.decode())
        conn.close()
    
    def start_client(self):
        """Run as Alice (sender)"""
        self.sock.connect((self.host, self.port))
        print(f"Alice ({self.key_size}-bit) connected to Bob at {self.host}:{self.port}")
        
        # Exchanging the ECDH keys
        self._alice_key_exchange()
        
        # Encrypt and send message
        message = input("Enter message to encrypt: ").encode()
        iv, ct = aes.aes_cbc_encrypt(message, self.shared_key, self.key_size)
        self.sock.send(iv + ct)
        print("Encrypted message sent!")
        self.sock.close()
    
    def _alice_key_exchange(self):
        """Alice initiates ECDH"""
        # Generate parameters with selected key size
        P = ecc.generate_large_prime(self.key_size)
        a, b = ecc.generate_curve_parameters(P)
        G = ecc.find_base_point(P, a, b)
        Ka = random.randrange(1, P)
        A = ecc.scalar_multiply(Ka, G, a, P)
        
        # Send to Bob (include key size in params)
        params = f"{self.key_size}|" + ecc.parameters_to_string(a, b, G, P, A)
        self.sock.send(params.encode())
        
        # Bob's public key is received
        B_data = self.sock.recv(1024).decode()
        xB, yB = map(int, B_data.split(','))
        B = (xB, yB)
        
        # Computing shared key (we will use first 'key_size' bits of x-coordinate)
        R = ecc.scalar_multiply(Ka, B, a, P)
        key_bytes = R[0].to_bytes((self.key_size//8)+1, 'big')[:self.key_size//8]
        self.shared_key = key_bytes
        print(f"Shared {self.key_size}-bit key established:", self.shared_key.hex())
        
    def _bob_key_exchange(self, conn):
        """Bob completes ECDH"""
        # Received Alice's parameters
        data = conn.recv(1024).decode()
        key_size_str, params = data.split('|', 1)
        self.key_size = int(key_size_str)  # Sync key size with Alice
        a, b, G, P, A = ecc.string_to_parameters(params)
        
        # we will generate and send public key
        Kb = random.randrange(1, P)
        B = ecc.scalar_multiply(Kb, G, a, P)
        conn.send(f"{B[0]},{B[1]}".encode())
        
        # Computing shared key
        R = ecc.scalar_multiply(Kb, A, a, P)
        key_bytes = R[0].to_bytes((self.key_size//8)+1, 'big')[:self.key_size//8]
        self.shared_key = key_bytes
        print(f"Shared {self.key_size}-bit key established:", self.shared_key.hex())

if __name__ == "__main__":
    key_size = 128  # default size of the key 
    if len(sys.argv) > 1:
        if sys.argv[1] == "bob":
            if len(sys.argv) > 2:
                key_size = int(sys.argv[2])
            CryptoSocket(key_size=key_size).start_server()
        else:
            if len(sys.argv) > 1:
                key_size = int(sys.argv[1])
            CryptoSocket(key_size=key_size).start_client()