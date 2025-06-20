import time
import random
from sympy import isprime


random.seed(42)  

def generate_large_prime(bit_size):
    """
    Generate a random prime number of the specified bit size (using a fixed seed).
    """
    while True:
        candidate = random.getrandbits(bit_size)
        candidate |= (1 << (bit_size - 1)) | 1  # Ensure correct bit length and odd
        if isprime(candidate):
            return candidate

def generate_curve_parameters(P):
    """
    Generate random a and b for the elliptic curve y^2 ≡ x^3 + ax + b mod P,
    ensuring non-singularity (4a³ + 27b² != 0 mod P).
    """
    while True:
        a = random.randrange(0, P)
        b = random.randrange(0, P)
        discriminant = (4 * pow(a, 3, P) + 27 * pow(b, 2, P)) % P
        if discriminant != 0:
            return a, b
        
def tonelli_shanks(n, p):
    """
    Tonelli-Shanks algorithm to find a square root of n modulo p.
    Returns y such that y^2 ≡ n mod p, or None if no such y exists.
    """
    if pow(n, (p - 1) // 2, p) != 1:  # Check if n is a quadratic residue
        return None

    # Case when p ≡ 3 mod 4 (faster method)
    if p % 4 == 3:
        y = pow(n, (p + 1) // 4, p)
        return y

    # General case (Tonelli-Shanks)
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    c = pow(z, Q, p)

    x = pow(n, (Q + 1) // 2, p)
    t = pow(n, Q, p)
    m = S
    while t != 1:
        # Find the smallest i such that t^(2^i) ≡ 1 mod p
        i, temp = 0, t
        while temp != 1 and i < m:
            temp = pow(temp, 2, p)
            i += 1

        if i == m:
            return None  # No solution exists

        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return x

def find_base_point(P, a, b):
    """
    Find a random point (x, y) on the elliptic curve y^2 ≡ x^3 + ax + b mod P.
    """
    while True:
        x = random.randrange(0, P)
        rhs = (pow(x, 3, P) + a * x + b) % P  # y^2 ≡ x^3 + ax + b mod P
        y = tonelli_shanks(rhs, P)
        if y is not None:
            return (x, y)
        
def point_add(P, Q, a, p):
    """
    Add two points P and Q on the elliptic curve y^2 ≡ x^3 + ax + b mod p.
    Handles the case of point at infinity (None) and P == -Q.
    """
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and y1 != y2:  # P == -Q (point at infinity)
        return None

    if P == Q:  # Point doubling (P == Q)
        m = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:  # Point addition (P != Q)
        m = (y2 - y1) * pow(x2 - x1, -1, p) % p

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p

    return (x3, y3)

def scalar_multiply(k, G, a, p):
    """
    Compute k * G using the double-and-add algorithm.
    """
    result = None
    current = G

    while k > 0:
        if k % 2 == 1:
            result = point_add(result, current, a, p)
        current = point_add(current, current, a, p)  # Double the point
        k = k // 2

    return result

def time_operations(bit_size, trials=5):
    """Measure average computation times for key operations"""
    times_A = []
    times_B = []
    times_R = []

    for _ in range(trials):
        # Generate curve parameters (timed separately)
        P = generate_large_prime(bit_size)
        a, b = generate_curve_parameters(P)
        G = find_base_point(P, a, b)
        
        # Generate keys
        Ka = random.randrange(1, P)
        Kb = random.randrange(1, P)
        
        # Time A = Ka * G
        start = time.time()
        A = scalar_multiply(Ka, G, a, P)
        times_A.append(time.time() - start)
        
        # Time B = Kb * G
        start = time.time()
        B = scalar_multiply(Kb, G, a, P)
        times_B.append(time.time() - start)
        
        # Time R = Ka * B
        start = time.time()
        R = scalar_multiply(Ka, B, a, P)
        times_R.append(time.time() - start)
    
    return {
        'A': sum(times_A)/trials,
        'B': sum(times_B)/trials,
        'R': sum(times_R)/trials
    }

# defined for serialization so that it can be sent over the network
def parameters_to_string(a, b, G, P, public_key):
    """Serialize ECC parameters for network transmission"""
    xG, yG = G
    xA, yA = public_key
    return f"{a},{b},{xG},{yG},{P},{xA},{yA}"

def string_to_parameters(data):
    """Deserialize received ECC parameters"""
    parts = data.split(',')
    a = int(parts[0])
    b = int(parts[1])
    G = (int(parts[2]), int(parts[3]))
    P = int(parts[4])
    public_key = (int(parts[5]), int(parts[6]))
    return a, b, G, P, public_key

#--for printing the performance report for task2 we need to uncomment it--
# results = {}
# for bits in [128, 192, 256]:
#     results[bits] = time_operations(bits)

# print("\nPerformance Report (Average of 5 trials in seconds):")
# print("| k    | Computation Time For    |")
# print("|------|-------------------------|")
# print("|      | A       | B       | R       |")
# print("|------|---------|---------|---------|")
# for bits in [128, 192, 256]:
#     t = results[bits]
#     print(f"| {bits:<4} | {t['A']:.6f} | {t['B']:.6f} | {t['R']:.6f} |")
#---------------------------------------------------------------------------------------