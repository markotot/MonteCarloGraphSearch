import concurrent.futures
import math

PRIMES = [
    1,2,3,4,5]

def is_prime(n, i):

    print(i)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(is_prime, [2,3,4,5], [-1, -1, -1, -1, -1])

        for result in results:
            print(result)


main()