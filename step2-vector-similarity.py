import math

# Euclidean distance 

a = [2, 3]
b = [5, 7]

sum = 0
for x,y in zip(a,b):
    sum += (x-y)**2

euclidean_distance = math.sqrt(sum)
print(euclidean_distance)



# Cosine similarity

def dot_product(x, y):
    total = 0
    for p,q in zip(x,y):
        total += p*q
    return total

def magnitude(numbers):
    total = 0
    for num in numbers:
        total += num**2
    return math.sqrt(total)

# dot_product = 10 + 21 = 31
# magnitude a = (2**2 + 3**2)**.5 = 13**.5 = 3.61
# magnitude b = (5**2 + 7**2)**.5 = (25+49)**.5 = 74**.5 = 8.6
cosine_similarity = dot_product(a, b) / (magnitude(a) * magnitude(b))
print(cosine_similarity)
