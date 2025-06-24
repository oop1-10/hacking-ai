vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(x * y for x, y in zip(v1, v2))

result = dot_product(vector1, vector2)
print("Dot product:", result)

# alternate version using a loop
# This version is more verbose and less efficient but illustrates the concept clearly
sum = 0
for x in range(len(vector1)):
    sum += vector1[x] * vector2[x]
print("Sum:", sum)