
def datal(A, B, chunk=2):
    A_ = A[:5] + B[5:]
    B_ = A[5:] + B[:5]
    for c, d in zip(A_, B_):
        yield (c, d)

a = [ i for i in range(10)]
b = [ i * 10 for i in range(10)]

for c, d in datal(a, b):
    print(c, d)
