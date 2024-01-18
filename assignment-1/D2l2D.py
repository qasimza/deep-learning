# D2l2D.py D2l 2.4 2023 cheng CS5173/6073
# Usage: python D2l2D.py

# D2l 2.4 Calculus
# D2l 2.4.1 Derivatives and Differentiation
import numpy as np
def f(x):
    return 3 * x ** 2 - 4 * x

for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')

# D2l 2.4.2 Visualization Utilities
import matplotlib.pyplot as plt
x = np.arange(0, 3, 0.1)
plt.plot(x, f(x))
plt.plot(x, 2 * x - 3)
plt.legend(['f(x)', 'Tangent line (x=1)'])
plt.show()