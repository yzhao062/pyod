from utils.data import generate_data_clusters
x_train, x_test, y_train, y_test = generate_data_clusters(n_samples=100, test_size=0.25, n_clusters=2, n_features=2, contamination=0.1, size='same', density='same', dist=0.25, random_state=0)

print(x_train)
print("\n------------------------------\n")
print(x_test)
print("\n------------------------------\n")
print(y_train)
print("\n------------------------------\n")
print(y_test)

import matplotlib.pyplot as plt
#plt.figure(1)
for cluster in x_train:
    plt.plot(cluster[:,0], cluster[:,1], 'o')
plt.show()
