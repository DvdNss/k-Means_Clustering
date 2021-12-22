import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import iris_config as config
# from config import recens_config as config
# from config import flags_config as config


def find_optimal_clusters(X, maxRange=config.ELBOW_MAX_RANGE, fitIter=300):
    wcss = []

    for i in range(1, maxRange):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=fitIter, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, maxRange), wcss)
    plt.title('Elbow Graph Method on ' + config.PROBLEM_NAME)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def clustering(path, columns):
    """"Applies k-Means clustering on a CSV file."""

    # We first transform our csv file into a pandas dataframe
    dataset = pd.read_csv(path)

    # Then we select the relevant columns and extract them into another pandas dataframe
    X = dataset.iloc[:, columns].values

    # We have to standardise the values of our dataframe it is scaled efficiently
    X = StandardScaler().fit_transform(X)

    # We generate a PCA algorithm which takes an integer dimension
    pca = PCA(n_components=config.PCA_DIM)

    # We apply the PCA to our standardised dataframe and extract the PCA value/dim dataframe
    X = pca.fit_transform(X)

    # We apply Elbow Graph Method to find optimal cluster number
    find_optimal_clusters(X)

    # For each iteration
    for i in range(1, config.ITERATIONS + 1):

        # We apply k-means algorithm to our dataframe with a defined number of clusters
        kmeans = KMeans(n_clusters=config.CLUSTERS, init='k-means++', max_iter=i, n_init=10, random_state=0)

        # We predict to which cluster each item belongs
        y_kmeans = kmeans.fit_predict(X)

        # We create a figure to draw the graph
        fig = plt.figure()

        # We set the graph to a 3 dimensional graph
        ax = fig.add_subplot(111, projection='3d')

        # For each cluster
        for j in range(0, config.CLUSTERS):
            # We draw the cluster items (points)
            ax.scatter(X[y_kmeans == j, 0], X[y_kmeans == j, 1], X[y_kmeans == j, 2], s=config.ITEM_SIZE,
                       c=config.COLORS[j], label="cluster #" + str(j))

            # We draw the cluster center
            ax.scatter(kmeans.cluster_centers_[j, 0], kmeans.cluster_centers_[j, 1], kmeans.cluster_centers_[j, 2],
                       s=config.CENTROID_SIZE,
                       c=config.COLORS[j], label='centroid #' + str(j))

        # We set the camera so it's easier to read the clusters
        ax.view_init(elev=config.PLT_ANGLE, azim=config.PLT_AZIM)
        ax.margins(config.PLT_ZOOM)

        # We set the names of the axes
        ax.set_xlabel(config.DIM1_NAME)
        ax.set_ylabel(config.DIM2_NAME)
        ax.set_zlabel(config.DIM3_NAME)

        # We display the legend
        ax.legend(loc='upper center', bbox_to_anchor=(-0.18, 1.0), shadow=True, ncol=1)

        # We set the title of the graph
        plt.title('k-Means Clusters #' + str(i) + ' on ' + config.PROBLEM_NAME)

        # We display the cluster graph
        plt.show()


def show_truth(path, columns, truth):
    """Shows the truth of the csv file."""

    # We first transform our csv file into a pandas dataframe
    dataset = pd.read_csv(path)

    # Then we select the relevant columns and extract them
    X = dataset.iloc[:, columns].values

    # We select the true labels from CSV
    Y = dataset.iloc[:, truth].values

    predictions = []
    for k in range(0, len(Y)):
        if Y[k] not in predictions:
            predictions.append(Y[k])

    # We have to standardise the values of our dataframe it is scaled efficiently
    X = StandardScaler().fit_transform(X)

    # We generate a PCA algorithm which takes an integer dimension
    pca = PCA(n_components=config.PCA_DIM)

    # We apply the PCA to our standardised dataframe and extract the PCA value/dim dataframe
    X = pca.fit_transform(X)

    # For each iteration
    for i in range(1, 3):

        # We get the true labels from the CSV
        truth = Y

        # We create a figure to draw the graph
        fig = plt.figure()

        # We set the graph to a 3 dimensional graph
        ax = fig.add_subplot(111, projection='3d')

        # For each label
        for j in range(0, len(predictions)):
            # We draw the true items (points)
            ax.scatter(X[truth == predictions[j], 0], X[truth == predictions[j], 1], X[truth == predictions[j], 2],
                       s=config.ITEM_SIZE,
                       c=config.COLORS2[j], label=predictions[j])

        # We set the camera so it's easier to read the clusters
        ax.view_init(elev=config.PLT_ANGLE, azim=config.PLT_AZIM)
        ax.margins(config.PLT_ZOOM)

        # We set the names of the axes
        ax.set_xlabel(config.DIM1_NAME)
        ax.set_ylabel(config.DIM2_NAME)
        ax.set_zlabel(config.DIM3_NAME)

        # We display the legend
        ax.legend(loc='upper center', bbox_to_anchor=(-0.16, 1.0), shadow=True, ncol=1)

        # We set the title of the graph
        plt.title('Truth for ' + config.PROBLEM_NAME)

        # We display the truth graph
        plt.show()


clustering(config.CSV_PATH, config.CSV_COLUMNS)
if config.CSV_TRUTH != -1:
    show_truth(config.CSV_PATH, config.CSV_COLUMNS, config.CSV_TRUTH)