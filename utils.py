import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from itertools import combinations
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def make_dataset(n):
    min_w, max_w = 50, 150
    noise_w = np.random.normal(0, 10, n)
    weights = np.random.uniform(min_w, max_w, n)

    min_h, max_h = 1.50, 1.90
    noise_h = np.random.normal(0, 10, n)
    heights = np.random.uniform(min_h, max_h, n)
    return np.vstack([weights, heights]).T


def plot(ax, dataset, title):
    weights, heights = dataset[:, 0], dataset[:, 1]
    noise = np.random.uniform(-0.2, 0.2, len(weights))
    ax.scatter(
        weights,
        np.full(len(weights), 1) + noise,
    )
    ax.scatter(
        heights,
        np.full(len(heights), 2) + noise,
    )
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([1, 2], ["Weight", "Height"])
    ax.set_title(title)
    return ax


def show_dataframe(dataset):
    return pd.DataFrame(dataset, columns=["Weight", "Height"])


def plot_dataset(*objects):
    objects = [(objects[i], objects[i + 1]) for i in range(0, len(objects) - 1, 2)]
    plots = len(objects)
    fig, axs = plt.subplots(1, plots, figsize=(5 * plots, 5))
    if len(objects) == 1:
        axs = [axs]
    for (dataset, title), ax in zip(objects, axs):
        plot(ax, dataset, title)
    fig.tight_layout()


def make_clusters():
    X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=1, random_state=42)
    y_pred_6 = KMeans(n_clusters=6, random_state=42, n_init="auto").fit_predict(X)
    y_pred_5 = KMeans(n_clusters=5, random_state=42, n_init="auto").fit_predict(X)
    y_pred_4 = KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(X)
    y_pred_3 = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X)
    y_wrong = np.random.randint(4, size=1000)

    return X, y_wrong, [y_pred_3, y_pred_4, y_pred_5, y_pred_6]


def plot_clusters(X, wrong_data, *predicted):
    fig, axs = plt.subplots(1, 2 + len(predicted), figsize=(25, 5))

    axs[0].scatter(X[:, 0], X[:, 1], c="k", alpha=0.5)
    axs[0].set_title("Datos originales")

    for idx, y_preds in enumerate(predicted, 1):
        axs[idx].scatter(X[:, 0], X[:, 1], c=y_preds)
        axs[idx].set_title(f"{idx+2} clusters encontrados")
    axs[-1].scatter(X[:, 0], X[:, 1], c=wrong_data)
    axs[-1].set_title("Mal clusttering")


def plot_regression_results(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, (yp, yt) in enumerate(zip(y_pred, y_true)):
        ax.plot((idx, idx), (yp, yt), "k--")
    ax.scatter(np.arange(0, len(y_true)), y_true, s=15, label="True value")
    ax.scatter(np.arange(0, len(y_pred)), y_pred, s=15, label="Predicted value")

    ax.set_xlabel("Data")
    ax.set_ylabel("Predictions")
    ax.set_title("True vs Predicted Values")
    ax.legend()


def load_split_iris():
    iris = load_iris()
    return train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


def load_custom_houses():
    data = np.loadtxt("custom_houses.txt")
    return data[:, :1], data[:, 1]


def load_complex_data():
    data = pd.read_csv("complex.csv")
    return data


def show_transformed_data(dataset, columns):
    cols = np.concatenate(columns).ravel()

    return pd.DataFrame(dataset, columns=cols)


def classification_report_comparison(y_true, model_predictions: dict):
    # accuracy_score, precision_score, recall_score, f1_score
    scores = (
        ("Accuracy", accuracy_score),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("F1", f1_score),
    )
    metrics_per_model = {}
    for model_name, y_pred in model_predictions.items():
        metrics_per_model[model_name] = {metric_name: metric(y_true, y_pred) for metric_name, metric in scores}

    return pd.DataFrame(metrics_per_model)


def load_trained_model():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    return model, X_test, y_test



def plot_boundary(classifier, X, y, title, ax):
    """Plot the decision boundary for a classifier."""
    # Train the classifier with the given hyperparameters
    classifier.fit(X, y)

    # Plot the original datapoints
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    ax.axis('tight')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    if hasattr(classifier, "support_vectors_"):
        # If the SVC is a kernel SVM, plot the support vectors
        ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=80,
                    facecolors='none', edgecolors='k')

    # Hide the ticks on the axes and set the corresponding title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def plot_boundaries(X, y, classifiers):
    # Compute the number of rows and columns based on the number of hyperparameter combinations
    num_plots = len(classifiers)
    num_cols = 3 # int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    size_per_plot = 3

    # Create a figure with the desired number of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * size_per_plot, num_rows*size_per_plot))

    # Flatten the axes array to simplify indexing
    axes = axes.ravel()
    
    for i, (title, svc) in enumerate(classifiers):
        plot_boundary(svc, X, y, title, axes[i])

    for i in range(num_plots, num_rows * num_cols):
        axes[i].set_visible(False)

    fig.tight_layout()


def plot_knn_boundaries(knn, knn_scaled, X, X_scaled, y):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
    scaled_xx, scaled_yy = np.meshgrid(np.linspace(X_scaled[:,0].min()-1, X_scaled[:,0].max()+1, 100),
                         np.linspace(X_scaled[:,1].min()-1, X_scaled[:,1].max()+1, 100))
    

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    Z_scaled = knn_scaled.predict(np.c_[scaled_xx.ravel(), scaled_yy.ravel()])
    Z_scaled = Z_scaled.reshape(scaled_xx.shape)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    axs[0].contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    axs[0].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdBu)
    axs[0].set_title('KNN sin escalar')
    
    axs[1].contourf(scaled_xx, scaled_yy, Z_scaled, cmap=plt.cm.RdBu, alpha=0.5)
    axs[1].scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap=plt.cm.RdBu)
    axs[1].set_title('KNN escalado')
    

    axs[0].set_xlabel('Característica 1')
    axs[0].set_ylabel('Característica 2')
    axs[0].set_xlim(xx.min(), xx.max())
    axs[0].set_ylim(yy.min(), yy.max())
    

    axs[1].set_xlabel('Característica 1')
    axs[1].set_ylabel('Característica 2')
    axs[1].set_xlim(scaled_xx.min(), scaled_xx.max())
    axs[1].set_ylim(scaled_yy.min(), scaled_yy.max())
    
    plt.tight_layout()
    plt.show()


def view_centroids(kmeans, input_features, 
                   title="Centroides de los clústers", xlabel="Característica 1", ylabel="Característica 2", ax=None):
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    no_ax = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        no_ax = True

    ax.scatter(input_features[:, 0], input_features[:, 1], c=labels)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='v', c='red', label='Centroids')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Calculate all the clustering metrics
    sscore = silhouette_score(input_features, labels)
    cscore = calinski_harabasz_score(input_features, labels)
    dscore = davies_bouldin_score(input_features, labels)

    # Add the clustering metrics as a text annotation
    annotation_text = "Silhouette score: {:.2f},\nCalinski-Harabasz score: {:.2f},\nDavies-Bouldin score: {:.2f}".format(sscore, cscore, dscore)
    ax.annotate(annotation_text, xy=(0.95, 0.15), xycoords='axes fraction', fontsize=8, ha='right', va='top')


    # Ajustar y mostrar la figura
    ax.legend()
    if no_ax:
        fig.tight_layout()


def plot_centroids(input_features, trained_kmeans, titles, xlabel="Característica 1", ylabel="Característica 2"):
    # Compute the number of rows and columns based on the number of hyperparameter combinations
    num_plots = len(trained_kmeans)
    num_cols = 3 # int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))

    size_per_plot = 4

    # Create a figure with the desired number of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * size_per_plot, num_rows*size_per_plot))

    # Flatten the axes array to simplify indexing
    axes = axes.ravel()
    
    for i, (title, kmeans) in enumerate(zip(titles, trained_kmeans)):
        view_centroids(kmeans, input_features, title=title, xlabel=xlabel, ylabel=ylabel, ax=axes[i])

    for i in range(num_plots, num_rows * num_cols):
        axes[i].set_visible(False)

    fig.tight_layout()


def view_dbscan(original_features, y_true, dbscans):

    num_plots = len(dbscans) + 1
    num_cols = 3
    num_rows = int(np.ceil(num_plots / num_cols))


    size_per_plot = 4

    # Create a figure with the desired number of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * size_per_plot, num_rows*size_per_plot))

    axes = axes.ravel()

    axes[0]

    # Gráfico de etiquetas predichas
    axes[0].scatter(original_features[:, 0], original_features[:, 1], c=y_true, cmap='viridis')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_title('Etiquetas "originales"')

    for idx, (title, dbscan) in enumerate(dbscans, 1):
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_labels = list(set(labels))
        
        ax2 = axes[idx]
        
        ax2.scatter(original_features[:, 0], original_features[:, 1], c=labels, cmap='viridis')
        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.set_title(f"{title} - {n_clusters} clusters")

        ruido_indices = np.where(labels == -1)[0]
        ax2.scatter(original_features[ruido_indices, 0], original_features[ruido_indices, 1], c='red')

    

    for i in range(num_plots, num_rows * num_cols):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def view_centroids_iris(kmeans, input_features):

    view_centroids(kmeans, input_features,
                     title="Centroides de los clústers", xlabel="Longitud del sépalo", ylabel="Ancho del sépalo")

def visualize_iris_pairplot(iris):

    bombs = list(combinations(range(4), 2))

    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    axes = axes.ravel()
    for i, comb in enumerate(bombs):
        ax = axes[i]
        ax.scatter(iris.data[:, comb[0]], iris.data[:, comb[1]], c=iris.target)
        ax.set_xlabel(iris.feature_names[comb[0]])
        ax.set_ylabel(iris.feature_names[comb[1]])

    plt.tight_layout()
    plt.show()
