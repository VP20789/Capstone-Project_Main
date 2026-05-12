from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from src.config import (
    KMEANS_RANDOM_STATE,
    KMEANS_N_INIT,
    PCA_COMPONENTS
)


def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


def perform_kmeans(X_scaled, n_clusters):

    model = KMeans(
        n_clusters=n_clusters,

        random_state=KMEANS_RANDOM_STATE,

        n_init=KMEANS_N_INIT
    )

    clusters = model.fit_predict(X_scaled)

    return model, clusters


def perform_pca(X_scaled):

    pca = PCA(
        n_components=PCA_COMPONENTS
    )

    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca