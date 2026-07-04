from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

class ClassifierModels:
    def __init__(self, name_clf):
        self.name_clf = name_clf

    def get_grid_knn(self):
        param_grid = {
            'n_neighbors': [5, 10, 15, 20, 25, 30], 
            'weights': ['uniform', 'distance'],
            'metric': ['cosine', 'euclidean']
        }

        base_knn = KNeighborsClassifier()

        grid_knn = GridSearchCV(
            base_knn,
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='f1_macro',
            verbose=1,
            n_jobs=-1
        )

        return grid_knn
    
    def get_mlp(self):
        return MLPClassifier(
            hidden_layer_sizes=(512, 256), 
            activation='relu', 
            max_iter=100, 
            random_state=42
        )
    
    def get_kmeans(self, n_clusters):
        return KMeans(
            n_clusters=n_clusters, 
            init='k-means++', 
            n_init=10, 
            random_state=42
        )
    
    def get_model_clf(self):
        if self.name_clf == "knn":
            return self.get_grid_knn()
        elif self.name_clf == "mlp":
            return self.get_mlp()
        else:
            self.get_kmeans(n_clusters=3)
        
    
