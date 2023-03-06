import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift, MiniBatchKMeans, Birch, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
import numpy as np

def get_performance_clustering(data, labels):
    siluetas = metrics.silhouette_score(data, labels, metric='euclidean')
    calinski = metrics.calinski_harabasz_score(data, labels)
    davies = metrics.davies_bouldin_score(data, labels)

    return siluetas, calinski, davies

dataset = {
    0 : "absorption",
    1 : "enantioselectivity",
    2 : "localization",
    3 : "T50"
}
method = {
    0:"FFT",
    1:"NLP",
    2:"Properties"
}
bioembedding = {
    0: "bepler",
    1: "esm",
    2: "fasttext",
    3: "plus_rnn",
    4: "prottrans"
}
distances = {
    1: "Euclidean",
    2: "Braycurtis",
    3: "Canberra",
    4: "Chebyshev",
    5: "Cityblock",
    6: "Correlation",
    7: "Cosine",
    8: "Minkowski",
    9: "Hamming"
}

resultados = [""]

for a in range(0,4):
    for b in range(0,3):
        if b == 1:
            for c in range(0,5):
                if a == 0:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/"+bioembedding[c]+"-absortion.csv")
                else:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/"+bioembedding[c]+"-"+dataset[a]+".csv")
                df_id = df_data['id']
                ignore_columns = pd.DataFrame()
                ignore_columns['id'] = df_data['id']
                ignore_columns['target'] = df_data['target']

                df_data = df_data.drop(columns=['id', 'target'])

                '''------------KMEANS-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    kmeans.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["k-means-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means-{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_kmeans.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/kmeans_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8]+strategies[0][9])), random_state=0)
                except:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8])), random_state=0)
                kmeans.fit(df_data)
                ignore_columns['labels'] = kmeans.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_kmeans.csv")

                '''------------DBSCAN-----------------------------------------------------------
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    dbscan = DBSCAN(eps=k,min_samples=2)
                    dbscan.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, dbscan.labels_)
                    row = ["dbscan-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_dbscan.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/dbscan_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    dbscan = KMeans(eps=(int(strategies[0][8]+strategies[0][9])), min_samples=2)
                except:
                    dbscan = KMeans(eps=(int(strategies[0][8])), min_samples=2)
                dbscan.fit(df_data)
                ignore_columns['labels'] = dbscan.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_dbscan.csv")
                '''
                '''------------MEANSHIFT-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    meanshift = MeanShift(bandwidth=k)
                    meanshift.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["meanshift-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['meanshift-{}'.format(k)]=meanshift.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_meanshift.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/meanshift_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10]+strategies[0][11])))
                except:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10])))
                meanshift.fit(df_data)
                ignore_columns['labels'] = meanshift.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_meanshift.csv")

                '''------------BIRCH-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    birch = Birch(n_clusters=k, threshold=0.006)
                    birch.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, birch.labels_)
                    row = ["birch-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['birch-{}'.format(k)]=birch.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_birch.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/birch_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    birch = Birch(n_clusters=(int(strategies[0][6]+strategies[0][7])), threshold=0.1)
                except:
                    birch = Birch(n_clusters=(int(strategies[0][6])), threshold=0.1)
                birch.fit(df_data)
                ignore_columns['labels'] = birch.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_birch.csv")

                '''------------AFFINITY-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in np.arange(0.5,1.0,0.05):
                    affinity = AffinityPropagation(damping=k)
                    affinity.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["affinity-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['affinity-{}'.format(k)]=affinity.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_affinity.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/affinity_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]+strategies[0][12]
                    affinity = AffinityPropagation(damping=(float(damp)))
                except:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]
                    affinity = AffinityPropagation(damping=(float(damp)))
                affinity.fit(df_data)
                ignore_columns['labels'] = affinity.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_affinity.csv")

                '''------------AGLOMERATIVE-----------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    aglomerative = AgglomerativeClustering(n_clusters=k)
                    aglomerative.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, aglomerative.labels_)
                    row = ["aglomerative-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['aglomerative-{}'.format(k)]=aglomerative.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/results_aglomerative.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/aglomerative_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/"+bioembedding[c]+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13]+strategies[0][14])))
                except:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13])))
                aglomerative.fit(df_data)
                ignore_columns['labels'] = aglomerative.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/"+bioembedding[c]+"/unsupervised_clustering_sequences_aglomerative.csv")
        if b == 0:
            for c in range (0,8):
                if a == 0:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/fft-Group_"+str(c)+"-absortion.csv")
                else:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/fft-Group_"+str(c)+"-"+dataset[a]+".csv")
                df_id = df_data['id']
                ignore_columns = pd.DataFrame()
                ignore_columns['id'] = df_data['id']
                ignore_columns['target'] = df_data['target']

                df_data = df_data.drop(columns=['id', 'target'])

                '''------------KMEANS-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    kmeans.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["k-means-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means-{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_kmeans.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/kmeans_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8]+strategies[0][9])), random_state=0)
                except:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8])), random_state=0)
                kmeans.fit(df_data)
                ignore_columns['labels'] = kmeans.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_kmeans.csv")

                '''------------DBSCAN-----------------------------------------------------------'''
                '''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in np.arange(0.05, 1, 0.05):
                    dbscan = DBSCAN(eps=k,min_samples=10)
                    dbscan.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, dbscan.labels_)
                    row = ["dbscan-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_dbscan.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/dbscan_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    dbscan = KMeans(eps=(int(strategies[0][8]+strategies[0][9])), min_samples=2)
                except:
                    dbscan = KMeans(eps=(int(strategies[0][8])), min_samples=2)
                dbscan.fit(df_data)
                ignore_columns['labels'] = dbscan.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_dbscan.csv")
                '''
                '''------------MEANSHIFT-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    meanshift = MeanShift(bandwidth=k)
                    meanshift.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["meanshift-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['meanshift-{}'.format(k)]=meanshift.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_meanshift.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/meanshift_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10]+strategies[0][11])))
                except:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10])))
                meanshift.fit(df_data)
                ignore_columns['labels'] = meanshift.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_meanshift.csv")

                '''------------BIRCH-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    birch = Birch(n_clusters=k, threshold=0.006)
                    birch.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, birch.labels_)
                    row = ["birch-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['birch-{}'.format(k)]=birch.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_birch.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/birch_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    birch = Birch(n_clusters=(int(strategies[0][6]+strategies[0][7])), threshold=0.1)
                except:
                    birch = Birch(n_clusters=(int(strategies[0][6])), threshold=0.1)
                birch.fit(df_data)
                ignore_columns['labels'] = birch.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_birch.csv")

                '''------------AFFINITY-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in np.arange(0.5,1.0,0.05):
                    affinity = AffinityPropagation(damping=k)
                    affinity.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["affinity-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['affinity-{}'.format(k)]=affinity.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_affinity.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/affinity_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]+strategies[0][12]
                    affinity = AffinityPropagation(damping=(float(damp)))
                except:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]
                    affinity = AffinityPropagation(damping=(float(damp)))
                affinity.fit(df_data)
                ignore_columns['labels'] = affinity.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_affinity.csv")

                '''------------AGLOMERATIVE-----------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    aglomerative = AgglomerativeClustering(n_clusters=k)
                    aglomerative.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, aglomerative.labels_)
                    row = ["aglomerative-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['aglomerative-{}'.format(k)]=aglomerative.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_aglomerative.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/aglomerative_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13]+strategies[0][14])))
                except:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13])))
                aglomerative.fit(df_data)
                ignore_columns['labels'] = aglomerative.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_aglomerative.csv")
        if b == 2:
            for c in range(0,8):
                if a == 0:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/physicochemical-Group_"+str(c)+"-absortion.csv")
                else:
                    df_data = pd.read_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/physicochemical-Group_"+str(c)+"-"+dataset[a]+".csv")
                df_id = df_data['id']
                ignore_columns = pd.DataFrame()
                ignore_columns['id'] = df_data['id']
                ignore_columns['target'] = df_data['target']

                df_data = df_data.drop(columns=['id', 'target'])

                '''------------KMEANS-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    kmeans = KMeans(n_clusters=k, random_state=0)
                    kmeans.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["k-means-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means-{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_kmeans.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/kmeans_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8]+strategies[0][9])), random_state=0)
                except:
                    kmeans = KMeans(n_clusters=(int(strategies[0][8])), random_state=0)
                kmeans.fit(df_data)
                ignore_columns['labels'] = kmeans.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_kmeans.csv")

                '''------------DBSCAN-----------------------------------------------------------
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    dbscan = DBSCAN(eps=k,min_samples=2)
                    dbscan.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, dbscan.labels_)
                    row = ["dbscan-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['K-means{}'.format(k)]=kmeans.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_dbscan.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/dbscan_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    dbscan = KMeans(eps=(int(strategies[0][8]+strategies[0][9])), min_samples=2)
                except:
                    dbscan = KMeans(eps=(int(strategies[0][8])), min_samples=2)
                dbscan.fit(df_data)
                ignore_columns['labels'] = dbscan.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_dbscan.csv")
                '''
                '''------------MEANSHIFT-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    meanshift = MeanShift(bandwidth=k)
                    meanshift.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["meanshift-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['meanshift{}'.format(k)]=meanshift.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_meanshift.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/meanshift_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10]+strategies[0][11])))
                except:
                    meanshift = MeanShift(bandwidth=(int(strategies[0][10])))
                meanshift.fit(df_data)
                ignore_columns['labels'] = meanshift.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_meanshift.csv")

                '''------------BIRCH-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    birch = Birch(n_clusters=k, threshold=0.006)
                    birch.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, birch.labels_)
                    row = ["birch-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['birch-{}'.format(k)]=birch.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_birch.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/birch_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    birch = Birch(n_clusters=(int(strategies[0][6]+strategies[0][7])), threshold=0.1)
                except:
                    birch = Birch(n_clusters=(int(strategies[0][6])), threshold=0.1)
                birch.fit(df_data)
                ignore_columns['labels'] = birch.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_birch.csv")

                '''------------AFFINITY-----------------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in np.arange(0.5,1.0,0.05):
                    affinity = AffinityPropagation(damping=k)
                    affinity.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, kmeans.labels_)
                    row = ["affinity-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['affinity-{}'.format(k)]=affinity.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_affinity.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/affinity_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]+strategies[0][12]
                    affinity = AffinityPropagation(damping=(float(damp)))
                except:
                    damp = strategies[0][9]+strategies[0][10]+strategies[0][11]
                    affinity = AffinityPropagation(damping=(float(damp)))
                affinity.fit(df_data)
                ignore_columns['labels'] = affinity.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_affinity.csv")

                '''------------AGLOMERATIVE-----------------------------------------------------'''
                df_concat = []
                matrix_result = []
                df_sub = pd.DataFrame()
                df_sub['secuencia']=df_id
                for k in range(2, 30):
                    aglomerative = AgglomerativeClustering(n_clusters=k)
                    aglomerative.fit(df_data)
                    siluetas, calinski, davies = get_performance_clustering(df_data, aglomerative.labels_)
                    row = ["aglomerative-{}".format(k), siluetas, calinski, davies]
                    matrix_result.append(row)
                    df_sub['aglomerative-{}'.format(k)]=aglomerative.labels_

                df_explore = pd.DataFrame(matrix_result, columns=['strategy', 'siluetas', 'calinski', 'davies'])
                df_explore.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/results_aglomerative.csv")
                df_sub.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/aglomerative_labels.csv")

                highest_siluetas = np.max(df_explore['siluetas'])
                highest_calinski = np.max(df_explore['calinski'])

                df_filter_by_siluetas = df_explore.loc[df_explore['siluetas'] >= highest_siluetas]
                df_filter_by_calinski = df_explore.loc[df_explore['calinski'] >= highest_calinski]

                df_concat = pd.concat([df_filter_by_siluetas, df_filter_by_calinski])
                strategies = df_concat['strategy'].unique()
                frase = dataset[a]+"/"+method[b]+"/Group"+str(c)+" dio "+strategies[0]
                resultados.append(frase)
                try:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13]+strategies[0][14])))
                except:
                    aglomerative = AgglomerativeClustering(n_clusters=(int(strategies[0][13])))
                aglomerative.fit(df_data)
                ignore_columns['labels'] = aglomerative.labels_

                ignore_columns.to_csv("../../results_demo/"+dataset[a]+"/"+method[b]+"/Group_"+str(c)+"/unsupervised_clustering_sequences_aglomerative.csv")

