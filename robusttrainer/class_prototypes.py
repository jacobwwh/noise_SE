import faiss


def get_prototypes(x, y, args):
    """
    Obtain class prototypes for each category via KMeans
    :param x: features for all examples
    :param y: targets for all examples (We use pseudo-label here)
    :param args: hyper-parameters
    :return:
    """
    cluster = []  # the set of class prototypes of the categories
    for i in range(args.nb_classes):
        y_x = x[y == i]  # the features of class i
        if len(y_x) < args.nb_prototypes:
            cluster.append(y_x)
        else:
            cluster.append(kmeans(y_x, args))
    return cluster


def kmeans(x, args):
    # obtain class prototypes by K-Means
    d = x.shape[1]
    k = int(args.nb_prototypes) ## number of prototypes per class?
    cluster = faiss.Clustering(d, k)
    cluster.verbose = True
    cluster.niter = 20
    cluster.nredo = 5

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    cluster.train(x, index)
    centroids = faiss.vector_to_array(cluster.centroids).reshape(k, d)

    return centroids
