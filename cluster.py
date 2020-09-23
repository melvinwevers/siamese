#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heatmap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

NUM_CLUSTERS = 5

df = pd.read_csv('ads.tsv', sep='\t')

X = df[['min_x','min_y','max_x','max_y']].as_matrix()
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=111).fit(X)

for label in range(NUM_CLUSTERS):
    cluster = df[kmeans.labels_ == label]
    heatmap.map_ads(cluster, df, 'heatmap_' + str(label))

