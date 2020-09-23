#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def map_ads(df, reference=None, output_file='heatmap'):

    if reference is None:
        reference = df

    # Init numpy array
    global_max_x = reference['max_x'].max()
    global_max_y = reference['max_y'].max()
    heatmap = np.zeros([global_max_y, global_max_x], dtype=np.float32)

    # Count the number of advertisements at each pixel
    for i, row in df.iterrows():
        heatmap[int(row['min_y'])-1:int(row['max_y']),
            int(row['min_x'])-1:int(row['max_x'])] += 1

    # Normalize to values between 0 - 1
    heatmap[:,:] /= np.amax(heatmap)
    #map[:,:] /= reference.shape[0]

    # Plot and save heatmap image
    plt.imshow(heatmap, cmap='jet')
    #plt.clim(0, 1)
    plt.colorbar()
    plt.savefig(output_file + '.png', bbox_inches='tight', dpi=300)
    plt.clf()

if __name__ == '__main__':
    df = pd.read_pickle("nrc-1945-1994")
    map_ads(df)

