# -*- coding: utf-8 -*-
"""
## Author: Sunil Patel
## Copyright: Copyright 2018-2019, Packt Publishing Limited
## Version: 0.0.1
## Maintainer: Sunil Patel
## Email: snlpatel01213@hotmail.com
## Linkedin: https://www.linkedin.com/in/linus1/
## Contributor : Shreyak Chakraborty
## Contributor Email : shreyak.rekshda@gmail.com
## Status: active
"""

import fasttext
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()

if __name__ == "__main__":

    # Skipgram model
    model = fasttext.train_unsupervised("data/testdata_en.txt", model='skipgram', lr=0.05, dim=100, ws=5, epoch=5)
    words = model.words  # list of words in dictionary
    print("words present in the model : ", words)

    # # CBOW model
    # model = fasttext.train_unsupervised("data/testdata_en.txt", model='cbow', lr=0.05, dim=100, ws=5, epoch=5)
    # print (model.words) # list of words in dictionary

    # I am using only  Skipgram model model

    # visualizing using tensorboard
    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")
    all_vectors = []
    for eachword in words:
        all_vectors.append(model[eachword])
    writer.add_embedding(np.asarray(all_vectors), words)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
