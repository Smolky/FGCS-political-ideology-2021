# Predicting Political Ideology from Linguistic Features and Embeddings: An Author Profiling Study on Spanish Politicians' tweets posted in 2020
In this work we explore the reliability of determining personality traits concerning political affiliation in different fine-grained levels. Our contribution is two fold. On the one hand, we evaluate several neural network architectures to build automatic classifiers grounded on explainable linguistic features in combination with state-of-the-art embeddings and, on the other, we release the PoliCorpus-2020, a dataset composed by Spanish politicians' tweets posted in 2020. As additional contributions, we evaluate other demographic traits, such as gender and age range and compare our methods with related datasets from the bibliography. Moreover, we carry out an authorship identification task to determine if it is possible to categorise the author of a text based on their writings from a political point of view. Our results indicate that incorporating linguistic features to neural networks classifiers is beneficial to improve the accuracy and the interpretability of the models. Results suggest lexical and morphosyntax are more effective on author profiling tasks whereas linguistic features based on stylometric are more effective in author identification tasks.


## Folders
    ```config/``` Contains the configuration of the PoliCorpus-2020
    ```code/``` Contains the scripts
    ```assets/``` contains assets (dataset, features, models, evaluations) for each dataset
    ```embeddings/``` Contains pretrained word embeddings models
