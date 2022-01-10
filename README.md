# RetrieveProject
our Retrieve Engine


So how did we implement our engine?
We have created three new inverted index classes using assingment 3.

We have added our own fields to each inverted index in order to facilitate and streamline the search.
( For example: To search the body of the articles, we added a dictionary that saves the entire length of the documents in the corpus so that we can calculate the Cosine Similarity Index. For a search on a document title we have added a dictionary containing the document ID and its title in order to optimize the search time and return the queries.)
We created posting list files into buckets in GCP for each class of inverted index.
General calculations like Cosine Similarity formula were done outside of functions in order to optimize the primary function of search.
