

# RetrieveProject
our Retrieve Engine: http://35.224.246.8:8080/search?query=bla


So how did i implement my engine?
I have created three new inverted index classes using assingment 3.

I have added my own fields to each inverted index in order to facilitate and streamline the search.
( For example: To search the body of the articles, we added a dictionary that saves the entire length of the documents in the corpus so that we can calculate the Cosine Similarity Index. For a search on a document title we have added a dictionary containing the document ID and its title in order to optimize the search time and return the queries.)
I created posting list files into buckets in GCP for each class of inverted index.
General calculations like Cosine Similarity formula were done outside of functions in order to optimize the primary function of search.

![welcome page](https://user-images.githubusercontent.com/82223056/148789041-a2eeb167-5317-41bd-b27e-73d0a067930e.jpg)
![migraine](https://user-images.githubusercontent.com/82223056/148814641-73a14cf7-fe17-4a1d-be7a-e075f6137d83.jpg)
