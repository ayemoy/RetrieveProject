
import math
from collections import Counter
from contextlib import closing
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex as bodyINV, MultiFileReaderBody
from inverted_index_title_gcp import InvertedIndex as titleINV, MultiFileReaderTitle
from inverted_index_anchor_gcp import InvertedIndex as anchorINV, MultiFileReaderAnchor



nltk.download('stopwords')
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing



# inverted_index_body = bodyINV.read_index(base_dir='/content/body/postings_gcp/', name='index')
# inverted_index_title = titleINV.read_index(base_dir='/content/title/postings_gcp/', name='title_index')
# inverted_index_anchor = anchorINV.read_index(base_dir='/content/anchor/postings_gcp/', name='anchor_index')


inverted_index_body = bodyINV.read_index(base_dir='/home/ayelet431/body/postings_gcp/', name='index')
inverted_index_title = titleINV.read_index(base_dir='/home/ayelet431/title/postings_gcp/', name='title_index')
inverted_index_anchor = anchorINV.read_index(base_dir='/home/ayelet431/anchor/postings_gcp/', name='anchor_index')




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/")
def hello():
  return "Welcome to Ayelet Moyal and Natalia Kataev search engine! :)"

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).

    '''
# ____ final ____#
    res = []
    query = request.args.get('query', '')
    # tokenize and filter the stopwords from the original queries.
    if len(query) == 0:
        return jsonify(res)
    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    res_body = body_func(tokenized_query)
    res_title = title_func(tokenized_query)
    total_dic = {}

    #now we will give score to the results of cosine and title, and return our best engine 
    for doc_id, score in res_body.items():
      if doc_id in total_dic:
        total_dic[doc_id] += 0.5*score
      else:
        total_dic[doc_id] = 0.5*score

    for doc_id, score in res_title.items():
      if doc_id in total_dic:
        total_dic[doc_id] += 0.5*score
      else:
        total_dic[doc_id] = 0.5*score

    sorted_dic = {}
    for doc_id, score in sorted(total_dic.items(), key=lambda x: x[1], reverse=True):
      sorted_dic[doc_id] = score

    num_of_res=0
    for doc_id in sorted_dic:
      if num_of_res<100:
        res.append((doc_id, inverted_index_body.doc_to_title[doc_id]))
        num_of_res +=1
      else:
        break

    return jsonify(res)




@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=political+hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    #_____________________body final___________________________________
    res = []

    #tokenize the query
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    cosine_calc_dic = body_func(tokenized_query)

    sorted_cosine_res = {}
    for key, value in sorted(cosine_calc_dic.items(), key=lambda x: x[1], reverse=True):
      sorted_cosine_res[key] = value

    counter_results = 0
    for doc_id in sorted_cosine_res:
        if counter_results >= 100:
            break
        res.append((doc_id, inverted_index_body.doc_to_title[doc_id]))
        counter_results += 1

    return jsonify(res)



#in this func we clalculate the cosine similarity and return the results in dic 
def body_func(tokenized_query):

    wiq_dic = {} #save the weight of every tokenized word in given query

    cosine_calc_dic = {} #save the cosine similarity of every word in body of the text 

    wiq_counter = Counter(tokenized_query)
    for word in wiq_counter:
        wiq_dic[word] = wiq_counter[word] / len(tokenized_query)

   
    query_mechane = 0

    for word in wiq_counter:
        weight = wiq_dic[word]
        query_mechane += math.pow(weight, 2)
        if word not in inverted_index_body.idf:
            continue
            
        idf_of_word = inverted_index_body.idf[word]
        word_in_posting_lst = read_posting_listB(inverted_index_body, word)

        if len(word_in_posting_lst) > 0:
            for tup in word_in_posting_lst:
                if tup[0] not in cosine_calc_dic:
                    cosine_calc_dic[tup[0]] = (tup[1] / inverted_index_body.doc_len[tup[0]]) * idf_of_word * weight
                else:
                    cosine_calc_dic[tup[0]] += (tup[1] / inverted_index_body.doc_len[tup[0]]) * idf_of_word * weight

    if len(cosine_calc_dic) < 1:
        return jsonify(res)

    # calc the bottom product for cosinesim
    for doc_id in cosine_calc_dic:
      total_calc = math.sqrt(inverted_index_body.tfidf_dominator[doc_id] * query_mechane)
      cosine_calc_dic[doc_id] = cosine_calc_dic[doc_id] / total_calc

    return cosine_calc_dic


def token_query(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)
    tokensQ = [token.group() for token in RE_WORD.finditer(query.lower())]
    filteredQ = [tok for tok in tokensQ if tok not in all_stopwords]

    return filteredQ


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    query_weight = title_func(tokenized_query)
    # sorted_query_similarity = {k: v for k, v in sorted(query_binary_similarity.items(), key=lambda item: item[1], reverse=True)}
    
    sorted_title_res = {}
    for key, value in sorted(query_weight.items(), key=lambda x: x[1], reverse=True):
      sorted_title_res[key] = value

    
    for key in sorted_title_res:
        res.append((key, inverted_index_title.doc_to_title[key]))

    # END SOLUTION
    return jsonify(res)


#this func calculate the similarity between tokenized query and title of article
#return dictionary
def title_func(tokenized_query):

    query_sim_dic = {}
    
    for query_token in tokenized_query:
        posting_lst_of_title = read_posting_listT(inverted_index_title, query_token)
        if len(posting_lst_of_title) > 0:

            for doc_id in posting_lst_of_title:
                if doc_id[0] in query_sim_dic:
                    query_sim_dic[doc_id[0]] += 1
                else:
                    query_sim_dic[doc_id[0]] = 1

    if len(query_sim_dic) == 0:
        return jsonify(res)
        
    return query_sim_dic



@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

        
    # BEGIN SOLUTION
    query_anchor_score = {}
    for word in tokenized_query:
        posting_lst_of_word = read_posting_listA(inverted_index_anchor, word)

        if len(posting_lst_of_word) > 0:
            for doc_id in posting_lst_of_word:
                if doc_id[0] in query_anchor_score:
                    query_anchor_score[doc_id[0]] += 1
                else:
                    query_anchor_score[doc_id[0]] = 1


    sorted_anchor_res = {}
    for key, value in sorted(query_anchor_score.items(), key=lambda x: x[1], reverse=True):
      sorted_anchor_res[key] = value

    if len(query_anchor_score) == 0:
        return jsonify(res)

    for key in sorted_anchor_res:
      if key in inverted_index_anchor.doc_to_title:
        res.append((key, inverted_index_anchor.doc_to_title[key]))

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    if len(wiki_ids) == 0:
        return jsonify(res)

    get_page_rank = pd.read_csv('/home/ayelet431/page_rank.csv', index_col=0, header=None, squeeze=True)
    
    page_rank_dic = get_page_rank.to_dict()


    for doc_id in wiki_ids:
      if doc_id in page_rank_dic:
        res.append(page_rank_dic[doc_id])

    # for i in range(len(wiki_ids)):
    #     if wiki_ids[i] in ranks_dict:
    #         res.append(ranks_dict[wiki_ids[i]])

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    with open('/home/ayelet431/page_view_smaller.pkl', 'rb') as f:
      loaded_pv_dic = pickle.load(f)

    for doc_id in wiki_ids:
      if doc_id in loaded_pv_dic:
        res.append(loaded_pv_dic[doc_id])

    # END SOLUTION
    return jsonify(res)




def read_posting_listB(inverted, w):
    with closing(MultiFileReaderBody()) as reader:
        locs = inverted.posting_locs[w]
        posting_list = []
        if len(locs) > 0:
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)

            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

def read_posting_listT(inverted, w):
    with closing(MultiFileReaderTitle()) as reader:
        locs = inverted.posting_locs[w]
        posting_list = []
        if len(locs) > 0:
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)

            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

def read_posting_listA(inverted, w):
    with closing(MultiFileReaderAnchor()) as reader:
        locs = inverted.posting_locs[w]
        posting_list = []
        if len(locs) > 0:
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)

            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list




if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)

