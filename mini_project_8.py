# Mini Project 8
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get a list of all .txt files in the current directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read()# Reading the content of each file and storing them in a list
                 for _file in student_files]

#function to vectorize the input text using TF-IDF
def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
#function to calculate the cosine similarity between two vectors
def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])

#Vectorize the student notes using TF-IDF
vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

# Defining a function to check for plagiarism among the documents
def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

# Calling the functions:
for data in check_plagiarism():
    print("Similarity data:\n", data)    