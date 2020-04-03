from src.text_preprocessing import text_preprocessing
from src.graph import CharacterGraph
from src.graph.properties_extraction import *
from src.graph.community_detection import *

# Import libraries
import argparse
import os
import pickle

if __name__ == "__main__":
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str, default='hp1',
                        help="name of the book (expect to have the book as txt file in data/raw_text")

    parser.add_argument("--reprocess_text", action='store_true', help="use to repreprocess the novel")
    parser.add_argument("--recreate_graph", action='store_true', help="use to recreate the graph")
    parser.add_argument("--bert_large", action='store_true', help="to use BERT LARGE for NER, by default BERT base")
    args = parser.parse_args()

    # NOVEL PREPROCESSING
    print("NOVEL PREPROCESSING")
    text_preprocessing(args.book, reprocess=args.reprocess_text, bert_large=args.bert_large)

    # GRAPH CREATION
    print("\nGRAPH CREATION")
    if os.path.exists('data/graph/' + args.book + '.pkl') and not args.recreate_graph:
        character_graph = pickle.load(open('data/graph/' + args.book + '.pkl', 'rb'))
        print("-- LOAD ENTITY GRAPH FROM CACHE --")
        entity_graph = pickle.load(open('data/graph/' + args.book + '-entity-graph.pkl', 'rb'))

        print("-- LOAD DYNAMIC GRAPH FROM CACHE --")
        dynamic_graph = pickle.load(open('data/graph/' + args.book + '-dynamic-graph.pkl', 'rb'))

        print("-- GENERATE ENTITY INTERACTION GRAPH FOR EACH CHAPTER --")
        entity_chapter_graph = character_graph.entity_graph_by_chapter()
    else:
        character_graph = CharacterGraph(args.book)
        pickle.dump(character_graph, open('data/graph/' + args.book + '.pkl', 'wb'))
        entity_graph, dynamic_graph, entity_chapter_graph = character_graph.generate_and_save()

    # GRAPH ANALYSIS
    # We compute, print and plot some graph properties that have been presented in the evaluation
    # section of the report
    print("\n\nGRAPH ANALYSIS")
    # Characters' importance
    print("-- ANALYSE ENTITY GRAPH OF FULL NOVEL --")
    entities_importance = importance_full_graph(entity_graph)
    print("\n-- ANALYSE ENTITY GRAPH BY CHAPTER --")
    most_central_characters, mcc_by_chapter = importance_subraphs(entity_chapter_graph)

    # Graph properties
    properties_full_graph(entity_graph, most_central_characters)
    properties_subgraphs(entity_graph, entity_chapter_graph, most_central_characters, mcc_by_chapter)

    # Community detection
    community_detection(entity_graph, entities_importance)

