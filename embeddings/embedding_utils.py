import os
import pickle
import re
from typing import List

import tqdm
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def simple_clean(corpus: List[str]) -> List[str]:
    """
    Make lowercase and remove special characters from a list of documents.
    :param corpus: List of documents.
    :return:  List of cleaned documents.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        cleaned_corpus.append(cleaned_document)
    return cleaned_corpus

def simple_clean_list(corpus: List[str]) -> List[List[str]]:
    """
    Make lowercase and remove special characters from a list of documents.
    :param corpus: List of documents.
    :return:  List of cleaned documents.
    """
    cleaned_corpus = []
    for document in tqdm.tqdm(corpus):
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        cleaned_corpus.append(cleaned_document.split())
    return cleaned_corpus

def clean_and_stem(corpus: List[str]) -> List[str]:
    """
    Clean a list of documents.

    Args:
        corpus: List of documents.

    Returns:
        Cleaned list word lists.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document_list = []
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        # remove stopwords and stem words
        stemmer = PorterStemmer()
        cleaned_document_list = [stemmer.stem(word) for word in cleaned_document.split() if word not in stopwords.words('english')]
        cleaned_corpus.append(" ".join(cleaned_document_list))
    return cleaned_corpus

def clean_and_stem_list(corpus: List[str]) -> List[List[str]]:
    """
    Clean a list of documents.

    Args:
        corpus: List of documents.

    Returns:
        Cleaned list word lists.
    """
    cleaned_corpus = []
    for document in corpus:
        cleaned_document_list = []
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)
        # remove stopwords and stem words
        stemmer = PorterStemmer()
        cleaned_document_list = [stemmer.stem(word) for word in cleaned_document.split() if word not in stopwords.words('english')]
        cleaned_corpus.append(cleaned_document_list)
    return cleaned_corpus

def clean_and_lemma_list(corpus: List[str]) -> List[List[str]]:
    cleaned_corpus = []
    for document in corpus:
        cleaned_document_list = []
        cleaned_document = document.lower()
        # Remove special characters.
        cleaned_document = re.sub(r'[^a-z]', ' ', cleaned_document)
        # Remove extra spaces.
        cleaned_document = re.sub(r'\s+', ' ', cleaned_document)

        #stopwords
        cleaned_document = [word for word in cleaned_document.split() if word not in stopwords.words('english')]

        tagged_words = pos_tag(cleaned_document)

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_document = []
        for word, tag in tagged_words:
            wordnet_tag = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=wordnet_tag)
            lemmatized_document.append(lemma)
        cleaned_corpus.append(lemmatized_document)

    return cleaned_corpus


def get_k_highest_values(dictionary, k):
    """
    Get the k highest values from a dictionary.

    Args:
        dictionary: Dictionary of values.
        k: Number of highest values to return.

    Returns:
        List of k highest values.
    """
    return sorted(dictionary, key=dictionary.get, reverse=True)[:k]


full_name_translator = {}
full_name_translator["cs"] = "Computer Science"
full_name_translator["cs.AI"] = "Artificial Intelligence"
full_name_translator["cs.AR"] = "Hardware Architecture"
full_name_translator["cs.CC"] = "Computational Complexity"
full_name_translator["cs.CE"] = "Computational Engineering, Finance, and Science"
full_name_translator["cs.CG"] = "Computational Geometry"
full_name_translator["cs.CL"] = "Computation and Language"
full_name_translator["cs.CR"] = "Cryptography and Security"
full_name_translator["cs.CV"] = "Computer Vision and Pattern Recognition"
full_name_translator["cs.CY"] = "Computers and Society"
full_name_translator["cs.DB"] = "Databases"
full_name_translator["cs.DC"] = "Distributed, Parallel, and Cluster Computing"
full_name_translator["cs.DL"] = "Digital Libraries"
full_name_translator["cs.DM"] = "Discrete Mathematics"
full_name_translator["cs.DS"] = "Data Structures and Algorithms"
full_name_translator["cs.ET"] = "Emerging Technologies"
full_name_translator["cs.FL"] = "Formal Languages and Automata Theory"
full_name_translator["cs.GL"] = "General Literature"
full_name_translator["cs.GR"] = "Graphics"
full_name_translator["cs.GT"] = "Computer Science and Game Theory"
full_name_translator["cs.HC"] = "Human-Computer Interaction"
full_name_translator["cs.IR"] = "Information Retrieval"
full_name_translator["cs.IT"] = "Information Theory"
full_name_translator["cs.LG"] = "Machine Learning"
full_name_translator["cs.LO"] = "Logic in Computer Science"
full_name_translator["cs.MA"] = "Multiagent Systems"
full_name_translator["cs.MM"] = "Multimedia"
full_name_translator["cs.MS"] = "Mathematical Software"
full_name_translator["cs.NA"] = "Numerical Analysis"
full_name_translator["cs.NE"] = "Neural and Evolutionary Computing"
full_name_translator["cs.NI"] = "Networking and Internet Architecture"
full_name_translator["cs.OH"] = "Other Computer Science"
full_name_translator["cs.OS"] = "Operating Systems"
full_name_translator["cs.PF"] = "Performance"
full_name_translator["cs.PL"] = "Programming Languages"
full_name_translator["cs.RO"] = "Robotics"
full_name_translator["cs.SC"] = "Symbolic Computation"
full_name_translator["cs.SD"] = "Sound"
full_name_translator["cs.SE"] = "Software Engineering"
full_name_translator["cs.SI"] = "Social and Information Networks"
full_name_translator["cs.SY"] = "Systems and Control"

full_name_translator["econ"] = "Economics"
full_name_translator["econ.EM"] = "Econometrics"
full_name_translator["econ.GN"] = "General Economics"
full_name_translator["econ.TH"] = "Theoretical Economics"

full_name_translator["eess"] = "Electrical Engineering and Systems Science"
full_name_translator["eess.AS"] = "Audio and Speech Processing"
full_name_translator["eess.IV"] = "Image and Video Processing"
full_name_translator["eess.SP"] = "Signal Processing"
full_name_translator["eess.SY"] = "Systems and Control"

full_name_translator["math"] = "Mathematics"
full_name_translator["math.AC"] = "Commutative Algebra"
full_name_translator["math.AG"] = "Algebraic Geometry"
full_name_translator["math.AP"] = "Analysis of PDEs"
full_name_translator["math.AT"] = "Algebraic Topology"
full_name_translator["math.CA"] = "Classical Analysis and ODEs"
full_name_translator["math.CO"] = "Combinatorics"
full_name_translator["math.CT"] = "Category Theory"
full_name_translator["math.CV"] = "Complex Variables"
full_name_translator["math.DG"] = "Differential Geometry"
full_name_translator["math.DS"] = "Dynamical Systems"
full_name_translator["math.FA"] = "Functional Analysis"
full_name_translator["math.GM"] = "General Mathematics"
full_name_translator["math.GN"] = "General Topology"
full_name_translator["math.GR"] = "Group Theory"
full_name_translator["math.GT"] = "Geometric Topology"
full_name_translator["math.HO"] = "History and Overview"
full_name_translator["math.IT"] = "Information Theory"
full_name_translator["math.KT"] = "K-Theory and Homology"
full_name_translator["math.LO"] = "Logic"
full_name_translator["math.MG"] = "Metric Geometry"
full_name_translator["math.MP"] = "Mathematical Physics"
full_name_translator["math.NA"] = "Numerical Analysis"
full_name_translator["math.NT"] = "Number Theory"
full_name_translator["math.OA"] = "Operator Algebras"
full_name_translator["math.OC"] = "Optimization and Control"
full_name_translator["math.PR"] = "Probability"
full_name_translator["math.QA"] = "Quantum Algebra"
full_name_translator["math.RA"] = "Rings and Algebras"
full_name_translator["math.RT"] = "Representation Theory"
full_name_translator["math.SG"] = "Symplectic Geometry"
full_name_translator["math.SP"] = "Spectral Theory"
full_name_translator["math.ST"] = "Statistics Theory"

full_name_translator["astro-ph"] = "Astrophysics"
full_name_translator["astro-ph.CO"] = "Cosmology and Nongalactic Astrophysics"
full_name_translator["astro-ph.EP"] = "Earth and Planetary Astrophysics"
full_name_translator["astro-ph.GA"] = "Astrophysics of Galaxies"
full_name_translator["astro-ph.HE"] = "High Energy Astrophysical Phenomena"
full_name_translator["astro-ph.IM"] = "Instrumentation and Methods for Astrophysics"
full_name_translator["astro-ph.SR"] = "Solar and Stellar Astrophysics"

full_name_translator["cond-mat"] = "Condensed Matter"
full_name_translator["cond-mat.dis-nn"] = "Disordered Systems and Neural Networks"
full_name_translator["cond-mat.mes-hall"] = "Mesoscale and Nanoscale Physics"
full_name_translator["cond-mat.mtrl-sci"] = "Materials Science"
full_name_translator["cond-mat.other"] = "Other Condensed Matter"
full_name_translator["cond-mat.quant-gas"] = "Quantum Gases"
full_name_translator["cond-mat.soft"] = "Soft Condensed Matter"
full_name_translator["cond-mat.stat-mech"] = "Statistical Mechanics"
full_name_translator["cond-mat.str-el"] = "Strongly Correlated Electrons"
full_name_translator["cond-mat.supr-con"] = "Superconductivity"

full_name_translator["gr-qc"] = "General Relativity and Quantum Cosmology"
full_name_translator["hep-ex"] = "High Energy Physics - Experiment"
full_name_translator["hep-lat"] = "High Energy Physics - Lattice"
full_name_translator["hep-ph"] = "High Energy Physics - Phenomenology"
full_name_translator["hep-th"] = "High Energy Physics - Theory"
full_name_translator["math-ph"] = "Mathematical Physics"

full_name_translator["nlin"] = "Nonlinear Sciences"
full_name_translator["nlin.AO"] = "Adaptation and Self-Organizing Systems"
full_name_translator["nlin.CD"] = "Chaotic Dynamics"
full_name_translator["nlin.CG"] = "Cellular Automata and Lattice Gases"
full_name_translator["nlin.PS"] = "Pattern Formation and Solitons"
full_name_translator["nlin.SI"] = "Exactly Solvable and Integrable Systems"

full_name_translator["nucl-ex"] = "Nuclear Experiment"
full_name_translator["nucl-th"] = "Nuclear Theory"

full_name_translator["physics"] = "Physics"
full_name_translator["physics.acc-ph"] = "Accelerator Physics"
full_name_translator["physics.ao-ph"] = "Atmospheric and Oceanic Physics"
full_name_translator["physics.app-ph"] = "Applied Physics"
full_name_translator["physics.atm-clus"] = "Atomic and Molecular Clusters"
full_name_translator["physics.atom-ph"] = "Atomic Physics"
full_name_translator["physics.bio-ph"] = "Biological Physics"
full_name_translator["physics.chem-ph"] = "Chemical Physics"
full_name_translator["physics.class-ph"] = "Classical Physics"
full_name_translator["physics.comp-ph"] = "Computational Physics"
full_name_translator["physics.data-an"] = "Data Analysis, Statistics and Probability"
full_name_translator["physics.ed-ph"] = "Physics Education"
full_name_translator["physics.flu-dyn"] = "Fluid Dynamics"
full_name_translator["physics.gen-ph"] = "General Physics"
full_name_translator["physics.geo-ph"] = "Geophysics"
full_name_translator["physics.hist-ph"] = "History and Philosophy of Physics"
full_name_translator["physics.ins-det"] = "Instrumentation and Detectors"
full_name_translator["physics.med-ph"] = "Medical Physics"
full_name_translator["physics.optics"] = "Optics"
full_name_translator["physics.plasm-ph"] = "Plasma Physics"
full_name_translator["physics.pop-ph"] = "Popular Physics"
full_name_translator["physics.soc-ph"] = "Physics and Society"
full_name_translator["physics.space-ph"] = "Space Physics"

full_name_translator["quant-ph"] = "Quantum Physics"

full_name_translator["q-bio"] = "Quantitative Biology"
full_name_translator["q-bio.BM"] = "Biomolecules"
full_name_translator["q-bio.CB"] = "Cell Behavior"
full_name_translator["q-bio.GN"] = "Genomics"
full_name_translator["q-bio.MN"] = "Molecular Networks"
full_name_translator["q-bio.NC"] = "Neurons and Cognition"
full_name_translator["q-bio.OT"] = "Other Quantitative Biology"
full_name_translator["q-bio.PE"] = "Populations and Evolution"
full_name_translator["q-bio.QM"] = "Quantitative Methods"
full_name_translator["q-bio.SC"] = "Subcellular Processes"
full_name_translator["q-bio.TO"] = "Tissues and Organs"

full_name_translator["q-fin"] = "Quantitative Finance"
full_name_translator["q-fin.CP"] = "Computational Finance"
full_name_translator["q-fin.EC"] = "Economics"
full_name_translator["q-fin.GN"] = "General Finance"
full_name_translator["q-fin.MF"] = "Mathematical Finance"
full_name_translator["q-fin.PM"] = "Portfolio Management"
full_name_translator["q-fin.PR"] = "Pricing of Securities"
full_name_translator["q-fin.RM"] = "Risk Management"
full_name_translator["q-fin.ST"] = "Statistical Finance"
full_name_translator["q-fin.TR"] = "Trading and Market Microstructure"

full_name_translator["stat"] = "Statistics"
full_name_translator["stat.AP"] = "Applications"
full_name_translator["stat.CO"] = "Computation"
full_name_translator["stat.ME"] = "Methodology"
full_name_translator["stat.ML"] = "Machine Learning"
full_name_translator["stat.OT"] = "Other Statistics"
full_name_translator["stat.TH"] = "Statistics Theory"
# old labels
alias_map = {}
alias_map["acc-phys"] = "physics.acc-ph"
alias_map["adap-org"] = "nlin.AO"
alias_map["chao-dyn"] = "nlin.CD"
alias_map["patt-sol"] = "nlin.PS"
alias_map["dg-ga"] = "math.DG"
alias_map["solv-int"] = "nlin.SI"
alias_map["bayes-an"] = "physics.data-an"
alias_map["comp-gas"] = "nlin.CG"
alias_map["alg-geom"] = "math.AG"
alias_map["funct-an"] = "math.FA"
alias_map["q-alg"] = "math.QA"
alias_map["ao-sci"] = "physics.ao-ph"
alias_map["atom-ph"] = "physics.atom-ph"
alias_map["chem-ph"] = "physics.chem-ph"
alias_map["plasm-ph"] = "physics.plasm-ph"
alias_map["mtrl-th"] = "cond-mat.mtrl-sci"
alias_map["cmp-lg"] = "cs.CL"
alias_map["supr-con"] = "cond-mat.supr-con"

def to_full_name(abbr: str) -> str:
    return full_name_translator.get(alias_map.get(abbr, abbr))


def get_queries():
    return {
        "Computer Science and AI": ["Transformer models", "Federated learning", "Quantum computing", "Explainable AI",
                                    "Graph neural networks"],
        "Physics and Engineering": ["Topological insulators", "Optical metamaterials", "Fission", "Soft robotics",
                                    "Health monitoring"],
        "Biology and Medicine": ["CRISPR", "Microbiome", "DNA sequencing", "Synthetic biology", "Drug delivery"],
        "Earth and Environmental Science": ["Climate model", "Remote sensing", "Greenhouse gas", "Biodiversity",
                                            "Light pollution"]
    }
def get_query_key(category, query):
    return (category[:5] + "_" + query[:10]).replace(" ", "_")