import re
import nltk
from nltk.stem import WordNetLemmatizer
from unicode import EMO_UNICODE
import regex

stemmer = WordNetLemmatizer()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
en_stop = set(nltk.corpus.stopwords.words('english'))


UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()}

def convert_emojis(text):
    emoji_list = []
    datas = regex.findall(r'\X', text)
    # print('converting text: ' + text)
    try:
        for emot in datas:
            text = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), text)
        return text
    except Exception as e:
        # print("Not able to convert ")
        return text


def preprocess_text(document):
    # convert Emojies into emotions
    document = convert_emojis(document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    # tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def preprocess_array(data):
    new_array = []   
    for text in data:
        new_array.append(preprocess_text(text))

    return new_array
