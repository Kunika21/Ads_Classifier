import re
import sys
from utils import write_status
from nltk.stem.porter import PorterStemmer


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(CONTENT):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    CONTENT = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', CONTENT)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    CONTENT = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', CONTENT)
    # Love -- <3, :*
    CONTENT = re.sub(r'(<3|:\*)', ' EMO_POS ', CONTENT)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    CONTENT = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', CONTENT)
    # Sad -- :-(, : (, :(, ):, )-:
    CONTENT = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', CONTENT)
    # Cry -- :,(, :'(, :"(
    CONTENT = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', CONTENT)
    return CONTENT


def preprocess_CONTENT(CONTENT):
    processed_CONTENT = []
    # Convert to lower case
    CONTENT = CONTENT.lower()
    # Replaces URLs with the word URL
    CONTENT = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', CONTENT)
    # Replace @handle with the word USER_MENTION
    CONTENT = re.sub(r'@[\S]+', 'USER_MENTION', CONTENT)
    # Replaces #hashtag with hashtag
    CONTENT = re.sub(r'#(\S+)', r' \1 ', CONTENT)
    # Remove RT (retweet)
    CONTENT = re.sub(r'\brt\b', '', CONTENT)
    # Replace 2+ dots with space
    CONTENT = re.sub(r'\.{2,}', ' ', CONTENT)
    # Strip space, " and ' from tweet
    CONTENT = CONTENT.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    CONTENT = handle_emojis(CONTENT)
    # Replace multiple spaces with a single space
    CONTENT = re.sub(r'\s+', ' ', CONTENT)
    words = CONTENT.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_CONTENT.append(word)

    return ' '.join(processed_CONTENT)


def preprocess_csv(csv_file_name, processed_file_name, test_file=True):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            COMMENT_ID = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            CONTENT = line
            processed_CONTENT = preprocess_CONTENT(CONTENT)
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (COMMENT_ID, positive, processed_CONTENT))
            else:
                save_to_file.write('%s,%s\n' %
                                   (COMMENT_ID, processed_CONTENT))
            write_status(i + 1, total)
    save_to_file.close()
    print '\nSaved processed comments to: %s' % processed_file_name
    return processed_file_name


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python preprocess.py <raw-CSV>'
        exit()
    use_stemmer = False
    csv_file_name = sys.argv[1]
    processed_file_name = sys.argv[1][:-4] + '-processed.csv'
    if use_stemmer:
        porter_stemmer = PorterStemmer()
        processed_file_name = sys.argv[1][:-4] + '-processed-stemmed.csv'
    preprocess_csv(csv_file_name, processed_file_name, test_file=True)
