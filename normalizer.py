import pandas as pd
import unicodedata

dev_path = "/home/mamun/Bangla-DeepSpeech/deepspeech-data/cv12-bn/dev.tsv"
test_path = "/home/mamun/Bangla-DeepSpeech/deepspeech-data/cv12-bn/test.tsv"
train_path = "/home/mamun/Bangla-DeepSpeech/deepspeech-data/cv12-bn/train.tsv"
validated_path = "/home/mamun/Bangla-DeepSpeech/deepspeech-data/cv12-bn/validated.tsv"

# Load TSV file into a pandas dataframe
dev_df = pd.read_csv(dev_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')
train_df = pd.read_csv(train_path, sep='\t')
validated_df = pd.read_csv(validated_path, sep='\t')

print(dev_df.head())

## NORMALIZATION
from bnunicodenormalizer import Normalizer 
import re
bnorm=Normalizer()

# import re
# chars_to_remove_regex = '[\"\“\%\‘\”\�]'
# def normalize(batch):
#     _words = [bnorm(word)['normalized']  for word in batch.split()]
#     batch = " ".join([word for word in _words if word is not None])
#     batch = re.sub(chars_to_remove_regex, '', batch)
#     return batch

# Regex for matching zero witdh joiner variations.
STANDARDIZE_ZW = re.compile(r'(?<=\u09b0)[\u200c\u200d]+(?=\u09cd\u09af)')

# Regex for removing standardized zero width joiner, except in edge cases.
DELETE_ZW = re.compile(r'(?<!\u09b0)[\u200c\u200d](?!\u09cd\u09af)')

# Regex matching punctuations to remove.
PUNC = re.compile(r'([\?\.।;:,!"\'])')

def removeOptionalZW(text):
    """
    Removes all optional occurrences of ZWNJ or ZWJ from Bangla text.
    """
    text = STANDARDIZE_ZW.sub('\u200D', text)
    text = DELETE_ZW.sub('', text)
    return text

def removePunc(text):
    """
    Remove for punctuations from text.
    """
    text = PUNC.sub(r"", text)
    return text

def normalize(text, normalize_nukta=True):
    """
    Normalizes unicode strings using the Normalization Form Canonical
    Composition (NFC) scheme where we first decompose all characters and then
    re-compose combining sequences in a specific order as defined by the
    standard in unicodedata module. Finally all zero-width joiners are
    removed.
    """
    if normalize_nukta:
        words = [ bnorm(word)['normalized']  for word in text.split() ]
        text = " ".join([word for word in words if word is not None])
        text = text.replace("\u2047", "-")

    text = text.replace(u"\u098c", u"\u09ef")
    text = unicodedata.normalize("NFC", text)
    text = removeOptionalZW(text)
    text = removePunc(text)

    return text



# Normalize transcript column
dev_df['sentence'] = dev_df["sentence"].apply(lambda x: normalize(x))
test_df['sentence'] = test_df["sentence"].apply(lambda x: normalize(x))
train_df['sentence'] = train_df["sentence"].apply(lambda x: normalize(x))
validated_df['sentence'] = validated_df["sentence"].apply(lambda x: normalize(x))

# Save the updated dataframe to a new TSV file
dev_df.to_csv('./deepspeech-data/cv12-bn/dev.tsv', sep='\t', index=False)
test_df.to_csv('./deepspeech-data/cv12-bn/test.tsv', sep='\t', index=False)
train_df.to_csv('./deepspeech-data/cv12-bn/train.tsv', sep='\t', index=False)
validated_df.to_csv('./deepspeech-data/cv12-bn/validated.tsv', sep='\t', index=False)

