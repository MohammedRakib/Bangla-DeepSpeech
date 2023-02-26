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


dev_list = dev_df["sentence"].to_list()
test_list = test_df["sentence"].to_list()
train_list = train_df["sentence"].to_list()
validated_list = validated_df["sentence"].to_list()


dev =  list(set("".join(dev_list)))
test =  list(set("".join(test_list)))
train =  list(set("".join([t for t in train_list if isinstance(t, str)])))
# train =  list(set("".join(train_list)))
validated =  list(set("".join([v for v in validated_list if isinstance(v, str)])))



# unique = list(set(dev) | set(test) | set(train) | set(validated))

unique2 = list(set(dev+test+train+validated))
unique2.sort()

print(f"\n\n Unique: {unique2} \n\n")

vocab_dict = {
#   "|": 0,
  "_": 1,
#   "a": 2,
#   "b": 3,
#   "c": 4,
#   "d": 5,
#   "e": 6,
#   "f": 7,
#   "g": 8,
#   "h": 9,
#   "i": 10,
#   "j": 11,
#   "k": 12,
#   "l": 13,
#   "m": 14,
#   "n": 15,
#   "o": 16,
#   "p": 17,
#   "r": 18,
#   "s": 19,
#   "t": 20,
#   "u": 21,
#   "v": 22,
#   "w": 23,
#   "x": 24,
#   "y": 25,
#   "z": 26,
#   '\x93': 27,
#   '\x94': 28,
  "œ": 29,
#   "।": 30,
  "ঁ": 31,
  "ং": 32,
  "ঃ": 33,
  "অ": 34,
  "আ": 35,
  "ই": 36,
  "ঈ": 37,
  "উ": 38,
  "ঊ": 39,
  "ঋ": 40,
  "এ": 41,
  "ঐ": 42,
  "ও": 43,
  "ঔ": 44,
  "ক": 45,
  "খ": 46,
  "গ": 47,
  "ঘ": 48,
  "ঙ": 49,
  "চ": 50,
  "ছ": 51,
  "জ": 52,
  "ঝ": 53,
  "ঞ": 54,
  "ট": 55,
  "ঠ": 56,
  "ড": 57,
  "ঢ": 58,
  "ণ": 59,
  "ত": 60,
  "থ": 61,
  "দ": 62,
  "ধ": 63,
  "ন": 64,
  "প": 65,
  "ফ": 66,
  "ব": 67,
  "ভ": 68,
  "ম": 69,
  "য": 70,
  "র": 71,
  "ল": 72,
  "শ": 73,
  "ষ": 74,
  "স": 75,
  "হ": 76,
  "়": 77,
  "া": 78,
  "ি": 79,
  "ী": 80,
  "ু": 81,
  "ূ": 82,
  "ৃ": 83,
  "ে": 84,
  "ৈ": 85,
  "ো": 86,
  "ৌ": 87,
  "্": 88,
  "ৎ": 89,
  "ৗ": 90,
  "ড়": 91,
  "ঢ়": 92,
  "য়": 93,
  "০": 94,
  "১": 95,
  "২": 96,
  "৩": 97,
  "৪": 98,
  "৫": 99,
  "৬": 100,
  "৭": 101,
  "৮": 102,
  "৯": 103,
  "ৰ": 104,
#   '\u200c': 105,
#   '\u200d': 106,
#   '\u200e': 107, 
#   "[UNK]": 108,
#   "[PAD]": 109,
#   "<s>": 110,
#   "</s>": 111,
}

# existing = list(vocab_dict.keys())

# print(f"\n\n Existing:{existing}")

# total = list(set(unique2 + existing))

# # Save the updated dataframe to a new TSV file
# print(unique)

# print("\n\n LOOOOL \n\n")

# print(unique2)

# lol = ["Rafi is a dog", "Meem is a cat", None]
# lol = ["Rafi is a dog", "Meem is a cat", None, 9]
# lol = ["Rafi is a dog", "Meem is a cat"]

# out = list(set("".join([l for l in lol if isinstance(l, str)])))
# out = list(set("".join([str(l) for l in lol])))
# print(out)


# Open file for writing
with open('./deepspeech-data/cv12-bn/alphabet.txt', 'w') as file:

    # Write each character to file on a new line
    for char in unique2:
        file.write(char + '\n')