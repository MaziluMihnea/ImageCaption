#%% md
# ## Import Modules
#%%
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
#%%
BASE_DIR = './dataset'
WORKING_DIR = './working'
#%% md
# ## Extract Image Features
#%%
# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
print(model.summary())
# %%
# Extract features from images
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tqdm import tqdm
import os

features = {}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    # Load the image from file
    img_path = os.path.join(directory, img_name)  # Use os.path.join for better compatibility
    image = load_img(img_path, target_size=(224, 224))

    # Convert image pixels to numpy array
    image = img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Preprocess the image for VGG16
    image = preprocess_input(image)

    # Extract features
    feature = model.predict(image, verbose=0)

    # Get image ID
    image_id = os.path.splitext(img_name)[0]  # Handles extensions like .jpg, .png

    # Store feature
    features[image_id] = feature

#%%
# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
#%%
# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
#%% md
# ## Load the Captions Data
#%%
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
#%%
# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

#%%

def create_image_caption_mapping(captions_doc):
    """
    Creates a mapping of image IDs to their corresponding captions.

    Args:
    captions_doc (str): A string containing image IDs and captions,
                        separated by commas and newlines.

    Returns:
    dict: A dictionary mapping image IDs (without file extensions) to lists of captions.
    """
    mapping = {}

    # Process each line in the document
    for line in tqdm(captions_doc.split('\n')):
        # Split the line by comma
        tokens = line.split(',')

        # Skip lines with less than two tokens
        if len(tokens) < 2:
            continue

        image_id, caption = tokens[0], tokens[1:]

        # Remove the extension from the image ID
        image_id = image_id.split('.')[0]

        # Convert the caption list to a single string
        caption = " ".join(caption).strip()

        # Initialize the list for the image ID if not already present
        if image_id not in mapping:
            mapping[image_id] = []

        # Add the caption to the list for the image ID
        mapping[image_id].append(caption)

    return mapping
#%%
len(mapping)
#%% md
# ## Preprocess Text Data
#%%
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in         caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

#%%
# before preprocess of text
mapping['1000268201_693b08cb0e']
#%%
# preprocess the text
clean(mapping)
#%%
# after preprocess of text
mapping['1000268201_693b08cb0e']
#%%
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
#%%
len(all_captions)
#%%
all_captions[:10]
#%%
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
#%%
vocab_size
#%%
# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length
#%% md
# ## Train Test Split
#%%
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

#%%
def process_caption_data(mapping, split_ratio=0.9):
    """
    Processes caption data by collecting all captions, tokenizing, and splitting into train and test sets.

    Args:
    mapping (dict): A dictionary where keys are image IDs and values are lists of captions.
    split_ratio (float): Ratio of the data to use for training (default is 0.9).

    Returns:
    dict: A dictionary containing:
        - 'all_captions': List of all captions.
        - 'vocab_size': Size of the vocabulary.
        - 'max_length': Maximum length of captions.
        - 'train_ids': List of image IDs for training.
        - 'test_ids': List of image IDs for testing.
        - 'tokenizer': The tokenizer fit on the captions.
    """
    # Collect all captions
    all_captions = [caption for key in mapping for caption in mapping[key]]

    # Tokenize the captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1

    # Get the maximum length of captions
    max_length = max(len(caption.split()) for caption in all_captions)

    # Train-test split
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * split_ratio)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]

    return {
        'all_captions': all_captions,
        'vocab_size': vocab_size,
        'max_length': max_length,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'tokenizer': tokenizer
    }

#%%
# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], 	num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0
#%% md
# ## Model Creation
#%%
# encoder model
# image feature layers
inputs1 = Input(shape=(4096,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)
#%%
# train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#%%
# save the model
model.save(WORKING_DIR+'/best_model.h5')
#%%
# save the model
model.save(WORKING_DIR + '/best_model.keras')

#%%
from tensorflow.keras.models import load_model

# Load the model from the .keras file
model_path = os.path.join(WORKING_DIR, 'best_model.keras')
loaded_model = load_model(model_path)

#%% md
# ## Generate Captions for Image
#%%
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
#%%
# generate caption for an image
def predict_caption(loaded_model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length, padding='post')
        # predict next word
        yhat = loaded_model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text
#%%
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(loaded_model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#%%
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#%%
from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(loaded_model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
#%%
generate_caption("1003163366_44323f5815.jpg")

#%%
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap

class ImageCaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Caption Generator")

        # Main widget and layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Image display label
        self.image_label = QLabel("No image uploaded")
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(400, 400)
        self.layout.addWidget(self.image_label)

        # Caption display label
        self.caption_label = QLabel("Caption will appear here")
        self.layout.addWidget(self.caption_label)

        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

    def upload_image(self):
        # Open file dialog to select image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp)",
            options=options
        )
        if file_path:
            # Display the full image
            self.display_image(file_path)
            # Pass both the full path and basename to caption generator
            self.generate_caption_action(file_path)

    def display_image(self, file_path):
        """Displays the selected image."""
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    def generate_caption_action(self, file_path):
        """Generates and displays the caption, showing only the filename in the caption label."""
        # Extract the file name from the full path
        file_name = os.path.basename(file_path)
        try:
            # If generate_caption needs the full path, pass file_path
            # But display only the file name to the user
            caption = generate_caption(file_path)
            self.caption_label.setText(f"Caption for {file_name}: {caption}")
        except Exception as e:
            self.caption_label.setText(f"Error generating caption for {file_name}: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageCaptionApp()
    window.show()
    sys.exit(app.exec_())

