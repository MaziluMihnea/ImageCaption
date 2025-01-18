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
    return y_pred, captions

#%%
import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget
)

class ImageCaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Caption Generator")

        # Make the window bigger
        self.resize(1000, 600)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # ----- MAIN LAYOUT -----
        # Create a central widget and assign a horizontal layout to it.
        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # ----- LEFT SIDE (IMAGE PREVIEW) -----
        # A container for the image
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)

        # Image display label
        self.image_label = QLabel("No image uploaded")
        self.image_label.setScaledContents(True)
        # Set a preferred size for the image area
        self.image_label.setFixedSize(450, 450)
        self.left_layout.addWidget(self.image_label)

        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.left_layout.addWidget(self.upload_button)

        # ----- RIGHT SIDE (CAPTIONS) -----
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)

        # Caption display label
        self.caption_label = QLabel("Caption will appear here")
        # If you have both a 'caption' and a 'predicted caption', you could add another label:
        self.predicted_caption_label = QLabel("Predicted caption will appear here")

        self.right_layout.addWidget(self.caption_label)
        self.right_layout.addWidget(self.predicted_caption_label)

        # Add left and right widgets to the main layout
        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)

    # ----- DRAG & DROP EVENTS -----
    def dragEnterEvent(self, event):
        """Allow drag if it contains URLs (i.e., file paths)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle file drop."""
        if event.mimeData().hasUrls():
            # Take the first file from the dropped URLs
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.display_image(file_path)
            self.generate_caption_action(file_path)

    # ----- UPLOAD IMAGE -----
    def upload_image(self):
        """Open a file dialog to select an image."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp)",
            options=options
        )
        if file_path:
            self.display_image(file_path)
            self.generate_caption_action(file_path)

    # ----- DISPLAY IMAGE -----
    def display_image(self, file_path):
        """Displays the selected image on the left side."""
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    # ----- GENERATE CAPTION -----
    def generate_caption_action(self, file_path):
        """Generates and displays the captions."""
        file_name = os.path.basename(file_path)
        try:
            pred_caption, caption = generate_caption(file_name)  # or pass file_path
            if isinstance(caption, list):
                # Convert the list into a single string with each item on a new line
                caption_text = "\n".join(caption)
                self.caption_label.setText(f"Caption:\n{caption_text}")
            else:
                # If it’s not a list, just display it as is
                self.caption_label.setText(f"Caption: {caption}")
            # If you have a separate "predicted caption," set it here:
            self.predicted_caption_label.setText(f"Predicted caption: {pred_caption}")
        except Exception as e:
            self.caption_label.setText(f"Error generating caption for {file_name}: {str(e)}")


def apply_dark_theme(app):
    """
    Applies a dark Fusion theme to the Qt application.
    """
    app.setStyle("Fusion")
    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a dark theme to the entire application
    apply_dark_theme(app)

    window = ImageCaptionApp()
    window.show()
    sys.exit(app.exec_())


