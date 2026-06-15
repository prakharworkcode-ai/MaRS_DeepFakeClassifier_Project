# Preprocessing

1. The raw audio is saved to drive and loaded into the colab session.
2. The audio is then augmented into various codecs such as .mp3, .wav, .ogg, and .m4a.
3. The augmented audio is sampled at sampling rate of 16000 samples/sec and padded to 2 seconds of audio.

---

# Feature Extraction
1. Librosa split the sampled audios into frames of 400 samples.
2. Then a FFT is performed on the frame at 31.25 Hz frequency resolution.
3. A power spectrum at each frequency is generated at each frequency.
4. A mel filter bank filters the power spectrum between f_min and f_max.
5. We use f_min as 20 Hz and f_max at 8000 Hz as human speech frequencies lie in this band. Some deepfake artifacts lie in higher frequencies so we can't risk to throw away potential information.
6. The deepfake speech is overly smoothened spectrally and human speech has high frequency details due to other noises. These features show up spectrally.
7. The human percieve speech logarithmically i.e. low frequency differences are more observables than differences in higher frequency. So we convert power mels into db mels so that CNN give importance to details in lower frequency bands too.
8. These mel db data is used to train a CNN classifier on one channel.

# Model Architecture

We use a Convulational neural network to classify fake and real samples. We treat the (128, 201) shape mel data as image tensor with only one channel and feed the data to the CNN.

The model uses the following types of layers:
1.The convolution layers learn local spectral and temporal patterns from the log Mel spectrogram while increasing the number of feature maps from 1 → 32 → 64 → 128.
2. ReLU activation layer to introduce non linearity in the model so it can learn complex patterns and features present in the dataset.
3. Max2d pooling layer which takes a max of 2 * 2 region.
4. flattening layer to convert multidimensional tensor into a single tensor.
5. A linear layer which takes the input from flattening layer converts into 256 features and then single value which represents the model prediction of the sample.
6. Dropout is used as a regularization technique to reduce overfitting by randomly disabling neurons

### Output Layer

The final linear layer produces a single logit value. During inference, a sigmoid activation function is applied to convert the logit into a probability between 0 and 1. A threshold of 0.5 is used to classify the sample as either genuine or deepfake speech.

   ## CNN Architecture
   

### Input

| Layer                     | Output Shape  |
| ------------------------- | ------------- |
| Input Log-Mel Spectrogram | (1, 128, 201) |

---

### Feature extraction block

| Layer                                  | Output Shape   |
| -------------------------------------- | -------------- |
| Conv2D (1 → 32, kernel=3, padding=1)   | (32, 128, 201) |
| ReLU                                   | (32, 128, 201) |
| MaxPool2D (2×2)                        | (32, 64, 100)  |
| Conv2D (32 → 64, kernel=3, padding=1)  | (64, 64, 100)  |
| ReLU                                   | (64, 64, 100)  |
| MaxPool2D (2×2)                        | (64, 32, 50)   |
| Conv2D (64 → 128, kernel=3, padding=1) | (128, 32, 50)  |
| ReLU                                   | (128, 32, 50)  |
| MaxPool2D (2×2)                        | (128, 16, 25)  |

---

### Classification head

| Layer                | Output Shape |
| -------------------- | ------------ |
| Flatten              | (51200)      |
| Linear (51200 → 256) | (256)        |
| ReLU                 | (256)        |
| Dropout              | (256)        |
| Linear (256 → 1)     | (1)          |

---

