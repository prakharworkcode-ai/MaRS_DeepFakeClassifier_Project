# DeepFakeClassifier Project
---
**Dataset** : https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?select=for-2sec

---
**Description**

This Deep Fake Classifier Project uses a Convulational neural network. The training data is augmented into various format such as .wav, .mp3, .ogg, .m4a. This new augmented training data is then converted into mel db and normalized. This augmented data is used to train the CNN and then evaluated using suggested metrics.

---

**Methodology**
1. Raw audio is uploaded to drive and retrieved by mounting the drive in colab session.
2. The audio is augmented into various codecs so that model doesnot confuse a codec artifact as a feature difference between real and fake samples. In naive training without augmentation. The model was able to confidently classify real and fake samples for validation data but not for testing. It was found that the training and validation data for fake samples had lots of mp3 files. So model was using codec artifacts to classify the data. In case of training with no codec difference it failed to classify implying it didn't learn the feature differences between real and fake samples.
3. The audio is sampled at 16000 samples / sec and then padded to 2 sec data.
4. The raw audio samples are converted into log mel spectrograms to extract necessary features from the data.
5. The mel data is normalized to improve training speed.
6. The mel data is stored in form of .npy files and saved to drive and loaded into session storage for training and testing purposes.
7. A Convulational Neural Network is used to classify the samples. The CNN is trained on the training data. We use binary cross entropy with logits loss function to get a loss of a run.

---

**Pipeline**

1. The raw audio is augmented into various formats so that model does not learn to differ samples on codec artifacts instead learn the feature differences between the two classes.
2. The raw audio is sampled at 16000Hz. Then we pad the samples such that there are not more than 32000 samples i.e. 2 seconds of raw audio.
3. DeepFake differ more in frequency domain than amplitude. So we convert the data to mel. Mel converts the audio such that we get a frequency spectrogram.We can train a CNN to classify between mels of deepfake and genuine samples.
4. Librosa first computes FFT using 512 points.Frequency resolution is 31Hz .Then we process chunks of 400 samples at a time with a hop of 160 samples. Then in mel conversion librosa converts frequency into mel bands.
5. We take the log to represent the power mapped into mel band to get decibels.
6. These decibels are then converted into .npy files and saved to the drive. These files are then loaded into session storage from drive from faster file IO.
7. We form a dataframe to the locations of .npy files. We use dataset to retrieve the data from the file locations in the session storage.
8. These datasets are then loaded into the dataloaders for batch training the model.
9. The model is then evaluated for the metrics suggested by problem statement.

   

```text
Raw audio files
       ↓
Converting into various codecs
       ↓
Mel spectrogram generation
       ↓
Save as .npy
       ↓
Create dataFrame
       ↓
DeepfakeDataset
       ↓
DataLoader
       ↓
Mini-Batches
       ↓
CNN forward pass
       ↓
BCEWithLogitsLoss
       ↓
Backpropagation
       ↓
Adam optimizer
       ↓
Updated weights
```

---

# **METRICS**

**Confusion Matrix**

| Actual \ Predicted | Real | Fake |
|-------------------|------|------|
| Real | 528 | 16 |
| Fake | 40 | 504 |
                   
**Real accuracy** : 97.06%

**Fake accuracy** : 92.65%

**F1 score** : 0.9473684210526315

**EER:** 0.038602

**Accuracy** : 94.85%





