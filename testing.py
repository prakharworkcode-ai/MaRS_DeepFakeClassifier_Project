from torch import nn
from pathlib import Path
import torch
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn.metrics as m

#PASTE THE PATH TO TEST SAMPLES HERE
TEST_SAMPLES_PATH_REAL = r""
TEST_SAMPLES_PATH_FAKE = r""
audio_files_testing_real = list(
    Path(TEST_SAMPLES_PATH_REAL)
    .rglob("*.wav")
)
audio_files_testing_fake = list(
    Path(TEST_SAMPLES_PATH_FAKE)
    .rglob("*.wav")
)

SAVE_PATH_REAL = r"content/processed_test/real"
SAVE_PATH_FAKE = r"content/processed_test/fake"

SAMPLING_FREQUENCY = 16000
MEL_BANDS = 128
def gen_meldb(audio_path, sampling_freq, mel_bands):

    y, sr = librosa.load(
        audio_path,
        sr=sampling_freq
    )

    target_len = 2 * sampling_freq

    if len(y) < target_len:
        y = np.pad(
            y,
            (0, target_len - len(y))
        )

    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=mel_bands,
        fmin=20,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db




def save_the_npy(save_path, file_list, prefix):
  Path(save_path).mkdir(
        parents=True,
        exist_ok=True
    )
  for i,file_name in (enumerate(tqdm(file_list))):
    mel_db = gen_meldb(file_name, SAMPLING_FREQUENCY, MEL_BANDS)
    save_name = save_path +'/' +prefix + "_" + str(i + 1)  + ".npy"
    np.save(save_name, mel_db.astype(np.float32))

save_the_npy(SAVE_PATH_REAL, audio_files_testing_real, "testing_real")
save_the_npy(SAVE_PATH_FAKE, audio_files_testing_fake, "testing_fake")

data_test = []
for file_name in Path(SAVE_PATH_REAL + '/').glob("*.npy"):
  data_test.append({
      "path" : str(file_name),
      "label" : 0
  })
for file_name in Path(SAVE_PATH_FAKE + '/').glob("*.npy"):
  data_test.append({
      "path" : str(file_name),
      "label" : 1
  })

df_test = pd.DataFrame(data_test)



class DeepfakeDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        path = self.df.iloc[idx]["path"]
        label = self.df.iloc[idx]["label"]

        mel = np.load(path)

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)


        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        mel = mel.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return mel, label

test_dataset = DeepfakeDataset(df_test)

BATCH_SIZE = 32
test_dataloader = DataLoader(test_dataset,
      batch_size = BATCH_SIZE,
      shuffle = True,
      num_workers = 0,
      pin_memory = False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class ClassifierModelV1(nn.Module):
  def __init__(self, input_shape : int, output_shape : int):
    super().__init__()
    self.block_1 = nn.Sequential(
    nn.Conv2d(input_shape, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    )
    self.classifier = nn.Sequential(
    nn.Linear(51200, 256),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(256, output_shape)
    )


  def forward(self, x : torch.tensor) -> torch.tensor:
    x = self.block_1(x)
    x = torch.flatten(x, start_dim=1)
    x = self.classifier(x)
    return x

model = ClassifierModelV1(input_shape=1, output_shape=1).to(device)
model.load_state_dict(torch.load(f = "models/MODEL_DFC.pth", map_location=device))


def evaluation(model, test_dataloader):
    all_preds = []
    all_labels = []
    all_scores = []

    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader):
            X = X.to(device)

            logits = model(X)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.flatten())
            preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long().cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    fp, tp, thresholds = m.roc_curve(all_labels, all_scores)
    fn = 1 - tp
    eer_idx = np.nanargmin(np.abs(fn - fp))
    eer = (fn[eer_idx] + fp[eer_idx]) / 2
    cm = m.confusion_matrix(all_labels, all_preds,labels=[0, 1])
    real_acc = cm[0, 0] / cm[0].sum()
    fake_acc = cm[1, 1] / cm[1].sum()
    accuracy = m.accuracy_score(all_labels, all_preds)

    f1_score = m.f1_score(all_labels, all_preds)

    print(f"Confusion Matrix : {cm}")
    print(f"Real accuracy : {real_acc:.4f}")
    print(f"Fake accuracy : {fake_acc:.4f}")
    print(f"F1 score : {f1_score}")
    print("EER:", eer)
    print(f"Accuracy : {accuracy:.4f}")

evaluation(model, test_dataloader)
