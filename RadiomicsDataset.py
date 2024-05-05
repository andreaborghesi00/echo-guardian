from torch.utils.data import Dataset
import SimpleITK as sitk
import radiomics

class RadiomicsDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extractor.enableAllFeatures()
        self.length = len(df)

        self.image_paths = df['image'].values
        self.mask_paths = df['mask'].values
        self.labels = df['label'].values

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx], sitk.sitkInt32)
        mask = sitk.ReadImage(self.mask_paths[idx], sitk.sitkInt32)
        features = self.extractor.execute(image, mask, voxelBased=False, label=255)
        return features, self.labels[idx]