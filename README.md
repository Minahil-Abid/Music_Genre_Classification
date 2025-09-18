# Music Genre Classification

This project classifies music into genres using the **GTZAN dataset**. By applying both **tabular models** (on extracted features) and **image-based models** (on spectrograms), it demonstrates how machine learning can be applied to music information retrieval.  

## Features
- Explored dataset distribution and class balance
- Preprocessed features (scaling, label encoding, train–test split)
- Trained and compared multiple models:
  - Random Forest
  - XGBoost (untuned, tuned)
  - CNN (from scratch)
  - Transfer Learning (VGG16, fine-tuned)
- Evaluated models using **Accuracy, Classification Report, and Confusion Matrix**  
- Compared **tabular vs. image-based approaches**  

## Dataset
- **Source**: [GTZAN Music Genre Dataset (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- **Features**:  
  - Tabular → Extracted audio features (MFCCs, chroma, spectral features, etc.)  
  - Image → Spectrograms of audio signals  
- **Target**: 10 music genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)  

## Models & Results

### 📊 Tabular Data (features_3_sec.csv)
| Model                   | Accuracy | Precision | Recall | F1-score | Notes |
|--------------------------|----------|-----------|--------|----------|-------|
| Random Forest (untuned)  | 0.883    | 0.88      | 0.88   | 0.88     | Strong baseline |
| XGBoost (untuned)        | 0.914    | 0.91      | 0.91   | 0.91     | Solid improvement |
| XGBoost (tuned)          | 0.943    | 0.94      | 0.94   | 0.94     |⭐ Best overall model |

✅ **XGBoost (tuned)** delivered the highest performance on tabular features, achieving ~94% accuracy.  

### 🖼 Image Data (Spectrograms)
| Model                      | Accuracy | Notes |
|-----------------------------|----------|-------|
| CNN (from scratch)          | ~50%     | Struggled to generalize |
| VGG16 (frozen layers)       | ~53%     | Slight boost from transfer learning |
| VGG16 (fine-tuned)          | ~58%     | Better than frozen, still behind tabular |

⚠️ Image-based models were slower to train and achieved **lower accuracy** than tabular models. With **larger pre-trained models** (ResNet, MobileNet, EfficientNet) and more epochs, performance could improve significantly.  
  

## Run the Project
```bash
git clone https://github.com/Minahil-Abid/Music_Genre_Classification.git
cd Music-Genre-Classification
pip install -r requirements.txt
jupyter notebook Music_Genre_Classification.ipynb
```

## License

MIT License – free to use and share.

