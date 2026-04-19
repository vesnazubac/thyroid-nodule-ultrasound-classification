# Klasifikacija nodula štitne žlezde pomoću CNN i Transfer Learning pristupa

Ovaj repozitorijum sadrži implementaciju različitih arhitektura dubokih neuronskih mreža za binarnu klasifikaciju nodula štitne žlezde na osnovu ultrazvučnih snimaka. Projekat obuhvata ceo proces: od obrade medicinskih snimaka i XML anotacija, preko augmentacije podataka, do poređenja performansi baznih i pretreniranih modela.

##  Pregled projekta

Glavni fokus je na primeni **Transfer Learning-a** kako bi se postigla visoka preciznost na specifičnom medicinskom skupu podataka.
Implementirani modeli uključuju:
- **Baseline CNN**: Prilagođena arhitektura sa 3 konvoluciona sloja.
- **Transfer Learning modeli**: DenseNet121, ResNet50, EfficientNet-B0, InceptionV3, VGG16 i ConvNeXt Tiny.
- **AutoML**: Automatizovano testiranje hiperparametara (optimizeri, learning rate, batch size).

##  Tehnologije i zavisnosti

Za pokretanje koda neophodno je imati instaliran Python 3.8+ i sledeće biblioteke:

- **PyTorch & torchvision**: Osnovni framework za modele i transformacije.
- **timm**: Za pristup naprednim pretreniranim arhitekturama (EfficientNet, ConvNeXt).
- **Scikit-learn**: Za stratifikovanu podelu podataka i evaluaciju (F1, Precision, Recall).
- **Pillow (PIL)**: Za učitavanje i obradu slika.
- **Pandas & NumPy**: Za rad sa podacima.
- **Matplotlib & Seaborn**: Za vizuelizaciju rezultata.

### Instalacija:
```bash
pip install torch torchvision timm scikit-learn pandas numpy matplotlib seaborn pillow
```

Ovaj projekat koristi duboko učenje (Deep Learning) za automatizovanu klasifikaciju ultrazvučnih snimaka štitne žlezde u dve kategorije (benigni/maligni). Fokus je na poređenju standardnih CNN arhitektura sa modernim pristupima poput Transfer Learning-a i AutoML-a.

### Google Colab & GPU Akceleracija

Projekat je primarno razvijen i optimizovan za rad u **Google Colab** okruženju. S obzirom na kompleksnost modela kao što su *ConvNeXt* i *DenseNet*, korišćenje grafičkog procesora (GPU) je neophodno za efikasno treniranje.

**Kako podesiti Colab:**
1. Otvorite `.ipynb` fajl u Google Colab-u.
2. Idite na **Runtime** -> **Change runtime type**.
3. Pod **Hardware accelerator** izaberite **T4 GPU** (ili jači).
4. Pokrenite prvu ćeliju za montiranje Google Drive-a kako biste pristupili podacima:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
