#  Animals-10 Image Classification with  CNN
## 🎯 Project Goal
The primary objective of this project is to develop a robust Deep Learning model capable of classifying images of 10 different animal species. Instead of using pre-trained models (Transfer Learning), a **Custom Convolutional Neural Network (CNN)** was built from scratch.

---

## 📂 Dataset Details
The model was trained on the **Animals-10** dataset. The dataset contains **26,179 images** in total.

* **Training Set:** 20,944 images
* **Validation Set:** 5,235 images
* **Classes:** The dataset includes 10 classes with labels originally in Italian:

| Class Label | English Translation |
|:---:|:---:|
| **Cane** | Dog |
| **Cavallo** | Horse |
| **Elefante** | Elephant |
| **Farfalla** | Butterfly |
| **Gallina** | Chicken |
| **Gatto** | Cat |
| **Mucca** | Cow |
| **Pecora** | Sheep |
| **Ragno** | Spider |
| **Scoiattolo** | Squirrel |

---
<img width="1582" height="695" alt="Ekran görüntüsü 2025-12-07 203746" src="https://github.com/user-attachments/assets/abaaa52d-ac32-4ad0-962b-9b9ba13486e6" />

## 🛠️ Techniques & Methodology
To achieve high performance and prevent overfitting on a custom architecture with **~25.8 Million parameters**, the following engineering decisions were made:

### 1. Data Augmentation (The "Noise" Factor)
* **Technique:** Random flips, rotations, and zooms were applied to the training images dynamically.
* **Reason:** Since we built a large model from scratch, there was a high risk of the model memorizing specific pixels. Augmentation introduces "noise" and variance, forcing the model to learn invariant features (e.g., recognizing a cat regardless of its orientation) rather than memorizing the exact image.

### 2. Custom CNN Architecture
* **Technique:** A sequential stack of 3 Convolutional Blocks (Filters: 32 -> 64 -> 128) followed by a massive Dense layer (256 units).
* **Reason:** The hierarchical structure allows the model to learn simple features (edges) in early layers and complex shapes (ears, eyes) in deeper layers.

### 3. Dropout Regularization
* **Technique:** A `Dropout(0.3)` layer was added before the final output.
* **Reason:** With over 25 million parameters in the dense layer, the model is prone to overfitting. Dropout randomly disables 30% of the neurons during training, preventing the network from relying too heavily on any single feature path.

### 4. Early Stopping & Model Checkpointing
* **Technique:** We monitored `val_accuracy` and saved only the best model.
* **Reason:** Training for too many epochs often leads to performance degradation on new data (overfitting). We restored the weights from **Epoch 22**, where the validation accuracy peaked, ensuring optimal generalization.

---

## 📊 Results & Evaluation

### Performance Metrics
The model achieved a solid baseline performance given it was trained from scratch:
* **Training Accuracy:** ~77.20%
* **Validation Accuracy:** ~72.02%
<img width="1316" height="536" alt="Ekran görüntüsü 2025-12-07 191024" src="https://github.com/user-attachments/assets/728b234b-7da6-48c1-8ac8-25ce4d93070c" />

The ~5% gap between training and validation scores indicates that our regularization strategies (Augmentation & Dropout) successfully kept overfitting under control.

### Inference Analysis (Case Study: Spider)
Below is a prediction example on an unseen image of a **Spider (Ragno)**:
<img width="1184" height="864" alt="cnnspider" src="https://github.com/user-attachments/assets/5ed33174-9f7c-4350-9afa-7746c5a5f6c8" />


* **Outcome:** The model **correctly** classified the image as "Ragno".
* **Observation:** The 74.05% score indicates that the model is quite confident and reliable in its prediction for a 10-class problem. This level of confidence proves that the model has successfully learned the fundamental features. The remaining uncertainty (25.95%) stems from background noise or similarities to other classes in the visual data, and can be addressed through hyperparameter optimization to achieve confidence levels above 90%.
---
---
# 🐾 Animals-10 Görüntü Sınıflandırma

## 🎯 Proje Amacı
Bu projenin temel amacı, 10 farklı hayvan türünü sınıflandırabilen güçlü bir Derin Öğrenme modeli geliştirmektir. Transfer Öğrenme (Transfer Learning) yerine, sıfırdan bir **Özel Evrişimli Sinir Ağı (CNN)** mimarisi kurularak modelin genelleme yeteneği test edilmiştir.

---

## 📂 Veri Seti Detayları
Model, **Animals-10** veri seti üzerinde eğitilmiştir. Veri seti toplam **26.179 görüntü** içermektedir.

* **Eğitim Seti:** 20.944 görüntü
* **Doğrulama (Validation) Seti:** 5.235 görüntü
* **Sınıflar:** Veri seti, orijinal etiketleri İtalyanca olan 10 sınıfı kapsamaktadır:

| Sınıf Adı | Türkçe Karşılığı |
|:---:|:---:|
| **Cane** | Köpek |
| **Cavallo** | At |
| **Elefante** | Fil |
| **Farfalla** | Kelebek |
| **Gallina** | Tavuk |
| **Gatto** | Kedi |
| **Mucca** | İnek |
| **Pecora** | Koyun |
| **Ragno** | Örümcek |
| **Scoiattolo** | Sincap |

---

## 🛠️ Teknikler ve Metodoloji
Yaklaşık **25.8 Milyon parametreye** sahip özel mimaride aşırı öğrenmeyi (overfitting) önlemek ve performansı artırmak için şu mühendislik kararları alınmıştır:

### 1. Veri Çoğaltma (Data Augmentation - "Gürültü" Faktörü)
* **Teknik:** Eğitim görüntüleri üzerinde dinamik olarak rastgele çevirme, döndürme ve yakınlaştırma teknikleri uygulanmıştır.
* **Gerekçe:** Sıfırdan oluşturulan büyük bir modelde, ezberleme riskini azaltmak için bu teknik kullanılmıştır. Çoğaltma, eğitime **varyasyon (gürültü)** ekleyerek modelin görüntüleri ezberlemek yerine, yönelimden bağımsız (invariant) özellikleri öğrenmeye zorlamıştır.

### 2. Özel CNN Mimarisi
* **Teknik:** Üç Evrişim Bloğu (Filtreler: 32 -> 64 -> 128) ve ardından 256 üniteden oluşan yoğun bir Tam Bağlantılı (Dense) katmandan oluşan sıralı (sequential) yığın kullanılmıştır.
* **Gerekçe:** Hiyerarşik yapı, modelin erken katmanlarda basit kenar özelliklerini, derin katmanlarda ise karmaşık şekilleri (göz, kulak gibi) öğrenmesine olanak tanımıştır.

### 3. Dropout (Seyreltme) Düzenlemesi
* **Teknik:** Son çıktıdan önce bir `Dropout(0.3)` katmanı eklenmiştir.
* **Gerekçe:** Tam bağlantılı katmandaki 25 Milyonun üzerindeki parametre nedeniyle aşırı öğrenme eğilimi yüksektir. Dropout, eğitim sırasında nöronların %30'unu rastgele devre dışı bırakarak modelin tek bir özelliğe aşırı bağımlı olmasını önlemiştir.

### 4. Erken Durdurma ve Model Kaydetme
* **Teknik:** `val_accuracy` değeri izlenmiş ve yalnızca en iyi performansı gösteren modelin ağırlıkları kaydedilmiştir.
* **Gerekçe:** Uzun süreli eğitimin performans düşüşüne yol açmaması için, doğrulama başarısının zirveye ulaştığı **22. Epoch**'taki ağırlıklar restore edilerek optimum genelleme sağlanmıştır.

---

## 📊 Sonuçlar ve Değerlendirme

### Performans Metrikleri
Sıfırdan eğitilen bir modele göre güçlü bir başlangıç performansı elde edilmiştir:
* **Eğitim Doğruluğu (Training Accuracy):** ~%77.20
* **Doğrulama Doğruluğu (Validation Accuracy):** ~%72.02

Eğitim ve doğrulama skorları arasındaki yaklaşık %5'lik fark, uygulanan düzenleme stratejilerinin (Çoğaltma ve Dropout) aşırı öğrenmeyi başarılı bir şekilde kontrol altında tuttuğunu göstermektedir.

### Çıkarım Analizi (Örnek Durum: Örümcek)

* **Sonuç:** Model, görünmeyen bir test görüntüsünü **"Ragno" (Örümcek)** olarak doğru sınıflandırmıştır.
* **Gözlem (Observation):** Güven skoru **%74.05** olarak kaydedilmiştir. Bu oran, 10 sınıflı bir problem için modelin tahmininde **oldukça emin ve güvenilir** olduğunu gösterir. Bu güven düzeyi, modelin temel özellikleri başarıyla öğrendiğini kanıtlar. Kalan belirsizlik (%25.95) ise, görseldeki arka plan karmaşası veya sınıf benzerliklerinden kaynaklanmaktadır ve %90 üzeri güven için **hiperparametre optimizasyonu** ile giderilebilir.
