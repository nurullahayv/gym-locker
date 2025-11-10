# Kaggle Training Setup Guide

Bu rehber, Gym-Locker environment'ını Kaggle'da GPU ile eğitmek için adım adım kurulum talimatlarını içerir.

## 📋 Gereksinimler

- Kaggle hesabı ([kaggle.com](https://kaggle.com))
- GitHub hesabı (kodları saklamak için)
- Telefon doğrulaması (GPU kullanımı için)

## 🚀 Adım Adım Kurulum

### Adım 1: GitHub Repository'yi Hazırla

1. **Repository'yi Fork Et veya Clone Et:**
   ```bash
   # Kendi GitHub hesabına fork et
   # Veya bu repo'yu kullan: https://github.com/YOUR_USERNAME/gym-locker
   ```

2. **Kodları Push Et:**
   ```bash
   git add .
   git commit -m "Ready for Kaggle training"
   git push origin main
   ```

### Adım 2: Kaggle Hesabını Ayarla

1. **Kaggle'a Giriş Yap:**
   - [kaggle.com](https://kaggle.com) adresine git
   - Hesabın yoksa oluştur

2. **Telefon Doğrulaması Yap (GPU için gerekli):**
   - Settings → Account → Phone Verification
   - Telefon numaranı doğrula

3. **GPU Erişimi Kontrol Et:**
   - Yeni notebook oluştur
   - Settings → Accelerator → GPU seçeneği varsa OK!

### Adım 3: Kaggle Notebook'u Yükle

#### **Yöntem 1: Notebook Dosyasını Upload Et (Önerilen)**

1. **Notebook Dosyasını İndir:**
   - Repository'den `notebooks/kaggle_training.ipynb` dosyasını indir

2. **Kaggle'a Yükle:**
   - Kaggle → Code → New Notebook
   - File → Upload Notebook
   - `kaggle_training.ipynb` dosyasını seç

3. **Settings Ayarları:**
   - Accelerator: **GPU T4 x2** (veya mevcut en iyi GPU)
   - Environment: **Python**
   - Internet: **ON** (git clone için gerekli)
   - Persistence: **Files only** (modelleri saklamak için)

#### **Yöntem 2: Manuel Oluşturma**

1. New Notebook oluştur
2. Her cell'i `kaggle_training.ipynb` dosyasından kopyala-yapıştır

### Adım 4: Repository'yi Kaggle'da Clone Et

**Notebook'taki 2. Cell'i Çalıştır:**

```python
# Repository'yi clone et
!git clone https://github.com/YOUR_USERNAME/gym-locker.git /kaggle/working/gym-locker

# Package'ı yükle
!pip install -e /kaggle/working/gym-locker

print("✓ Repository cloned and installed!")
```

**ÖNEMLİ:** `YOUR_USERNAME` kısmını kendi GitHub kullanıcı adınla değiştir!

### Adım 5: Training Parametrelerini Ayarla

**CONFIG Cell'ini Düzenle:**

```python
CONFIG = {
    'algorithm': 'PPO',  # Options: PPO, SAC, TD3
    'state_mode': 'vector',  # 'vector' önerilen (8D observation)
    'architecture': 'simple',  # 'simple' veya 'deep'
    'total_timesteps': 500000,  # 500K timestep (~1-2 saat GPU ile)
    'evader_difficulty': 0.5,
    'evader_speed_multiplier': 1.5,
    'observation_delay': 0,  # 0 = no delay (Markovian)
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'checkpoint_freq': 10000,
    'eval_freq': 5000,
}
```

**Önerilen Başlangıç Ayarları:**
- **Algorithm**: PPO (en stabil)
- **Timesteps**: 500K (orta seviye, ~1-2 saat)
- **State Mode**: vector (daha hızlı)

### Adım 6: Training'i Başlat

1. **Run All Cells:**
   - Notebook → Run → Run All
   - Veya her cell'i tek tek çalıştır (daha kontrollü)

2. **İlerlemeyi İzle:**
   - TensorBoard logs: `/kaggle/working/tensorboard/`
   - Progress bar: Her 100 timestep'te güncellenir
   - Evaluation: Her 5000 timestep'te

3. **Beklenen Süre:**
   - 100K timesteps: ~20-30 dakika
   - 500K timesteps: ~1-2 saat
   - 1M timesteps: ~3-4 saat

### Adım 7: Modelleri İndir

**Training Bittiğinde:**

1. **Output Panel'i Aç:**
   - Sağ taraftaki Output sekmesi

2. **Model Dosyalarını İndir:**
   ```
   ✓ final_PPO_model.zip (son model)
   ✓ best_models/best_model.zip (en iyi model)
   ✓ checkpoints/*.zip (ara checkpointler)
   ```

3. **Yerel Bilgisayarda Test Et:**
   ```bash
   python test.py --mode rl --model final_PPO_model.zip
   ```

## 📊 Training Sonuçlarını Anlama

### Evaluation Metrikleri

```python
Mean Reward: -450.23 ± 125.45
Mean Episode Length: 234.5
Mean Lock Time: 45.2
```

**Ne Demek?**
- **Mean Reward**: Ortalama ödül (yüksek = iyi)
  - Negatif normal (başlangıç)
  - Zamanla artar
  - 0'a yaklaşırsa iyi performans

- **Episode Length**: Ortalama episode süresi (step)
  - Çok kısa: Target hemen kaçıyor
  - Orta (~200-500): İyi
  - Çok uzun: Takip ediyor ama lock olmuyor

- **Lock Time**: Lock-on süresi (step)
  - 150+ step (5 saniye): Mükemmel! Bonus kazanıyor
  - 50-150 step: İyi, lock başarılı
  - <50 step: Zayıf, lock sürdüremiyor

### TensorBoard (Opsiyonel)

Kaggle'da TensorBoard çalıştır:

```python
# Yeni cell ekle
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/tensorboard/
```

## 🔧 Troubleshooting

### Problem 1: "ModuleNotFoundError: gym_locker"

**Çözüm:**
```python
# Cell'i tekrar çalıştır:
!pip install -e /kaggle/working/gym-locker
import gym_locker
```

### Problem 2: "CUDA out of memory"

**Çözüm:**
- `batch_size`'ı azalt: 64 → 32
- `n_steps`'i azalt: 2048 → 1024
- Notebook'u restart et

### Problem 3: "No module named 'simple_pid'"

**Çözüm:**
```python
# İlk cell'de:
!pip install simple-pid
```

### Problem 4: Training Çok Yavaş

**Optimizasyonlar:**
- GPU seçili mi kontrol et (Settings → Accelerator)
- `state_mode`: image → vector (daha hızlı)
- `architecture`: deep → simple
- Gereksiz cell'leri comment out et

### Problem 5: Model Converge Etmiyor

**Deneyebileceğin Şeyler:**
```python
CONFIG = {
    'learning_rate': 1e-4,  # Daha düşük LR
    'n_steps': 4096,  # Daha fazla step
    'batch_size': 128,  # Daha büyük batch
    'total_timesteps': 1000000,  # Daha uzun training
}
```

## 💡 İpuçları

### Training Hızlandırma

1. **Vector state kullan** (image değil)
2. **Simple architecture** seç (deep değil)
3. **Checkpoint frequency'yi artır** (10K → 50K)
4. **Eval frequency'yi azalt** (5K → 10K)

### Model Quality İyileştirme

1. **Daha uzun train et** (500K → 1M timesteps)
2. **Evader difficulty'yi kademeli artır** (0.5 → 0.7 → 1.0)
3. **Multiple runs yap** (farklı random seeds)
4. **Hyperparameter tuning** (learning rate, batch size)

### Kaggle Session Sınırları

- **GPU Quota**: Haftada 30 saat
- **Session Timeout**: 9-12 saat (idle durumda daha az)
- **Otomatik Save**: Notebook otomatik kaydedilir
- **Checkpoint Kullan**: Uzun training için

**Uzun Training İçin:**
```python
# Checkpoint'lerden devam et
if os.path.exists('./checkpoints/PPO_model_500000_steps.zip'):
    model = PPO.load('./checkpoints/PPO_model_500000_steps.zip', env=train_env)
    model.learn(total_timesteps=500000)  # Devam et
```

## 📚 Ek Kaynaklar

- **Stable-Baselines3 Docs**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- **Kaggle GPU Guide**: [https://www.kaggle.com/docs/efficient-gpu-usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- **Gym-Locker README**: `../README.md`

## ✅ Checklist

Training başlatmadan önce kontrol et:

- [ ] GPU seçili (Settings → Accelerator)
- [ ] Internet açık (git clone için)
- [ ] Repository doğru URL (YOUR_USERNAME değiştirildi)
- [ ] CONFIG parametreleri ayarlandı
- [ ] Checkpoint ve best_model klasörleri oluşturuldu
- [ ] Telefon doğrulaması yapıldı (GPU için)

## 🎯 Beklenen Timeline

| Timesteps | Süre | Beklenen Performans |
|-----------|------|---------------------|
| 100K | 20-30 dk | Baseline, henüz zayıf |
| 250K | 45-60 dk | Orta, bazı locklar başarılı |
| 500K | 1-2 saat | İyi, tutarlı lock |
| 1M | 3-4 saat | Çok iyi, uzun lock süresi |

İyi eğitimler! 🚀
