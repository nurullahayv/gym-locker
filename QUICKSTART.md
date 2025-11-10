# Quick Start Guide

Bu rehber Gym-Locker projesini hızlıca başlatmanız için adım adım talimatlar içerir.

## 1. Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/YOUR_USERNAME/gym-locker.git
cd gym-locker

# Virtual environment oluşturun (önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Paketi development modda kurun
pip install -e .
```

## 2. Ortamı Test Edin

```bash
# Rastgele aksiyonlar ile ortamı test edin
python demo.py --mode random

# Manuel kontrol (klavye ile)
python demo.py --mode manual
```

## 3. PID Baseline'ı Değerlendirin

```bash
# PID kontrolcüyü 100 bölüm boyunca değerlendirin
python train.py --mode baseline --n-episodes 100
```

Tipik PID baseline skoru: **-1500 ile -500 arası**

## 4. RL Ajanını Eğitin

### Başlangıç İçin (Vektör Durumu + PPO)

```bash
python train.py \
    --mode train \
    --algorithm PPO \
    --state-mode vector \
    --timesteps 100000 \
    --evader-difficulty 0.5
```

Bu eğitim yaklaşık **10-20 dakika** sürer (CPU'ya bağlı).

### İleri Seviye (Görüntü Durumu + SAC)

```bash
python train.py \
    --mode train \
    --algorithm SAC \
    --state-mode image \
    --timesteps 500000
```

Bu eğitim **1-2 saat** sürebilir. Kaggle GPU kullanımı önerilir.

## 5. Eğitilmiş Modeli Test Edin

```bash
# Görselleştirme ile test
python test.py \
    --mode rl \
    --model-path models/PPO_vector_YYYYMMDD_HHMMSS/PPO_final.zip \
    --render

# Hybrid PID+RL test
python test.py \
    --mode hybrid \
    --model-path models/PPO_vector_YYYYMMDD_HHMMSS/PPO_final.zip \
    --pid-weight 0.7 \
    --rl-weight 0.3
```

## 6. Video Kaydı

```bash
# RL ajanının performansını kaydedin
python test.py \
    --mode video \
    --model-path models/PPO_final.zip \
    --output my_agent_demo.mp4 \
    --n-episodes 3
```

## 7. TensorBoard ile İzleme

Eğitim sırasında metrikleri izleyin:

```bash
tensorboard --logdir logs/
```

Tarayıcınızda `http://localhost:6006` adresine gidin.

## 8. Kaggle'da GPU ile Eğitim

1. `notebooks/kaggle_training.ipynb` dosyasını Kaggle'a yükleyin
2. Settings > Accelerator > GPU seçin
3. Tüm hücreleri çalıştırın
4. Eğitilmiş modelleri indirin

## Sık Sorulan Sorular

### Q: Eğitim çok yavaş, nasıl hızlandırırım?
A:
- Vector state kullanın (image yerine)
- Batch size'ı azaltın
- Kaggle GPU kullanın
- `--timesteps` parametresini azaltın (test için)

### Q: Agent öğrenmiyor, ne yapmalıyım?
A:
- Evader difficulty'yi azaltın (0.3'e düşürün)
- Daha fazla timestep eğitin (500k+)
- SAC veya TD3 deneyin
- TensorBoard'da reward grafiklerini kontrol edin

### Q: Pygame/SDL hatası alıyorum
A:
```python
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
```

### Q: PID'den daha iyi skorlar alamıyorum
A:
- PID oldukça iyi bir baseline'dır
- En az 200k-500k timestep eğitin
- Hybrid yaklaşımı deneyin
- Hyperparameter tuning yapın

## Beklenen Performans

| Kontrolcü | Ortalama Ödül | Eğitim Süresi | Zorluk |
|-----------|---------------|---------------|--------|
| Random | -5000 | - | - |
| PID | -1000 | - | Kolay |
| RL (100k) | -800 | 15 dk | Orta |
| RL (500k) | -400 | 1 saat | Orta |
| Hybrid | -300 | - | Kolay |

## Sonraki Adımlar

1. **Hyperparameter Tuning**: Learning rate, batch size vb. ayarlayın
2. **Evader Difficulty**: Zorluk seviyesini artırın
3. **Custom Networks**: Kendi network mimarinizi tasarlayın
4. **MARL**: Öğrenen evader ajanı ekleyin (Phase 2)

## Yardım

Sorunlarla karşılaşırsanız:
- README.md dosyasını inceleyin
- GitHub Issues'a bakın
- Yeni issue açın

Başarılar! 🎯
