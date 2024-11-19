# Google ReRanker - Arama Sonuçlarını Analiz ve Yeniden Sıralama Aracı

FOR ENGLISH EXPLANATION read here; [https://metehan.ai/blog/an-open-source-google-reranker-transforming-google-results-with-ai-magic/]

Google ReRanker, arama sonuçlarını analiz ederek yeniden sıralamak ve belirli SEO faktörlerini optimize etmek için geliştirilmiş bir Python aracıdır. Bu araç, NLP ve Transformer modellerini kullanarak, alaka düzeyi, içerik kalitesi, domain otoritesi gibi çeşitli faktörlere dayalı sıralama yapar.

## Özellikler / Features

- **Alaka Düzeyi (Relevance Score)**: Arama sorgusu ile içerik arasındaki benzerliği hesaplar.
- **İçerik Kalitesi (Content Quality)**: İçeriğin okunabilirliğini, kelime çeşitliliğini ve uzunluğunu analiz eder.
- **Domain Otoritesi (Domain Authority)**: Belirli domain'lerin otoritesini ölçer.
- **Güncellik (Freshness)**: İçeriğin ne kadar güncel olduğunu değerlendirir.
- **Kullanıcı Niyeti Uyum Skoru (User Intent Match)**: Sorgunun kullanıcı niyetiyle uyumluluğunu analiz eder.
- **Teknik SEO Faktörleri (Technical SEO)**: Başlık, meta açıklama ve URL uzunluklarını değerlendirir.
- **Kullanıcı Sinyalleri (User Signals)**: Varsayılan olarak bounce rate ve oturum süresini hesaba katar.

## Kurulum / Installation

### Gerekli Kütüphaneler (Dependencies)

Aşağıdaki Python kütüphanelerini yükleyerek başlayabilirsiniz:

```bash
pip install numpy pandas scikit-learn spacy requests beautifulsoup4 torch transformers
```

### Spacy Modeli

Spacy'nin dil modelini yükleyin:

```bash
python -m spacy download en_core_web_sm
```

## Nasıl Kullanılır? (How to Use)

### 1. Kodun İndirilmesi ve Çalıştırılması

Bu depoyu klonlayın ve GoogleReRanker sınıfını kullanarak analiz yapabilirsiniz.

```bash
git clone https://github.com/username/google-reranker.git
cd google-reranker
python main.py
```

### 2. Arama Sorgusu Girin

Program çalıştırıldığında, analiz etmek istediğiniz sorguyu girin:

```plaintext
Arama sorgusunu girin: python data analysis tools
```

### 3. Sonuçların Yeniden Sıralanması

Araç, Google'dan arama sonuçlarını çeker, içerikleri analiz eder ve belirlenen faktörlere göre yeniden sıralar.

## Türkçe Örnek Kullanım (Sample Usage in Turkish)

### Kodda Yapılabilecek Özelleştirmeler

**Faktör Ağırlıkları**: Kodda `self.weights` parametresini değiştirerek farklı sıralama faktörlerine ağırlık verebilirsiniz. Örneğin:

```python
def _initialize_weights(self):
    return {
        'relevance_score': 0.30,
        'content_quality': 0.25,
        'domain_authority': 0.10,
        'user_intent_match': 0.15,
        'freshness': 0.10,
        'technical_seo': 0.05,
        'user_signals': 0.05
    }
```

**Domain Otoritesi Listesi**: `_load_domain_authority()` fonksiyonuna özel bir domain listesi ekleyebilirsiniz.

### Arama Sonuçlarının Analizi

Arama sonuçlarını analiz eder ve sıralar:

```python
reranker = GoogleReRanker()
results = reranker.analyze_and_rerank("Python programlama araçları")
for result in results['reranked_results']:
    print(f"{result['title']} - Skor: {result['total_score']}")
```

### Analiz Sonuçlarının Kaydedilmesi

Sonuçlar bir metin dosyasına kaydedilir:

```plaintext
Re-Ranking Analiz Raporu - 2024-11-18 15:42:23

Sorgu: Python programlama araçları
================================================================================

YENİDEN SIRALANMIŞ SONUÇLAR:
--------------------------------------------------------------------------------

1. Python.org - Skor: 0.8943
2. Real Python - Skor: 0.8721
3. DataCamp - Skor: 0.8417
```

## English Usage

### Customizations

**Adjusting Factor Weights**: Modify `self.weights` to prioritize specific ranking factors. Example:

```python
def _initialize_weights(self):
    return {
        'relevance_score': 0.30,
        'content_quality': 0.25,
        'domain_authority': 0.10,
        'user_intent_match': 0.15,
        'freshness': 0.10,
        'technical_seo': 0.05,
        'user_signals': 0.05
    }
```

**Domain Authority List**: Add your custom domain authority list in `_load_domain_authority()`.

### Analyze Search Results

Fetch and rerank results:

```python
reranker = GoogleReRanker()
results = reranker.analyze_and_rerank("Best Python data analysis tools")
for result in results['reranked_results']:
    print(f"{result['title']} - Score: {result['total_score']}")
```

### Save Results to File

Results are saved in a text file:

```plaintext
Re-Ranking Analysis Report - 2024-11-18 15:42:23

Query: Best Python data analysis tools
================================================================================

RERANKED RESULTS:
--------------------------------------------------------------------------------

1. Python.org - Score: 0.8943
2. Real Python - Score: 0.8721
3. DataCamp - Score: 0.8417
```

## Kısıtlamalar / Limitations

- Şu and Google arama sonuçlarından çekilen snippet'ler içerik analizinde kullanılır; tüm içerik alınmaz, bu özelliği geliştireceğim.
- Tam içerik analizi geldiğinde her sayfa scrape edilecek, etik kurallar çerçevesinde kullanın.
- Kullandığınız kod terminalini kapatırsanız analiz yarıda kesilir. 
