import numpy as np
from sklearn.preprocessing import MinMaxScaler
import spacy
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class GoogleReRanker:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Load NLP model
        self.nlp = spacy.load('en_core_web_sm')
        
        # BERT model initialization
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Feature scaler
        self.scaler = MinMaxScaler()
        
        # Weights parameters
        self.weights = self._initialize_weights()
        
        # Domain authority database
        self.domain_authority = self._load_domain_authority()

    def _initialize_weights(self):
        """initialize Re-ranking factors"""
        return {
            'relevance_score': 0.25,
            'content_quality': 0.20,
            'domain_authority': 0.15,
            'user_intent_match': 0.15,
            'freshness': 0.10,
            'technical_seo': 0.10,
            'user_signals': 0.05
        }

    def _load_domain_authority(self):
        """Domain authority scores(enter your other domains)"""
        return {
            'wikipedia.org': 95,
            'github.com': 90,
            'microsoft.com': 95,
            'google.com': 98,
            'medium.com': 85,
            'stackoverflow.com': 93,
            'amazon.com': 96,
            'youtube.com': 95
        }

    def _get_bert_embeddings(self, text):
        """Create BERT embedding"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _calculate_relevance_score(self, query, title, snippet):
        """Calculate content relevance score"""
        # BERT embeddings
        query_embedding = self._get_bert_embeddings(query)
        content_embedding = self._get_bert_embeddings(f"{title} {snippet}")
        
        # Cosine similarity
        similarity = np.dot(query_embedding[0], content_embedding[0]) / \
                    (np.linalg.norm(query_embedding[0]) * np.linalg.norm(content_embedding[0]))
        
        return float(similarity)

    def _analyze_content_quality(self, content):
        """Analyze content quality"""
        doc = self.nlp(content)
        
        # Calculate base metrics
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
        
        # Readability score (Flesch Reading Ease benzeri)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            readability = max(0, min(1, 1 - (avg_sentence_length - 15) / 30))
        else:
            readability = 0
            
        # Merge factors
        quality_score = np.mean([
            min(1, word_count / 300),  # Content length
            min(1, unique_words / word_count if word_count > 0 else 0),  # unique words count
            readability  # Okunabilirlik
        ])
        
        return quality_score

    def _calculate_domain_authority_score(self, url):
        """Get domain authority score"""
        domain = urlparse(url).netloc
        base_domain = '.'.join(domain.split('.')[-2:])
        
        # Get scores from domains, entered in this file
        return self.domain_authority.get(base_domain, 50) / 100

    def _calculate_user_intent_match(self, query, title, snippet):
        """Calculate User Intent Match"""
        query_doc = self.nlp(query.lower())
        content_doc = self.nlp(f"{title} {snippet}".lower())
        
        # Intent keywords
        intent_patterns = {
            'informational': ['how', 'what', 'why', 'who', 'when', 'guide', 'tutorial'],
            'transactional': ['buy', 'price', 'shop', 'deal', 'discount', 'order'],
            'navigational': ['login', 'sign in', 'official', 'website'],
            'local': ['near', 'location', 'directions', 'maps']
        }
        
        # Set Query Intent
        query_intent = None
        max_intent_score = 0
        
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query.lower())
            if score > max_intent_score:
                max_intent_score = score
                query_intent = intent
                
        if not query_intent:
            query_intent = 'informational'  # Default intent
            
        # Content intent match
        content_text = f"{title} {snippet}".lower()
        intent_match_score = sum(1 for keyword in intent_patterns[query_intent] 
                               if keyword in content_text) / len(intent_patterns[query_intent])
        
        return min(1, intent_match_score)

    def _calculate_freshness_score(self, content):
        """Calculate Freshness"""
        # Date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s,]\d{1,2}[\s,]\d{4}'  # Month DD, YYYY
        ]
        
        latest_date = None
        
        # Look for date pattern in SERP
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    try:
                        date = datetime.strptime(match, '%Y-%m-%d')
                        if not latest_date or date > latest_date:
                            latest_date = date
                    except:
                        continue
        
        if not latest_date:
            return 0.5  # Default score
        
        # Analyze freshness score
        days_old = (datetime.now() - latest_date).days
        freshness_score = max(0, min(1, 1 - (days_old / 365)))  # 1 yıl baz alındı
        
        return freshness_score

    def _analyze_technical_seo(self, url, title, snippet):
        """Analyze technical SEO scores"""
        scores = []
        
        # URL analysis
        url_length = len(url)
        url_score = max(0, min(1, 1 - (url_length - 50) / 100))
        scores.append(url_score)
        
        # Title analysis
        title_length = len(title)
        title_score = max(0, min(1, 1 - abs(60 - title_length) / 60))
        scores.append(title_score)
        
        # Snippet analysis
        snippet_length = len(snippet)
        snippet_score = max(0, min(1, 1 - abs(160 - snippet_length) / 160))
        scores.append(snippet_score)
        
        return np.mean(scores)

    def _calculate_user_signals(self, url):
        """Calculate User Signals"""
        # Real world results
        domain = urlparse(url).netloc
        
        # Example metrics (Needs to be changed)
        bounce_rate = np.random.uniform(0.2, 0.8)
        time_on_site = np.random.uniform(30, 300)  # seconds
        
        # Normalized score
        signals_score = (1 - bounce_rate) * 0.5 + min(1, time_on_site / 180) * 0.5
        
        return signals_score

    def rerank_results(self, query, search_results):
        """Re-rank Results"""
        reranked_results = []
        
        for result in search_results:
            # Calculate factors for each result
            relevance = self._calculate_relevance_score(query, result['title'], result['snippet'])
            content_quality = self._analyze_content_quality(f"{result['title']} {result['snippet']}")
            domain_authority = self._calculate_domain_authority_score(result['url'])
            intent_match = self._calculate_user_intent_match(query, result['title'], result['snippet'])
            freshness = self._calculate_freshness_score(f"{result['title']} {result['snippet']}")
            technical_seo = self._analyze_technical_seo(result['url'], result['title'], result['snippet'])
            user_signals = self._calculate_user_signals(result['url'])
            
            # Calculate total score
            total_score = sum([
                relevance * self.weights['relevance_score'],
                content_quality * self.weights['content_quality'],
                domain_authority * self.weights['domain_authority'],
                intent_match * self.weights['user_intent_match'],
                freshness * self.weights['freshness'],
                technical_seo * self.weights['technical_seo'],
                user_signals * self.weights['user_signals']
            ])
            
            reranked_results.append({
                'original_position': result['position'],
                'url': result['url'],
                'title': result['title'],
                'snippet': result['snippet'],
                'total_score': total_score,
                'score_breakdown': {
                    'relevance': relevance,
                    'content_quality': content_quality,
                    'domain_authority': domain_authority,
                    'intent_match': intent_match,
                    'freshness': freshness,
                    'technical_seo': technical_seo,
                    'user_signals': user_signals
                }
            })
        
        # Order results by score
        reranked_results.sort(key=lambda x: x['total_score'], reverse=True)
        
        return reranked_results

    def get_search_results(self, query, num_results=10):
        """Get Google SERP"""
        url = f"https://www.google.com/search?q={query}&num={num_results}"
        
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for i, div in enumerate(soup.find_all('div', class_='g'), 1):
                title_elem = div.find('h3')
                if not title_elem:
                    continue
                    
                title = title_elem.text
                url = div.find('a')['href'] if div.find('a') else ''
                snippet = div.find('div', class_='VwiC3b').text if div.find('div', class_='VwiC3b') else ''
                
                results.append({
                    'position': i,
                    'url': url,
                    'title': title,
                    'snippet': snippet
                })
                
            return results
        except Exception as e:
            print(f"Error on getting search results: {e}")
            return []

    def analyze_and_rerank(self, query):
        """Complete Analysis and Re-Ranking Process"""
        # Get Original Results
        original_results = self.get_search_results(query)
        
        # Sonuçları yeniden sırala
        reranked_results = self.rerank_results(query, original_results)
        
        return {
            'query': query,
            'original_results': original_results,
            'reranked_results': reranked_results
        }

    def save_analysis(self, analysis_results, filename="rerank_analysis.txt"):
        """Save Analysis Results"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Re-Ranking Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Query: {analysis_results['query']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RE-RANKED RESULTS:\n")
            f.write("-" * 80 + "\n")
            
            for i, result in enumerate(analysis_results['reranked_results'], 1):
                f.write(f"\n{i}. Result (Original Position: {result['original_position']})\n")
                f.write(f"URL: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Snippet: {result['snippet']}\n")
                f.write(f"Total Score: {result['total_score']:.4f}\n\n")
                
                f.write("Score Details:\n")
                for factor, score in result['score_breakdown'].items():
                    f.write(f"- {factor}: {score:.4f}\n")
                
                f.write("-" * 40 + "\n")

def main():
    reranker = GoogleReRanker()
    
    query = input("Enter search query: ")
    print("\nAnalysis in progress...")
    
    try:
        # Complete analysis and rerank
        results = reranker.analyze_and_rerank(query)
        
        # Save results
        reranker.save_analysis(results)
        
        print("\nAnalysis completed!")
        print(f"Results are in the 'rerank_analysis.txt' file.")
        
        # Show Summary
        print("\nSummary:")
        print(f"Total results: {len(results['reranked_results'])}")
        print("\nTop 3 results:")
        for i, result in enumerate(results['reranked_results'][:3], 1):
            print(f"{i}. {result['title']} (Skor: {result['total_score']:.4f})")
            
    except Exception as e:
        print(f"an error occured: {e}")
        
if __name__ == "__main__":
    main()
