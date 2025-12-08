import os
import re
import string
import jieba
import pandas as pd
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np

DetectorFactory.seed = 0  # for reproducible language detection
def chinese_tokenize(text: str, cut_all=False, HMM=True):
    """
    最常用的分词函数，返回列表
    """
    return list(jieba.cut(text, cut_all=cut_all, HMM=HMM))

def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


# -----------------------------
# Load spaCy model (English + disable unnecessary components for speed)
# -----------------------------
nlp_en = spacy.blank("en")          # super fast blank model
nlp_en.add_pipe("sentencizer")
nlp_en.max_length = 2_000_000
# Chinese personal pronouns (common ones)


# English personal pronouns (spaCy tags)
EN_PRONOUNS_POS = {"PRP", "PRP$"}

# ----------------- Main: Process all genres -----------------


CHINESE_PRONOUNS = {
    # ────────────── 1st person ──────────────
    "我", "俺", "吾", "在下", "本座", "老子", "大爷", "大爺", "小的", "臣", "朕", "孤", "寡人",
    "人家", "人家自己", "自己", "俺们", "俺們", "咱们", "咱們", "我们", "我們",

    # ────────────── 2nd person ──────────────
    "你", "妳", "您", "尔", "爾", "汝", "乃", "阁下", "閣下", "大人", "兄台", "道友",
    "你们", "你們", "诸位", "諸位", "各位", "大家",

    # ────────────── 3rd person ──────────────
    "他", "她", "它", "牠", "祂", "伊", "彼", "其", "之", "这位", "這位", "那位", "那位",
    "他们", "他們", "她们", "她們", "它们", "它們", "牠们", "牠們",

    # ────────────── Reflexive / reciprocal ──────────────
    "自己", "彼此", "互相", "各自", "自家",

    # ────────────── Indefinite / generic ──────────────
    "别人", "別人", "人家", "有人", "某些人", "大家", "大夥", "大伙", "大伙儿", "大伙兒"
}



def safe_read_text(filepath):
    encodings = ['utf-8', 'gbk', 'gb18030', 'big5', 'utf-8-sig']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    print(f"Failed to decode (tried all encodings): {filepath}")
    return None

def clean_df(df, verbose=True):
    initial = len(df)
    
    # 1. Remove only real garbage (NaN / inf / empty)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["pronoun_ratio", "info_density", 
                           "avg_sent_len","total_words"])
    
    # 2. Minimum realistic book size (skip samples/chapters)
    df = df[df["total_words"] >= 10_000]   # ← 10k words minimum (very safe)
    
    # 3. VERY LOOSE outlier removal — use 5×IQR or percentile clipping
    numeric_cols = ["pronoun_ratio", "info_density", "avg_sent_len"]
    
    for col in numeric_cols:
        # For all ratios: winsorize at 1% and 99% (keeps real variation)
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        removed = len(df[(df[col] < lower) | (df[col] > upper)])
        df = df[df[col].between(lower, upper)]
        
        if verbose and removed > 0:
            print(f"  Removed {removed} extreme values in {col}")
    
    
    final = len(df)
    if verbose:
        print(f"\nData cleaning complete:")
        print(f"   Before → {initial} books")
        print(f"   After  → {final} books ({initial-final} removed)")
        print(f"   Kept {100*final/initial:.1f}% of the data\n")
    
    return df.reset_index(drop=True)

def analyze_book(text, lang="en"):
    text = clean_text(text)

    if len(text) < 100:
        return None  # skip very short files

    if lang == "ch" :
        # ----------------- Chinese processing with jieba -----------------
        words = [w for w in jieba.cut(text) if w.strip()]

        # Remove pure punctuation tokens (very common in jieba output: ，。！？；：、“”‘’()[] etc.)
        words = [w for w in words if not all(c in '，。！？；：、“”‘’()[]【】《》—… \t\n\r' for c in w)]
        
        total_words = len(words)
        if total_words == 0:
            return None
        
        # Personal pronouns
        pronouns = [w for w in words if w in CHINESE_PRONOUNS]
        pronoun_ratio = len(pronouns) / total_words
        
        # Stop words + punctuation already filtered above
        STOP_WORDS = {
            # Core particles & auxiliaries
            "的", "地", "得", "着", "了", "之", "於", "于", "在", "是", "为", "為",
            "和", "與", "与", "或", "而", "則", "则", "以", "以", "及", "且",
        
            # Common discourse markers / modal particles
            "啊", "呀", "啦", "吧", "呢", "吗", "嘛", "喔", "哦", "呃", "嗯", "嘿", "哎", "哎呀",
        
            # Demonstratives
            "这", "那", "这个", "那个", "这些", "那些", "這", "那", "這個", "那個", "這些", "那些",
        
            # Extremely frequent verbs/adverbs that carry little content
            "有", "没有", "沒有", "是", "不是", "很", "非常", "太", "都", "也", "就", "才", "到",
            "来", "來", "去", "說", "说", "道", "想", "知道", "知道", "可以", "能", "会", "會",
            "想要", "看到", "看到", "覺得", "觉得",
        
            # Pronouns – remove if you want them counted as content
            # (most stylometry studies exclude them from "information density")
            "我", "你", "妳", "他", "她", "它", "牠", "祂", "我们", "我們", "你们", "你們",
            "他们", "他們", "她们", "她們", "它们", "它們", "牠們",
        
            # Common measure words & quantifiers (usually not content-bearing)
            "一个", "一個", "一些", "很多", "许多", "許多", "每个", "每個", "大家", "人家",
        
            # Extra high-frequency words common in novels
            "自己", "這樣", "这样", "那樣", "那样", "然后", "然後", "但是", "但是", "如果", "雖然", "虽然",
            "因為", "因为", "所以", "而且", "並", "并", "又", "再", "被", "把", "將", "将", "對", "对"
        }
        
        # Content words = real lexical words (nouns, verbs, adjectives, etc.)
        content_words = [w for w in words if len(w) > 1 and w not in STOP_WORDS]
        info_density = len(content_words) / total_words
        
        # Sentence segmentation (robust version)
        sentences = [s.strip() for s in re.split(r'[。！？；\n]+', text) if s.strip()]
        # fallback if no sentence-ending punctuation
        if len(sentences) <= 1:
            sentences = [s.strip() for s in re.split(r'[,.!?;]\s+', text) if s.strip()]
        
        avg_sent_len = total_words / len(sentences) if sentences else 10
        
        return {
            "pronoun_ratio": pronoun_ratio,
            "info_density": info_density,
            "avg_sent_len": avg_sent_len,
            "language": "ch",
            "total_words": total_words
        }
    
    else:
        # Personal pronouns (lower-cased, most common forms + possessive)
        EN_PRONOUNS = {
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves"
        }
        
        # Very small stop-word list (function words we don’t count as “content”)
        EN_STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "if", "while", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        }
        # ----------------- English processing with spaCy -----------------
        text = text.lower()
    
        # Tokenize: split on whitespace and punctuation
        tokens = re.findall(r"\w+(?:['’-]\w+)?", text)   # handles don't, Mary's, etc.
        
        # Remove pure punctuation that might slip through
        real_tokens = [t for t in tokens if t not in string.punctuation]
        
        total_words = len(real_tokens)
        if total_words == 0:
            return None
    
        # 1. Personal pronoun ratio
        pronouns = [t for t in real_tokens if t in EN_PRONOUNS]
        pronoun_ratio = len(pronouns) / total_words
    
        # 2. Content words = everything except stop words
        content_words = [t for t in real_tokens if t not in EN_STOPWORDS]
        info_density = len(content_words) / total_words
    
        # 3. Average sentence length – split on . ! ? (robust for novels)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) == 0:
            sentences = [text]  # fallback
        avg_sent_len = total_words / len(sentences)
    
        return {
            "pronoun_ratio": pronoun_ratio,
            "info_density": info_density,
            "avg_sent_len": avg_sent_len,
            "language": "en",
            "total_words": total_words
        }

def process_genre_folder(base_path,lang="en"):
    results = []
    total_files = 0
    success = 0

    for genre in sorted(os.listdir(base_path)):               # sorted = predictable order
        genre_path = os.path.join(base_path, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\nProcessing genre: {genre}")
        files_in_genre = [f for f in os.listdir(genre_path) if f.lower().endswith(".txt")]
        total_files += len(files_in_genre)

        for filename in files_in_genre:
            filepath = os.path.join(genre_path, filename)

            # Read file safely
            text = safe_read_text(filepath)
            if text is None:
                print(f"  Failed reading: {filename}")
                continue

            # Skip extremely short or empty files
            if len(text.strip()) < 1000:
                print(f"  Too short, skipping: {filename}")
                continue

            # Analyze
            try:
                metrics = analyze_book(text,lang=lang)
                if metrics:
                    metrics["book"] = filename
                    metrics["genre"] = genre
                    metrics["filepath"] = filepath          # optional: for debugging
                    results.append(metrics)
                    success += 1
                    print(f"  Success: {filename[:50]:50} → {metrics['total_words']:,} words")
                else:
                    print(f"  No metrics returned: {filename}")
            except Exception as e:
                print(f"  Analysis failed: {filename} | Error: {e}")

    print(f"\nFinished! Successfully processed {success}/{total_files} files.")
    df = pd.DataFrame(results)
    df = clean_df(df)
    df = add_tfidf_features(df, base_path)
    return df

def load_corpus(base_path, lang="en"):
    """Load raw corpus into a DataFrame for downstream BERT training."""
    records = []
    total_files = 0
    success = 0

    for genre in sorted(os.listdir(base_path)):
        genre_path = os.path.join(base_path, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\nProcessing genre: {genre}")
        files_in_genre = [f for f in os.listdir(genre_path) if f.lower().endswith(".txt")]
        total_files += len(files_in_genre)

        for filename in files_in_genre:
            filepath = os.path.join(genre_path, filename)

            text = safe_read_text(filepath)
            if text is None:
                print(f"  Failed reading: {filename}")
                continue

            if len(text.strip()) < 1000:
                print(f"  Too short, skipping: {filename}")
                continue

            try:
                records.append(
                    {
                        "book": filename,
                        "genre": genre,
                        "language": lang,
                        "text": clean_text(text),
                    }
                )
                success += 1
                print(f"  Success: {filename[:50]:50} → {len(text.split()):,} words")
            except Exception as e:
                print(f"  Load failed: {filename} | Error: {e}")

    print(f"\nFinished! Successfully loaded {success}/{total_files} files.")
    return pd.DataFrame(records)

def add_tfidf_features(df, base_path):
    """
    Computes TF-IDF separately per language, then combines.
    This gives fair and meaningful scores for both English and Chinese.
    """
    corpus = []
    book_indices = []  # to keep order

    for idx, row in df.iterrows():
        genre_path = os.path.join(base_path, row["genre"])
        filepath = os.path.join(genre_path, row["book"])
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            # Clean and prepare text
            text = clean_text(raw_text)
            
            if row["language"] == "ch":
                # Chinese: use jieba tokenizer
                text = " ".join(chinese_tokenize(text))
            else:
                # English: use default whitespace tokenizer (good enough)
                text = text.lower()
                
            corpus.append(text)
            book_indices.append(idx)
        except:
            corpus.append("")
            book_indices.append(idx)

    # Now use a single vectorizer but with language-aware preprocessing already done
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,           # appear in at least 2 books
        max_df=0.95,        # ignore terms in >95% of books
        ngram_range=(1,2),  # include bigrams for richer features
        norm='l2'
    )
    
    try:
        X = vectorizer.fit_transform(corpus)
        tfidf_avg = X.mean(axis=1)
        df.loc[book_indices, "tfidf_avg"] = tfidf_avg
    except:
        df["tfidf_avg"] = 0.0  # fallback if all empty

    return df

#test 

def debug_tfidf_features(df, base_path):
    """
    Computes TF-IDF separately per language, then combines.
    This gives fair and meaningful scores for both English and Chinese.
    """
    corpus = []
    book_indices = []  # to keep order
    judge = False
    for idx, row in df.iterrows():
        genre_path = os.path.join(base_path, row["genre"])
        filepath = os.path.join(genre_path, row["book"])
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            # Clean and prepare text
            text = clean_text(raw_text)
            
            if row["language"] == "ch":
                # Chinese: use jieba tokenizer
                text = " ".join(chinese_tokenize(text))
            else:
                # English: use default whitespace tokenizer (good enough)
                text = text.lower()
                
            corpus.append(text)
            book_indices.append(idx)
        except:
            corpus.append("")
            book_indices.append(idx)

    # Now use a single vectorizer but with language-aware preprocessing already done
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,           # appear in at least 2 books
        max_df=0.95,        # ignore terms in >95% of books
        ngram_range=(1,2),  # include bigrams for richer features
        norm='l2'
    )
    
    try:
        X = vectorizer.fit_transform(corpus)
        non_zero_counts = X.getnnz(axis=1)
        tfidf_avg = np.array(X.sum(axis=1)).flatten() / non_zero_counts
        df.loc[book_indices, "tfidf_avg"] = tfidf_avg
    except:
        df["tfidf_avg"] = 0.0  # fallback if all empty
        judge = True

    return judge