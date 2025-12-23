# import re

# class PersianPreprocessor:
#     def __init__(self):
#         # Mapping for character unification (Arabic to Persian)
#         self.translation_table = str.maketrans({
#             'k': 'ک', 'y': 'ی', 'u': 'و', 
#             'ك': 'ک', 'ي': 'ی', 'ؤ': 'و', 'إ': 'ا', 'أ': 'ا', 'آ': 'ا',
#             'ة': 'ه', 'ۀ': 'ه',
#         })
        
#         # Regex patterns for cleaning
#         self.punc_pattern = re.compile(r'[!@#%^&*()_+\-=\[\]{};:\'",.<>/?\\|~`؟،«»]')
#         self.space_pattern = re.compile(r'\s+')
#         self.zwnj_pattern = re.compile(r'\u200c') # Zero-width non-joiner

#     def normalize(self, text):
#         if not isinstance(text, str):
#             return ""
        
#         # 1. Lowercase (irrelevant for Persian chars but good for mixed English)
#         text = text.lower()
        
#         # 2. Character Translation (Arabic -> Persian)
#         text = text.translate(self.translation_table)
        
#         # 3. Handle Zero-width non-joiner (replace with space for better token matching)
#         # In e-commerce, 'میز‌تحریر' (Desk) vs 'میز تحریر' should match. 
#         # Separating them aids bag-of-words models.
#         text = self.zwnj_pattern.sub(' ', text)
        
#         # 4. Remove Punctuation
#         text = self.punc_pattern.sub(' ', text)
        
#         # 5. Remove extra whitespace and strip
#         text = self.space_pattern.sub(' ', text).strip()
        
#         return text

#     def tokenize(self, text):
#         """Simple whitespace tokenizer after normalization."""
#         return self.normalize(text).split()
    

# # Add this to your text_processor.py or as a new utility
# class MultilingualNormalizer:
#     def __init__(self):
#         # Mapping common English/Finglish terms to Persian equivalents
#         self.mapping = {
#             'samsung': 'سامسونگ',
#             'apple': 'اپل',
#             'iphone': 'ایفون',
#             'xiaomi': 'شیائومی',
#             'asus': 'ایسوس',
#             'lenovo': 'لنوو',
#             'nokia': 'نوکیا',
#             'huawei': 'هوآوی',
#             'sony': 'سونی',
#             'hp': 'اچ پی',
#             'gard': 'گارد',
#             'gooshi': 'گوشی',
#             'mobile': 'موبایل',
#             'kaver': 'کاور',
#             'lap tap': 'لپ تاپ',
#             'laptop': 'لپ تاپ',
#             'headset': 'هدست',
#             'هدفون': 'هدفون'
#         }

#     def normalize_query(self, text):
#         text = text.lower() # Ensure English is lowercased
#         words = text.split()
#         normalized_words = [self.mapping.get(w, w) for w in words]
#         return " ".join(normalized_words)

# # Usage Example Verification
# if __name__ == "__main__":
#     pp = PersianPreprocessor()
#     sample = "گوشی موبایل  سامسونگ مدل Galaxy S21"
#     print(f"Original: {sample}")
#     print(f"Normalized: {pp.normalize(sample)}")


import re

class PersianPreprocessor:
    def __init__(self):
        self.translation_table = str.maketrans({
            'k': 'ک', 'y': 'ی', 'u': 'و', 
            'ك': 'ک', 'ي': 'ی', 'ؤ': 'و', 'إ': 'ا', 'أ': 'ا', 'آ': 'ا',
            'ة': 'ه', 'ۀ': 'ه',
        })
        
        # Mapping Persian/Arabic numbers to English
        self.num_map = str.maketrans({
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', 
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        })

        # Common Finglish/English -> Persian mapping for E-commerce
        self.term_map = {
            'samsung': 'سامسونگ', 'apple': 'اپل', 'iphone': 'آیفون', 
            'xiaomi': 'شیائومی', 'huawei': 'هواوی', 'nokia': 'نوکیا',
            'sony': 'سونی', 'lg': 'ال جی', 'asus': 'ایسوس', 'lenovo': 'لنوو',
            'hp': 'اچ پی', 'dell': 'دل', 'acer': 'ایسر', 'msi': 'ام اس ای',
            'gooshi': 'گوشی', 'mobile': 'گوشی', 'phone': 'گوشی',
            'tablet': 'تبلت', 'laptop': 'لپ تاپ', 'lap': 'لپ تاپ',
            'watch': 'ساعت', 'smart': 'هوشمند', 'case': 'کاور', 
            'cover': 'کاور', 'gard': 'کاور', 'glass': 'گلس',
            'headphone': 'هدفون', 'handsfree': 'هندزفری', 'headset': 'هدست',
            'speaker': 'اسپیکر', 'powerbank': 'پاوربانک', 'charger': 'شارژر',
            'cable': 'کابل', 'airpod': 'ایرپاد', 'airpods': 'ایرپاد',
            'pro': 'پرو', 'max': 'مکس', 'ultra': 'الترا', 'plus': 'پلاس',
            'note': 'نوت', 'red': 'قرمز', 'blue': 'آبی', 'black': 'مشکی',
            'white': 'سفید', 'gold': 'طلایی', 'silver': 'نقره ای'
        }
        
        self.punc_pattern = re.compile(r'[!@#%^&*()_+\-=\[\]{};:\'",.<>/?\\|~`؟،«»]')
        self.space_pattern = re.compile(r'\s+')
        self.zwnj_pattern = re.compile(r'\u200c') 

    def normalize(self, text):
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # 0. Lowercase first (for English matching)
        text = text.lower()
        
        # 1. Number Normalization (Persian -> English)
        text = text.translate(self.num_map)
        
        # 2. Token mapping (Finglish/English -> Persian)
        # We process tokens to find exact word matches
        tokens = text.split()
        new_tokens = [self.term_map.get(t, t) for t in tokens]
        text = " ".join(new_tokens)
        
        # 3. Character Unification (Arabic -> Persian)
        text = text.translate(self.translation_table)
        
        # 4. Handle ZWNJ
        text = self.zwnj_pattern.sub(' ', text)
        
        # 5. Remove Punctuation
        text = self.punc_pattern.sub(' ', text)
        
        # 6. Cleanup Spaces
        text = self.space_pattern.sub(' ', text).strip()
        
        return text

    def tokenize(self, text):
        return self.normalize(text).split()