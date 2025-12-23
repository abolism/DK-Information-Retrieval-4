from text_processor import PersianPreprocessor

class QueryParser:
    def __init__(self, kb, intent_dict):
        """
        kb: KnowledgeBase instance from Step 2
        intent_dict: IntentDictionary instance from Step 4
        """
        self.kb = kb
        self.intent_dict = intent_dict
        self.preprocessor = PersianPreprocessor()

    def parse(self, query):
        tokens = self.preprocessor.tokenize(query)
        entities = {
            'detected_brands': set(),
            'detected_product_types': set(),
            'detected_categories': set(),
            'tokens': tokens
        }

        for token in tokens:
            # 1. Check Knowledge Base (Explicit Match)
            if self.kb.is_brand(token):
                entities['detected_brands'].add(token)
            
            if self.kb.is_product_type(token):
                entities['detected_product_types'].add(token)
            
            if self.kb.is_category(token):
                entities['detected_categories'].add(token)

            # 2. Check Intent Dictionary (Implicit/Statistical Match)
            intent = self.intent_dict.get_intent(token)
            if intent['brand']:
                entities['detected_brands'].add(intent['brand'])
            if intent['category']:
                entities['detected_categories'].add(intent['category'])

        return entities

    def get_structured_query(self, query):
        """Helper to return a clean dictionary of detected features."""
        parsed = self.parse(query)
        return {
            'brands': list(parsed['detected_brands']),
            'product_types': list(parsed['detected_product_types']),
            'categories': list(parsed['detected_categories']),
            'token_count': len(parsed['tokens'])
        }