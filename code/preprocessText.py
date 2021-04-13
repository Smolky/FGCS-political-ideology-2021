"""
    preprocess Library
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Ricardo Colomo-Palacios <ricardo.colomo-palacios@hiof.no>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import regex
import string
import emoji
import argparse
import numpy as np
import pandas as pd


# @link https://stackoverflow.com/questions/24893977/whats-the-best-way-to-regex-replace-a-string-in-python-but-keep-its-case
def replace_keep_case (word, replacement, text):
    def func (match):
        g = match.group ()
        if g.islower (): return replacement.lower ()
        if g.istitle (): return replacement.title ()
        if g.isupper (): return replacement.upper ()
        return replacement
    return regex.sub (word, func, text)


class PreProcessText ():
    """
    PreProcessText
    """
    
    # @var msg_language Dict A list of common Spanish phrases
    msg_language = {
        'ad\+': 'además',
        'bno': 'bueno',
        'd': 'de',
        'dl': 'del',
        'dra': 'doctora',
        'dr': 'doctor',
        'dsps': 'después',
        'gob': 'gobierno',
        'm da': 'me da',
        'mñn': 'mañana',
        'knto': 'cuánto',
        'h1n1': 'gripe',
        'k': 'que',
        'kx': 'porque',
        'q': 'que',
        'qndo': 'cuando',
        'toy': 'estoy',
        'tb': 'también',
        'pa': 'para',
        'ppio': 'principio',
        'tdv': 'todavía',
        'tmb': 'también',
        'tkm': 'te quiero mucho',
        'tp': 'tampoco',
        'salu2': 'saludos',
        'x': 'por',
        'xq': 'porque',
        'xq\?': 'por qué',
        '100pre': 'siempre',
        '1ra': 'primera',
        '2da': 'segunda',
        '3ra': 'tercera',
        '€+': 'euros',
        
        '([0-3][0-9])(\s?de\s?)?ene': r'\1 de enero',
        '([0-3][0-9])(\s?de\s?)?feb': r'\1 de febrero',
        '([0-3][0-9])(\s?de\s?)?mar': r'\1 de marzo',
        '([0-3][0-9])(\s?de\s?)?abr': r'\1 de abril',
        '([0-3][0-9])(\s?de\s?)?may': r'\1 de mayo',
        '([0-3][0-9])(\s?de\s?)?jun': r'\1 de junio',
        '([0-3][0-9])(\s?de\s?)?jul': r'\1 de julio',
        '([0-3][0-9])(\s?de\s?)?ago': r'\1 de agosto',
        '([0-3][0-9])(\s?de\s?)?sep': r'\1 de septiembre',
        '([0-3][0-9])(\s?de\s?)?oct': r'\1 de ocbubre',
        '([0-3][0-9])(\s?de\s?)?nov': r'\1 de noviembre',
        '([0-3][0-9])(\s?de\s?)?dic': r'\1 de diciembre',
         
    }
    
    
    # @var english_contractions Dict A list of common English contractions
    english_contractions = {
        "won\'t": "will not",
        "can\'t": "can not",
        "n\'t": " not",
        "\'re": " are",
        "\'s": " is",
        "\'d": " would",
        "\'ll": " will",
        "\'t": " not",
        "\'ve": " have",
        "\'m": " am"
    }
    
    
    # @var acronyms Dict A list of common Spanish acronyms
    acronyms = {
        'TVE': 'Televisión Española',
        'OMS': 'Organización Mundial de la Salud',
        'EEUU': 'Estados Unidos',
        'SM': 'Su Majestad',
        'GAL': 'Grupos Antiterroristas de Liberación',
        'UE': 'Unión Europa',
        'PAC': 'Política Agraria Común',
        'CCAA': 'Comunidades Autónomas',
        'ERTE': 'Expediente de Regulación de Empleo',
        'ERTES': 'Expedientes de Regulación de Empleo',
        'MEFP': 'Ministerio de Educación y Formación Profesional',
        'BCE': 'Banco Central Europeo',
        'FMI': 'Fondo Monetario Internacional',
        'OCDE': 'Organización para la Cooperación y el Desarrollo Económicos',
        'MIR': 'Médico Interno Residente',
        'ETA': 'Euskadi Ta Askatasuna',
        'CSIC': 'Consejo Superior de Investigaciones Científicas',
        'LGTB': 'Lesbianas, Gais,​ Bisexuales y Transgénero',
        'LGTB[\w\+]?': 'Lesbianas, Gais,​ Bisexuales y Transgénero',
        'IVA': 'Impuesto al valor agregado',
        'CE': 'Constitución Española',
        'CM': 'Congreso de Ministros',
        'CLM': 'Castilla La Mancha',
        'CyL': 'Castilla y León',
        'CAM': 'Comunidad de Madrid',
        'BCN': 'Barcelona',
        'MWC': 'Mobile World Congress',
        'G. Mixto': 'Grupo Mixto',
        'PGE': 'Presupuestos Generales del Estado',
        'PNV': 'Partido Nacionalista Vasco',
        'PP': 'Partido Popular',
        'PSOE': 'Partido Socialista Obrero Español',
        'UP': 'Unidas Podemos',
    }
    
    
    # Patterns
    whitespace_pattern = r'\s+'
    quotations_pattern = r'["“”\'«»‘’]'
    
    # @todo Add (.)\1{2,}|[aá]{2,}|[eé]{2,}|[ií]{2,}|[oó]{2,}|[uú]{2,})
    elongation_pattern = regex.compile (r'(.)\1{2,}')
    
    gender_contraction_pattern = r'(?i)\b(\p{L}+)@s\b'
    orphan_dots_pattern = r'[\.\s]{2,}'
    dashes_pattern = r' [\-\—] '
    orphan_exclamatory_or_interrogative_pattern = regex.compile (r'\s+([\?\!])+')
    url_pattern = regex.compile (r'https?://\S+')
    hashtag_pattern = regex.compile (r'#([\p{L}0-9\_]+)')
    begining_mentions_pattern = regex.compile (r"^(@[A-Za-z0-9\_]+\s?)+")
    middle_mentions_pattern = regex.compile (r'(?<!\b)@([A-Za-z0-9\_]+)\b(?<!user)')
    laughs_pattern = regex.compile (r'(?i)\b(mua)?j[ja]+a?\b')
    digits_pattern = regex.compile (r"\b(\d+[\.,]?\d*|\d{2}[AP]M)\b")
    percents_pattern = regex.compile (r"\b(\d+[\.,]?\d*|\d{2}[AP]M)%")
    punctuation_pattern = regex.compile (r"(?![\[\]])[.\|\+\(\)]+")
    
    

    def camel_case_split (self, identifier):
        """
        camel_case_split
        
        Use this function to split hashtags by its CamelCase
        
        @param identifier String
        
        @link https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python/29920015
        """
        matches = regex.finditer ('.+?(?:(?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|[0-9]+|$)', identifier)
        return ' '.join ([m.group (0) for m in matches])

    def remove_urls (self, sentences):
        """
        remove_urls
        
        Strip hyperlinks from text
        
        @param sentences
        """
        return sentences.apply (lambda x: regex.sub (self.url_pattern, '', x))
    
    def remove_mentions (self, sentences, placeholder = "[USUARIO]"):
        """
        remove mentions
        
        Removes or replaces mentions from text
        
        @param placeholder
        
        """
        sentences = sentences.apply (lambda x: regex.sub (self.begining_mentions_pattern, '', x))
        sentences = sentences.apply (lambda x: regex.sub (self.middle_mentions_pattern, placeholder, x))
        sentences = sentences.apply (lambda x: regex.sub (r'(@USER ){2,}', placeholder + " ", x))

        return sentences

    def expand_hashtags (self, sentences):
        """
        Expand hashtags into meaningful words by camel case
        """
        return sentences.apply (lambda x: regex.sub (self.hashtag_pattern, lambda match: self.camel_case_split (match.group (1)), x))


    def remove_hashtags (self, sentences, replace = r'#[HASHTAG]_\1'):
        """
        Remove hashtags hashtags into meaningful words by camel case
        
        @param sentences 
        @param replace
        """
        return sentences.apply (lambda x: regex.sub (self.hashtag_pattern, replace, x))
    
    
    def remove_percentages (self, sentences, replace = '[PORCENTAJE]'):
        """
        remove_percentages
        
        Strips percents
        
        @param sentences Series
        @param replace String
        
        """
        return sentences.apply (lambda x: regex.sub (self.percents_pattern, replace, x))


    def remove_digits (self, sentences, replace = '[NUMERO]'):
        """
        remove_digits
        
        Strips digits and numbers from the text
        
        @param sentences Series
        @param replace String
        
        """
        return sentences.apply (lambda x: regex.sub (self.digits_pattern, replace, x))
    
    
    def remove_whitespaces (self, sentences):
        """
        remove_whitespaces
        
        Removes whitespaces from the texts
        """
        sentences = sentences.replace (to_replace = self.whitespace_pattern, value = ' ', regex = True)
        sentences = sentences.replace (to_replace = self.orphan_dots_pattern, value = '. ', regex = True)
        sentences = sentences.replace (to_replace = self.dashes_pattern, value = '. ', regex = True)
        sentences = sentences.replace (to_replace = self.dashes_pattern, value = '. ', regex = True)
        sentences = sentences.apply (lambda x: regex.sub (self.orphan_exclamatory_or_interrogative_pattern, r"\1 ", x))
        
        return sentences
        

    def remove_emojis (self, sentences):
        """
        remove_emojis
        
        @param sentences
        """
        return sentences.apply (lambda x: emoji.get_emoji_regexp ().sub (u'', x))
    
    
    def remove_quotations (self, sentences):
        """
        remove_quotations
        
        @param sentences
        """
        return sentences.replace (to_replace = self.quotations_pattern, value = '', regex = True)


    def remove_elongations (self, sentences):
        """
        remove_elongations
        
        Removes elongations from texts
        
        @param sentences
        """
        
        # Remove laughs
        sentences = sentences.apply (lambda x: regex.sub (self.laughs_pattern, 'jajaja', x))

        
        # Remove exclamatory and interrogative
        for character in ['!', '¡', '?', '¿']:
            pattern = regex.compile ('\\' + character + '{2,}')
            sentences = sentences.apply (lambda x: regex.sub (pattern, character, x))
        
        
        # Remove letters longer than 2
        sentences = sentences.apply (lambda x: regex.sub (self.elongation_pattern, r'\1', x))
        
        sentences = sentences.apply (lambda x: replace_keep_case (regex.compile (r'(?i)[aá]{2,}'), "á", x))
        sentences = sentences.apply (lambda x: replace_keep_case (regex.compile (r'(?i)[eé]{2,}'), "é", x))
        sentences = sentences.apply (lambda x: replace_keep_case (regex.compile (r'(?i)[ií]{2,}'), "í", x))
        sentences = sentences.apply (lambda x: replace_keep_case (regex.compile (r'(?i)[oó]{2,}'), "ó", x))
        sentences = sentences.apply (lambda x: replace_keep_case (regex.compile (r'(?i)[uú]{2,}'), "ú", x))
        
        
        return sentences
        
    def to_lower (self, sentences):
        """
        to_lower
        
        Transform a text into its lowercase form
        
        @param sentences
        """
        return sentences.str.lower ()
    
    
    def remove_punctuation (self, sentences):
        """
        remove_punctuation
        
        @param sentences
        """
        return sentences.apply (lambda x: regex.sub (self.punctuation_pattern, '', x))
    

    def expand_gender (self, sentences):
        """
        expand_gender. IT makes sense for Spanish texts
        
        Example: amig@s: amigos y amigas
                 nosotr@s: nosotros y nosotras
        """
        return sentences.apply (lambda x: regex.sub (self.gender_contraction_pattern, r'\1os y \1as', x))

    def expand_acronyms (self, sentences, acronyms):
        """
        expand_acronyms
        
        @param sentences
        @param acronyms
        """
    
        for key, value in acronyms.items ():
        
            if key[-2:] == "\+":
                pattern = regex.compile (r"(?i)\b" +  (key[:-2]) + r"[\b\+]")
                
            else:
                pattern = regex.compile (r"(?i)\b" + key + r"\b")
            
            sentences = sentences.apply (lambda x: regex.sub (pattern, value, x))
            
        return sentences
    
    
    def replace_substring (self, sentences, substring, replace = ""):
        """
        replace_substring
        
        @param pattern
        @param words 
        @param replace
        """
        return sentences.apply (lambda x: x.replace (substring, replace))
    
    
    
    def strip (self, sentences):
        """
        strip (trims) white spaces
        
        @param sentences
        """
        return sentences.apply (lambda x: x.strip (" \r\n\t,.?"))
        
        
        
def main ():
    """ To use from command line """
    
    # Parser
    parser = argparse.ArgumentParser (description = 'Test preprocessing model')
    parser.add_argument ('--text', dest = 'text', default = '¡Suena una CANCIÓÓÓOÓÓN! CANCIÓN cancion canción cancióoáóon!', help = "A text to test")


    # Get args
    args = parser.parse_args ()
    
    
    # @var preprocess PreProcessText
    preprocess = PreProcessText ()


    # var texts list
    texts = pd.Series ([args.text])
    
    
    # @var preprocessing List
    preprocessing = [
        'expand_hashtags', 'expand_gender', 'remove_urls', 'remove_mentions', 'remove_percentages', 'remove_digits', 'remove_whitespaces', 
        'remove_elongations', 'remove_emojis', 'to_lower', 'remove_quotations', 
        'remove_punctuation', 'remove_whitespaces', 'strip'
    ]
    
    
    # Custom per language
    texts = preprocess.expand_acronyms (texts, preprocess.msg_language)
    texts = preprocess.expand_acronyms (texts, preprocess.acronyms)

    print ("original: " + args.text)
    print ()
    
    print ("acronyms: " + texts[0])
    
    for pipe in preprocessing:
        texts = getattr (preprocess, pipe)(texts)
        print (pipe + ": " + texts[0])


if __name__ == "__main__":
    main ()