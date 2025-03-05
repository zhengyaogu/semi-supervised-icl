from string import Template
from in_context_ssl.template import ABCInstanceTemplate, ABCTemplate


class BigBenchInstanceTemplateWithRationale(ABCInstanceTemplate):
    
    def __init__(self):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            "Question: $question\n"
            "Rationale: $rationale\n"
            "Answer: $answer\n"
        )

class BigBenchInstanceTemplate(ABCInstanceTemplate):
    
    def __init__(self):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            "Question: $question\n"
            "Answer: $answer\n"
        )

class BigBenchInstanceTemplateUnsupervised(ABCInstanceTemplate):
    
    def __init__(self):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            "Question: $question\n"
        )
    
class BigBenchTemplate(ABCTemplate):

    def __init__(self):
        self.template = Template(
            "First, I am going to give you a series of demonstrations of Questions.\n"
            "___\n"
            "$demonstrations\n"
            "___\n"
            "Now, Answer the following Question. Also give the Rationale as well as the Confidence of your given Answer. Think step by step.\n"
            "Question: $query"
        )

class LiveBenchMathTemplate(ABCTemplate):

    def __init__(self):
        self.template = Template(
            "First, I am going to give you a series of demonstrations of Questions.\n"
            "___\n"
            "$demonstrations\n"
            "___\n"
            "Now, Answer the following Question. Think step by step.\n"
            "Question: $query\n"
            "Also give the Confidence of your given Answer in the following format: **Confidence**: <a confidence score between 0 and 1>."
        )

class TranslationInstanceTemplate(ABCInstanceTemplate):
    
    def __init__(self, source, target):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            Template(
                "$source: $question\n"
                "$target: $answer\n"
            ).safe_substitute(
                source=source,
                target=target
            )
        )

class TranslationInstanceTemplateUnsupervised(ABCInstanceTemplate):
    
    def __init__(self, source):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            Template(
                "$source: $question\n"
            ).safe_substitute(
                source=source
            )
        )

class TranslationTemplate(ABCTemplate):
    def __init__(self, source, target):
        self.template = Template(
                Template(
                "You are an expert translator. I am going to give you zero or more example pairs of text snippets where the "
                "first is in the source language and the second is a translation of the first snippet into the target language. The sentences will be written\n"  
                "<source language>: <first sentence>\n"
                "<target language>: <translated first sentence>\n"
                "___\n"
                "$demonstrations\n"
                "___\n"
                "Now, Translate the following $source text into $target. Also give the Confidence of your given Answer in the following format: **Confidence**: <a confidence score between 0 and 1>:\n"
                "English: $query\n"
                "$target: "
            ).safe_substitute(
                source=source,
                target=target
            )
        )

class BackTranslationInstanceTemplate(ABCInstanceTemplate):
    
    def __init__(self, source, target):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            Template(
                "$source: $answer\n"
                "$target: $question\n"
            ).safe_substitute(
                source=source,
                target=target
            )
        )

class TranslationTemplate(ABCTemplate):
    def __init__(self, source, target):
        self.template = Template(
                Template(
                "You are an expert translator. I am going to give you zero or more example pairs of text snippets where the "
                "first is in the source language and the second is a translation of the first snippet into the target language. The sentences will be written\n"  
                "<source language>: <first sentence>\n"
                "<target language>: <translated first sentence>\n"
                "___\n"
                "$demonstrations\n"
                "___\n"
                "Now, Translate the following $source text into $target:\n"
                "$source: $query\n"
                "$target: "
            ).safe_substitute(
                source=source,
                target=target
            )
        )


        