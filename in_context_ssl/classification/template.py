from string import Template
from in_context_ssl.template import ABCInstanceTemplate, ABCTemplate

class ClassificationTemplate(ABCTemplate):

    def __init__(self, confidence=False):
        self.template = Template(
            "You are a helpful assistant who is capable of performing a classification task (mapping an Input to a Label) with the following possible labels:\n"
            "$labels\n"
            "___\n"
            "Here are zero or more Input and Label pairs sampled from the classification task.\n"
            "___\n"
            "$demonstrations\n"
            "___\n"
            "Now, Label the following Input among the following \n"
            "Input: $query\n"
        )

class ClassificationInstanceTemplate(ABCInstanceTemplate):
    
    def __init__(self):
        super().__init__("\n___\n") # the connector between instances

        self.template = Template(
            "Input: $input\n"
            "Label: $label"
        )