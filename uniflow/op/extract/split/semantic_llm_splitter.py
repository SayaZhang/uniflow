"""Semantic splitter op."""

import copy
from typing import Sequence

from uniflow.node import Node
from uniflow.op.model.abs_llm_processor import AbsLLMProcessor
from uniflow.op.model.llm_processor import LLMDataProcessor
from uniflow.op.model.model_config import OpenAIModelConfig
from uniflow.op.op import Op
from uniflow.op.prompt import PromptTemplate, Context


class SemanticLLMSplitter(Op):
    """Semantic Splitter Op Class"""

    default_guided_prompt = PromptTemplate(
        instruction="If you have a long article like an instruction manual, please segment according to semantics and combine semantically similar sentences into one paragraph.\
            Following the format of the examples below to include the same context and the segment",
        few_shot_prompt=[
            Context(
                context="""
                    ## Getting Started
                    An overview of Markdown, how it works, and what you can do with it.
                    ## What is Markdown?
                    Markdown is a lightweight markup language that you can use to add formatting elements to plaintext text documents. Created by John Gruber in 2004, Markdown is now one of the world’s most popular markup languages.
                    Using Markdown is different than using a WYSIWYG editor. In an application like Microsoft Word, you click buttons to format words and phrases, and the changes are visible immediately. Markdown isn’t like that. When you create a Markdown-formatted file, you add Markdown syntax to the text to indicate which words and phrases should look different.
                    """,
                segment="""
                    ## Getting Started
                    An overview of Markdown, how it works, and what you can do with it.
                    \n
                    \n
                    ## What is Markdown?
                    Markdown is a lightweight markup language that you can use to add formatting elements to plaintext text documents. Created by John Gruber in 2004, Markdown is now one of the world’s most popular markup languages.
                    Using Markdown is different than using a WYSIWYG editor. In an application like Microsoft Word, you click buttons to format words and phrases, and the changes are visible immediately. Markdown isn’t like that. When you create a Markdown-formatted file, you add Markdown syntax to the text to indicate which words and phrases should look different.
                """,
            ),
            Context(
                context="""
                    Deep learning models have achieved remarkable results in computer vision [11] and speech recognition [1] in recent years. Within natural language processing, much of the work with deep learning methods has involved learning word vector representations through neural language models [1, 1, 2] and performing composition over the learned word vectors for classification [1]. Word vectors, wherein words are projected from a sparse, 1-of-\\(V\\) encoding (here \\(V\\) is the vocabulary size) onto a lower dimensional vector space via a hidden layer, are essentially feature extractors that encode semantic features of words in their dimensions. In such dense representations, semantically close words are likewise close--in euclidean or cosine distance--in the lower dimensional vector space.
                    Convolutional neural networks (CNN) utilize layers with convolving filters that are applied to local features [1]. Originally invented for computer vision, CNN models have subsequently been shown to be effective for NLP and have achieved excellent results in semantic parsing [13], search query retrieval [2], sentence modeling [1], and other traditional NLP tasks [1].
                    In the present work, we train a simple CNN with one layer of convolution on top of word vectors obtained from an unsupervised neural language model. These vectors were trained by Mikolov et al. (2013) on 100 billion words of Google News, and are publicly available.1 We initially keep the word vectors static and learn only the other parameters of the model. Despite little tuning of hyperparameters, this simple model achieves excellent results on multiple benchmarks, suggesting that the pre-trained vectors are 'universal' feature extractors that can be utilized for various classification tasks. Learning task-specific vectors through fine-tuning results in further improvements. We finally describe a simple modification to the architecture to allow for the use of both pre-trained and task-specific vectors by having multiple channels."
                """,
                segment="""
                    Deep learning models have achieved remarkable results in computer vision [11] and speech recognition [1] in recent years. Within natural language processing, much of the work with deep learning methods has involved learning word vector representations through neural language models [1, 1, 2] and performing composition over the learned word vectors for classification [1]. Word vectors, wherein words are projected from a sparse, 1-of-\\(V\\) encoding (here \\(V\\) is the vocabulary size) onto a lower dimensional vector space via a hidden layer, are essentially feature extractors that encode semantic features of words in their dimensions. In such dense representations, semantically close words are likewise close--in euclidean or cosine distance--in the lower dimensional vector space.
                    Convolutional neural networks (CNN) utilize layers with convolving filters that are applied to local features [1]. Originally invented for computer vision, CNN models have subsequently been shown to be effective for NLP and have achieved excellent results in semantic parsing [13], search query retrieval [2], sentence modeling [1], and other traditional NLP tasks [1].
                    \n
                    \n
                    In the present work, we train a simple CNN with one layer of convolution on top of word vectors obtained from an unsupervised neural language model. These vectors were trained by Mikolov et al. (2013) on 100 billion words of Google News, and are publicly available.1 We initially keep the word vectors static and learn only the other parameters of the model. Despite little tuning of hyperparameters, this simple model achieves excellent results on multiple benchmarks, suggesting that the pre-trained vectors are 'universal' feature extractors that can be utilized for various classification tasks. Learning task-specific vectors through fine-tuning results in further improvements. We finally describe a simple modification to the architecture to allow for the use of both pre-trained and task-specific vectors by having multiple channels."
                """,
            ),
        ],
    )

    def __init__(self, name: str, model: AbsLLMProcessor = None) -> None:
        """Semantic Splitter Op Constructor

        Args:
            name (str): Name of the op.
            model (AbsLLMProcessor): Model to run.
        """
        super().__init__(name)
        self._model = model

        if not model:
            self._model =LLMDataProcessor(
                prompt_template=SemanticLLMSplitter.default_guided_prompt,
                model_config=OpenAIModelConfig(response_format={"type": "json_object"}),
            )

    def __call__(
        self,
        nodes: Sequence[Node],
    ) -> Sequence[Node]:
        """Run Pattern Splitter Op

        Args:
            nodes (Sequence[Node]): Nodes to run.

        Returns:
            Sequence[Node]: Nodes after running the split.
        """
        output_nodes = []
        for node in nodes:
            value_dict = copy.deepcopy(node.value_dict)
            text = value_dict["text"]
            value_dict = self._model.run(value_dict)
            text = value_dict["response"][0]
            output_nodes.append(
                Node(
                    name=self.unique_name(),
                    value_dict={"text": text},
                    prev_nodes=[node],
                )
            )

        return output_nodes
