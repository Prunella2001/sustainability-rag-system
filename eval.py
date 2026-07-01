

import numpy as np
from trulens.core import Metric, Selector, Select
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

provider = OpenAI(model_engine="gpt-5-nano")

from trulens.core import Metric


#f_groundedness = Metric(
#    implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
#    name="Groundedness",
#    selectors={
#        # Grabs the text list returned from your retrieval step
#        "source": Selector(
#            span_type=SpanAttributes.SpanType.RETRIEVAL,
#            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
#            collect_list=True
#        ),
#        
#        "statement": Selector(
#            span_type=SpanAttributes.SpanType.RECORD_ROOT,
#            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT
#        ),
#
#        "question": Selector(
#            span_type=SpanAttributes.SpanType.RECORD_ROOT,
#            span_attribute=SpanAttributes.RECORD_ROOT.INPUT
#        ),
#    },
#)
#
## 2. Answer Relevance Evaluation
#f_answer_relevance = Metric(
#    implementation=provider.relevance_with_cot_reasons,
#    name="Answer Relevance",
#    selectors={
#        # Evaluates the root prompt input vs final response
#        "prompt": Selector(
#            span_type=SpanAttributes.SpanType.RECORD_ROOT,
#            span_attribute=SpanAttributes.RECORD_ROOT.INPUT
#        ),
#        "response": Selector(
#            span_type=SpanAttributes.SpanType.RECORD_ROOT,
#            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT
#        ),
#    },
#)
#
## 3. Context Relevance Evaluation
#f_context_relevance = Metric(
#    implementation=provider.context_relevance_with_cot_reasons,
#    name="Context Relevance",
#    selectors={
#        "question": Selector(
#            span_type=SpanAttributes.SpanType.GENERATION,
#            span_attribute="args.query"#SpanAttributes.RECORD_ROOT.INPUT
#        ),
#        "context": Selector(
#            span_type=SpanAttributes.SpanType.RETRIEVAL,
#            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
#            collect_list=True
#        ),
#    },
#    agg=np.mean,
#)
#
#
#def debug_context(question, context):
#    print("\n====================")
#    print("QUESTION TYPE:", type(question))
#    print(question)
#
#    print("\nCONTEXT TYPE:", type(context))
#    print(context)
#
#    return 1.0
#
#f_debug_context = Metric(
#    implementation=debug_context,
#    name="DEBUG_CONTEXT",
#    selectors={
#        "question": Selector(
#            span_type=SpanAttributes.SpanType.RECORD_ROOT,
#            span_attribute=SpanAttributes.RECORD_ROOT.INPUT
#        ),
#        "context": Selector(
#            span_type=SpanAttributes.SpanType.RETRIEVAL,
#            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS
#        )
#    }
#)

f_groundedness = Metric(
    name="Groundedness",
    implementation=provider.groundedness_measure_with_cot_reasons, 
    agg=np.mean,
    selectors={
        "source": Selector.select_context(collect_list=True), 
        "statement": Selector.select_record_output() 
    }
)

f_context_relevance = Metric(
    name="Context Relevance",
    implementation=provider.context_relevance,
    selectors={
        "question": Selector.select_record_input(), # Ajusté en "question" au cas où
        "context": Selector.select_context(collect_list=True) 
    }
)

f_answer_relevance = Metric(
    name="Answer Relevance",
    implementation=provider.relevance_with_cot_reasons,
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output()
    }
)