import os
import json
from google import genai
from google.genai import types
from google.colab import userdata

client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))
turn = 0
import pandas as pd
train_df = pd.read_csv('drive/Shareddrives/LLM MP/Datasets - Training.csv')[1:]
# shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# train_df
eval_df = pd.read_csv('drive/Shareddrives/LLM MP/Datasets - Human-Labeled Validation Set.csv')
# rename 'Final Score                          (1 - lowest, 5 - highest)' to Final Score, Source Text (English) to English, Target Text (Filipino) to Filipino
eval_df = eval_df.rename(columns={'Final Score                          (1 - lowest, 5 - highest)': 'Final Score', 'Source Text (English)': 'English', 'Target Text (Filipino)': 'Filipino'})
eval_df = eval_df.dropna(subset=['Final Score'])
# eval_df
from pydantic import BaseModel, Field
from typing import Literal
# load wiktionary_tl
import pandas as pd
from ast import literal_eval
tmp = pd.read_csv('drive/Shareddrives/LLM MP/wiktionary_tl.csv')
tl_dictionary = {}
def extract_gloss_senses(row, dictionary=tl_dictionary):
	senses = literal_eval(row['senses'])
	word = row['word']
	ret = []
	for sense in senses:
		if 'glosses' not in sense or not sense['glosses']:
			continue
		ret.append('; '.join(sense['glosses']))
	if ret and (word not in dictionary or len(dictionary[word]) < len(ret)):
		dictionary[word] = ret
tmp.apply(lambda row: extract_gloss_senses(row, tl_dictionary), axis=1)
tmp = pd.read_csv('drive/Shareddrives/LLM MP/wiktionary_en.csv')
en_dictionary = {}
tmp.apply(lambda row: extract_gloss_senses(row, en_dictionary), axis=1)
del tmp
from typing import Dict
def dictionary_tool(word: str, lang: str="tl") -> Dict[str, str]:
	if lang == "tl":
		if word in tl_dictionary:
			return {"word": word, "definition": '\n\t'.join(tl_dictionary[word])}
		else:
			return {"error": "No definition found for {}.".format(word)}
	if lang == "en":
		if word in en_dictionary:
			return {"word": word, "definition": '\n\t'.join(en_dictionary[word])}
		else:
			return {"error": "No definition found for {}.".format(word)}
	return {"error": f"Language '{lang}' is not supported. Only 'tl' and 'en' are supported."}
# sacrebleu
import sacrebleu
def sacrebleu_tool(english_text: str, tagalog_translation: str) -> Dict[str, float]:
	"""Evaluate a Tagalog translation using sacreBLEU."""
	# Prepare the reference and hypothesis
	reference = [english_text]
	hypothesis = tagalog_translation

	# Calculate BLEU score
	bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
	return {"score": bleu.score}
# ===== TRANSLATION EVALUATION AGENT SYSTEM =====
import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from typing import Literal
from collections import Counter
from google.colab import userdata

class Criteria(BaseModel):
	"""A class to hold the analysis, comment, and type for a specific criterion."""
	analysis: str = Field(description="The analysis of the segment or document.")
	type: Literal["Correct", "Minor", "Major"] = Field(
		description="The type for the analysis, one of 'Correct', 'Minor', or 'Major'."
	)
class TranslationEvaluation(BaseModel):
	"""A class to hold the translation evaluation results."""
	term_accuracy: Criteria = Field(description="Evaluation of term accuracy and consistency.")
	meaning_fidelity: Criteria = Field(description="Evaluation of meaning fidelity.")
	grammar_structure: Criteria = Field(description="Evaluation of grammar and structure.")
	style_naturalness: Criteria = Field(description="Evaluation of style and readability.")
	cultural_biases: Criteria = Field(description="Evaluation of cultural and audience fit.")
	analysis: str = Field(description="Overall analysis of the translation.")
	rating: int = Field(description="Overall rating of the translation, from 1 (incomprehensible) to 5 (perfect translation).")
# --- Pydantic Models for Structured Output ---
class SubAgentAnalysis(BaseModel):
	"""Output format for specialized sub-agents"""
	analysis: str = Field(description="Detailed analysis of the specific criterion")
	comment: str = Field(description="Clear comment with correction if needed")

class SelfEvaluationOutput(BaseModel):
	"""Output from self-evaluation agent"""
	accuracy_assessment: str = Field(description="Assessment of analyzer accuracy")
	needs_reevaluation: bool = Field(description="Whether re-evaluation is needed for the main analyzer")
	feedback_for_analyzer: Optional[str] = Field(description="Feedback to improve analysis")

# Initialize client
client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))

class TranslationEvaluationAgent:
	def __init__(self):
		self.model_name = "gemini-2.0-flash"
		self.turn = 0
		self.analyzer_state = []
		self.tools = []
		self.tool_stats = Counter()
		self.MAX_REEVALUATIONS = 5

	def analyze_translation(self, english_text: str, tagalog_translation: str) -> Dict[str, Any]:
		"""Main orchestration method"""
		self.turn = 0
		self.analyzer_state = None
		self.subagent_cache = {}
		num_repeats = 0
		while True:
			# Step 1: Main analyzer agent
			analyzer_result = self._run_analyzer_agent(english_text, tagalog_translation)

			data = self._run_extractor_agent(analyzer_result)
			subagent_results = self._get_subagent_results(english_text, tagalog_translation)
			# Step 3: Self-evaluation
			self_eval_result = self._run_self_evaluation(english_text, tagalog_translation, data, subagent_results)
			if not self_eval_result.needs_reevaluation or num_repeats >= self.MAX_REEVALUATIONS:
				break
			num_repeats += 1
			print("Re-evaluating: ", english_text, tagalog_translation, self_eval_result.feedback_for_analyzer)
			feedback = "Please re-evaluate the translation based on the feedback provided.\n\n" + self_eval_result.feedback_for_analyzer
			self.analyzer_state.append(
				types.Content(
					role="user",
					parts=[types.Part.from_text(text=feedback)]
				)
			)

		return {
			"final_evaluation": data,
			"self_evaluation": self_eval_result,
			"logs": self.analyzer_state,
			"subagemt_results": subagent_results
		}

	def _run_analyzer_agent(self, english_text: str, tagalog_translation: str) -> str:
		"""Main analyzer agent - simplified rubric evaluation"""
		system_prompt = """You are an expert Filipino linguist specializing in translation, and the main analyzer in an agentic system."""
		analyzer_prompt = """Evaluate a translation pair using these criteria:
1. Term Accuracy & Consistency - Are roots of terms translated correctly and consistently?
2. Meaning Fidelity - Does it preserve the source meaning faithfully? With no additions or omissions?
3. Grammar & Structure - Is it grammatically correct?
4. Style & Naturalness - Does it sound natural and appropriate?
5. Cultural & Audience Fit - Is it culturally appropriate for Filipino readers?

Your response must contain an analysis and a rating for each criterion.
- **Correct (No Error)** — The translation is acceptable for the intended audience and purpose. No action required.
- **Minor** — Small problem that does not materially affect meaning or usability (e.g., slightly-off terminology, awkward phrasing, minor grammatical slip, minor punctuation error). Fix recommended but not urgent.
- **Major** — Significant problem that seriously affects understandability, fidelity, or usability. This includes errors that change meaning, omit essential info, or are highly inappropriate for the audience.
Provide an overall analysis and rating (1-5), with 1 being incomprehensible, 3 being poor but understandable, and 5 being an acceptably correct translation.
You may call tools or ask for a detailed analysis from specialized sub-agents. You may also pass optional notes to sub-agents to give more context or analyze a specific aspect (no need to give the translation pair, the agent already has it).
"""

		tools = self._get_analyzer_tool_schemas()
		if not self.analyzer_state:
			self.analyzer_state = [
				types.Content(role="user", parts=[types.Part.from_text(
					text=analyzer_prompt + f"\n\nEnglish: {english_text}\nTagalog: {tagalog_translation}"
				)])
			]

		while True:
			self.turn += 1
			resp = client.models.generate_content(
				model=self.model_name,
				contents=self.analyzer_state,
				config=types.GenerateContentConfig(
					tools=tools,
            temperature=0,
            top_p=0,
            top_k=1,
					system_instruction=system_prompt,
				)
			)

			self.analyzer_state.append(resp.candidates[0].content)

			if not resp.function_calls:
				return resp.candidates[0].content.parts[0].text

			# Handle tool calls - collect all responses first
			tool_responses = []
			for fc in resp.function_calls:
				result = self._handle_tool_call(fc, english_text, tagalog_translation)
				tool_responses.append(
					types.Part.from_function_response(name=fc.name, response=result)
				)

			# Add all tool responses as a single content message
			self.analyzer_state.append(
				types.Content(
					role="tool",
					parts=tool_responses
				)
			)

	def _term_accuracy_agent(self, english_text: str, tagalog_translation: str, note: str) -> SubAgentAnalysis:
		"""Specialized agent for term accuracy evaluation"""
		note = 'Additional Notes: ' + note if note else ''
		system_prompt = """You are an expert Filipino linguist specializing in word translation."""
		prompt = f"""Focus on:
- Correct translations of specific terms
- Consistency within the text
- Technical/domain term handling
- Proper nouns and numbers
- Spelling accuracy

Examples:
- "Break time" → "nasirang oras" (Major error - wrong meaning), "pahinga" (Correct)
- "100" → "isang daan" (Correct), "isang libo" (Major error - wrong number)
- "Glucose level" → "lebel ng asukal" (Major error - wrong term), "lebel ng glucose" (Minor error - Glucose level may be preferred in medical contexts for clarity)

English: {english_text}
Tagalog: {tagalog_translation}
{note}
Analyze term accuracy and consistency specifically."""

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SubAgentAnalysis,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _meaning_fidelity_agent(self, english_text: str, tagalog_translation: str, note: str) -> SubAgentAnalysis:
		"""Specialized agent for meaning fidelity evaluation"""
		note = 'Additional Notes: ' + note if note else ''
		system_prompt = """You are an expert Filipino linguist specializing in translation analysis."""
		prompt = f"""Focus on:
- Accurate meaning preservation
- No additions or omissions
- Proper handling of idioms/metaphors
- Verb focus accuracy (actor vs object focus)
- Tense/aspect correctness

Examples:
- "We asked for help" → "Humingi kami ng tulong" (Correct) vs "Hiningian kami ng tulong" (Major error - wrong focus)
- "Please sign here for a bit" → "Pakilagda dito ng saglit" (Correct) vs "Pakilagda dito" (Major error - missing context)

English: {english_text}
Tagalog: {tagalog_translation}
{note}
Analyze meaning fidelity specifically."""

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SubAgentAnalysis,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _grammar_structure_agent(self, tagalog_translation: str, note: str) -> SubAgentAnalysis:
		"""Specialized agent for grammar and structure evaluation"""
		note = 'Additional Notes: ' + note if note else ''
		system_prompt = """You are an expert Filipino linguist specializing in grammar and syntax."""
		prompt = f"""Focus on:
- Correct conjugation and syntax
- Proper punctuation and capitalization
- Correct affixation (mag-isip vs magisip)
- Proper use of linkers (na, ng, nang)
- Sentence structure correctness

Tagalog: {tagalog_translation}
{note}
Analyze grammar and structure specifically."""

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SubAgentAnalysis,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _style_naturalness_agent(self, english_text: str, tagalog_translation: str, note: str) -> SubAgentAnalysis:
		"""Specialized agent for style and naturalness evaluation"""
		note = 'Additional Notes: ' + note if note else ''
		system_prompt = """You are an expert Filipino linguist specializing in style and naturalness."""
		prompt = f"""Focus on:
- Natural word order and constructions
- Appropriate tone and register (with respect to the source text)
- Proper/Reasonable code-switching usage
- Fluency and readability
- Avoiding overly literal translations
- Avoiding unnatural phrasing or awkward inversions

Examples:
- "A 4.8 magnitude earthquake hit yesterday" → "Ang lindol na may lakas na 4.8 ay tumama kahapon" (unnatural word order and ay inversion, should be "Tumama ang magnitude 4.8 na lindol kahapon.")

English: {english_text}
Tagalog: {tagalog_translation}
{note}
Analyze style, fluency, and naturalness specifically."""

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SubAgentAnalysis,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _cultural_biases_agent(self, english_text: str, tagalog_translation: str, note: str) -> SubAgentAnalysis:
		"""Specialized agent for cultural biases evaluation"""
		note = 'Additional Notes: ' + note if note else ''
		system_prompt = """You are a specialist in Filipino culture."""
		prompt = f"""Focus on:
- Cultural sensitivity and appropriateness
- Proper localization vs literal translation
- Gender/age, Religious/cultural bias detection
- Audience appropriateness

Examples:
- "The nurse went outside" → "Lumabas ang babaeng nars" (assumes gender not in source)
- "under the weather" → "sa ilalim ng panahon" (literal translation, should be localized to "hindi maganda ang pakiramdam")

English: {english_text}
Tagalog: {tagalog_translation}
{note}
Analyze cultural appropriateness and biases specifically."""

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SubAgentAnalysis,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _run_self_evaluation(self, english_text: str, tagalog_translation: str,
						analyzer_result: TranslationEvaluation,
						subagent_results: Dict[str, SubAgentAnalysis]) -> SelfEvaluationOutput:
		"""Self-evaluation agent to assess accuracy of analysis"""
		self.tool_stats['self_evaluation'] += 1
		analyzer_result = json.dumps(analyzer_result.model_dump(), indent=2)
		subagent_results = json.dumps(subagent_results, indent=2)
		system_prompt = """You are an expert Filipino linguist, and a meta-evaluator assessing the accuracy of translation evaluation."""
		prompt = f"""Compare the main analyzer's assessment with the specialized sub-agents' detailed analyses.
Look for:
- Consistency between main analysis and sub-agent findings
- Consistency between main analysis and the original texts
- Any missed critical issues or inaccuracies
- Overpenalization or underpenalization
- Need for re-evaluation

Original texts:
English: {english_text}
Tagalog: {tagalog_translation}

Main analyzer assessment: {analyzer_result}
Sub-agent detailed analyses: {subagent_results}

Evaluate the accuracy and provide feedback.
Note: If the main analyzer's assessment is correct, but the sub-agents' analyses do not align with it, this should not be treated as a reason to re-evaluate. In your feedback, simply note that the analysis for that criterion is correct.
"""
		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=SelfEvaluationOutput,
				response_mime_type="application/json",
				system_instruction=system_prompt
			)
		)
		return resp.parsed

	def _run_extractor_agent(self, analyzer_result: str) -> TranslationEvaluation:
		"""Extract final structured output in the required format"""
		self.tool_stats['structure_extractor'] += 1
		prompt = """You are an output formatter. Convert the evaluation results into the final TranslationEvaluation format.

Analyzer result: \"\"\"{analyzer_result}\"\"\"

Criteria format:
{
  "analysis": "Detailed analysis of the segment or document.",
  "comment": "A clear and concise comment explaining the analysis and providing a corrected alternative.",
  "type": "Correct" | "Minor" | "Major"
}
Output format:
{
  "term_accuracy": Criteria,
  "meaning_fidelity": Criteria,
  "grammar_structure": Criteria,
  "style_naturalness": Criteria,
  "cultural_biases": Criteria,
  "analysis": "Overall analysis of the translation.",
  "rating": 1 | 2 | 3 | 4 | 5
}""".replace("{analyzer_result}", analyzer_result)

		resp = client.models.generate_content(
			model=self.model_name,
			contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
			config=types.GenerateContentConfig(
            temperature=0,
            top_p=0,
            top_k=1,
				response_schema=TranslationEvaluation,
				response_mime_type="application/json"
			)
		)
		return resp.parsed

	def _get_subagent_results(self, english_text: str, tagalog_translation: str) -> Dict[str, SubAgentAnalysis]:
		"""Run all specialized sub-agents and return their results"""
		cache = self.subagent_cache
		subagent_results = {}
		subagent_results['term_accuracy'] = cache.get(
			(None, 'term_accuracy_tool'),
		) or self._term_accuracy_agent(english_text, tagalog_translation, '').model_dump()
		subagent_results['meaning_fidelity'] = cache.get(
			(None, 'meaning_fidelity_tool'),
		) or self._meaning_fidelity_agent(english_text, tagalog_translation, '').model_dump()
		subagent_results['grammar_structure'] = cache.get(
			(None, 'grammar_structure_tool'),
		) or self._grammar_structure_agent(tagalog_translation, '').model_dump()
		subagent_results['style_naturalness'] = cache.get(
			(None, 'style_naturalness_tool'),
		) or self._style_naturalness_agent(english_text, tagalog_translation, '').model_dump()
		subagent_results['cultural_biases'] = cache.get(
			(None, 'cultural_biases_tool'),
		) or self._cultural_biases_agent(english_text, tagalog_translation, '').model_dump()
		return subagent_results
	def _get_analyzer_tool_schemas(self):
		if self.tools:
			return self.tools
		sacrebleu = types.FunctionDeclaration(
			name="sacrebleu",
			description="Calculate BLEU score for translation quality",
		)
		dictionary = types.FunctionDeclaration(
			name="dictionary",
			description="Look up word definitions in Filipino or English dictionary",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"word": types.Schema(type="STRING"),
					"lang": types.Schema(type="STRING", enum=["tl", "en"])
				},
				required=["word", "lang"]
			)
		)
		term_accuracy = types.FunctionDeclaration(
			name="term_accuracy_tool",
			description="Generate a detailed analysis of term accuracy in the translation",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"note": types.Schema(type="STRING", description="Additional note for the analysis"),
				},
				required=[]
			)
		)
		meaning_fidelity = types.FunctionDeclaration(
			name="meaning_fidelity_tool",
			description="Generate a detailed analysis of meaning fidelity in the translation",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"note": types.Schema(type="STRING", description="Additional note for the analysis"),
				},
				required=[]
			)
		)
		grammar_structure = types.FunctionDeclaration(
			name="grammar_structure_tool",
			description="Generate a detailed analysis of grammar and structure in the translation",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"note": types.Schema(type="STRING", description="Additional note for the analysis"),
				},
				required=[]
			)
		)
		style_naturalness = types.FunctionDeclaration(
			name="style_naturalness_tool",
			description="Generate a detailed analysis of style and naturalness in the translation",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"note": types.Schema(type="STRING", description="Additional note for the analysis"),
				},
				required=[]
			)
		)
		cultural_biases = types.FunctionDeclaration(
			name="cultural_biases_tool",
			description="Generate a detailed analysis of cultural biases in the translation",
			parameters=types.Schema(
				type="OBJECT",
				properties={
					"note": types.Schema(type="STRING", description="Additional note for the analysis"),
				},
				required=[]
			)
		)
		tools = [types.Tool(function_declarations=[
			dictionary, # sacrebleu,
			term_accuracy, meaning_fidelity,
			grammar_structure, style_naturalness, cultural_biases
		])]
		self.tools = tools
		return tools

	def _handle_tool_call(self, fc, english_text, tagalog_translation):
		"""Handle function calls from the model"""
		args = fc.args if not isinstance(fc.args, str) else json.loads(fc.args)
		note = args.get("note") or None
		if fc.name not in ('dictionary_tool', 'sacrebleu_tool') and \
			(note, fc.name) in self.subagent_cache:
			return self.subagent_cache[(note, fc.name)]
		self.tool_stats[fc.name] += 1

		if fc.name == "dictionary_tool":
			ret = dictionary_tool(**args)
		elif fc.name == "sacrebleu_tool":
			ret = sacrebleu_tool(**args)
		elif fc.name == "term_accuracy_tool":
			ret = self._term_accuracy_agent(
				english_text=english_text,
				tagalog_translation=tagalog_translation,
				note=note
			)
		elif fc.name == "meaning_fidelity_tool":
			ret = self._meaning_fidelity_agent(
				english_text=english_text,
				tagalog_translation=tagalog_translation,
				note=note
			)
		elif fc.name == "grammar_structure_tool":
			ret = self._grammar_structure_agent(
				tagalog_translation=tagalog_translation,
				note=note
			)
		elif fc.name == "style_naturalness_tool":
			ret = self._style_naturalness_agent(
				english_text=english_text,
				tagalog_translation=tagalog_translation,
				note=note
			)
		elif fc.name == "cultural_biases_tool":
			ret = self._cultural_biases_agent(
				english_text=english_text,
				tagalog_translation=tagalog_translation,
				note=note
			)
		else:
			return {"error": f"Unknown function {fc.name}"}
		if ret is None:
			self._handle_tool_call(fc, english_text, tagalog_translation)
		if fc.name not in ('dictionary_tool', 'sacrebleu_tool'):
			# Cache the result for future calls
			ret = ret.model_dump()
			self.subagent_cache[(note, fc.name)] = ret
		return ret

# Initialize the agent system
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Tuple, Dict, Any, List

_thread_local = threading.local()

def _get_thread_agent():
    """Return a TranslationEvaluationAgent unique to this thread (lazy init)."""
    if not hasattr(_thread_local, "agent"):
        _thread_local.agent = TranslationEvaluationAgent()
    return _thread_local.agent

def _worker_with_stats(args: Tuple[int, str, str]) -> Tuple[int, Any, str, Dict[str,int]]:
    """
    Worker returns:
      (index, result, thread_name, delta_tool_stats_dict)
    """
    idx, english, filipino = args
    agent = _get_thread_agent()
    thread_name = threading.current_thread().name

    # Snapshot before
    before = agent.tool_stats.copy()

    # Run the evaluation (use the free function or agent method as appropriate)
    while True:
      try:
          print(idx, '----- analyzing', english, filipino)
          res = agent.analyze_translation(english, filipino)
          break
      except Exception as e:
          print(e)
          continue

    # Snapshot after
    after = agent.tool_stats.copy()

    # Delta attributable to this call
    delta = after - before  # Counter subtraction yields non-negative counts for keys present in after
    # convert Counter to plain dict of ints
    delta_dict = {k: int(v) for k, v in delta.items() if v}

    return idx, res, thread_name, delta_dict

def parallel_evaluate_threaded(df, n_workers=None, show_progress=False):
    """
    Runs evaluate_translation in parallel threads, collects predictions into df['predictions1'],
    and returns (df_with_preds, total_tool_stats_counter, per_thread_tool_stats_dict).

    per_thread_tool_stats_dict maps thread_name -> Counter (as dict).
    total_tool_stats_counter is a Counter (dict-like) summing all thread counters.
    """
    import os
    from tqdm import tqdm  # optional; only used if show_progress True

    if n_workers is None:
        n_workers = min(32, (os.cpu_count() or 1) * 5)  # tune for I/O-bound

    # prepare work items: list of (idx, english, filipino)
    items = list(enumerate(zip(df['English'].tolist(), df['Filipino'].tolist())))
    items = [(i, eng, fil) for i, (eng, fil) in items]

    # storage for results
    results: List[Any] = [None] * len(df)

    # aggregate stats
    total_stats: Counter = Counter()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        mapped = ex.map(_worker_with_stats, items)
        if show_progress:
            mapped = tqdm(mapped, total=len(items))

        for idx, res, thread_name, delta_dict in mapped:
            # put prediction result in correct slot
            results[idx] = res
            # update per-thread and total counters
            delta_counter = Counter(delta_dict)
            total_stats.update(delta_counter)

    # attach predictions to a copy of df
    df_out = df.copy()
    df_out['predictions1'] = results

    # convert per_thread_stats counters to normal dicts for nicer display/serialization
    total_stats_dict = dict(total_stats)

    return df_out, total_stats_dict
new_eval_df = eval_df.copy()
stats = [None] * 10
i = 0
for i in range(i, 10):
    print(f"Iteration {i+1}")
    new_eval_df, stats[9-i] = parallel_evaluate_threaded(new_eval_df, n_workers=100, show_progress=True)
    new_eval_df[f"predictions{10-i}"] = new_eval_df['predictions1']
import pandas as pd
from scipy.stats import pearsonr

# Assuming eval_df is your DataFrame
correlation, v = pearsonr(new_eval_df['predictions1'].apply(lambda x: int(x['final_evaluation'].rating)), new_eval_df['Final Score'])
print("Pearson's correlation coefficient:", correlation, v)
import matplotlib.pyplot as plt

# Plot histograms of Final Score and predicted_rating
plt.figure(figsize=(10, 6))
plt.hist(new_eval_df['Final Score'], bins=5, alpha=0.7, label='Human-Labeled Final Score', range=(1, 5))
plt.hist(new_eval_df['predictions1'].apply(lambda x: int(x['final_evaluation'].rating)), bins=5, alpha=0.7, label='Predicted Rating', range=(1, 5))
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Human-Labeled and Predicted Ratings')
plt.xticks(range(1, 6))
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()
pred_cols = [f"predictions{i}" for i in range(1, 11)]
pred_ratings = new_eval_df[pred_cols].applymap(lambda x: x['final_evaluation'].rating)
avg_row_std = pred_ratings.std(axis=1, ddof=0).mean()
print("avg_row_std:", avg_row_std)
import numpy as np
mean_per_input = pred_ratings.mean(axis=1)
std_per_input  = pred_ratings.std(axis=1, ddof=0)   # population std (ddof=0). use ddof=1 if you prefer sample std
cv_per_input   = std_per_input / mean_per_input.replace(0, np.nan)  # avoid div-by-zero

# 2) Dataset-level summaries
summary = {
    'mean_CV': cv_per_input.mean(),                     # average CV across inputs
    'median_CV': cv_per_input.median(),
    'pct_inputs_CV_lt_10pct': (cv_per_input < 0.10).mean(),   # fraction meeting your <10% target
    'mean_rel_range': rel_range_per_input.mean(),
    'pct_exact_same_rating': (std_per_input == 0).mean(),     # proportion of inputs with zero variation
    'overall_CV': pred_ratings.stack().std(ddof=0) / pred_ratings.stack().mean(),  # global CV
}

# 3) Attach the per-input metrics to the dataframe (optional)
per_input_df = pd.DataFrame({
    'mean': mean_per_input,
    'std': std_per_input,
    'cv': cv_per_input,
    'rel_range': rel_range_per_input,
})

print("Dataset summary:", summary)
per_input_df.head()
