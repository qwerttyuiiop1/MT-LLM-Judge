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
class Criteria(BaseModel):
	"""A class to hold the analysis, comment, and type for a specific criterion."""
	analysis: str = Field(description="The analysis of the segment or document.")
	comment: str = Field(
		description="A clear and concise comment explaining the analysis and providing a corrected alternative."
	)
	type: Literal["Correct", "Minor", "Major"] = Field(
		description="The type for the analysis, one of 'Correct', 'Minor', or 'Major'."
	)
class TranslationEvaluation(BaseModel):
	"""A class to hold the translation evaluation results."""
	term_accuracy: Criteria = Field(
		description="Evaluation of term accuracy and consistency."
	)
	meaning_fidelity: Criteria = Field(
		description="Evaluation of meaning fidelity."
	)
	grammar_structure: Criteria = Field(
		description="Evaluation of grammar and structure."
	)
	style_naturalness: Criteria = Field(
		description="Evaluation of style and readability."
	)
	cultural_biases: Criteria = Field(
		description="Evaluation of cultural and audience fit."
	)
	analysis: str = Field(
		description="Overall analysis of the translation."
	)
	rating: int = Field(
		description="Overall rating of the translation, from 1 (incomprehensible) to 5 (perfect translation)."
	)

base_system_prompt = "You are an expert Filipino linguist."
base_prompt_template = """# Filipino Translation Evaluation

## Severity categories (how to mark each criterion)
Use one label for every criterion:
- **Correct (No Error)** — The translation is acceptable for the intended audience and purpose. No action required.
- **Minor** — Small problem that does not materially affect meaning or usability (e.g., awkward phrasing, minor grammatical slip, minor punctuation error). Fix recommended but not urgent.
- **Major** — Significant problem that seriously affects understandability, fidelity, or usability. This includes errors that change meaning, omit essential info, or are highly inappropriate for the audience.

---
## Evaluation criteria

### 1) Term Accuracy & Consistency
**Definition:** The correctness or accuracy of translations of terms (root words).

**Error Types:**
- **Inconsistent term use:** — inconsistent translations of the same term within the segment or document.
- **Incorrect terms:** Incorrect translations for words.
  _Break time_: **nasirang oras** (Major), **pahinga** (Minor, related meanings and may not significantly alter semantics, depending on context).
- **Technical / clinical / legal / jargon errors:** Poor handling of domain terms.
  _Glucose level_: **lebel ng asukal** (Major, alters meaning), **lebel ng glucose** (Minor, Glucose level may be preferred in medical contexts for clarity).
- **Names, numbers, and factual errors:** Misspelled place names or wrong proper nouns should be marked Major.
  _100_: **isang daan** (Correct, spelling variation), **isang libo** (Major)
  _Juan_: **John** (Major)
- **Spelling:** typographical or orthographic errors.

---

### 2) Meaning Fidelity
**Definition:** Faithfulness to the source message — no distortion, omission, or addition.

**Error Types:**
- **Mistranslation:** incorrect meaning conveyed.
- **Overtranslation / Addition:** extra information not in source.
- **Undertranslation / Omission:** missing information.
- **Loss of idiom, metaphor, tone or emotional weight:** euphemisms or humor that become flat — mark Minor or Major depending on whether meaning or tone is lost.

**Examples:**
- Source: "The medication is for temporary use." → **Major:** Permanenteng gamit ang gamot.
- Source: "Please sign here for a bit." →  "Pakilagda dito ng saglit" (Correct, for a bit is captured by saglit).

**Filipino-specific explanations & examples:**
- **Mistranslation of nuance / semantic flips:** In Filipino, verb focus (actor vs object) can change who performs an action. If translation flips focus and changes responsibility or intent, mark Major.
   Source: "We asked for help." → **Major:** "Hiningian kami ng tulong". (implies we are the ones who were asked, not the ones who asked)
- **Tense/aspect/focus errors:** Incorrect tense/voice (e.g., translating general truths as past events). Filipino is a morphologically complex language. Classify as Major if the error significantly alters semantics.

---

### 3) Grammar & Structure
**Definition:** Basic linguistic correctness and sentence formation.

**Error Types:**
- **Grammar:** wrong conjugation, incorrect syntax.
- **Punctuation:** missing/extra commas, incorrect use of quotation marks, spacing, incorrect capitalization.
- **Wrong affixation:** **magisip** (Minor) vs **mag-isip**/**mag isip** (Correct)
- **Missing or incorrect linkers:** Dropped or wrong "na", "ng", "nang" — mark Minor when purely grammatical, Major if meaning changes.

---

### 4) Style, Fluency, Naturalness
**Definition:** Is it culturally appropriate for Filipino readers?

**Error Types:**
- **Unnatural word order and grammatical constructions:** Overuse of _ay_ inversion or awkward inversions that sound non-native — usually Minor.
  A 4.8 magnitude earthquake hit yesterday → **Major:** Ang lindol na may lakas na 4.8 ay tumama kahapon. (ay inversion with unnatural word order, should be "Tumama ang magnitude 4.8 na lindol kahapon.")
- **Unnatural wording / awkward phrasing:** phrasing that sounds non-native or overly formal — usually Minor.
  I know what I know. → **Major:** Ang alam ko lang ay ang nalalaman ko. (unnatural ay inversion, unnatural wording of "ang nalalaman ko" for "what I know")
- **Tone mismatch:** Translations that are unnaturally stiff or too slangy compared to the source text, usually Minor for small tone slips.
- **Improper code-switching and inconsistent register:** Inappropriate mixing of English and Filipino that breaks tone, Minor.
- **Style / flow / readability issues:** Awkward phrasing or repetition — typically Minor, escalate to Major if it blocks comprehension.
- **Overly literal / word-for-word translation:** Mechanical renderings that ignore idiom (e.g., literal metaphors) — mark Minor or Major depending on whether meaning or tone is lost.

---

### 5) Cultural Biases
**Definition:** Use of culture-specific references, offensiveness, or language that won’t be understood by target audience.

**Error Types:**
- **Culture-specific reference:** literal carryover of source culture references that confuse readers.
- **Offensive:** translations that create unintended insult or culturally insensitive phrasing.
- **Localization & cultural insensitivity:** Poor choices for culturally loaded terms (incorrectly as a translation for Creator in a secular context), mistranslated place names or history facts — Major when sensitive.
- **Audience appropriateness:** Culture-specific references that the intended Filipino audience won't understand should be adapted or explained; rate Minor or Major based on impact.
- **Age / gender, Cultural (preferring Western values over Filipino norms):** Biases are present in the translation that are not originally in the source text.
  The nurse went outside. → **Major:** Lumabas ang babaeng nars. (Incorrectly assumed the female gender of the nurse, which is not specified in the source text.)

---

## Scoring & process guidance for raters
- Rate every criterion for the segment or document required by workflow.
- Correct if the entire translation is acceptable, Minor if there are slight issues, Major if at least one significant issue exists.
- When in doubt between Minor and Major, consider impact on user understanding or safety. If the issue could cause user harm, legal problems, or serious misinterpretation, mark Major.
- If the translation intentionally adapts content for target-culture relevance (and this behavior is allowed by brief), mark **Correct** and note the adaptation; do not penalize for acceptable localization.
- Provide a short comment for every rating explaining what is right/wrong and provide a corrected alternative (one sentence or a span suggestion).
- After evaluating all criteria, provide an overall analysis of the translation and a rating from 1 to 5:
  - **1**: Incomprehensible translation, major issues in most criteria.
  - **2**: Poor but somewhat understandable, major issues in some criteria.
  - **3**: Poor but understandable, minor issues in some criteria.
  - **4**: Good translation, minor issues in one or two criteria.
  - **5**: Acceptably correct translation, with no issues in any criteria.

---

# Output format
The output should be a JSON object that conforms to the `TranslationEvaluation` model. It should include detailed analysis, comments, and types for each evaluation criterion, as well as an overall analysis and rating.
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
}

---

## Example Evaluation
English text: "Please be careful, the floor is wet."
Tagalog translation: "Maging maingat, ang sahig ay basa."

### Expected JSON Output:
{
  "term_accuracy": {
    "analysis": "The terms 'maingat' (careful), 'sahig' (floor), and 'basa' (wet) are all accurate.",
    "comment": "No correction needed.",
    "type": "Correct"
  },
  "meaning_fidelity": {
    "analysis": "The translation perfectly preserves the warning and meaning of the source text.",
    "comment": "No correction needed.",
    "type": "Correct"
  },
  "grammar_structure": {
    "analysis": "The grammar is correct and intelligible.",
    "comment": "No correction needed.",
    "type": "Correct"
  },
  "style_naturalness": {
    "analysis": "The phrase 'ang sahig ay basa' uses the 'ay' inversion, which is grammatically correct but slightly less natural than the alternative. The more common, natural-sounding construction is 'basa ang sahig'.",
    "comment": "The translation is acceptable, but a more natural phrasing would be: 'Mag-ingat, basa ang sahig.'",
    "type": "Minor"
  },
  "cultural_biases": {
    "analysis": "The warning is culturally appropriate and universally understood.",
    "comment": "No correction needed.",
    "type": "Correct"
  },
  "analysis": "The translation is very good and fully understandable. There is only a minor stylistic issue where the word order could be more natural for everyday Filipino speech, but this does not impact comprehension.",
  "rating": 4
}
---

Now, please evaluate the following translation segment based on the criteria above.
English text: {english_text}
Tagalog translation: {tagalog_translation}
"""
def evaluate_translation(english_text: str, tagalog_translation: str) -> TranslationEvaluation:
	"""Evaluate a Tagalog translation of an English text."""
	# Prepare the prompt with the provided texts
	prompt = base_prompt_template.replace(
		"{english_text}", english_text
  ).replace(
		"{tagalog_translation}", tagalog_translation
	)

	# Call the model to generate the evaluation
	response = client.models.generate_content(
		model="gemini-2.0-flash",
		contents=[
			types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
		],
		config=types.GenerateContentConfig(
			temperature=0,
			top_p=0,
			top_k=1,
			system_instruction=base_system_prompt,
			response_schema=TranslationEvaluation,
			response_mime_type="application/json"
		)
	)

	# Parse and return the structured output
	return response.parsed
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from tqdm import tqdm  # optional progress bar

def _worker_thread(args):
    idx, (english, filipino) = args
    while True:
      try:
          res = evaluate_translation(english, filipino)
          break
      except Exception as e:
          print(e)
          continue
    return idx, res

def parallel_apply_threaded(df, max_workers=16, show_progress=False):
    items = list(enumerate(zip(df['English'].tolist(), df['Filipino'].tolist())))
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if show_progress:
            for idx, res in tqdm(ex.map(_worker_thread, items), total=len(items)):
                results[idx] = res
        else:
            for idx, res in ex.map(_worker_thread, items):
                results[idx] = res
    df = df.copy()
    df['predictions1'] = results
    return df
for i in range(0, 10):
    print(f"Iteration {i+1}")
    eval_df = parallel_apply_threaded(eval_df, max_workers=100, show_progress=True)
    eval_df[f"predictions{i+1}"] = eval_df['predictions1']
pred_cols = [f"predictions{i}" for i in range(1, 11)]
pred_ratings = eval_df[pred_cols].applymap(lambda x: x.rating)
avg_row_std = pred_ratings.std(axis=1, ddof=0).mean()
print("avg_row_std:", avg_row_std)
import numpy as np
mean_per_input = pred_ratings.mean(axis=1)
std_per_input  = pred_ratings.std(axis=1, ddof=0)   # population std (ddof=0). use ddof=1 if you prefer sample std
cv_per_input   = std_per_input / mean_per_input.replace(0, np.nan)  # avoid div-by-zero
rel_range_per_input = (pred_ratings.max(axis=1) - pred_ratings.min(axis=1)) / mean_per_input.replace(0, np.nan)

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
import pandas as pd
from scipy.stats import pearsonr

# Assuming eval_df is your DataFrame
correlation, v = pearsonr(eval_df['predictions1'].apply(lambda x: int(x.rating)), eval_df['Final Score'])
print("Pearson's correlation coefficient:", correlation, v)
import matplotlib.pyplot as plt

# Plot histograms of Final Score and predicted_rating
plt.figure(figsize=(10, 6))
plt.hist(eval_df['Final Score'], bins=5, alpha=0.7, label='Human-Labeled Final Score', range=(1, 5))
plt.hist(eval_df['predictions1'].apply(lambda x: int(x.rating)), bins=5, alpha=0.7, label='Predicted Rating', range=(1, 5))
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Human-Labeled and Predicted Ratings')
plt.xticks(range(1, 6))
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()
# Calculate the absolute difference between predicted and final scores
eval_df['rating_difference'] = abs(eval_df['predictions1'].apply(lambda x: int(x.rating)) - eval_df['Final Score'])

# Sort the DataFrame by the absolute difference in descending order
eval_df_sorted = eval_df.sort_values(by='rating_difference', ascending=False)
i=12
print(eval_df_sorted.iloc[i].English)
print(eval_df_sorted.iloc[i].Filipino)
for key, value in eval_df_sorted.iloc[i].predictions1.model_dump().items():
    print(f"{key}: {value}")

print(eval_df_sorted.iloc[i]['Final Score'])
print(eval_df_sorted.iloc[i]['Rater 1 Explanation'])
print(eval_df_sorted.iloc[i]['Rater 2 Explanation'])