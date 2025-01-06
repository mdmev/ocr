# Prompts #

prompts ={
############################################################################
############################################################################
"Classifier":
    """
Classify the given image of a book page based on whether it is a candidate for containing parallel data (e.g., text in Rapa Nui and Spanish presented side by side or one above the other).

Output a single-word response:
- "Candidate": If the image shows a structured parallel text layout (e.g., left-right or top-bottom alignment of text in Rapa Nui and Spanish).
- "No Candidate": If the image does not show such a layout (e.g., text is mixed combining both languages, diagrams dominate, or no clear structure is evident, or just pure plain text).

Follow these criteria:
1. Look for **clear separation** of text regions, indicating distinct sections for each language.
2. Detect the **languages** in the image to confirm the presence of Rapa Nui and Spanish text in separate blocks.
3. Avoid classifying pages with a **dominance of diagrams, captions, or images** as "Candidate."
4. Ensure the text layout suggests a **parallel structure** (left-right, top-bottom, etc.) for translation.

Examples:
1. A page with two distinct columns: one in Rapa Nui and the other in Spanish. -> "Candidate"
2. A page with text in Rapa Nui and Spanish mixed together without clear separation. -> "No Candidate"
3. A page with diagrams and a mix of Rapa Nui and Spanish captions. -> "No Candidate"
""",
############################################################################
############################################################################
"JSON_extractor": """
Please carefully analyze the provided image. Extract all text pairs where a phrase is in Spanish and its equivalent in Rapanui. The texts may appear side by side, one above the other, or in separate sections of the image.

Ignore any text that is not in Spanish or Rapanui, or that does not form part of a translated pair.

**Important Instructions:**
- You must not include any additional commentary, explanations, or text outside of the JSON.
- Use the `print_pairs` tool exactly once to output your final answer.
- The output must be a strictly valid JSON array following this structure:

[
  {
    "spanish": "text in Spanish",
    "rapanui": "text in Rapanui"
  },
  {
    "spanish": "another text in Spanish",
    "rapanui": "another text in Rapanui"
  }
]

If no pairs are found, return an empty array: []

No other text is allowed outside the JSON. Do not explain or summarize. Your final response must be the `print_pairs` tool usage with the JSON.
""",
############################################################################
############################################################################
"nahuatl": """
System/Instruction:

You are an advanced language model with robust data-extraction and reasoning capabilities. Your task is to process images that contain two columns of text separated by a black line. The left column is Spanish; the right column is Nahuatl. You must output aligned text pairs (Spanish–Nahuatl) in JSON format, capturing each segment’s position, as well as the textual content.

Follow these steps carefully, and present your final result in a valid JSON structure. If you detect ambiguities or missing data, note them and propose corrections or re-analysis steps.

Step-by-Step Chain of Thought (internal reasoning instructions)
(You do not need to show these intermediate thoughts in the final answer. They are your internal process to ensure accuracy.)

    1. Identify Text Regions:
        - Detect all text regions in the left column (Spanish).
        - Detect all text regions in the right column (Nahuatl).

    2. Match Corresponding Text Blocks:
        - For each Spanish text block, locate the matching Nahuatl block based on relative vertical alignment and proximity to the dividing line.
        - Maintain a consistent pairing order from top to bottom to preserve the original sequence.

    3. Extract Content and Positions:
        - For each matched pair, record:
            1. Spanish text (verbatim from the left column).
            2. Nahuatl text (verbatim from the right column).

    4. Format the Output in JSON:
        - Create a list or array of objects where each object has keys such as "spanishText" and "nahuatlText".
        - Example:
'''
[
  {
    "spanishText": "Spanish text snippet",
    "nahuatlText": "Nahuatl text snippet"
  },
  ...
]
'''
5. Ensure Both Paragraph-Level and Word-Level Detail Are Supported:

    - For paragraph-level extraction, group contiguous lines if needed.
    - For word-level extraction, split paragraphs into words or tokens, ensuring each is aligned with its corresponding text on the other side.

6. Validate and Refine:
    - Check if the extracted information seems coherent and complete.
    - If any content appears missing, inaccurate, or randomly inserted, flag it and provide a short explanation of how you would attempt a second pass or a refined approach to fix the error.

Final Output Requirements:

    1. Must be valid JSON (no trailing commas, correct quotation marks, properly closed brackets, etc.).
    2. Clearly mark any uncertain or low-confidence extractions (e.g., "spanishText": "Possibly missing text...").
    3. Avoid “random” text insertions by re-checking the consistency of Spanish vs. Nahuatl content.
""",
############################################################################
############################################################################
"extract_raw_nahuatl": """
You will be provided with an image containing text. Your task is to extract the raw text exactly as it appears in the image, without adding any introductory phrases like "Here is the text" or additional explanations. The text is in Nahuatl, a low-resource language, and may include uncommon characters or structures.

To ensure accuracy, follow these steps:

    1.Carefully analyze the image to identify and read all visible text.
    2.Pay special attention to uncommon characters, diacritical marks, or unique orthographic features of Nahuatl.
    3.Double-check the extracted text for accuracy, verifying that no errors or omissions occurred during the process.
    4.Output only the extracted text, exactly as it appears, maintaining original spacing, punctuation, and line breaks.

Remember, your goal is precise transcription, respecting the original format and language features.
Avoid any modifications, translations, or interpretations of the text.
""",
############################################################################
############################################################################
"extract_raw_spanish": """
You will be provided with an image containing text. Your task is to extract the raw text exactly as it appears in the image, without adding any introductory phrases like "Here is the text" or additional explanations. The text is in Spanish and may include accents, special characters, or unique punctuation.

To ensure accuracy, follow these steps:

    1.Carefully examine the image to identify all visible text, including accents (e.g., á, é, í, ó, ú) and special punctuation marks (e.g., ¿, ¡).
    2.Verify that the extracted text matches the image exactly, including all line breaks, spacing, and formatting.
    3.Double-check for any errors or omissions, ensuring the text is accurately transcribed.
    4.Output only the extracted text, exactly as it appears, preserving the original format.

Your goal is precise and faithful transcription, respecting the original text's language-specific details and formatting.
""",
############################################################################
############################################################################
    "parallel": "Extract the parallel data",
    "single": "Extract the single data",
    "test": "Extract the text from the image",
}


## Tool structure obtained from 
## https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
tools = {
    "extract_json": [
        {
            "name": "print_pairs",
            "description": "Prints the extracted Rapa Nui-Spanish text pairs in a structured JSON format.",
            "input_schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "spanish": {"type": "string", "description": "The Spanish text segment."},
                        "rapanui": {"type": "string", "description": "The Rapanui text segment."}
                    },
                    "required": ["spanish", "rapanui"]
                }
            }
        }
    ]
}