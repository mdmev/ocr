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
    "parallel": "Extract the parallel data",
    "single": "Extract the single data",
    "test": "What do you see in this image?",
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