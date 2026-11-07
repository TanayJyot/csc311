# Synthetic survey generation spec

You are simulating University of Toronto students (CSC311, Fall 2026) filling in a short survey.
Each simulated student (a "respondent") answers the SAME 12 questions three times: once about
Soccer, once about Baseball, once about Basketball. So every respondent produces exactly 3 rows.

Output format: JSON Lines (one JSON object per line, no surrounding array, no comments, no code fences).
Keys (all values are strings except "id"):
- "id": integer respondent id (given in your task; the same id on all 3 of that respondent's rows)
- "label": exactly one of "Soccer", "Baseball", "Basketball" (each used once per respondent)
- "ease": answer to "How easy do you think it would be for someone with no prior experience to start
  playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult
  and 10 means extremely easy." -> a number "0".."10" as a string
- "strategy": "Success in this sport depends heavily on strategy and planning."
- "injury": "Players must be comfortable with a high risk of injury."
- "individual": "Individual brilliance can outweigh teamwork in this sport."
  -> each one of: "Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"
- "places": "Where might you play this sport?" -> comma-separated (no spaces after commas) subset of:
  Indoor facility,Outdoor public space,School or university facility,Community recreation facility,Private sports facility,At home or nearby,Others
- "seasons": "During which seasons would you be most likely to play this sport recreationally in Toronto?"
  -> comma-separated subset of: Winter,Spring,Summer,Fall
- "music": "What types of music do you most associate with this sport?" -> comma-separated subset of:
  Pop,Rock,Hip-hop or rap,Electronic or dance,Latin,Folk or traditional,Country,Classical or jazz,Other,No particular type
- "food": free text, "What foods and drinks do you associate with watching this sport in the place or culture you know best?"
- "colour": free text, "What colour do you associate with this sport?"
- "song": free text, "Which song do you associate with this sport?"
- "feeling": free text, "How do you feel when you watch this sport?"
- "price": free text, "How much would you be willing to pay for a decent seat at a professional competition of this sport?"

Realism rules (important):
- Write like real, busy students typing into a form: mostly short answers (1-5 words), casual,
  lowercase sometimes, occasional typos, occasional "idk", "none", "n/a" or blank "" answers,
  especially for song. Prices in many styles: "80", "$50-100", "around 200 CAD", "idk", "0".
- Students differ a lot: some love a sport, some never watch it; some know songs, many do not.
  Answers should follow from each student's own background, not from a stereotype checklist.
- Do NOT make every row of a sport look the same. Vary every field across respondents.
- The three rows of one respondent should read as the same person (consistent style and background).
- Multi-select answers can have 1 to several options; do not always pick the same ones.
