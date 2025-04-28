"""
Define all prompts used in this project.
"""

ICL_USER_PROMPT_2 = """Paths:
Path 0. (Caribbean, location.location.contains, Netherlands Antilles), (Netherlands Antilles, location.country.official_language, Dutch Language)
Path 1. (Spanish Language, language.human_language.countries_spoken_in, Barbados), (Barbados, location.location.containedby, Caribbean)

Triplets:
(Anguilla, location.location.containedby, Caribbean)
(English Language, language.human_language.countries_spoken_in, Bahamas)
(Barbados, location.country.languages_spoken, English Language)
(Barbados, location.location.containedby, British West Indies)
(Sint Maarten, location.country.official_language, Dutch Language)
(Caribbean, location.location.contains, British Virgin Islands)
(Flemish language, language.human_language.main_country, Belgium)


Question:
Of the countries that are in the Caribbean, which has the Dutch Language as an official language?
"""

ICL_ASS_PROMPT_2_brief = """ans: Netherlands Antilles
"""

ICL_ASS_PROMPT_2 = """To answer the question, we need to identify countries in the Caribbean that have Dutch as an official language. Our first step is to check whether any paths explicitly mention both the Caribbean and the Dutch language.

From the paths, we find that the only country explicitly confirmed to be in the Caribbean with Dutch as an official language is the Netherlands Antilles (based on path 0).

Therefore, the answer is:

ans: Netherlands Antilles
"""


ICL_USER_PROMPT_3 = """Triplets:
(Phoenix Islands, location.administrative_division.country, Kiribati)
(m.04c6sl7, location.imports_and_exports.exported_to, Kiribati)
(Kiribati, location.statistical_region.places_imported_from, m.04c6sl7)
(m.04c6sj8, location.imports_and_exports.imported_from, Kiribati)
(Asia, base.locations.continents.countries_within, Japan)
(Japan, location.statistical_region.places_exported_to, m.0493792)
(Japan, location.statistical_region.places_exported_to, m.04c6sl7)
('m.0493792', 'location.imports_and_exports.exported_to', 'United States of America')


Question:
What country does Japan export to that contains the Phoenix Islands?


"""

ICL_ASS_PROMPT_3_brief = """ans: Kiribati
"""


ICL_USER_PROMPT_triple = """Triplets:
(Lou Seal,sports.mascot.team,San Francisco Giants)
(San Francisco Giants,sports.sports_team.championships,{2012 World Series, 2014 World Series, 2010 World Series})
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,2014 Major League Baseball season)
(San Francisco Giants,time.participant.event,2010 World Series)
(San Francisco Giants,time.participant.event,2010 Major League Baseball season)
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco Giants,time.participant.event,2012 World Series)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?"""


ICL_ASS_PROMPT_triple = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 World Series
ans: 2012 World Series
ans: 2010 World Series"""


ICL_USER_PROMPT_triple_Apr = """Evidence Chains:
Chain 1. Lou Seal → [sports.mascot.team] → San Francisco Giants → [sports.sports_team.championships] → {(1) 2012 World Series (2) 2014 World Series (3) 2010 World Series}
Chain 2. San Francisco Giants → [sports.sports_championship_event.champion] → {(1) 2014 World Series}
Chain 3. San Francisco Giants → [time.participant.event] → {(1) 2014 Major League Baseball season}
Chain 4. San Francisco Giants → [time.participant.event] → {(1) 2010 World Series}
Chain 5. San Francisco Giants → [sports.sports_team.team_mascot] → {(1) Crazy Crab}
Chain 6. San Francisco Giants → [sports.professional_sports_team.owner_s] → {(1) Bill Neukom}
Chain 7. San Francisco Giants → [time.participant.event] → {(1) 2012 World Series}
Chain 8. San Francisco → [sports.sports_team_location.teams] → {(1) San Francisco Giants}
Chain 9. San Francisco Giants → [sports.sports_team.arena_stadium] → {(1) AT&T Park}
Chain 10. AT&T Park → [location.location.events] → {(1) 2012 World Series}


Question:
What year did the team with mascot named Lou Seal win the World Series?"""


ICL_ASS_PROMPT_triple_Apr = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From Chain 1, we can see that Lou Seal is the mascot of the San Francisco Giants, and that San Francisco Giants won the 2010, 2012, and 2014 World Series.

Therefore, the formatted answers are:

ans: 2014 World Series
ans: 2012 World Series
ans: 2010 World Series"""


ICL_USER_PROMPT_triple_webqsp = """Evidence Chains:
Chain 1. Atlanta → [travel.travel_destination.tourist_attractions] → {(1) Georgia State Capitol (2) Georgia World Congress Center (3) Four Seasons Hotel Atlanta (4) Atlanta Jewish Film Festival (5) Margaret Mitchell House & Museum (6) Martin Luther King, Jr. National Historic Site (7) Georgia Dome (8) Woodruff Arts Center (9) Six Flags White Water}
Chain 2. Atlanta → [travel.travel_destination.tourist_attractions] → {(1) Omni Coliseum (2) Masquerade (3) Arbor Place Mall (4) Turner Field (5) Atlanta History Center (6) Atlanta Ballet (7) Jimmy Carter Library and Museum (8) Variety Playhouse (9) Philips Arena (10) Zoo Atlanta}
Chain 3. Atlanta → [sports.sports_team_location.teams] → {(1) Chain Lightning (2) Georgia Tech Yellow Jackets football (3) Atlanta Thrashers (4) Atlanta Falcons}
Chain 4. Atlanta → [location.adjoining_relationship.adjoins] → {(1) Duluth}
Chain 5. Atlanta → [travel.transportation.transport_operator] → {(1) Amtrak (2) Greyhound Lines}
Chain 6. Atlanta → [travel.transportation.mode_of_transportation] → {(1) Air travel (2) Bus (3) Train}


Question:
what to do today in atlanta with kids?"""


ICL_ASS_PROMPT_triple_webqsp = """To find activities to do today in Atlanta with kids, we can look at the tourist attractions and activities suitable for families.

From Chain 1 and Chain 2, we see several attractions that are family-friendly. Therefore, we extract and list all answer entities from Chain 1 (1)–(9) and Chain 2 (1)–(10) as the formatted answers.

ans: Georgia State Capitol
ans: Georgia World Congress Center
ans: Four Seasons Hotel Atlanta
ans: Atlanta Jewish Film Festival
ans: Margaret Mitchell House & Museum
ans: Martin Luther King, Jr. National Historic Site
ans: Georgia Dome
ans: Woodruff Arts Center
ans: Six Flags White Water
ans: Omni Coliseum
ans: Masquerade
ans: Arbor Place Mall
ans: Turner Field
ans: Atlanta History Center
ans: Atlanta Ballet
ans: Jimmy Carter Library and Museum
ans: Variety Playhouse
ans: Philips Arena
ans: Zoo Atlanta"""












ICL_USER_PROMPT_path_level_inf = """Paths:
0. ('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XLVII')
1. ('Super Bowl XLVII', 'sports.sports_championship_event.champion', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.fight_song', 'The Baltimore Fight Song')
2. ('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XXXV')
3. ('Super Bowl XXXV', 'sports.sports_championship_event.champion', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.fight_song', 'The Baltimore Fight Song')


Question:
What year did the team with Baltimore Fight Song win the Superbowl?


Hints:
The Baltimore Fight Song
Superbowl
"""


ICL_ASS_PROMPT_brief_path_level_inf = """ans: Super Bowl XXXV
ans: Super Bowl XLVII
"""


ICL_ASS_PROMPT_3 = """To determine what country Japan exports to that contains the Phoenix Islands, we will focus on the hints provided: "Japan" and "Phoenix Islands." 

From the triplets, we can see that:

1. The Phoenix Islands are contained within Kiribati, as indicated by the triplet:
   - ('Phoenix Islands', 'location.administrative_division.country', 'Kiribati')

2. We need to find the exports from Japan. The relevant triplets that mention Japan's exports include:
   - ('Japan', 'location.statistical_region.places_exported_to', 'm.04c6sl7')
   - ('Japan', 'location.statistical_region.places_exported_to', 'm.0493792')

3. Then, we need to identify what these abstract identifiers (like 'm.048prwp') correspond to. The triplet:
   - ('m.04c6sl7', 'location.imports_and_exports.exported_to', 'Kiribati')
   - ('m.0493792', 'location.imports_and_exports.exported_to', 'United States of America')

4. Among these, the triplet:
   - ('m.04c6sl7', 'location.imports_and_exports.exported_to', 'Kiribati')
   indicates that Japan exports to Kiribati.

Since the Phoenix Islands are part of Kiribati, we conclude that Japan exports to Kiribati, which contains the Phoenix Islands.

Therefore, the answer is:

ans: Kiribati
"""


ICL_USER_PROMPT = """Paths:
Path 0. (Lou Seal,sports.mascot.team,San Francisco Giants), (San Francisco Giants,sports.sports_team.championships,{2012 World Series, 2014 World Series, 2010 World Series})


Scattered Triplets:
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,{2014 Major League Baseball season, 2010 World Series, 2010 Major League Baseball season, 2012 World Series})
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?
"""

ICL_ASS_PROMPT_brief = """ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)
"""

ICL_ASS_PROMPT = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From path 0, we can see that Lou Seal is the mascot of the San Francisco Giants, and San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""

SYS_PROMPT = (
    "Given the reasoning paths and additional scattered triplets retrieved from a knowledge graph, please answer the question."
    " If a triplet contains curly braces {}, it means the relation applies to multiple entities."
    " For example, the triplet (James K. Polk, people.person.profession, {Lawyer, Politician, Farmer}) shows that James K. Polk has multiple professions: Lawyer, Politician, and Farmer."
    ' Please return formatted answers, each on a new line and prefixed with "ans:".'
)

SYS_PROMPT_triple = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question."
    " If a triplet contains curly braces {}, it means the relation applies to multiple entities."
    " For example, the triplet (James K. Polk, people.person.profession, {Lawyer, Politician, Farmer}) shows that James K. Polk has multiple professions: Lawyer, Politician, and Farmer."
    ' Please return formatted answers, each on a new line and prefixed with "ans:".'
)

# SYS_PROMPT_triple_Apr = (
#     "Answer the question using evidence chains retrieved from a knowledge graph. Each evidence line follows this structure:\n"
#     "   - Entities and relations are connected by - symbols\n"
#     "   - Relations are enclosed in square brackets []\n"
#     "   - Multiple entities for the same relation are separated by |\n"
#     "For example, the evidence James K. Polk - [people.person.profession] - Lawyer | Politician | Farmer shows that James K. Polk has multiple professions: Lawyer, Politician, and Farmer."
#     "Please return formatted answers, each on a new line and prefixed with 'ans:'."
# )

SYS_PROMPT_EVIDENCE = (
    "Answer the question using evidence chains from a knowledge graph, where each evidence line represents a continuous, directional logical path flowing from left to right. "
    # "Each chain yields one or more target entities, which are either at the beginning or end of the chain, enclosed in '{}' and numbered from (1) to (N). "

    "For example, the evidence 'James K. Polk → [people.person.profession] → {(1) Lawyer (2) Politician}' shows that James K. Polk has multiple professions: Lawyer and Politician. "

    # "When identifying a chain critical for answering the query, extract all target entities listed inside '{ }' — from (1) to (N) — without missing any. "
    "Please return formatted answers, each on a new line and prefixed with 'ans:'."
)

SYS_PROMPT_EVIDENCE_QWQ = (
    "You are an analytical assistant that answers questions using provided knowledge graph evidence chains. Each line is a directed logical path, but only a small subset is relevant to any given question. "
    "For example, the evidence 'James K. Polk - [people.person.profession] - Lawyer | Politician | Farmer' shows that James K. Polk has multiple professions: Lawyer, Politician, and Farmer.\n"
    
    "When answering questions, follow this concise two-step reasoning process:\n\n"
    "1. **Question Analysis**\n"
    "   - Identify the key entities, properties, and relationships needed to answer the question.\n"
    "2. **Evidence Selection and Answer Derivation**\n"
    "   - QUICKLY Scan the provided evidence chains and select **only** the line(s) whose logic is directly relevant to the question. Ignore unrelated evidence.\n"
    "   - Derive the answer(s) based on the selected chain(s).\n"

    "**Note:**\n"
    "   - The questions are generally straightforward and do not require complex reasoning.\n"
    "   - Avoid 1) overthinking the question wording or interpreting it in a complicated or nuanced way; 2) giving detailed explanations for each chain.\n"
    "   - Keep your thinking process **brief, direct, and focused**—stay within 1000 tokens. \n\n"

    "After completing the reasoning, return formatted answers, each on a new line and prefixed with 'ans:'."
)

# Don't overthink the question wording. The questions are generally straightforward and direct. Focus on extracting the correct answer based on relevant evidence, rather than interpreting the question in a complicated or nuanced way. Avoid over-analyzing the phrasing.
# "   - Some later evidence lines may repeat the same information already presented earlier. You can safely ignore these without additional analysis.\n"


# SYS_PROMPT_EVIDENCE_QWQ = (
#     "You are an analytical assistant that answers questions using provided knowledge graph evidence chains, where each evidence line represents a directed logical path. "
#     "For example, the evidence 'James K. Polk - [people.person.profession] - Lawyer | Politician | Farmer' shows that James K. Polk has multiple professions: Lawyer, Politician, and Farmer.\n"
#     "When answering questions, follow this two-step reasoning process with a note:\n"
#     "1. Question Analysis: Identify key entities, properties, and relationships needed to answer the question.\n"
#     "2. Evidence Selection and Answer Derivation: Scan all the provided evidence chains and identify ONLY the line number whose logic is directly relevant to the question. Ignore irrelevant lines. Derive the answers using the identified chains.\n"
#     "**Note**: The questions are generally straightforward. Keep your reasoning short (less than 3000 tokens) and to the point — no need for lengthy explanations for each chain line.\n\n"
#     "After completing the reasoning, return formatted answers, each on a new line and prefixed with 'ans:'."
#     # "3. ANSWER FORMATION: Format the final answer based on the identified evidence. Each answer must start with 'ans:' on a new line.\n"
# )

# SYS_PROMPT_EVIDENCE_QWQ = (
#     "You are an analytical assistant that answers questions using provided knowledge graph evidence chains, where each evidence line represents a directed logical path."
#     "Required Workflow:\n"
#     "(1). QUESTION ANALYSIS: Identify key entities and relationships needed to answer the question.\n"
#     "(2). EVIDENCE SELECTION: Quickly scan the evidence chains and identify the line numbers directly relevant to the query demand. Ignore irrelevant lines. Then derive answers based on identified evidence.\n"
#     "(3). ANSWER FORMATION: Return formatted answers, each on a new line and prefixed with 'ans:'.\n"
#     "Follow (1) and (2) to generate a concise reasoning process between <think> and </think>. Follow (3) to generate final sanwers."
# )

ICL_USER_PROMPT_triple_QWQ = """Evidence Chains:
Chain 1. Watt per square metre per steradian - [measurement_unit.radiance_unit.measurement_system] - International System of Units
Chain 2. Watt per square metre per steradian - [measurement_unit.radiance_unit.measurement_system] - International System of Units - [measurement_unit.measurement_system.weight_units] - Gram | Kilogram | Tonne | Milligram | Microgram
Chain 3. Watt per square metre per steradian - [measurement_unit.radiance_unit.measurement_system] - International System of Units - [measurement_unit.measurement_system.data_rate_units] - Kilobit per second | Gigabit per second | Megabit per second | bit per second | Terabit per second
Chain 4. Watt per square metre per steradian - [measurement_unit.radiance_unit.measurement_system] - International System of Units - [measurement_unit.measurement_system.length_units] - Decametre | Micrometer | Decimetre | Millimeter | Kilometer | centimeters | Megametre | Picometre | Nanometre | Meter | Hectometre
Chain 5. Watt per square metre per steradian - [measurement_unit.radiance_unit.measurement_system] - International System of Units - [measurement_unit.measurement_system.radioactivity_units] - becquerels


Question:
watt per square metre per steradian is used in what unit of length?
"""

ICL_ASS_PROMPT_triple_QWQ = """<think>
Okay, let's tackle this question: "watt per square metre per steradian is used in what unit of length?" The main subject here is "Watt per square metre per steradian," which is a radiance unit in the International System of Units (SI). The question is asking about a unit of length connected to this.
Looking through the evidence lines, only Chain 4 shows the units of length associated with "watt per square metre per steradian" are: Decametre, Micrometer, Decimetre, etc. Therefore, I will present all length units referenced in Chain 4 as the answer.
</think>

ans: Decametre
ans: Micrometer
ans: Decimetre
ans: Millimeter
ans: Kilometer
ans: centimeters
ans: Megametre
ans: Picometre
ans: Nanometre
ans: Meter
ans: Hectometre
"""

# ICL_ASS_PROMPT_triple_QWQ = """<think>
# Okay, let's tackle this question: "watt per square metre per steradian is used in what unit of length?" 
# QUESTION ANALYSIS: The main subject here is "Watt per square metre per steradian," which is a radiance unit in the International System of Units (SI). The question is asking about length units connected to this.
# EVIDENCE SELECTION: After examining all evidence lines, only Chain 4 shows that the units of length associated with "watt per square metre per steradian" are: Decametre, Micrometer, Decimetre, etc. Therefore, I will present all length units referenced in Chain 4 as the answer.

# ans: Decametre
# ans: Micrometer
# ans: Decimetre
# ans: Millimeter
# ans: Kilometer
# ans: centimeters
# ans: Megametre
# ans: Picometre
# ans: Nanometre
# ans: Meter
# ans: Hectometre
# """

# SYS_PROMPT = (
#     " Based solely on the provided triplets from the knowledge graph, please answer the question."
#     " Do not rely on any common knowledge. Only use the information explicitly stated in the triplets."
#     " Your response should strictly adhere to the relationships, entities, and properties mentioned in the triplets. If any entity or relationship is not present in the triplets, do not infer or assume it."
#     " Please return your answer as a list, with each valid response prefixed with 'ans:'."
#     # For example, if the question asks about a country's official language, check for a triplet like ('Country', 'location.country.official_language', 'Language') and use it to form your answer.
# )

SYS_PROMPT_brief = (
"Based solely on the provided triplets, please answer the question. Pay special attention to the following:"
" 1. You will be provided with hints containing a list of entities which are extracted from the question. When answering the question, prioritize considering triplets that explicitly mention these entities. This will help you focus on relevant context for generating the most accurate response."
" 2. If the triplet you identify links the hint entity to an abstract identifier like 'm.0hpny13', simply look for other triplets involving this abstract entity, as it often acts as a bridge connecting the question entities to the answer." 
" 3. When answering the question, only use the information provided in the triplets. Do not rely on your inherent knowledge. Only the entities, relationships, and facts in the triplets are allowed to be used in your reasoning process."
" 4. You only need to return your answer(s), each on a new line and prefixed with 'ans:'. If no answers are found from the given triplets, simply return 'ans: Not available'."
)

SYS_PROMPT_brief_path_level_inf = (
"You are given a question along with several reasoning paths. Each path contains one or more triplets that are linked end-to-end."
" Based solely on information of the provided paths, please answer the question. Pay special attention to the following:"
" 1. You should combine multiple relevant paths to the question context in order to make a precise answer(s)."
" 2. You will be provided with hints that list entities which are extracted from the question. When answering, prioritize paths that explicitly mention these entities."
# " 2. If the triplet you identify links the hint entity to an abstract identifier like 'm.0hpny13', simply look for other triplets involving this abstract entity, as it often acts as a bridge connecting the question entities to the answer." 
" 3. Only use the information available in the reasoning paths to form your answer. Do not rely on your inherent knowledge. Your reasoning should be based solely on the entities, relationships, and facts present in the paths."
" 4. You only need to return the answer(s), each on a new line and prefixed with 'ans:'. If no answer can be derived from the given paths, simply return 'ans: Not available'."
)

# SYS_PROMPT = (
# "Based on the provided triplets, please answer the question. Pay special attention to the following:"
# " 1. You will be provided with hints containing a list of entities which are extracted from the question. When answering the question, prioritize considering triplets that explicitly mention these entities. This will help you focus on relevant context for generating the most accurate response."
# " 2. If the triplet you identify links the hint entity to an abstract identifier like 'm.0hpny13', simply look for other triplets involving this abstract entity, as it often acts as a bridge connecting the question entities to the answer." 
# # " 3. When answering the question, only use the information provided in the triplets. Do not rely on your inherent knowledge. Only the entities, relationships, and facts in the triplets are allowed to be used in your reasoning process."
# " 3. After thinking, return your answer(s), each on a new line and prefixed with 'ans:', like 'ans: A'. If no answers are found from the given triplets, return 'ans: Not available'."
# )


COT_PROMPT = (
    "Let's think step by step."
    ' Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'
)

SYS_PROMPT_PATH_old = (
    # oracal subset detection
    # TODO: whether or not ignore or preserve paths with repeating meaning, 
    "You are provided with a question, its answer(s), and a set of reasoning paths."
    " Your task is to select the paths that explicitly cover all entities (e.g., people, dates, locations) and relationships (e.g., geographical, temporal) mentioned in the question."
    " Pay attention that if the question involves multiple entities or relationships that span across different paths, identify all relevant paths and combine them together."
    # " If the question involves multiple entities or relationships that are spread across different paths, identify all such paths and combine them together."
    # " Pay special attention to ensuring that all critical concepts, including specific entities (e.g., people, dates, locations) and relationships (e.g., geographical, hierarchical, temporal), are fully represented in the selected paths."
    " Return all identified paths, each on a new line and prefixed with 'ans:', e.g., 'ans: Path 0'."
    # " If you need to combine information from multiple paths to answer the question, include all relevant paths that contribute to the answer."
    # assessment of detection
   #  " Once the paths are identified, assess whether they sufficiently cover all entities and relationships in the question."
   #  " If they do, append 'sign: STOP' to your answer. If any entity or relationship is missing, append 'sign: CONTINUE' to your answer."
)

SYS_PROMPT_PATH_old1 = (
   # description
   "You are provided with a question, its answer(s), and several reasoning paths. Each path represents a logical sequence of steps that leads toward an answer."
   # task
   " Your task is to evaluate each path individually and determine whether it should be selected based on the following criteria:"
   # notice
   " 1. A path should be selected if it connects some (but not necessarily all) of the entities and relationships mentioned in the question."
   " 2. Even if a single evaluated path does not cover all the entities and relationships, it should still be selected if it covers part of them."
   # " However, it should only be selected if it logically connects some (not necessarily all) key entities and relationships mentioned in the question and meaningfully contributes to answering (or partially answering) the question."
   # " However, if the reasoning is completely misaligned with the question or does not provide any meaningful information to answer it, the path should not be selected, even if it points to the correct answer."
   " 3. However, if the reasoning does not cover any entities or relationships, the path should not be selected, even if it points to the correct answer."   
   # return
   " 4. After evaluation, return the selected paths, each prefixed with 'ans:', such as 'ans: Path 0\nans: Path 1'. "
   # final assessment 
   # TODO: consider using.
   # " Finally, assess whether the selected paths sufficiently cover all necessary entities and relationships to answer the question." 
   # " If all entities and relationships are covered, append 'sign: STOP' to your answer. If any key entity or relationship is still missing, append 'sign: CONTINUE' to your answer."
)

SYS_PROMPT_PATH_grailqa = """Given a question, an answer, and reasoning paths (each representing a directed chain of entities connected by semantic relations), select paths based on these criteria:

1. Relevance: 
   - Relations in the selected path lexically match or semantically align with certain question keywords

2. Exclusivity:
   - Reject paths that:
     * Contain only peripheral information
     * Include irrelevant details
     * Demonstrate logical inconsistencies with question requirements

Return all selected paths prefixed with 'ans:', one per line. If none apply, return 'No valid paths found'.
"""

# ('Foveon X3 sensor', 'digicams.camera_color_filter_array_type.cameras', 'Sigma SD1'), ('Sigma SD1', 'digicams.digital_camera.supported_storage_types', 'CompactFlash')
# Foveon X3 sensor -> [digicams.camera_color_filter_array_type.cameras] -> Sigma SD1 -> [digicams.digital_camera.supported_storage_types] -> CompactFlash
# 

# in webqsp
# SYS_PROMPT_PATH = (
#     "You are provided with a question, its answer(s), and several reasoning paths."
#     " While all paths may arrive at a certain answer, some may use reasoning that deviates from the context and logic presented in the question."
#     " Your task is to carefully evaluate each path and select only those that strictly adhere to the question's context and logical framework."
#     " Filter out any paths that rely on irrelevant information or reasoning inconsistent with the question's requirements."
#     " After evaluation, return only the selected paths that meet the criteria, each on a new line and prefixed with 'ans:', such as 'ans: Path 0'."
#     " If no paths meet the criteria, return 'No valid paths found'."
# )

# SYS_PROMPT_PATH = (

#    "You are provided with a question, its answer(s), and several reasoning paths. Each path represents a logical sequence of steps that leads to an answer."

#    " Your task is to evaluate each path individually and assign it a score based on its relevance to answering the question, according to the following criteria:"
   
#    "\nScore = 1: If the path covers all the key entities and relationships mentioned in the question and provides sufficient information to fully answer the question."
   
#    "\nScore = 0: If the path covers some of the key entities or relationships from the question and provides useful information that can help justify or validate the final answer, even if it doesn't fully answer the question."
   
#    "\nScore = -1: If the path does not cover any key entities or relationships, or does not contribute any meaningful information to answering the question."

#    "\nFor example, for the question 'What country speaks Arabic in the Central Time Zone?' and its answer 'Canada', the path ('Canada', 'location.location.time_zones', 'Central Time Zone') should be marked as 0,"
#    " because this path cannot fully answer the question, but it captures the key entity 'Central Time Zone' and helps validate the answer by confirming the Canada is in the Central Time Zone."
   
#    " For the question 'Which college includes Newt Gincrich as an alumni?' and its answer 'Tulane University', the path ('Tulane University', 'common.topic.notable_types', 'College/University') should be marked as -1,"
#    " because this path does not provide any meaningful information for the question."

#    "\nFor each path, return its score in the format 'Path X: score' (e.g., 'Path 0: 1')."
#    # TODO: brief version
#    # "\nProvide a brief explanation for each path, justifying the assigned score."
#    )

# deprecated.
# SYS_PROMPT_PATH_STRICT = (
#     "You are provided with a question, the answer(s) to the question, and a set of reasoning paths which lead to the answer(s)."
#     " Your goal is to identify the reasoning path(s) that are strictly relevant to the context of the question."
#     " A reasoning path is considered relevant only if it explicitly mentions all the entities or concepts of the question."
#     " If a path does not meet this criterion—meaning it lacks a direct mention of one or more key entities or concepts—it should be considered non-relevant, even if it might indirectly hint at the answer."
#     " Please return the number of the identified relevant paths as a list, each prefixed with 'ans:', or return an empty list if no paths are relevant."
# )

ICL_USER_PROMPT_PATH_0 = """Paths:
Path0.
('Tupac Shakur', 'film.actor.film', 'm.0jz0c4'), ('m.0jz0c4', 'film.performance.film', "Gridlock'd")


Question:
What movie with film character named Mr. Woodson did Tupac star in?


Answer(s):
Gridlock'd
"""

ICL_ASS_PROMPT_PATH_0_brief = """Path 0: 0
"""

ICL_ASS_PROMPT_PATH_0 = """Path 0: 0
This path connects 'Tupac Shakur' to 'Gridlock'd', but it does not mention 'Mr. Woodson'. Therefore, it does not fully answer the question but provides useful information which helps validate the answer by confirming Tupac's involvement in Gridlock'd.
"""

ICL_USER_PROMPT_PATH_1 = """Paths:
Path0.
('Super Bowl', 'time.recurring_event.instances', 'Super Bowl XLVII')
Path1.
('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XLVII')


Question:
What year did the team with Baltimore Fight Song win the Superbowl?


Answers:
Super Bowl XLVII, Super Bowl XXXV
"""

ICL_ASS_PROMPT_PATH_1 = """Path 0: -1
This path mentions 'Super Bowl' and 'Super Bowl XLVII', but it does not provide any information about the team associated with the 'Baltimore Fight Song' or the year they won the Super Bowl. Therefore, it does not contribute meaningfully to answering the question.

Path 1: 1  
This path connects 'The Baltimore Fight Song' to 'Baltimore Ravens' and then links 'Baltimore Ravens' to 'Super Bowl XLVII'. This path identifies the team associated with the fight song and provides a year they won the Super Bowl, thus fully answering the question.
"""

ICL_ASS_PROMPT_PATH_1_brief = """Path 0: -1
Path 1: 1  
"""

ICL_USER_PROMPT_PATH_2 = """Paths:
Path0.
('Canada', 'location.location.time_zones', 'Central Time Zone')


Question:
What country speaks Arabic in the Central Time Zone?


Answer(s):
Canada
"""

ICL_ASS_PROMPT_PATH_2 = """Path 0: 0
This path identifies Canada as being in the Central Time Zone, which is relevant to the question. However, it does not mention the Arabic language, a key part of the question. Therefore, it does not fully answer the question but provides useful information about the time zone.
"""

ICL_ASS_PROMPT_PATH_2_brief = """Path 0: 0
"""

ICL_USER_PROMPT_PATH_3 = """Paths:
Path0.
('Spain', 'location.location.adjoin_s', 'm.03z95_p'), ('m.03z95_p', 'location.adjoining_relationship.adjoins', 'Portugal')
Path1.
('Portugal', 'common.topic.notable_types', 'Country')


Question:
What country sharing borders with Spain does the SetÃºbal District belong to?


Answer(s):
Portugal
"""

ICL_ASS_PROMPT_PATH_3 = """Path 0: 0
This path establishes that Spain adjoins Portugal, which is the country that shares a border with Spain. However, it does not connect to the Setúbal District. Therefore, it does not fully answer the question but provides useful information about the bordering information.

Path 1: -1
This path identifies Portugal as a country but does not address the specific relationship between Portugal and the Setúbal District or its borders with Spain. It fails to mention the key entities of "Setúbal District" and "borders with Spain," which are crucial for answering the question. Therefore, it does not contribute meaningful information relevant to the question and should be scored as -1.
"""

ICL_ASS_PROMPT_PATH_3_brief = """Path 0: 0
Path 1: -1
"""

ICL_USER_PROMPT_PATH = """Paths:
Path 0. 
('United States of America', 'location.location.partially_contains', 'American Falls'), ('American Falls', 'location.location.partially_containedby', 'Canada')
Path 1. 
('Mexico', 'location.country.form_of_government', 'Federal republic'), ('Federal republic', 'government.form_of_government.countries', 'United States of America')
Path 2. 
('United States of America', 'sports.sport_country.multi_event_tournaments_participated_in', '2012 World Mountain Running Championships'), ('2012 World Mountain Running Championships', 'sports.multi_event_tournament.participating_countries', 'Mexico')


Question:
which countries border the us?


Answer(s):
Mexico
Canada
"""

ICL_ASS_PROMPT_PATH = """The correct answer is Canada and Mexico. To identify the relevant reasoning paths, we need to focus on relationships that establish geographical borders with the United States. 

Path 0 involves the shared location of American Falls, which connects the United States and Canada through a geographical relationship. This path is valid as it adheres to the question's logical framework and context.

Path 1 links Mexico and the US via government form, which is irrelevant to the question's focus on geographical borders. 

Path 2 connects Mexico and the US through a sports event, also irrelevant.

Therefore, the relevant paths are: 

ans: Path 0
"""


ICL_USER_PROMPT_PATH_grailqa = """Paths:
Path 0. 
[('boeing company', 'spaceflight.rocket_manufacturer.rockets_manufactured', 'saturn v rocket'), ('saturn v rocket', 'spaceflight.rocket.manufacturer', 'North American Aviation')]
Path 1. 
[('boeing company', 'business.business_operation.industry', 'Aerospace'), ('Aerospace', 'business.industry.companies', 'North American Aviation')]
Path 2. 
[('Little Joe', 'spaceflight.rocket.manufacturer', 'North American Aviation')]
Path 3.
[('North American Aviation', 'aviation.aircraft_manufacturer.aircraft_models_made', 'North American XB-70A')]
Path 4.
[('saturn v rocket', 'spaceflight.rocket.height_meters', '110.6')]
Path 5.
[('saturn v rocket', 'spaceflight.rocket.country_of_origin', 'United States of America')]
Path 6.
[('saturn v rocket', 'spaceflight.rocket.mass', '3038500.0')]
Path 7.
[('Aerospace', 'business.industry.companies', 'North American Aviation')]


Question:
What rocket manufacturer produced Little Joe and also collaborated with Boeing on making a rocket with mass over 2.916e+06?


Answer(s):
North American Aviation
"""

ICL_ASS_PROMPT_PATH_grailqa = """To identify the relevant reasoning paths, we need to focus on relationships that:

1) Confirm the manufacturer produced Little Joe, 2) Show collaboration with Boeing on a rocket, and 3) Provide information about the rocket mass.

Path 0 is valid because it establishes collaboration with Boeing (via Saturn V) and confirms North American Aviation as the manufacturer.

Path 2 is valid because it directly proves North American Aviation produced Little Joe.

Path 6 is valid because it provides information about the rocket mass (3,038,500 - 2.916e+06).

For irrelevant paths, Path 1/7 link Boeing and North American Aviation via Aerospace, which is indirect (no collaboration on making a rocket).

Path 3 discusses aircraft models, which is irrelevant to rockets.

Path 4/5 Describe background info of Saturn V (irrelevant to its mass).

Therefore, the relevant paths are:

ans: Path 0
ans: Path 2
ans: Path 6
"""


SYS_PROMPT_TRI = (
    "You are provided with a question, the answer(s) to the question, and a set of triplets."
    " Your goal is to identify the triplet(s) that are relevant to the context of the question and lead to the correct answer(s)."
    ' Please return the number of the identified triplet(s) as a list.'
)


ICL_USER_PROMPT_TRI = """Triplets:
Triplet1.
('Super Bowl', 'time.recurring_event.instances', 'Super Bowl XLVII')
Triplet2.
('Super Bowl XLVII', 'sports.sports_championship_event.championship', 'Super Bowl')
Triplet3.
('Super Bowl', 'time.recurring_event.instances', 'Super Bowl XXXV')
Triplet4.
('Super Bowl XXXV', 'time.event.instance_of_recurring_event', 'Super Bowl')
Triplet5.
('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens')
Triplet6.
('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XLVII')
Triplet7.
('Super Bowl XLVII', 'sports.sports_championship_event.champion', 'Baltimore Ravens')
Triplet8.
('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XXXV')
Triplet9.
('Super Bowl XXXV', 'sports.sports_championship_event.champion', 'Baltimore Ravens')
Triplet10.
('Baltimore Ravens', 'sports.sports_team.fight_song', 'The Baltimore Fight Song')


Question:
What year did the team with Baltimore Fight Song win the Superbowl?


Answers:
Super Bowl XLVII, Super Bowl XXXV
"""


ICL_ASS_PROMPT_TRI = """The question asks about the year the team associated with "The Baltimore Fight Song" won the Super Bowl. Based on this, the relevant triplets are those linking "The Baltimore Fight Song" to the Baltimore Ravens and their Super Bowl victories. 

Triplet 5 establishes that "The Baltimore Fight Song" is associated with the Baltimore Ravens. Triplet 6 indicates that the Baltimore Ravens won Super Bowl XLVII, and Triplet 8 shows they also won Super Bowl XXXV. 

Therefore, the relevant triplets are:

ans: Triplet 5
ans: Triplet 6
ans: Triplet 8
"""


SYS_PROMPT_R = (
    "You are provided with a question and a set of relations."
    " Your goal is to identify the relations that can be useful for answering this question."
    ' Please return the number of the identified relation(s) as a list.'
)


ICL_USER_PROMPT_R = """Relations:
Relation1.
time.recurring_event.instances
Relation2.
sports.sports_championship_event.championship
Relation3.
time.event.instance_of_recurring_event
Relation4.
sports.fight_song.sports_team
Relation5.
sports.sports_team.championships
Relation6.
sports.sports_championship_event.champion
Relation7.
sports.sports_team.fight_song


Question:
What year did the team with Baltimore Fight Song win the Superbowl?
"""


ICL_ASS_PROMPT_R = """To answer the question "What year did the team with Baltimore Fight Song win the Superbowl?", we need to identify the relevant relations that can help us trace the team with the Baltimore Fight Song and its Superbowl victories.

Relation 4 links a fight song to a sports team. This helps identify the team with the Baltimore Fight Song.

Relation 5 connects a sports team to its championships. This will help us identify if the team has won a Superbowl.

Relation 6 provides the champion for a championship event, which would be relevant to identify the winning team of the Superbowl.

Thus, the relations that are useful for answering the question are:

ans: Relation4
ans: Relation5
ans: Relation6
"""

ICL_USER_PROMPT_R_W = """Relations:
Relation0.
people.person.parents
Relation1.
people.person.children
Relation2.
people.person.sibling_s
Relation3.
people.sibling_relationship.sibling


Question:
what is the name of justin bieber brother?
"""


ICL_ASS_PROMPT_R_W = """To answer the question, we need to identify relations that are relevant to sibling relationships, as the question specifically asks about a brother.

Relation2 indicates the siblings of a person, which directly relates to finding a brother.

Relation3 is useful because it provides sibling information, which can be helpful for identifying a brother.

Thus, the relations that are useful for answering the question are:

ans: Relation2
ans: Relation3
"""