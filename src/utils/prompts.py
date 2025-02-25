"""
Define all prompts used in this project.
"""

ICL_USER_PROMPT_2 = """Triplets:
(Barbados, location.location.containedby, Caribbean)
(Netherlands Antilles, location.location.containedby, Caribbean)
(Cura�ao, location.country.official_language, Dutch Language)
(Anguilla, location.location.containedby, Caribbean)
(English Language, language.human_language.countries_spoken_in, Bahamas)
(Spanish Language, language.human_language.countries_spoken_in, Barbados)
(Barbados, location.country.languages_spoken, English Language)
(Barbados, location.location.containedby, British West Indies)
(Sint Maarten, location.country.official_language, Dutch Language)
(Caribbean, location.location.contains, British Virgin Islands)
(Netherlands Antilles, location.country.official_language, Dutch Language)
(Flemish language, language.human_language.main_country, Belgium)


Question:
Of the countries that are in the Caribbean, which has the Dutch Language as an official language?


Hints:
Caribbean
Dutch Language
"""

ICL_ASS_PROMPT_2_brief = """ans: Netherlands Antilles
"""

ICL_ASS_PROMPT_2 = """To find the countries in the Caribbean that have Dutch Language as an official language, taking into account the hints, we will look for triplets that link countries to the location 'Caribbean' and also link countries to 'Dutch Language' as their official language.

From the triplets, we can identify the following:

1. Countries in the Caribbean:
   - Barbados (mentioned in the triplet: 'Barbados', 'location.location.containedby', 'Caribbean')
   - Netherlands Antilles (mentioned in the triplet: 'Netherlands Antilles', 'location.location.containedby', 'Caribbean')
   - Anguilla (mentioned in the triplet: 'Anguilla', 'location.location.containedby', 'Caribbean')
   - British Virgin Islands (mentioned in the triplet: 'Caribbean', 'location.location.contains', 'British Virgin Islands')

2. Countries with Dutch Language as an official language:
   - Cura�ao (mentioned in the triplet: 'Cura�ao', 'location.country.official_language', 'Dutch Language')
   - Sint Maarten (mentioned in the triplet: 'Sint Maarten', 'location.country.official_language', 'Dutch Language')
   - Netherlands Antilles (mentioned in the triplet: 'Netherlands Antilles', 'location.country.official_language', 'Dutch Language')

3. **Construct the sets**:
   - Caribbean countries: Barbados, Netherlands Antilles, Anguilla, British Virgin Islands
   - Countries with Dutch Language as an official language: Cura�ao, Sint Maarten, Netherlands Antilles

4. **Strict set intersection**: When finding the intersection of the two sets, we only include entities that appear in both sets. Do **not** include entities that only appear in one set.
   - Netherlands Antilles

In this process, we **strictly use** the entities mentioned in the triplets, and **ensure that only entities that appear in both sets are included in the intersection**.
   
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


Hints:
Japan
Phoenix Islands
"""

ICL_ASS_PROMPT_3_brief = """ans: Kiribati
"""

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


ICL_USER_PROMPT = """Triplets:
(Lou Seal,sports.mascot.team,San Francisco Giants)
(San Francisco Giants,sports.sports_team.championships,2012 World Series)
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,2014 Major League Baseball season)
(San Francisco Giants,time.participant.event,2010 World Series)
(San Francisco Giants,time.participant.event,2010 Major League Baseball season)
(San Francisco Giants,sports.sports_team.championships,2014 World Series)
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.sports_team.championships,2010 World Series)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco Giants,time.participant.event,2012 World Series)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?


Hints:
Lou Seal
"""

ICL_ASS_PROMPT_brief = """ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)
"""

ICL_ASS_PROMPT = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""

# SYS_PROMPT = (
#     "Based on the triplets retrieved from a knowledge graph, please answer the question."
#     ' Please return formatted answers as a list, each prefixed with "ans:".'
# )

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

SYS_PROMPT = (
"Based solely on the provided triplets, please answer the question. Pay special attention to the following:"
" 1. You will be provided with hints containing a list of entities which are extracted from the question. When answering the question, prioritize considering triplets that explicitly mention these entities. This will help you focus on relevant context for generating the most accurate response."
" 2. If the triplet you identify links the hint entity to an abstract identifier like 'm.0hpny13', simply look for other triplets involving this abstract entity, as it often acts as a bridge connecting the question entities to the answer." 
" 3. When answering the question, only use the information provided in the triplets. Do not rely on your inherent knowledge. Only the entities, relationships, and facts in the triplets are allowed to be used in your reasoning process."
" 4. After thinking, return your answer(s), each on a new line and prefixed with 'ans:', like 'ans: A'. If no answers are found from the given triplets, return 'ans: Not available'."
)


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

SYS_PROMPT_PATH = (

   "You are provided with a question, its answer(s), and several reasoning paths. Each path represents a logical sequence of steps that leads to an answer."

   " Your task is to evaluate each path individually and assign it a score based on its relevance to answering the question, according to the following criteria:"
   
   "\nScore = 1: If the path covers all the key entities and relationships mentioned in the question and provides sufficient information to fully answer the question."
   
   "\nScore = 0: If the path covers some of the key entities or relationships from the question and provides useful information that can help justify or validate the final answer, even if it doesn't fully answer the question."
   
   "\nScore = -1: If the path does not cover any key entities or relationships, or does not contribute any meaningful information to answering the question."

   "\nFor example, for the question 'What country speaks Arabic in the Central Time Zone?' and its answer 'Canada', the path ('Canada', 'location.location.time_zones', 'Central Time Zone') should be marked as 0,"
   " because this path cannot fully answer the question, but it captures the key entity 'Central Time Zone' and helps validate the answer by confirming the Canada is in the Central Time Zone."
   
   " For the question 'Which college includes Newt Gincrich as an alumni?' and its answer 'Tulane University', the path ('Tulane University', 'common.topic.notable_types', 'College/University') should be marked as -1,"
   " because this path does not provide any meaningful information for the question."

   "\nFor each path, return its score in the format 'Path X: score' (e.g., 'Path 0: 1')."
   # TODO: brief version
   # "\nProvide a brief explanation for each path, justifying the assigned score."
   )

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

ICL_USER_PROMPT_PATH_W = """Paths:
Path0.
('Justin Bieber', 'people.person.parents', 'Jeremy Bieber'), ('Jeremy Bieber', 'people.person.children', 'Jaxon Bieber')
Path1.
('Justin Bieber', 'people.person.sibling_s', 'm.0gxnnwp'), ('m.0gxnnwp', 'people.sibling_relationship.sibling', 'Jaxon Bieber')
Path2.
('Jaxon Bieber', 'people.person.sibling_s', 'm.0gxnnwp'), ('m.0gxnnwp', 'people.sibling_relationship.sibling', 'Justin Bieber')
Path3.
('Jaxon Bieber', 'people.person.parents', 'Jeremy Bieber'), ('Jeremy Bieber', 'people.person.children', 'Justin Bieber')


Question:
what is the name of justin bieber brother?


Answer(s):
Jaxon Bieber
"""

ICL_ASS_PROMPT_PATH_W = """The correct answer is Jaxon Bieber. To identify the relevant reasoning paths, we need to focus on relationships that establish Jaxon Bieber as Justin Bieber's brother.

Path0 involves Justin Bieber's father, Jeremy Bieber, and his children (including Jaxon Bieber), which directly establishes the family relationship.

Path3 also connects Jaxon Bieber and Justin Bieber through their shared parent, Jeremy Bieber, which confirms that they are siblings.

Therefore, the relevant paths are:

ans: Path 0
ans: Path 3
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