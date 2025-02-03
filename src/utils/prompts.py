"""
Define all prompts used in this project.
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
(AT&T Park,location.location.events,2010 World Series)
(Crazy Crab,sports.mascot.team,San Francisco Giants)
(New York Giants,baseball.baseball_team.league,National League)
(San Francisco Giants,sports.sports_team.colors,Black)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Gothams)
(m.0k079qm,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(m.0k079ry,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(2010 World Series,time.event.locations,AT&T Park)
(San Francisco Giants,time.participant.event,2012 Major League Baseball season)
(San Francisco Giants,baseball.baseball_team.league,National League)
(m.0crtd80,sports.sports_league_participation.league,National League West)
(San Francisco Giants,sports.sports_team.location,San Francisco)
(San Francisco Giants,sports.sports_team.sport,Baseball)
(m.05n6dtn,baseball.baseball_team_stats.team,San Francisco Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?"""


ICL_ASS_PROMPT = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""

SYS_PROMPT = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question."
    ' Please return formatted answers as a list, each prefixed with "ans:".'
)


COT_PROMPT = (
    "Let's think step by step."
    ' Return the most possible answers based on the given triplets by listing each answer on a separate line, starting with the prefix "ans:".'
    ' Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'
)

SYS_PROMPT_PATH = (
    # oracal subset detection
    # TODO: whether or not ignore or preserve paths with repeating meaning, 
    "You are provided with a question, its answer(s), and a set of reasoning paths."
    " Your task is to select the paths that explicitly cover all entities (e.g., people, dates, locations) and relationships (e.g., geographical, temporal) mentioned in the question."
    " Pay attention that if the question involves multiple entities or relationships that span across different paths, identify all relevant paths and combine them together."
    # " If the question involves multiple entities or relationships that are spread across different paths, identify all such paths and combine them together."
    # " Pay special attention to ensuring that all critical concepts, including specific entities (e.g., people, dates, locations) and relationships (e.g., geographical, hierarchical, temporal), are fully represented in the selected paths."
    " Return the number of all identified paths as a list, each prefixed with 'ans:'."
    # " If you need to combine information from multiple paths to answer the question, include all relevant paths that contribute to the answer."
    # assessment of detection
    " Once the paths are identified, assess whether they sufficiently cover all entities and relationships in the question."
    " If they do, append 'sign: STOP' to your answer. If any entity or relationship is missing, append 'sign: CONTINUE' to your answer."
)

# deprecated.
# SYS_PROMPT_PATH_STRICT = (
#     "You are provided with a question, the answer(s) to the question, and a set of reasoning paths which lead to the answer(s)."
#     " Your goal is to identify the reasoning path(s) that are strictly relevant to the context of the question."
#     " A reasoning path is considered relevant only if it explicitly mentions all the entities or concepts of the question."
#     " If a path does not meet this criterion—meaning it lacks a direct mention of one or more key entities or concepts—it should be considered non-relevant, even if it might indirectly hint at the answer."
#     " Please return the number of the identified relevant paths as a list, each prefixed with 'ans:', or return an empty list if no paths are relevant."
# )

ICL_USER_PROMPT_PATH = """Paths:
Path 0.
('Super Bowl', 'time.recurring_event.instances', 'Super Bowl XLVII')
Path 1.
('Super Bowl', 'time.recurring_event.instances', 'Super Bowl XXXV')
Path 2.
('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XLVII')
Path 3.
('The Baltimore Fight Song', 'sports.fight_song.sports_team', 'Baltimore Ravens'), ('Baltimore Ravens', 'sports.sports_team.championships', 'Super Bowl XXXV')


Question:
What year did the team with Baltimore Fight Song win the Superbowl?


Answers:
Super Bowl XLVII, Super Bowl XXXV
"""

ICL_ASS_PROMPT_PATH = """To answer the question about the years the team associated with "The Baltimore Fight Song" won the Super Bowl, we need to identify paths that connect the fight song to the Baltimore Ravens and then show the Super Bowls they won.

The relevant paths are:

- **Path 2**: This path connects "The Baltimore Fight Song" to the Baltimore Ravens and indicates that they won Super Bowl XLVII.
- **Path 3**: This path also connects "The Baltimore Fight Song" to the Baltimore Ravens and indicates that they won Super Bowl XXXV.

Paths 0 and 1 only mention the Super Bowls but do not connect them to the Baltimore Ravens or the fight song.

Thus, the identified paths that lead to the correct answer are:

ans: Path 2  
ans: Path 3

After reviewing the identified paths, Path 2 and Path 3 together cover all the key elements of the question: they link 'The Baltimore Fight Song' to the Baltimore Ravens and specify the years they won the Super Bowl (XLVII and XXXV).

All relevant concepts (team, fight song, Super Bowl years) are present and connected.

sign: STOP
"""

# ICL_USER_PROMPT_PATH_NEG = """Paths:
# Path 0.
# ("George Washington Colonials men's basketball", 'sports.sports_team.arena_stadium', 'Charles E. Smith Center'), ('Charles E. Smith Center', 'location.location.containedby', 'Washington, D.C.')


# Question:
# What state is home to the university that is represented in sports by George Washington Colonials men's basketball?


# Answers:
# Washington, D.C.
# """


# ICL_ASS_PROMPT_PATH_NEG = """To answer the question about the state home to the university represented in sports by George Washington Colonials men's basketball, we need to identify paths that connect the George Washington Colonials men's basketball team to its home university and then determine the location.

# Path 0 connects the George Washington Colonials men's basketball team to the Charles E. Smith Center and shows that the center is located in Washington, D.C. However, it does not explicitly mention George Washington University, as required by the question.

# Since Path 0 connects to Washington, D.C., but doesn't mention the university explicitly, it is not strictly relevant to the question context.

# Thus, the identified paths that lead to the correct answer are:

# ans: []
# """


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