"""
Define all prompts used in this project.
"""

PLACEHOLDER = "."  # The effect of placeholder

INSTRUCTION = (
    "Based on the following triplets from a knowledge graph, please answer the given question. "
    "Please keep the answers as simple as possible and return all the possible answers as a list.\n\n"
)

RETRIEVAL_CONTEXT = """Triplets:\n"""  # + triplet token-ids

QUESTION = """\n\nQuestion:\n{question}"""

BOS = '<s>'
BOS_INST = '[INST]'
EOS_INST = '[/INST]'
EOS = '</s>'

FORMER = BOS + BOS_INST + INSTRUCTION + RETRIEVAL_CONTEXT
# masked triplet
LATTER = QUESTION + EOS_INST
LABEL = """{label}""" + EOS


# v1 prompt:
# bos + bos_inst + inst + triplet + query + eos_inst + labels + eos.
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