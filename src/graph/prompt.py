"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["graph_construction_associa"] = """—Goal—
Please extract information about the users' events, user' behavioral trends and personal
information from the above text.
—Requirement—
1. Please infer the time of the event's occurrence and provide a specific, absolute time, rather
than simply expressing it as 'soon', 'later', or 'yesterday'.
2. Please extract entities as completely as possible, especially those related to the user.
Attention should be given not only to the facts about the user, but also to the user's needs,
intentions, sentiments, and reactions.
3. The entities in triplets can include types such as [Object, Person/User/Organization,
Resource, Place, Event, Goal/Intention, Time, Interest/Skill, Sentiment].
4. Extract the relevant events in the following format:
[{"event": "{event}",
"event_time": "{specific event time inferred}",
"description": "{description}",
"triplets": [(subject, predicate, object),]},]
If there is no user's information, please reply with [].
Output:
"""


# PROMPTS[
#     "claim_extraction"
# ] = """-Target activity-
# You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

# -Goal-
# Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

# -Steps-
# 1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
# 2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
# For each claim, extract the following information:
# - Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
# - Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
# - Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
# - Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
# - Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
# - Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
# - Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

# Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

# 3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

# 4. When finished, output {completion_delimiter}

# -Examples-
# Example 1:
# Entity specification: organization
# Claim description: red flags associated with an entity
# Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
# Output:

# (COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
# {completion_delimiter}

# Example 2:
# Entity specification: Company A, Person C
# Claim description: red flags associated with an entity
# Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
# Output:

# (COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
# {record_delimiter}
# (PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
# {completion_delimiter}

# -Real Data-
# Use the following input for your answer.
# Entity specification: {entity_specs}
# Claim description: {claim_description}
# Text: {input_text}
# Output: """

PROMPTS[
    "community_report"
] = """You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""

# PROMPTS[
#     "entity_extraction"
# ] = """-Goal-
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, capitalized
# - entity_type: One of the following types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
#  Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

# 3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

# 4. When finished, output {completion_delimiter}

# ######################
# -Examples-
# ######################
# Example 1:

# Entity_types: [person, technology, mission, organization, location]
# Text:
# while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

# Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

# The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

# It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
# ################
# Output:
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
# ("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
# ("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 2:

# Entity_types: [person, technology, mission, organization, location]
# Text:
# They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

# Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

# Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
# #############
# Output:
# ("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
# ("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
# ("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 3:

# Entity_types: [person, role, technology, organization, event, location, concept]
# Text:
# their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

# "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

# Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

# Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

# The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
# #############
# Output:
# ("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
# ("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
# ("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
# ("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
# ("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
# ("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
# #############################
# -Real Data-
# ######################
# Entity_types: {entity_types}
# Text: {input_text}
# ######################
# Output:
# """


PROMPTS[
    "entity_extraction"
] = """-Goal-
Given a multi-turn conversation consisting only of the user's messages (each turn separated by "\n"), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Steps-
1. Treat the entire conversation as one continuous narrative reflecting the user's life. Integrate information across all turns to infer complete and coherent entities and relationships.

2. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time `{dialogue_time}` as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to `{dialogue_time}`.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

4. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

5. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: use a **concise predicate** describing the relationship type (e.g., "use", "own", "buy", "track", "prefer", "plan", "occur_on", "come_from", "obtained_on", "used_with").  
  Avoid repeating detailed information already included in entity descriptions.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

6. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

7. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example Input:
Conversation time: 2023/05/20 (Sat) 02:57 
Text: "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!\nI've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality.\nThat's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions.\nI've also been tracking my blood pressure regularly with my new wireless blood pressure monitor from Omron, which I got on March 10th. I've been trying to keep an eye on it since my last check-up showed slightly higher than normal readings. Do you have any tips on how to lower blood pressure naturally?\nI'm also experimenting with essential oils for stress relief, and I recently got a new diffuser on March 22nd. It's been a game-changer for unwinding before bed. I've been using a lavender and chamomile blend that I got from a local health food store.\nI've been meaning to get a flu shot, but I haven't gotten around to it yet. I need to schedule an appointment with my doctor for that."

Expected Output:
("entity"{tuple_delimiter}"User"{tuple_delimiter}"User"{tuple_delimiter}"The user is focused on improving physical health, sleep quality, and stress management through various wellness habits including fitness tracking, yoga, stretching, and monitoring blood pressure."){record_delimiter}
("entity"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user's main goal is to increase daily step count and maintain fitness progress."){record_delimiter}
("entity"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"Object"{tuple_delimiter}"A fitness tracker purchased on February 15th, used by the user to monitor step count and activity progress."){record_delimiter}
("entity"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"time"{tuple_delimiter}"2023/02/15 (February 15th) is the date when the user bought the Fitbit Inspire HR."){record_delimiter}
("entity"{tuple_delimiter}"Yoga"{tuple_delimiter}"Interest/Skill"{tuple_delimiter}"A morning exercise routine practiced by the user to improve flexibility and sleep quality."){record_delimiter}
("entity"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"Health"{tuple_delimiter}"A health aspect the user aims to improve through yoga and relaxation techniques."){record_delimiter}
("entity"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Object"{tuple_delimiter}"A stretching and recovery tool purchased from Amazon and received on March 2nd, used three times a week after morning yoga to reduce muscle soreness."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/02 (March 2nd) is the date when the foam roller ordered from Amazon arrived."){record_delimiter}
("entity"{tuple_delimiter}"Amazon"{tuple_delimiter}"Organization"{tuple_delimiter}"An online store where the user purchased the foam roller."){record_delimiter}
("entity"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Object"{tuple_delimiter}"A wireless monitor from Omron, purchased on March 10th, used to track blood pressure regularly."){record_delimiter}
("entity"{tuple_delimiter}"Omron"{tuple_delimiter}"Organization"{tuple_delimiter}"The manufacturer of the user's blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/10 (March 10th) is the date when the user obtained the Omron blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"Behavior"{tuple_delimiter}"The user monitors blood pressure regularly due to slightly elevated readings from a past check-up."){record_delimiter}
("entity"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Object"{tuple_delimiter}"A lavender and chamomile blend purchased from a local health food store, used for stress relief and relaxation before bed."){record_delimiter}
("entity"{tuple_delimiter}"Diffuser"{tuple_delimiter}"Object"{tuple_delimiter}"A new diffuser purchased on March 22nd, used to diffuse essential oils for relaxation and better sleep."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/22 (March 22nd) is the date when the user bought the diffuser."){record_delimiter}
("entity"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"Place"{tuple_delimiter}"A local shop where the user bought lavender and chamomile essential oils."){record_delimiter}
("entity"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user aims to relieve stress through aromatherapy and relaxation habits such as using essential oils and yoga."){record_delimiter}
("entity"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Event"{tuple_delimiter}"A planned vaccination that the user intends to schedule with a doctor but has not yet completed."){record_delimiter}
("entity"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"Event"{tuple_delimiter}"A future medical appointment the user needs to schedule for the flu shot."){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"obtained_on"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Yoga"{tuple_delimiter}"practice"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Amazon"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Omron"{tuple_delimiter}"made_by"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"perform"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"use"|8){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Diffuser"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Diffuser"{tuple_delimiter}"used_with"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"aim_for"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"aim_for"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"plan"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"scheduled_for"{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

#############################
-Real Data-
######################
Conversation time: {dialogue_time}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "entity_extraction_short"
] = """-Goal-
Given a multi-turn conversation consisting only of the user's messages (each turn separated by "\n"), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Steps-
1. Treat the entire conversation as one continuous narrative reflecting the user's life. Integrate information across all turns to infer complete and coherent entities and relationships.

2. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time `{dialogue_time}` as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to `{dialogue_time}`.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

4. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

5. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: use a **concise predicate** describing the relationship type (e.g., "use", "own", "buy", "track", "prefer", "plan", "occur_on", "come_from", "obtained_on", "used_with").  
  Avoid repeating detailed information already included in entity descriptions.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

6. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

7. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example Input:
Conversation time: 2023/05/20 (Sat) 02:57 
Text: "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!\nI've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality.\nThat's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions."

Expected Output:
("entity"{tuple_delimiter}"User"{tuple_delimiter}"User"{tuple_delimiter}"The user is focused on maintaining fitness and improving overall health through workouts, yoga, and recovery routines. They are actively tracking progress and motivated by wearable technology."){record_delimiter}
("entity"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user aims to stay on top of fitness objectives, particularly increasing daily step count and maintaining consistent workout habits."){record_delimiter}
("entity"{tuple_delimiter}"Workouts"{tuple_delimiter}"Behavior"{tuple_delimiter}"Physical activities performed by the user to improve fitness and step count."){record_delimiter}
("entity"{tuple_delimiter}"Step Count"{tuple_delimiter}"Statistic"{tuple_delimiter}"The number of daily steps the user seeks to increase as part of their fitness goals."){record_delimiter}
("entity"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"Object"{tuple_delimiter}"A fitness tracker purchased on February 15th used by the user to monitor activity progress and step count."){record_delimiter}
("entity"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"time"{tuple_delimiter}"2023/02/15 (February 15th) is the date when the user bought the Fitbit Inspire HR."){record_delimiter}
("entity"{tuple_delimiter}"Yoga"{tuple_delimiter}"Interest/Skill"{tuple_delimiter}"A morning exercise routine practiced by the user to enhance flexibility and relaxation."){record_delimiter}
("entity"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"Health"{tuple_delimiter}"An aspect of health the user seeks to improve through yoga practice."){record_delimiter}
("entity"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Object"{tuple_delimiter}"A stretching and recovery tool purchased from Amazon, received on March 2nd, and used regularly after yoga sessions to reduce muscle soreness."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/02 (March 2nd) is the date when the foam roller ordered from Amazon arrived."){record_delimiter}
("entity"{tuple_delimiter}"Amazon"{tuple_delimiter}"Organization"{tuple_delimiter}"An online retailer where the user purchased the foam roller."){record_delimiter}
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user uses the foam roller approximately three times per week, typically after morning yoga sessions."){record_delimiter}
("entity"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"Behavior"{tuple_delimiter}"A recovery activity performed by the user using a foam roller to alleviate muscle soreness and aid flexibility."){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"aim_for"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Workouts"{tuple_delimiter}"perform"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"use"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"obtained_on"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Yoga"{tuple_delimiter}"practice"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Yoga"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"improve"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Amazon"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"used_for"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"frequency"{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

#############################
-Real Data-
######################
Conversation time: {dialogue_time}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "entity_relation_extraction_short"
] = """-Goal-
Given a multi-turn conversation consisting only of the user's messages (each turn separated by "\n"), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Steps-
1. Treat the entire conversation as one continuous narrative reflecting the user's life. Integrate information across all turns to infer complete and coherent entities and relationships.

2. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time `{dialogue_time}` as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to `{dialogue_time}`.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

4. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

5. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: a natural-language description explaining the relationship or connection between the source entity and the target entity.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

6. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

7. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example Input:
Conversation time: 2023/05/20 (Sat) 02:57 
Text: "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!\nI've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality.\nThat's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions."

Expected Output:
("entity"{tuple_delimiter}"User"{tuple_delimiter}"User"{tuple_delimiter}"The user is focused on maintaining fitness and improving overall health through workouts, yoga, and recovery routines. They are actively tracking progress and motivated by wearable technology."){record_delimiter}
("entity"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user aims to stay on top of fitness objectives, particularly increasing daily step count and maintaining consistent workout habits."){record_delimiter}
("entity"{tuple_delimiter}"Workouts"{tuple_delimiter}"Behavior"{tuple_delimiter}"Physical activities performed by the user to improve fitness and step count."){record_delimiter}
("entity"{tuple_delimiter}"Step Count"{tuple_delimiter}"Statistic"{tuple_delimiter}"The number of daily steps the user seeks to increase as part of their fitness goals."){record_delimiter}
("entity"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"Object"{tuple_delimiter}"A fitness tracker purchased on February 15th used by the user to monitor activity progress and step count."){record_delimiter}
("entity"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"time"{tuple_delimiter}"2023/02/15 (February 15th) is the date when the user bought the Fitbit Inspire HR."){record_delimiter}
("entity"{tuple_delimiter}"Yoga"{tuple_delimiter}"Interest/Skill"{tuple_delimiter}"A morning exercise routine practiced by the user to enhance flexibility and relaxation."){record_delimiter}
("entity"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"Health"{tuple_delimiter}"An aspect of health the user seeks to improve through yoga practice."){record_delimiter}
("entity"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Object"{tuple_delimiter}"A stretching and recovery tool purchased from Amazon, received on March 2nd, and used regularly after yoga sessions to reduce muscle soreness."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/02 (March 2nd) is the date when the foam roller ordered from Amazon arrived."){record_delimiter}
("entity"{tuple_delimiter}"Amazon"{tuple_delimiter}"Organization"{tuple_delimiter}"An online retailer where the user purchased the foam roller."){record_delimiter}
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user uses the foam roller approximately three times per week, typically after morning yoga sessions."){record_delimiter}
("entity"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"Behavior"{tuple_delimiter}"A recovery activity performed by the user using a foam roller to alleviate muscle soreness and aid flexibility."){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"The user is actively working toward these fitness goals."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Workouts"{tuple_delimiter}"The user performs workouts to improve fitness and increase step count."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"The Fitbit Inspire HR is used by the user to track their activity progress."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"This is the date when the user obtained the Fitbit Inspire HR."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Yoga"{tuple_delimiter}"The user practices yoga regularly in the morning."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Yoga"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"The user believes that practicing yoga could help improve sleep quality."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"The user uses the foam roller during post-workout stretching to reduce soreness."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Amazon"{tuple_delimiter}"The foam roller was purchased by the user from Amazon."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"This is the date when the foam roller arrived."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"The foam roller is used by the user specifically for post-workout stretching."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Post-Workout Stretching"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"The user performs post-workout stretching with this frequency."{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

#############################
-Real Data-
######################
Conversation time: {dialogue_time}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "entity_relation_extraction_clonemem"
] = """-Goal-
Given a User-generated content (such as notes, messages, etc.), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Steps-
1. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time `{dialogue_time}` as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to `{dialogue_time}`.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

3. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

4. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: a natural-language description explaining the relationship or connection between the source entity and the target entity.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

5. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

6. When finished, output {completion_delimiter}.

#############################
-Real Data-
######################
Time: {dialogue_time}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "entity_extraction_with_judge"
] = """-Goal-
Given a multi-turn conversation consisting only of the user's messages (each turn separated by "\n"), extract structured information that reflects the user's activities, possessions, goals, behaviors and reactions. Identify all relevant entities and their relationships to build a knowledge graph representing the user's context and life events.

-Conversation Relevance Filter-
If the conversation is **purely a temporary or impersonal task not related to the user** (e.g., translation, calculation, coding, puzzle-solving, creative writing, trivia, or reasoning not related to the user), do not perform extraction and return a short explanation less than 50 words.

-Steps-
1. Treat the entire conversation as one continuous narrative reflecting the user's life. Integrate information across all turns to infer complete and coherent entities and relationships.

2. Identify all entities mentioned or implied by the user. For each entity, extract:
- entity_name: Name of the entity, capitalized.
- entity_type: One of the following types: [User, Person, Object, Resource, Event, Goal/Intention, Time, Statistic, Duration, Place, Organization, Interest/Skill, Sentiment, Health, Behavior, Other]
- entity_description: A comprehensive description summarizing how this entity relates to the user and any attributes mentioned (e.g., purpose, frequency, purchase time, emotional tone).
Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Time Normalization and Extraction
Whenever a specific or relative date is mentioned in the conversation, standardize it as a separate entity of type `"time"`.  
Follow these rules:
- Use the provided conversation time `{dialogue_time}` as reference.
- If an explicit date is mentioned (e.g., “March 2nd”), convert it to `YYYY/MM/DD` format.
- If a relative time (e.g., “yesterday”, “last week”) appears, infer its absolute date relative to `{dialogue_time}`.
- Do **not** create separate entities for recurring or habitual times (e.g., “every morning”, “three times a week”); include such patterns only in related entity/relationship descriptions.
- Each time entity should describe **what happened at/before/after that time**.

4. Quantitative & Frequency Extraction
Explicitly extract any quantity, count, frequency, or duration mentioned in the conversation that describes the user's actions, achievements, or possessions.
Include these as separate `"Statistic"` or `"Duration"` entities.
Examples:
("entity"{tuple_delimiter}"Three Goals"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user has scored 3 goals in the indoor soccer league until YYYY/MM/DD.")
("entity"{tuple_delimiter}"Three Times A Week"{tuple_delimiter}"Statistic"{tuple_delimiter}"The user performs an activity 3 times a week.")
("entity"{tuple_delimiter}"Five Weeks"{tuple_delimiter}"Duration"{tuple_delimiter}"The activity lasted for 5 weeks.")

5. From the identified entities, detect all pairs of (source_entity, target_entity) that have a meaningful or causal relationship in the context of the user's life. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: use a **concise predicate** describing the relationship type (e.g., "use", "own", "buy", "track", "prefer", "plan", "occur_on", "come_from", "obtained_on", "used_with").  
  Avoid repeating detailed information already included in entity descriptions.
- relationship_strength: a numeric score (1–10) estimating how strong or explicit this connection is.
Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

6. Return output in English as a single list of all identified entities and relationships. Use **{record_delimiter}** as the list delimiter.

7. When finished, output {completion_delimiter}.

######################
-Examples-
######################
Example Input:
Conversation time: 2023/05/20 (Sat) 02:57 
Text: "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!\nI've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality.\nThat's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions.\nI've also been tracking my blood pressure regularly with my new wireless blood pressure monitor from Omron, which I got on March 10th. I've been trying to keep an eye on it since my last check-up showed slightly higher than normal readings. Do you have any tips on how to lower blood pressure naturally?\nI'm also experimenting with essential oils for stress relief, and I recently got a new diffuser on March 22nd. It's been a game-changer for unwinding before bed. I've been using a lavender and chamomile blend that I got from a local health food store.\nI've been meaning to get a flu shot, but I haven't gotten around to it yet. I need to schedule an appointment with my doctor for that."

Expected Output:
("entity"{tuple_delimiter}"User"{tuple_delimiter}"User"{tuple_delimiter}"The user is focused on improving physical health, sleep quality, and stress management through various wellness habits including fitness tracking, yoga, stretching, and monitoring blood pressure."){record_delimiter}
("entity"{tuple_delimiter}"Fitness Goals"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user's main goal is to increase daily step count and maintain fitness progress."){record_delimiter}
("entity"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"Object"{tuple_delimiter}"A fitness tracker purchased on February 15th, used by the user to monitor step count and activity progress."){record_delimiter}
("entity"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"time"{tuple_delimiter}"2023/02/15 (February 15th) is the date when the user bought the Fitbit Inspire HR."){record_delimiter}
("entity"{tuple_delimiter}"Yoga"{tuple_delimiter}"Interest/Skill"{tuple_delimiter}"A morning exercise routine practiced by the user to improve flexibility and sleep quality."){record_delimiter}
("entity"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"Health"{tuple_delimiter}"A health aspect the user aims to improve through yoga and relaxation techniques."){record_delimiter}
("entity"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Object"{tuple_delimiter}"A stretching and recovery tool purchased from Amazon and received on March 2nd, used three times a week after morning yoga to reduce muscle soreness."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/02 (March 2nd) is the date when the foam roller ordered from Amazon arrived."){record_delimiter}
("entity"{tuple_delimiter}"Amazon"{tuple_delimiter}"Organization"{tuple_delimiter}"An online store where the user purchased the foam roller."){record_delimiter}
("entity"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Object"{tuple_delimiter}"A wireless monitor from Omron, purchased on March 10th, used to track blood pressure regularly."){record_delimiter}
("entity"{tuple_delimiter}"Omron"{tuple_delimiter}"Organization"{tuple_delimiter}"The manufacturer of the user's blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/10 (March 10th) is the date when the user obtained the Omron blood pressure monitor."){record_delimiter}
("entity"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"Behavior"{tuple_delimiter}"The user monitors blood pressure regularly due to slightly elevated readings from a past check-up."){record_delimiter}
("entity"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Object"{tuple_delimiter}"A lavender and chamomile blend purchased from a local health food store, used for stress relief and relaxation before bed."){record_delimiter}
("entity"{tuple_delimiter}"Diffuser"{tuple_delimiter}"Object"{tuple_delimiter}"A new diffuser purchased on March 22nd, used to diffuse essential oils for relaxation and better sleep."){record_delimiter}
("entity"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"time"{tuple_delimiter}"2023/03/22 (March 22nd) is the date when the user bought the diffuser."){record_delimiter}
("entity"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"Place"{tuple_delimiter}"A local shop where the user bought lavender and chamomile essential oils."){record_delimiter}
("entity"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"Goal/Intention"{tuple_delimiter}"The user aims to relieve stress through aromatherapy and relaxation habits such as using essential oils and yoga."){record_delimiter}
("entity"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Event"{tuple_delimiter}"A planned vaccination that the user intends to schedule with a doctor but has not yet completed."){record_delimiter}
("entity"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"Event"{tuple_delimiter}"A future medical appointment the user needs to schedule for the flu shot."){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Fitbit Inspire HR"{tuple_delimiter}"2023/02/15"{tuple_delimiter}"obtained_on"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Yoga"{tuple_delimiter}"practice"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"Amazon"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Foam Roller"{tuple_delimiter}"2023/03/02"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"use"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"Omron"{tuple_delimiter}"made_by"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Wireless Blood Pressure Monitor"{tuple_delimiter}"2023/03/10"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Blood Pressure Tracking"{tuple_delimiter}"perform"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"use"|8){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Local Health Food Store"{tuple_delimiter}"buy_from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Diffuser"{tuple_delimiter}"2023/03/22"{tuple_delimiter}"obtained_on"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Essential Oils"{tuple_delimiter}"Diffuser"{tuple_delimiter}"used_with"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Stress Relief"{tuple_delimiter}"aim_for"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Sleep Quality"{tuple_delimiter}"aim_for"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"User"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"plan"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Flu Shot"{tuple_delimiter}"Doctor Appointment"{tuple_delimiter}"scheduled_for"{tuple_delimiter}8){record_delimiter}
{completion_delimiter}

#############################
-Real Data-
######################
Conversation time: {dialogue_time}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

# PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Object", "Person", "User", "Organization", "Resource", "Place", "Event", "Goal", "Intention", "Time", "Interest", "Skill", "Sentiment"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "naive_rag_response"
] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
