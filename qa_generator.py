from uuid import uuid4

from utils import load_json


def generate_qas(interactions_file: str, context_id: str):
    """
    Generates Q&A rows for each interaction preference in three modes:
    'future', 'related', 'contradictory'.
    Returns a list of dicts and does NOT include the context itself.
    """
    interactions = load_json(interactions_file)
    # predefined queries per type (aligned to interactions order)
    future_qs = [
        "Any recommendations for travel-friendly coffee gear that’s quick to use on field trips?",
        "What are some unique audio stations or streaming services you'd suggest for work hours?",
        "Could you suggest indoor weekend hobbies that are mentally engaging but relaxing?",
        "Any big events or competitions worth watching this month—sports, games, anything?",
        "Where’s the best place to get unusual fruits or ingredients without leaving the house?",
        "Know of any really well-designed eco-friendly containers or gear lately?",
        "Any thoughts on practical cars that still have some character to them?",
        "Can you recommend a nostalgic snack that’s not too sweet but still indulgent?",
        "I’m updating my wellness routine—any high-impact or calming recs?",
        "Got any dark sci-fi book recs that aren’t too cliché?",
        "Any recommendations for travel-friendly coffee gear that’s quick to use on field trips?",
        "What are some unique audio stations or streaming services you'd suggest for work hours?",
        "Could you suggest indoor weekend hobbies that are mentally engaging but relaxing?",
        "Any big events or competitions worth watching this month—sports, games, anything?",
        "Where’s the best place to get unusual fruits or ingredients without leaving the house?",
        "Where’s the best place to get unusual fruits or ingredients without leaving the house?",
        "Any thoughts on practical cars that still have some character to them?",
        "Can you recommend a nostalgic snack that’s not too sweet but still indulgent?",
        "I’m updating my wellness routine—any high-impact or calming recs?",
        "Got any dark sci-fi book recs that aren’t too cliché?"
    ]
    related_qs = [
        "I'm putting together a survival kit for the field—got coffee stuff covered, just need to sort the rest.",
        "I’ve been playing audio a lot while debugging lately—it really helps me focus.",
        "I’ve cleared some time this weekend and I’m debating whether to stay in or head out.",
        "I’ve kept up with a few matches recently—definitely makes downtime more fun.",
        "I was trying to make something new and realized my ingredient stash is super basic.",
        "I’ve been trying to stay consistent with reducing waste, especially in the lab.",
        "My car’s holding up well on field drives, though it’s definitely got quirks.",
        "I was clearing out my drawer and found some old snacks—really took me back.",
        "I feel like my current pace is decent, but could use more physical balance.",
        "I stayed up writing again last night—must’ve gotten inspired by something I read.",
        "I'm putting together a survival kit for the field—got coffee stuff covered, just need to sort the rest.",
        "I’ve been playing audio a lot while debugging lately—it really helps me focus.",
        "I’ve cleared some time this weekend and I’m debating whether to stay in or head out.",
        "I’ve kept up with a few matches recently—definitely makes downtime more fun.",
        "I was trying to make something new and realized my ingredient stash is super basic.",
        "I was trying to make something new and realized my ingredient stash is super basic.",
        "My car’s holding up well on field drives, though it’s definitely got quirks.",
        "I was clearing out my drawer and found some old snacks—really took me back.",
        "I feel like my current pace is decent, but could use more physical balance.",
        "I stayed up writing again last night—must’ve gotten inspired by something I read."
    ]
    contradictory_qs = [
        "Thinking of trying out a matcha latte routine in the mornings instead—what do you think?",
        "I’m considering a full silent work policy this week to see how it impacts my concentration.",
        "I’m actually signing up for a weekend wilderness retreat—figure I should try something new.",
        "I might skip all the game streams this month, kind of burnt out on screen stuff.",
        "Thinking of checking out the Sunday local market this weekend just for fun—worth it?",
        "Do you think single-use biodegradable stuff is better than reusables in some cases?",
        "I’ve been eyeing a sleek EV coupe—feels like time for a switch.",
        "Thinking of trying a new chocolate dessert recipe tonight—feel like experimenting.",
        "Do you know any calming guided yoga series online? I might want to slow down a bit.",
        "I might check out this year's Booker Prize shortlist after all—worth it?",
        "Thinking of trying out a matcha latte routine in the mornings instead—what do you think?",
        "I’m considering a full silent work policy this week to see how it impacts my concentration.",
        "I’m actually signing up for a weekend wilderness retreat—figure I should try something new.",
        "I might skip all the game streams this month, kind of burnt out on screen stuff.",
        "Thinking of checking out the Sunday local market this weekend just for fun—worth it?",
        "Thinking of checking out the Sunday local market this weekend just for fun—worth it?",
        "I’ve been eyeing a sleek EV coupe—feels like time for a switch.",
        "Thinking of trying a new chocolate dessert recipe tonight—feel like experimenting.",
        "Do you know any calming guided yoga series online? I might want to slow down a bit.",
        "I might check out this year's Booker Prize shortlist after all—worth it?"
    ]

    qas = []
    for i, inter in enumerate(interactions):
        pref = inter.get('preference', f'pref_{i}')
        # future
        qas.append({
            'query_id': str(uuid4()),
            'user_query': future_qs[i],
            'query_type': 'future',
            'context_id': context_id
        })
        # related
        qas.append({
            'query_id': str(uuid4()),
            'user_query': related_qs[i],
            'query_type': 'related',
            'context_id': context_id
        })
        # contradictory
        qas.append({
            'query_id': str(uuid4()),
            'user_query': contradictory_qs[i],
            'query_type': 'contradictory',
            'context_id': context_id
        })
    return qas