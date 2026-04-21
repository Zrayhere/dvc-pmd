"""
build_boundary_tokens_yc2.py
----------------------------
STT (Semantic Transition Token) generation for YouCook2.

Parses GT captions with spaCy to extract (action, object) pairs, classifies
each adjacent-event transition into one of {strong, mid-action, mid-object,
weak}, and emits time-anchored typed boundary phrases for all non-weak
transitions. Weak boundaries are deliberately skipped (paper Sec. III-C).

Output JSON schema (per video):
    {
        "<video_id>": {
            "boundary_parts": [
                {"text": "boundary: after cutting the onion, stir the sauce",
                 "t_end_prev": 12.4, "t_start_next": 14.1},
                ...
            ],
            "n_boundaries": 4,
            "details": [...]   # full per-boundary diagnostic info
        },
        ...
    }

Usage:
    python build_boundary_tokens_yc2.py \\
        --json_path   ./data/yc2/train.json \\
        --output_path ./data/yc2/boundary_tokens.json
"""

import json
import argparse
from collections import Counter

import spacy


# ============================================================================
# 1. Action groups and object categories (YouCook2; cooking domain only)
# ============================================================================

ACTION_GROUPS = {
    "CUT":    {"cut", "slice", "chop", "dice", "mince", "trim", "peel",
               "shred", "grate", "julienne"},
    "ADD":    {"add", "put", "place", "pour", "drizzle", "sprinkle", "drop",
               "toss"},
    "MIX":    {"mix", "stir", "whisk", "combine", "blend", "fold", "beat",
               "knead"},
    "COOK":   {"cook", "fry", "bake", "roast", "boil", "simmer", "grill",
               "saute", "sear", "broil", "steam", "braise", "toast", "heat",
               "warm", "microwave"},
    "SERVE":  {"serve", "garnish", "plate", "arrange", "top", "decorate",
               "finish"},
    "REMOVE": {"remove", "take", "drain", "strain", "lift", "transfer",
               "scoop"},
    "SPREAD": {"spread", "coat", "brush", "rub", "smear", "glaze", "marinate",
               "season"},
    "SHAPE":  {"roll", "flatten", "shape", "form", "press", "fold", "wrap",
               "stuff"},
    "COVER":  {"cover", "wrap", "seal", "close", "lid"},
    "FLIP":   {"flip", "turn", "rotate", "toss"},
}

OBJECT_CATEGORIES = {
    "VEGETABLE": {"onion", "garlic", "tomato", "pepper", "carrot", "potato",
                  "broccoli", "celery", "cucumber", "lettuce", "spinach",
                  "cabbage", "mushroom", "corn", "pea", "bean", "zucchini",
                  "eggplant", "squash", "ginger", "scallion", "shallot",
                  "leek", "chili", "jalapeno", "parsley", "cilantro", "basil",
                  "oregano", "thyme", "rosemary", "dill", "mint"},
    "MEAT":      {"chicken", "beef", "pork", "lamb", "meat", "steak",
                  "sausage", "bacon", "turkey", "duck", "ham", "veal", "ribs"},
    "SEAFOOD":   {"fish", "shrimp", "salmon", "tuna", "crab", "lobster",
                  "squid", "prawn", "mussel", "clam", "oyster", "octopus"},
    "DAIRY":     {"cheese", "butter", "cream", "milk", "yogurt", "egg",
                  "eggs", "parmesan", "mozzarella", "cheddar", "margarine"},
    "GRAIN":     {"rice", "pasta", "noodles", "bread", "flour", "dough",
                  "tortilla", "couscous", "quinoa", "oats", "cereal",
                  "cracker", "bun", "roll"},
    "SAUCE":     {"sauce", "oil", "vinegar", "soy", "ketchup", "mustard",
                  "mayo", "dressing", "broth", "stock", "wine", "water",
                  "juice", "honey", "syrup", "paste", "puree", "marinade"},
    "SEASONING": {"salt", "pepper", "sugar", "spice", "cumin", "paprika",
                  "cinnamon", "nutmeg", "turmeric", "cayenne", "chili powder",
                  "garlic powder", "onion powder", "seasoning", "herb"},
    "MIXTURE":   {"mixture", "batter", "filling", "stuffing", "coating",
                  "glaze", "topping", "garnish", "crumb"},
    "TOOL":      {"pan", "pot", "bowl", "plate", "tray", "skillet", "wok",
                  "oven", "grill", "blender", "mixer", "knife", "spoon",
                  "spatula", "whisk", "colander", "cutting board"},
}

# Words removed before object-category matching (measurements + modifiers).
FILTER_WORDS = {
    # Measurement units
    "cup", "cups", "tbsp", "tsp", "pinch", "oz", "ml", "gram", "grams",
    "lb", "lbs", "half", "quarter", "teaspoon", "tablespoon", "handful",
    "dash", "bunch",
    # Common cooking modifiers
    "chopped", "diced", "sliced", "minced", "grated", "crushed", "ground",
    "dried", "fresh", "raw", "cooked", "roasted", "fried", "boiled",
    "steamed", "melted", "shredded", "peeled", "toasted", "frozen", "extra",
    "virgin", "little", "small", "large", "thin", "thick",
}


# ============================================================================
# 2. spaCy-based action / object extraction
# ============================================================================

def extract_action_object(sentence, nlp):
    """
    Extract (action_lemma, object_phrase) from a single GT caption using
    dependency parsing.

    The action is the lemma of the ROOT verb. The object is the direct
    object of the ROOT verb together with any compound modifiers. Returns
    (None, None) if parsing fails or no direct object is present.
    """
    if not sentence or not sentence.strip():
        return None, None

    doc = nlp(sentence.strip())
    action = None
    obj = None

    for token in doc:
        if token.dep_ == "ROOT":
            action = token.lemma_.lower()
            for child in token.children:
                if child.dep_ in ("dobj", "obj", "attr", "oprd"):
                    compounds = [c.text.lower() for c in child.children
                                 if c.dep_ == "compound"]
                    obj = " ".join(compounds + [child.text.lower()])
                    break
            break

    return action, obj


def clean_obj(obj_text):
    """Remove measurement words and modifiers from an object phrase."""
    if obj_text is None:
        return None
    words = obj_text.lower().split()
    cleaned = [w for w in words if w not in FILTER_WORDS
               and not w.replace(".", "").isdigit()]
    if not cleaned:
        return None
    # If the phrase is too long, prefer the last in-vocabulary noun;
    # otherwise truncate to the last word.
    if len(cleaned) > 2:
        for w in reversed(cleaned):
            if any(w in nouns for nouns in OBJECT_CATEGORIES.values()):
                return w
        cleaned = cleaned[-1:]
    return " ".join(cleaned)


def match_action_group(action):
    """Map an action lemma to its action group, or None if unmatched."""
    if action is None:
        return None
    a = action.lower()
    for group, verbs in ACTION_GROUPS.items():
        if a in verbs:
            return group
    return None


def match_object_category(obj_text):
    """Map an object phrase to its object category, or None if unmatched."""
    if obj_text is None:
        return None
    o = obj_text.lower()
    for cat, nouns in OBJECT_CATEGORIES.items():
        for noun in nouns:
            if noun in o:
                return cat
    return None


# ============================================================================
# 3. Boundary classification (paper Eq. 3 / Sec. III-C)
# ============================================================================

def classify_boundary(act_group_from, act_group_to, obj_cat_from, obj_cat_to):
    """
    Classify an adjacent-event boundary into one of:
        strong      -- both action group and object category change
        mid-action  -- only action group changes
        mid-object  -- only object category changes
        weak        -- neither changes (or any side is unmatched)
    """
    action_changed = (
        act_group_from is not None
        and act_group_to is not None
        and act_group_from != act_group_to
    )
    object_changed = (
        obj_cat_from is not None
        and obj_cat_to is not None
        and obj_cat_from != obj_cat_to
    )

    if action_changed and object_changed:
        return "strong"
    elif action_changed:
        return "mid-action"
    elif object_changed:
        return "mid-object"
    else:
        return "weak"


# ============================================================================
# 4. Caption-style boundary text generation
# ============================================================================

# -ing form for irregular / consonant-doubling verbs
_ING_IRREGULAR = {
    "cut": "cutting", "put": "putting", "stir": "stirring",
    "chop": "chopping", "dice": "dicing", "slice": "slicing",
    "trim": "trimming", "rub": "rubbing", "drop": "dropping",
    "flip": "flipping", "wrap": "wrapping", "top": "topping",
    "set": "setting", "get": "getting", "run": "running",
    "dip": "dipping", "whip": "whipping", "stop": "stopping",
    "shop": "shopping", "snap": "snapping", "strip": "stripping",
    "sip": "sipping",
}


def to_ing(verb):
    """Convert a base-form verb to its -ing form."""
    if not verb:
        return "preparing"
    v = verb.lower().strip()
    if v in _ING_IRREGULAR:
        return _ING_IRREGULAR[v]
    if v.endswith("ie"):
        return v[:-2] + "ying"
    if v.endswith("e") and not v.endswith("ee"):
        return v[:-1] + "ing"
    return v + "ing"


def generate_boundary_text(level, act_from, act_to, obj_from, obj_to):
    """
    Produce a typed, caption-style boundary phrase. Returns None for weak
    boundaries (which are skipped from the output).
    """
    a_from = act_from or "processing"
    a_to = act_to or "processing"
    o_from = obj_from or "the ingredients"
    o_to = obj_to or "the ingredients"

    # Prepend "the" only if the object phrase is not already prefixed
    def _the(o):
        return o if o.startswith("the ") else f"the {o}"

    if level == "strong":
        return f"boundary: after {to_ing(a_from)} {_the(o_from)}, " \
               f"{a_to} {_the(o_to)}"
    elif level == "mid-action":
        return f"action shift: then {a_to} {_the(o_from)}"
    elif level == "mid-object":
        return f"object shift: continue to {a_from} {_the(o_to)}"
    else:
        return None


# ============================================================================
# 5. Main pipeline
# ============================================================================

def build_boundary_tokens(json_path, output_path):
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading annotations from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    stats = {
        "total_videos": len(data),
        "total_events": 0,
        "total_boundaries": 0,
        "action_extracted": 0,
        "object_extracted": 0,
        "action_matched": 0,
        "object_matched": 0,
        "level_counts": Counter(),
        "group_counts": Counter(),
        "cat_counts": Counter(),
    }

    result = {}

    video_ids = list(data.keys())
    for vid_idx, video_id in enumerate(video_ids):
        anno = data[video_id]
        timestamps = anno["timestamps"]
        sentences = anno["sentences"]
        n_events = len(sentences)
        stats["total_events"] += n_events

        # Step 1: parse every event caption
        parsed = []
        for sent in sentences:
            action, obj_raw = extract_action_object(sent, nlp)
            obj_clean = clean_obj(obj_raw)
            act_group = match_action_group(action)
            obj_cat = match_object_category(obj_clean) \
                or match_object_category(obj_raw)

            if action:
                stats["action_extracted"] += 1
            if obj_raw:
                stats["object_extracted"] += 1
            if act_group:
                stats["action_matched"] += 1
                stats["group_counts"][act_group] += 1
            if obj_cat:
                stats["object_matched"] += 1
                stats["cat_counts"][obj_cat] += 1

            parsed.append({
                "action": action,
                "object_raw": obj_raw,
                "object_clean": obj_clean,
                "action_group": act_group,
                "object_category": obj_cat,
            })

        # Step 2: compare adjacent events, classify, and generate tokens
        boundary_parts = []
        details = []

        for i in range(n_events - 1):
            curr, nxt = parsed[i], parsed[i + 1]

            t_end_prev = timestamps[i][1]
            t_start_next = timestamps[i + 1][0]

            level = classify_boundary(
                curr["action_group"], nxt["action_group"],
                curr["object_category"], nxt["object_category"],
            )
            stats["level_counts"][level] += 1
            stats["total_boundaries"] += 1

            if level == "weak":
                details.append({
                    "t_end_prev": t_end_prev,
                    "t_start_next": t_start_next,
                    "level": "weak",
                    "description": "(skipped)",
                })
                continue

            obj_from = curr["object_clean"] or curr["object_raw"]
            obj_to = nxt["object_clean"] or nxt["object_raw"]

            text = generate_boundary_text(
                level,
                curr["action"], nxt["action"],
                obj_from, obj_to,
            )
            if text is None:
                continue

            boundary_parts.append({
                "text": text,
                "t_end_prev": t_end_prev,
                "t_start_next": t_start_next,
            })
            details.append({
                "t_end_prev": t_end_prev,
                "t_start_next": t_start_next,
                "action_from": curr["action"],
                "action_to": nxt["action"],
                "object_from": obj_from,
                "object_to": obj_to,
                "action_group_from": curr["action_group"],
                "action_group_to": nxt["action_group"],
                "object_cat_from": curr["object_category"],
                "object_cat_to": nxt["object_category"],
                "level": level,
                "description": text,
            })

        result[video_id] = {
            "boundary_parts": boundary_parts,
            "n_boundaries": len(boundary_parts),
            "details": details,
        }

        if (vid_idx + 1) % 200 == 0:
            print(f"  Processed {vid_idx + 1}/{len(video_ids)} videos...")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ------- Statistics -------
    n_events = max(stats["total_events"], 1)
    n_bd = max(stats["total_boundaries"], 1)

    print(f"\n{'='*60}")
    print(f"YouCook2 STT generation statistics")
    print(f"{'='*60}")
    print(f"Videos:              {stats['total_videos']}")
    print(f"Total events:        {stats['total_events']}")
    print(f"Total boundaries:    {stats['total_boundaries']}")
    print(f"Action extracted:    {stats['action_extracted']}/{n_events} "
          f"({100 * stats['action_extracted'] / n_events:.1f}%)")
    print(f"Object extracted:    {stats['object_extracted']}/{n_events} "
          f"({100 * stats['object_extracted'] / n_events:.1f}%)")
    print(f"Action group match:  {stats['action_matched']}/{n_events} "
          f"({100 * stats['action_matched'] / n_events:.1f}%)")
    print(f"Object cat match:    {stats['object_matched']}/{n_events} "
          f"({100 * stats['object_matched'] / n_events:.1f}%)")

    print(f"\nAction group distribution:")
    for group, cnt in stats["group_counts"].most_common():
        print(f"  {group:12s}: {cnt:6d}")

    print(f"\nObject category distribution:")
    for cat, cnt in stats["cat_counts"].most_common():
        print(f"  {cat:12s}: {cnt:6d}")

    print(f"\nBoundary level distribution:")
    for level in ["strong", "mid-action", "mid-object", "weak"]:
        cnt = stats["level_counts"].get(level, 0)
        pct = 100 * cnt / n_bd
        tag = "-> generated" if level != "weak" else "-> skipped"
        print(f"  {level:12s}: {cnt:6d} ({pct:5.1f}%)  {tag}")

    non_zero = [v for v in result.values() if v["n_boundaries"] > 0]
    print(f"\nVideos with >=1 boundary: {len(non_zero)}/{stats['total_videos']}")
    if non_zero:
        avg = sum(v["n_boundaries"] for v in non_zero) / len(non_zero)
        print(f"Avg boundaries per such video: {avg:.1f}")

    print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to YouCook2 annotations JSON "
                             "(train.json or val.json).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write the boundary tokens JSON.")
    args = parser.parse_args()
    build_boundary_tokens(args.json_path, args.output_path)