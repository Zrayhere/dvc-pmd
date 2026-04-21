"""
build_boundary_tokens_vitt.py
-----------------------------
STT (Semantic Transition Token) generation for ViTT.

Differences from the YouCook2 version:
  1. STRUCTURE action group covers structural tags such as "intro", "outro",
     "ingredients", "finished", "closing" that are common in ViTT.
  2. PRESENT action group covers meta verbs such as "showing", "explaining",
     "demonstrating".
  3. APPLY / PLAY / ASSEMBLE action groups extend coverage to beauty,
     music, and craft / DIY domains.
  4. BEAUTY / MUSIC / CRAFT / VISUAL object categories are added accordingly.
  5. Structure <-> Content transitions are treated as strong boundaries.
  6. Tag-style annotations (often phrases rather than full sentences) are
     handled with a hybrid extraction strategy.

Usage:
    python build_boundary_tokens_vitt.py \\
        --json_path   ./data/vitt/train.json \\
        --output_path ./data/vitt/boundary_tokens.json
"""

import json
import argparse
from collections import Counter

import spacy


# ============================================================================
# 1. Action groups (extended for ViTT multi-domain coverage)
# ============================================================================

ACTION_GROUPS = {
    # Structural tags (specific to ViTT, occur very frequently)
    "STRUCTURE": {
        "intro", "introduction", "outro", "closing", "opening", "beginning",
        "ending", "conclusion", "closure", "start", "end", "finish",
        "finished", "final", "result", "started", "starting", "concluding",
        "finishing",
    },

    # Meta verbs: showing / explaining (common in tutorial videos)
    "PRESENT": {
        "show", "showing", "explain", "explaining", "demonstrate",
        "demonstrating", "display", "displaying", "showcase", "showcasing",
        "present", "presenting", "detail", "detailing", "discuss",
        "discussing", "list", "listing", "describe", "describing",
        "introduce", "introducing",
    },

    # Cooking actions (shared with YouCook2 dictionary)
    "CUT":    {"cut", "chop", "slice", "dice", "mince", "trim", "peel",
               "grate", "shred", "julienne",
               "cutting", "chopping", "slicing", "peeling", "grating",
               "shredding"},
    "ADD":    {"add", "put", "pour", "place", "drop", "insert", "squeeze",
               "adding", "putting", "pouring", "placing", "dropping",
               "inserting", "incorporating"},
    "MIX":    {"mix", "stir", "whisk", "blend", "combine", "fold", "beat",
               "mash", "knead",
               "mixing", "stirring", "whisking", "blending", "combining",
               "folding", "beating", "mashing", "kneading"},
    "COOK":   {"cook", "boil", "fry", "bake", "roast", "grill", "saute",
               "sauté", "simmer", "steam", "heat", "warm", "broil", "toast",
               "sear", "braise", "poach", "blanch", "melt",
               "cooking", "boiling", "frying", "baking", "roasting",
               "grilling", "sauteing", "simmering", "steaming", "heating",
               "warming", "toasting", "melting"},
    "SERVE":  {"serve", "plate", "transfer", "arrange", "garnish", "top",
               "serving", "plating", "garnishing", "topping", "tasting"},
    "REMOVE": {"remove", "take", "drain", "strain", "scoop", "scrape",
               "pull", "lift", "pick",
               "removing", "taking", "draining", "pulling", "scraping",
               "cleaning"},
    "SPREAD": {"spread", "brush", "drizzle", "coat", "sprinkle", "dust",
               "glaze", "rub", "spray",
               "spreading", "brushing", "sprinkling", "coating", "spraying",
               "lining"},
    "SHAPE":  {"shape", "roll", "flatten", "press", "form", "mold", "stretch",
               "wrap", "twist", "pinch",
               "shaping", "rolling", "forming", "wrapping", "folding",
               "assembling"},
    "COVER":  {"cover", "seal", "close", "lid", "covering", "closing"},
    "FLIP":   {"flip", "turn", "rotate", "invert", "flipping", "turning"},
    "FILL":   {"fill", "stuff", "filling", "stuffing"},

    # Beauty / personal care
    "APPLY":  {"apply", "applying", "set", "setting",
               "contour", "contouring", "highlight", "highlighting",
               "prime", "priming", "color", "coloring", "colouring",
               "curl", "curling", "straighten", "straightening",
               "style", "styling", "braid", "braiding"},

    # Music
    "PLAY":   {"play", "playing", "strum", "strumming", "sing", "singing",
               "tune", "tuning", "practice", "practicing"},

    # Craft / DIY
    "ASSEMBLE": {"attach", "attaching", "install", "installing", "connect",
                 "connecting", "secure", "securing", "screw", "screwing",
                 "sew", "sewing", "knit", "knitting", "crochet", "crocheting",
                 "glue", "gluing", "tape", "taping", "tie", "tying",
                 "paint", "painting", "draw", "drawing", "sketch", "sketching",
                 "build", "building", "construct", "constructing",
                 "measure", "measuring", "mark", "marking"},

    # Generic preparation / closing actions
    "PREPARE": {"prepare", "preparing", "prep", "prepping", "gather",
                "gathering", "setup", "setting", "get", "getting",
                "check", "checking", "test", "testing", "repeat",
                "repeating", "do", "doing", "use", "using", "make",
                "making", "create", "creating", "work", "working"},
}


# ============================================================================
# 2. Object categories (extended for ViTT multi-domain coverage)
# ============================================================================

OBJECT_CATEGORIES = {
    # Cooking domain (shared with YouCook2 dictionary)
    "VEGETABLE": {"onion", "onions", "garlic", "tomato", "tomatoes", "pepper",
                  "carrot", "potato", "potatoes", "celery", "lettuce",
                  "spinach", "broccoli", "cabbage", "cucumber", "zucchini",
                  "mushroom", "corn", "pea", "bean", "beans", "eggplant",
                  "squash", "ginger", "scallion", "avocado", "asparagus",
                  "kale", "cauliflower", "vegetables", "salad", "herb"},
    "MEAT":      {"chicken", "beef", "pork", "lamb", "turkey", "bacon",
                  "sausage", "ham", "steak", "meat", "meatball", "wing",
                  "thigh", "breast", "drumstick", "rib", "ribs", "shrimp",
                  "fish", "salmon", "seafood"},
    "DAIRY":     {"cheese", "butter", "cream", "milk", "yogurt", "egg",
                  "eggs", "mozzarella", "parmesan", "yolk", "white",
                  "ice cream"},
    "GRAIN":     {"rice", "pasta", "noodle", "noodles", "bread", "flour",
                  "dough", "tortilla", "batter", "crust", "cake", "cookie",
                  "pancake", "waffle", "roll"},
    "SAUCE":     {"sauce", "broth", "stock", "soup", "gravy", "marinade",
                  "dressing", "vinegar", "ketchup", "mustard", "salsa",
                  "pesto", "glaze", "syrup", "honey", "paste", "juice",
                  "water", "wine", "liquid", "drink", "oil"},
    "SEASONING": {"salt", "pepper", "sugar", "spice", "spices", "cumin",
                  "paprika", "cinnamon", "powder", "seasoning", "lemon",
                  "chocolate"},
    "MIXTURE":   {"mixture", "mix", "filling", "topping", "base", "coating",
                  "blend", "combination", "ingredient", "ingredients",
                  "food", "dish", "product"},
    "TOOL":      {"pan", "pot", "bowl", "plate", "oven", "knife", "board",
                  "skillet", "wok", "grill", "tray", "rack", "blender",
                  "spoon", "spatula", "foil", "paper", "towel", "container",
                  "bag", "box", "cup", "glass"},

    # Beauty / personal care
    "BEAUTY":    {"hair", "face", "eye", "eyes", "lip", "lips", "skin",
                  "brow", "brows", "eyebrow", "eyebrows", "eyelash",
                  "lash", "lashes", "nail", "nails", "cheek", "cheeks",
                  "forehead", "nose", "chin", "neck", "head",
                  "mascara", "foundation", "concealer", "primer",
                  "eyeshadow", "eyeliner", "lipstick", "blush",
                  "highlighter", "contour", "powder", "brush",
                  "palette", "sponge", "wax", "gel", "cream",
                  "color", "colour", "shade", "look", "style"},

    # Music
    "MUSIC":     {"chord", "chords", "note", "notes", "string", "strings",
                  "guitar", "piano", "drum", "drums", "bass", "melody",
                  "song", "tune", "rhythm", "beat", "scale", "key",
                  "finger", "fingers", "fret", "pick"},

    # Craft / DIY
    "CRAFT":     {"pattern", "fabric", "thread", "needle", "yarn", "stitch",
                  "stitches", "line", "lines", "edge", "edges", "side",
                  "sides", "piece", "pieces", "part", "parts", "layer",
                  "strip", "sheet", "card", "frame", "wire", "screw",
                  "screws", "bolt", "bracket", "pipe", "tube",
                  "wood", "board", "panel", "tile",
                  "paint", "pencil", "pen", "marker", "crayon",
                  "glue", "tape", "ribbon", "paper",
                  "tool", "tools", "supplies", "materials"},

    # Visual / presentation references
    "VISUAL":    {"video", "camera", "screen", "image", "photo",
                  "result", "results", "step", "steps", "process",
                  "tutorial", "tip", "tips", "instruction", "instructions"},
}

# Words removed before object-category matching.
FILTER_WORDS = {
    # Measurement units
    "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "cup", "cups", "oz", "ounce", "ounces", "lb", "pound", "pounds",
    "piece", "pieces", "handful", "pinch", "dash", "bunch", "half",
    # Modifiers
    "chopped", "diced", "minced", "sliced", "grated", "shredded",
    "roasted", "baked", "fried", "boiled", "steamed", "grilled",
    "fresh", "frozen", "dried", "canned", "raw", "cooked",
    "hot", "cold", "warm", "cool", "dry", "wet",
    "small", "medium", "large", "big", "little", "thin", "thick",
    "more", "some", "few", "remaining", "rest", "other", "additional",
    "enough", "extra", "whole", "new", "first", "second", "final",
    "all", "needed",
}


# ============================================================================
# 3. Extraction (adapted to ViTT's tag-style annotations)
# ============================================================================

def extract_action_object_vitt(text, nlp):
    """
    Extract (action, object, action_group, object_category) from a single
    ViTT annotation. Returns Nones for any field that cannot be resolved.

    ViTT annotations come in several forms:
      - bare structural tags ("intro", "outro", "ingredients")
      - gerund phrases ("adding seasoning", "mixing ingredients")
      - short sentences ("cutting the onion")

    The extraction strategy is:
      1. Direct match against the STRUCTURE keyword set.
      2. spaCy dependency parsing to find the ROOT verb and its dobj.
      3. Token-level fallbacks for action (first -ing word) and object
         (first content noun after the action position).
    """
    text_lower = text.strip().lower()
    words = text_lower.split()

    # Strategy 1: structural tag
    for w in words:
        if w in ACTION_GROUPS["STRUCTURE"]:
            return w, None, "STRUCTURE", None

    # Strategy 2: spaCy dependency parsing
    doc = nlp(text.strip())
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

    # Fallback action: first -ing word
    if action is None:
        for w in words:
            if w.endswith("ing") and len(w) > 4:
                action = w
                break

    # Fallback object: first content word after the action position
    if obj is None and action is not None:
        action_idx = -1
        for i, w in enumerate(words):
            if w == action or w == action.replace("ing", ""):
                action_idx = i
                break
        if 0 <= action_idx < len(words) - 1:
            remaining = words[action_idx + 1:]
            skip = {"the", "a", "an", "of", "to", "in", "on", "with",
                    "for", "into", "up", "out", "off"}
            remaining = [w for w in remaining if w not in skip]
            if remaining:
                obj = remaining[0]

    return action, obj, match_action_group(action), match_object_category(obj)


def clean_obj(obj_text):
    """Remove filter words from an object phrase."""
    if obj_text is None:
        return None
    words = obj_text.split()
    cleaned = [w for w in words if w not in FILTER_WORDS]
    if not cleaned:
        return None
    if len(cleaned) > 2:
        cleaned = cleaned[-1:]
    return " ".join(cleaned)


def match_action_group(action):
    """Map an action to its group. Handles -ing forms with simple lemmatization."""
    if action is None:
        return None
    a = action.lower()
    for group, verbs in ACTION_GROUPS.items():
        if a in verbs:
            return group
    # Strip -ing and try again
    if a.endswith("ing"):
        base = a[:-3]
        for group, verbs in ACTION_GROUPS.items():
            if base in verbs:
                return group
        # Double-consonant case (cutting -> cut)
        if len(base) > 2 and base[-1] == base[-2]:
            base2 = base[:-1]
            for group, verbs in ACTION_GROUPS.items():
                if base2 in verbs:
                    return group
        # -e ending case (placing -> place)
        base_e = base + "e"
        for group, verbs in ACTION_GROUPS.items():
            if base_e in verbs:
                return group
    return None


def match_object_category(obj_text):
    """Map an object phrase to its category, or None if unmatched."""
    if obj_text is None:
        return None
    o = obj_text.lower()
    for cat, nouns in OBJECT_CATEGORIES.items():
        for noun in nouns:
            if noun in o:
                return cat
    return None


# ============================================================================
# 4. Boundary classification (adapted for ViTT structure tags)
# ============================================================================

def classify_boundary_vitt(act_group_from, act_group_to,
                           obj_cat_from, obj_cat_to):
    """
    Classify an adjacent-event boundary, with two ViTT-specific rules:

      - Structure <-> Content: a video-phase change, always treated as strong.
      - Structure -> Structure: weak (consecutive structural tags carry no
        meaningful transition).

    The remaining Content -> Content cases follow the standard rule
    (paper Eq. 3 / Sec. III-C).
    """
    is_struct_from = (act_group_from == "STRUCTURE")
    is_struct_to = (act_group_to == "STRUCTURE")

    if is_struct_from != is_struct_to:
        return "strong"
    if is_struct_from and is_struct_to:
        return "weak"

    action_changed = (act_group_from is not None and act_group_to is not None
                      and act_group_from != act_group_to)
    object_changed = (obj_cat_from is not None and obj_cat_to is not None
                      and obj_cat_from != obj_cat_to)

    if action_changed and object_changed:
        return "strong"
    elif action_changed:
        return "mid-action"
    elif object_changed:
        return "mid-object"
    else:
        return "weak"


# ============================================================================
# 5. Caption-style boundary text generation (with structural cases)
# ============================================================================

def generate_boundary_text_vitt(level, act_from, act_to, obj_from, obj_to,
                                act_group_from, act_group_to):
    """
    Produce a typed, caption-style boundary phrase. Returns None for weak
    boundaries (which are skipped).

    Structural transitions get special templates:
        Structure -> Content : "boundary: after intro, start adding ingredients"
        Content -> Structure : "boundary: after cooking the onion, move to outro"
    """
    a_from = act_from or "processing"
    a_to = act_to or "processing"
    o_from = obj_from or "items"
    o_to = obj_to or "items"

    is_struct_from = (act_group_from == "STRUCTURE")
    is_struct_to = (act_group_to == "STRUCTURE")

    if level == "strong":
        if is_struct_from and not is_struct_to:
            return f"boundary: after {a_from}, start {a_to} the {o_to}"
        elif not is_struct_from and is_struct_to:
            return f"boundary: after {a_from} the {o_from}, move to {a_to}"
        else:
            return f"boundary: after {a_from} the {o_from}, {a_to} the {o_to}"
    elif level == "mid-action":
        return f"action shift: then {a_to} the {o_from}"
    elif level == "mid-object":
        return f"object shift: continue to {a_from} the {o_to}"
    else:
        return None


# ============================================================================
# 6. Main pipeline
# ============================================================================

def build_boundary_tokens(json_path, output_path):
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading dataset from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    video_ids = list(data.keys())
    print(f"  {len(video_ids)} videos")

    stats = {
        "total_videos": len(video_ids),
        "total_events": 0,
        "total_boundaries": 0,
        "action_extracted": 0,
        "object_extracted": 0,
        "action_matched": 0,
        "object_matched": 0,
        "structure_events": 0,
        "present_events": 0,
        "level_counts": Counter(),
        "group_counts": Counter(),
        "cat_counts": Counter(),
    }

    result = {}

    for vid_idx, vid in enumerate(video_ids):
        ann = data[vid]
        timestamps = ann["timestamps"]
        sentences = ann["sentences"]
        n_events = len(sentences)
        stats["total_events"] += n_events

        # Step 1: parse each event caption
        parsed = []
        for sent in sentences:
            action, obj_raw, act_group, obj_cat = \
                extract_action_object_vitt(sent, nlp)
            obj_clean = clean_obj(obj_raw)

            # Retry category match using the cleaned object
            if obj_cat is None and obj_clean:
                obj_cat = match_object_category(obj_clean)

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
            if act_group == "STRUCTURE":
                stats["structure_events"] += 1
            if act_group == "PRESENT":
                stats["present_events"] += 1

            parsed.append({
                "action": action,
                "object_raw": obj_raw,
                "object_clean": obj_clean,
                "action_group": act_group,
                "object_category": obj_cat,
            })

        # Step 2: classify adjacent-event boundaries and generate tokens
        boundary_parts = []
        details = []

        for i in range(n_events - 1):
            curr, nxt = parsed[i], parsed[i + 1]
            t_end_prev = timestamps[i][1]
            t_start_next = timestamps[i + 1][0]

            level = classify_boundary_vitt(
                curr["action_group"], nxt["action_group"],
                curr["object_category"], nxt["object_category"],
            )
            stats["level_counts"][level] += 1
            stats["total_boundaries"] += 1

            if level == "weak":
                continue

            obj_from = curr["object_clean"] or curr["object_raw"]
            obj_to = nxt["object_clean"] or nxt["object_raw"]

            text = generate_boundary_text_vitt(
                level,
                curr["action"], nxt["action"],
                obj_from, obj_to,
                curr["action_group"], nxt["action_group"],
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

        result[vid] = {
            "boundary_parts": boundary_parts,
            "n_boundaries": len(boundary_parts),
            "details": details,
        }

        if (vid_idx + 1) % 1000 == 0:
            print(f"  Processed {vid_idx + 1}/{len(video_ids)} videos...")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ------- Statistics -------
    n_events = max(stats["total_events"], 1)
    n_bd = max(stats["total_boundaries"], 1)

    print(f"\n{'='*60}")
    print(f"ViTT STT generation statistics")
    print(f"{'='*60}")
    print(f"Videos:              {stats['total_videos']}")
    print(f"Total events:        {stats['total_events']}")
    print(f"  - STRUCTURE:       {stats['structure_events']} "
          f"({100 * stats['structure_events'] / n_events:.1f}%)")
    print(f"  - PRESENT:         {stats['present_events']} "
          f"({100 * stats['present_events'] / n_events:.1f}%)")
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
                        help="Path to ViTT annotations JSON.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write the boundary tokens JSON.")
    args = parser.parse_args()
    build_boundary_tokens(args.json_path, args.output_path)