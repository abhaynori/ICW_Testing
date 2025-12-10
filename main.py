import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
import Levenshtein
import json
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ===== CONFIGURATION =====
# Read from environment variables if set (for CLI usage), otherwise use defaults
MEMORY_STRATEGY = os.getenv('ICW_MEMORY_STRATEGY', "4bit")  # Options: "small" (1.5B), "4bit" (7B), "8bit" (7B), "full" (7B)
TEMPERATURE = float(os.getenv('ICW_TEMPERATURE', '0.7'))
NUM_SAMPLES = int(os.getenv('ICW_NUM_SAMPLES', '50'))  # Increase to 200+ for more reliable statistics
OUTPUT_DIR = os.getenv('ICW_OUTPUT_DIR', "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-configure based on memory strategy
from memory_config import get_model_config

# Check if a custom model path is provided (e.g., GRPO-trained model)
CUSTOM_MODEL_PATH = os.getenv('ICW_MODEL_PATH', None)

if CUSTOM_MODEL_PATH:
    MODEL_NAME = CUSTOM_MODEL_PATH
    config = {"model_name": MODEL_NAME, "device_map": "auto"}
    print(f"Using custom trained model: {MODEL_NAME}")
else:
    config = get_model_config(MEMORY_STRATEGY)
    MODEL_NAME = config["model_name"]

print(f"\n{'='*80}")
print(f"ICW WATERMARKING EVALUATION (Paper-Accurate Implementation)")
print(f"{'='*80}")
print(f"Memory Strategy: {MEMORY_STRATEGY}")
print(f"Model: {MODEL_NAME}")
print(f"Temperature: {TEMPERATURE}")
print(f"Samples: {NUM_SAMPLES}")
print(f"{'='*80}")

# Warning for small models
if "1.5B" in MODEL_NAME or "1.5b" in MODEL_NAME:
    print(f"\n⚠️  WARNING: Using a 1.5B model")
    print(f"   The paper shows that even GPT-4o-mini struggles with some ICW methods.")
    print(f"   Expected results:")
    print(f"   - Unicode ICW: Will likely fail (model can't insert actual Unicode)")
    print(f"   - Initials ICW: Will likely fail (requires strong instruction-following)")
    print(f"   - Lexical ICW: Will likely fail (can't track 36-word list)")
    print(f"   - Acrostics ICW: May show weak signal (easiest task)")
    print(f"\n   For better results, use MEMORY_STRATEGY='4bit' for 7B model")
print(f"{'='*80}\n")

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model_kwargs = {
    "device_map": config.get("device_map", "auto"),
    "trust_remote_code": True,
    "low_cpu_mem_usage": True
}

if config.get("quantization"):
    model_kwargs["quantization_config"] = config["quantization"]
elif config.get("dtype"):
    model_kwargs["torch_dtype"] = config["dtype"]

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded successfully!\n")

# Generation configuration
generation_config = {
    "max_new_tokens": 500,
    "min_new_tokens": 100,
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
}

def generate_response(messages):
    """Generate response using chat template."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(input_ids, **generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

def log_generation(query, prompt, output, method, filename="generation_log.jsonl"):
    """Log generation to file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "query": query,
        "prompt": prompt,
        "output": output
    }
    with open(os.path.join(OUTPUT_DIR, filename), "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Load dataset
print("Loading dataset...")
eli5 = load_dataset("sentence-transformers/eli5", "pair", split="train")["question"][:NUM_SAMPLES]
print(f"✓ Loaded {len(eli5)} questions\n")

# ============================================================================
# WATERMARKING STRATEGIES (Paper-Accurate)
# ============================================================================

# --- 1. Unicode ICW ---
def unicode_embed_prompt(query):
    """Paper prompt: Insert zero-width space (U+200B) after every word."""
    system_msg = (
        "You are an adaptive language model. In the following conversation, "
        "please insert a zero-width space Unicode (U+200B) after every word in your response."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def unicode_detector(text):
    """
    Paper detector: D(y|ku,τu) := |y|ku / N
    where |y|ku = number of inserted Unicode characters
    and N = total number of words
    """
    unicode_count = text.count('\u200B')
    words = text.split()
    N = len(words)
    
    if N == 0:
        return 0
    
    return unicode_count / N

# --- 2. Initials ICW ---
# Canterbury Corpus letter frequencies (initial letter probabilities)
CANTERBURY_FREQUENCIES = {
    'a': 0.0856, 'b': 0.0139, 'c': 0.0254, 'd': 0.0387, 'e': 0.1270,
    'f': 0.0289, 'g': 0.0199, 'h': 0.0528, 'i': 0.0627, 'j': 0.0013,
    'k': 0.0042, 'l': 0.0339, 'm': 0.0254, 'n': 0.0710, 'o': 0.0797,
    'p': 0.0199, 'q': 0.0012, 'r': 0.0677, 's': 0.0607, 't': 0.1045,
    'u': 0.0243, 'v': 0.0105, 'w': 0.0169, 'x': 0.0017, 'y': 0.0143,
    'z': 0.0008
}

green_letters = set('abcdefghijklm')

def initials_embed_prompt(query):
    """Paper prompt: Maximize words starting with green letters."""
    green_list = ', '.join(sorted(green_letters))
    red_list = ', '.join(sorted(set('nopqrstuvwxyz')))
    
    system_msg = (
        f"### Green Letter List: {green_list}\n"
        f"### Red Letter List: {red_list}\n\n"
        "### Command:\n"
        "You are provided a Green Letter List and a Red Letter List. For each user query, generate a response that is:\n"
        "1. Clear & Coherent: Easy to follow and logically organized.\n"
        "2. Accurate & Concrete: Provides precise facts, examples, or steps.\n"
        "3. Contextually Relevant: Directly addresses the user's intent and context.\n"
        "4. 'Green Letter' Enriched (Most Important!): Try to increase the use of words beginning "
        "with letters from the Green List and reduce the use of words that start with letters from the Red List."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def initials_detector(text, green_letters):
    """
    Paper detector: D(y|kc,τc) := (|y|G - γ|y|) / sqrt(γ(1-γ)|y|)
    where γ = sum of P_A(a) for a in green_letters
    """
    words = [w for w in text.lower().split() if w and w[0].isalpha()]
    
    if len(words) == 0:
        return 0
    
    green_count = sum(1 for w in words if w[0] in green_letters)
    gamma = sum(CANTERBURY_FREQUENCIES.get(letter, 0) for letter in green_letters)
    
    n = len(words)
    numerator = green_count - gamma * n
    denominator = np.sqrt(gamma * (1 - gamma) * n)
    
    if denominator == 0:
        return 0
    
    z_score = numerator / denominator
    return z_score

# --- 3. Lexical ICW ---
# Comprehensive green word list (~3000 words as per paper: adjectives, verbs, adverbs)
green_words = {
    'StudY', 'abandon', 'able', 'abolish', 'abruptly', 'absolutely', 'absorb', 'abundantly', 'accept', 'access',
    'accompany', 'accomplish', 'accord', 'accordingly', 'account', 'accumulate', 'accurately', 'accuse', 'achieve', 'acknowledge',
    'acquire', 'act', 'activate', 'active', 'actively', 'actual', 'actually', 'acutely', 'adapt', 'add',
    'address', 'adequate', 'adequately', 'adjust', 'administer', 'admirably', 'admit', 'adopt', 'advance', 'advanced',
    'adversely', 'advertise', 'advise', 'advocate', 'affect', 'afford', 'afterwards', 'again', 'aggressively', 'agree',
    'aid', 'aim', 'alert', 'align', 'allege', 'allocate', 'allow', 'almost', 'alone', 'already',
    'also', 'alter', 'alternatively', 'altogether', 'always', 'amazing', 'amazingly', 'amend', 'amplify', 'analyze',
    'announce', 'annually', 'anticipate', 'anxiously', 'anyway', 'anywhere', 'apart', 'apologize', 'apparently', 'appeal',
    'appear', 'applaud', 'apply', 'appoint', 'appreciate', 'approach', 'appropriate', 'approve', 'approximately', 'arguably',
    'argue', 'arise', 'around', 'arrange', 'arrest', 'arrive', 'articulate', 'ascertain', 'aside', 'ask',
    'assemble', 'assert', 'assess', 'assign', 'assist', 'associate', 'assume', 'assure', 'attach', 'attack',
    'attain', 'attempt', 'attend', 'attract', 'attribute', 'audit', 'authorize', 'automate', 'automatically', 'available',
    'avoid', 'await', 'award', 'away', 'back', 'badly', 'balance', 'ban', 'barely', 'base',
    'basic', 'basically', 'bear', 'beat', 'beautiful', 'beautifully', 'become', 'beforehand', 'begin', 'behave',
    'behind', 'believe', 'belong', 'below', 'beneficial', 'benefit', 'besides', 'best', 'bet', 'better',
    'bid', 'big', 'bind', 'bitterly', 'blame', 'blend', 'block', 'bold', 'boldly', 'boost',
    'borrow', 'bother', 'bounce', 'bound', 'brain', 'brave', 'breach', 'break', 'breathe', 'breed',
    'brief', 'briefly', 'bright', 'brightly', 'brilliant', 'bring', 'broad', 'broadcast', 'broadly', 'browse',
    'budget', 'build', 'burn', 'burst', 'bury', 'buy', 'calculate', 'call', 'calm', 'cancel',
    'capable', 'capture', 'care', 'careful', 'carefully', 'carry', 'cast', 'casually', 'catch', 'cause',
    'cease', 'celebrate', 'center', 'central', 'certain', 'certainly', 'certify', 'challenge', 'change', 'channel',
    'characterize', 'charge', 'chart', 'chase', 'check', 'choose', 'cite', 'claim', 'clarify', 'classic',
    'classify', 'clean', 'clear', 'clearly', 'clever', 'climb', 'close', 'closely', 'coach', 'coincide',
    'collaborate', 'collapse', 'collect', 'collectively', 'combine', 'come', 'comfort', 'comfortably', 'command', 'comment',
    'commit', 'common', 'commonly', 'communicate', 'comparatively', 'compare', 'compel', 'compensate', 'compete', 'compile',
    'complain', 'complete', 'completely', 'complex', 'complicate', 'comply', 'compose', 'compound', 'comprehend', 'comprehensive',
    'comprehensively', 'compress', 'comprise', 'compromise', 'compute', 'conceal', 'concede', 'conceive', 'concentrate', 'conceptually',
    'concern', 'concisely', 'conclude', 'conclusively', 'concretely', 'concurrently', 'conduct', 'confer', 'confess', 'confident',
    'confidently', 'confine', 'confirm', 'conflict', 'conform', 'confront', 'confuse', 'connect', 'conquer', 'consciously',
    'consecutively', 'consent', 'consequently', 'conserve', 'consider', 'considerably', 'consist', 'consistent', 'consistently', 'consolidate',
    'constant', 'constantly', 'constitute', 'constrain', 'construct', 'constructively', 'consult', 'consume', 'contact', 'contain',
    'contemplate', 'contend', 'content', 'continually', 'continue', 'continuous', 'continuously', 'contract', 'contradict', 'contrary',
    'contrast', 'contribute', 'control', 'conversely', 'convert', 'convey', 'convict', 'convince', 'convincingly', 'cool',
    'cooperate', 'coordinate', 'cope', 'copy', 'correct', 'correctly', 'correlate', 'correspond', 'correspondingly', 'cost',
    'count', 'counter', 'couple', 'cover', 'crack', 'craft', 'crash', 'create', 'creative', 'credit',
    'critical', 'critically', 'criticize', 'cross', 'crowd', 'crucial', 'crucially', 'cry', 'cultivate', 'cure',
    'current', 'currently', 'cut', 'daily', 'damage', 'dare', 'date', 'deal', 'dearly', 'debate',
    'decay', 'deceive', 'decent', 'decide', 'decidedly', 'declare', 'decline', 'decorate', 'decrease', 'dedicate',
    'deduce', 'deem', 'deep', 'deeply', 'defeat', 'defend', 'defer', 'define', 'definite', 'definitely',
    'delegate', 'delete', 'deliberate', 'deliberately', 'delicate', 'delicately', 'deliver', 'demand', 'demonstrate', 'dense',
    'densely', 'deny', 'depart', 'depend', 'depict', 'deploy', 'deposit', 'depreciate', 'depress', 'derive',
    'describe', 'deserve', 'design', 'designate', 'desirable', 'desire', 'desperately', 'destroy', 'detach', 'detail',
    'detailed', 'detect', 'determine', 'determined', 'determinedly', 'develop', 'deviate', 'devise', 'devote', 'diagnose',
    'dictate', 'differ', 'different', 'differentiate', 'differently', 'difficult', 'dig', 'digest', 'diligently', 'diminish',
    'dimly', 'dip', 'direct', 'directly', 'disagree', 'disappear', 'disappoint', 'discard', 'discharge', 'disclose',
    'discourage', 'discover', 'discreetly', 'discriminate', 'discuss', 'dismiss', 'display', 'dispose', 'dispute', 'disrupt',
    'dissolve', 'distance', 'distinct', 'distinctly', 'distinguish', 'distort', 'distract', 'distribute', 'disturb', 'dive',
    'diverge', 'diverse', 'divert', 'divide', 'divine', 'document', 'dominant', 'dominate', 'donate', 'double',
    'doubt', 'draft', 'drag', 'drain', 'dramatic', 'dramatically', 'dramatize', 'drastically', 'draw', 'dream',
    'dress', 'drift', 'drill', 'drink', 'drive', 'drop', 'drown', 'dry', 'due', 'duly',
    'dump', 'duplicate', 'during', 'dwell', 'dynamic', 'eager', 'eagerly', 'early', 'earn', 'earnestly',
    'ease', 'easily', 'easy', 'eat', 'echo', 'economic', 'economically', 'edit', 'educate', 'effect',
    'effective', 'effectively', 'efficient', 'efficiently', 'elaborate', 'elaborately', 'elderly', 'elect', 'elegant', 'elegantly',
    'elementary', 'elevate', 'elicit', 'eligible', 'eliminate', 'eloquently', 'elsewhere', 'embark', 'embed', 'embody',
    'embrace', 'emerge', 'emerging', 'emotional', 'emotionally', 'emphasize', 'emphatically', 'employ', 'empower', 'empty',
    'enable', 'enact', 'encounter', 'encourage', 'end', 'endanger', 'endless', 'endlessly', 'endorse', 'endure',
    'energetically', 'enforce', 'engage', 'engineer', 'enhance', 'enjoy', 'enlarge', 'enlighten', 'enlist', 'enormously',
    'enough', 'enrich', 'enroll', 'ensure', 'entail', 'enter', 'entertain', 'enthusiastic', 'entire', 'entirely',
    'entitle', 'equal', 'equally', 'equate', 'equip', 'equivalent', 'erase', 'erect', 'erode', 'err',
    'escape', 'especially', 'essential', 'essentially', 'establish', 'estimate', 'eternal', 'eternally', 'ethical', 'ethically',
    'evaluate', 'even', 'eventually', 'ever', 'evermore', 'every', 'everybody', 'everyday', 'everyone', 'everything',
    'everywhere', 'evident', 'evidently', 'evoke', 'evolve', 'exact', 'exactly', 'examine', 'exceed', 'excel',
    'excellent', 'except', 'excessively', 'exchange', 'exciting', 'exclude', 'exclusive', 'exclusively', 'execute', 'exemplify',
    'exercise', 'exert', 'exhaust', 'exhibit', 'exist', 'existing', 'exit', 'exotic', 'expand', 'expect',
    'expected', 'expedite', 'expel', 'expensive', 'experience', 'experienced', 'experiment', 'expire', 'explain', 'explicit',
    'explicitly', 'explode', 'exploit', 'explore', 'export', 'expose', 'express', 'expressly', 'extend', 'extensive',
    'extensively', 'external', 'externally', 'extra', 'extract', 'extreme', 'extremely', 'face', 'facilitate', 'factor',
    'fail', 'fair', 'fairly', 'faithful', 'faithfully', 'fall', 'familiar', 'famous', 'famously', 'fancy',
    'far', 'fashion', 'fast', 'fasten', 'fatal', 'fatally', 'favor', 'favorable', 'favorably', 'favorite',
    'fear', 'feasible', 'feature', 'federal', 'feed', 'feel', 'fetch', 'few', 'fiercely', 'fight',
    'figure', 'file', 'fill', 'film', 'filter', 'final', 'finally', 'finance', 'find', 'fine',
    'finely', 'finish', 'fire', 'firm', 'firmly', 'first', 'firstly', 'fiscal', 'fit', 'fix',
    'fixed', 'flag', 'flash', 'flat', 'flatten', 'flee', 'flexible', 'float', 'flood', 'flourish',
    'flow', 'fluctuate', 'fluid', 'fly', 'focus', 'fold', 'follow', 'fond', 'forbid', 'force',
    'forecast', 'foresee', 'forge', 'forget', 'forgive', 'form', 'formal', 'formally', 'former', 'formerly',
    'formulate', 'forth', 'fortunate', 'fortunately', 'forward', 'foster', 'found', 'fracture', 'fragile', 'frame',
    'frankly', 'free', 'freely', 'freeze', 'frequent', 'frequently', 'fresh', 'freshly', 'friendly', 'frustrate',
    'fuel', 'fulfill', 'full', 'fully', 'fun', 'function', 'functional', 'functionally', 'fund', 'fundamental',
    'fundamentally', 'furnish', 'further', 'furthermore', 'future', 'gain', 'gamble', 'gather', 'gauge', 'gaze',
    'general', 'generally', 'generate', 'generous', 'generously', 'gentle', 'gently', 'genuine', 'genuinely', 'get',
    'giant', 'give', 'glad', 'gladly', 'glance', 'glimpse', 'global', 'globally', 'glorious', 'go',
    'golden', 'good', 'gorgeous', 'govern', 'grab', 'grace', 'grade', 'gradual', 'gradually', 'grand',
    'grant', 'grasp', 'grateful', 'gratefully', 'great', 'greatly', 'green', 'greet', 'grind', 'grip',
    'gross', 'grossly', 'ground', 'group', 'grow', 'growing', 'guarantee', 'guard', 'guess', 'guide',
    'halt', 'hand', 'handle', 'hang', 'happen', 'happily', 'happy', 'harass', 'hard', 'hardly',
    'harm', 'harmless', 'harsh', 'harshly', 'harvest', 'hasten', 'hastily', 'hate', 'haul', 'haunt',
    'have', 'head', 'heal', 'heap', 'hear', 'heat', 'heavily', 'heavy', 'help', 'helpful',
    'hence', 'here', 'hereby', 'herein', 'heroically', 'hesitate', 'hidden', 'hide', 'high', 'higher',
    'highest', 'highly', 'highlight', 'hinder', 'hint', 'hire', 'historically', 'hit', 'hold', 'home',
    'honest', 'honestly', 'honor', 'hope', 'hopeful', 'hopefully', 'horizontally', 'host', 'hot', 'hourly',
    'house', 'hover', 'however', 'hug', 'huge', 'hugely', 'human', 'humanly', 'humble', 'humbly',
    'humorous', 'hunt', 'hurry', 'hurt', 'hypothesize', 'ideal', 'ideally', 'identical', 'identically', 'identify',
    'ideologically', 'ignore', 'illegal', 'illustrate', 'imagine', 'imitate', 'immediate', 'immediately', 'immense', 'immensely',
    'immerse', 'immune', 'impact', 'impair', 'imperial', 'impede', 'implement', 'implicate', 'implicit', 'implicitly',
    'imply', 'import', 'important', 'importantly', 'impose', 'impossible', 'impossibly', 'impress', 'impressive', 'impressively',
    'improve', 'improved', 'improvise', 'inaccurately', 'inadvertently', 'inappropriate', 'inappropriately', 'inaugurate', 'incidentally', 'incline',
    'inclined', 'include', 'incorporate', 'increase', 'increasingly', 'incredible', 'incredibly', 'incur', 'indeed', 'indefinitely',
    'independent', 'independently', 'indicate', 'indirect', 'indirectly', 'individual', 'individually', 'indoor', 'induce', 'indulge',
    'industrial', 'industrially', 'infect', 'infer', 'inferior', 'infinite', 'infinitely', 'inflate', 'inflict', 'influence',
    'influential', 'inform', 'informal', 'informally', 'inhabit', 'inherent', 'inherently', 'inherit', 'inhibit', 'initial',
    'initially', 'initiate', 'inject', 'injure', 'inland', 'inner', 'innocently', 'innovate', 'innovative', 'input',
    'inquire', 'inscribe', 'insert', 'inside', 'insist', 'inspect', 'inspire', 'install', 'instant', 'instantly',
    'instead', 'instinctively', 'institute', 'instruct', 'insulate', 'insure', 'integrate', 'integrally', 'intellectual', 'intellectually',
    'intelligent', 'intelligently', 'intend', 'intense', 'intensely', 'intensify', 'intensive', 'intensively', 'intentionally', 'interact',
    'interactive', 'intercept', 'interest', 'interested', 'interesting', 'interestingly', 'interfere', 'internal', 'internally', 'international',
    'internationally', 'interpret', 'interrupt', 'intervene', 'interview', 'intimate', 'intimately', 'introduce', 'intrinsically', 'intuitively',
    'invade', 'invalid', 'invariably', 'invent', 'invest', 'investigate', 'invisible', 'invite', 'invoke', 'involve',
    'inwardly', 'ironically', 'irregularly', 'irritate', 'isolate', 'issue', 'itemize', 'jeopardize', 'join', 'joint',
    'jointly', 'joyful', 'joyfully', 'judge', 'jump', 'just', 'justly', 'justify', 'keen', 'keenly',
    'keep', 'key', 'kick', 'kill', 'kind', 'kindly', 'know', 'knowingly', 'label', 'lack',
    'land', 'large', 'largely', 'last', 'lastly', 'lasting', 'late', 'lately', 'later', 'latest',
    'lateral', 'latter', 'launch', 'lay', 'lead', 'leading', 'lean', 'leap', 'learn', 'lease',
    'least', 'leave', 'lecture', 'left', 'legal', 'legally', 'legendary', 'legislate', 'legitimate', 'legitimately',
    'lend', 'lengthen', 'lengthwise', 'lengthy', 'lessen', 'lesser', 'less', 'let', 'level', 'leverage',
    'levy', 'liberal', 'liberate', 'license', 'lick', 'lie', 'lift', 'light', 'lighten', 'lightly',
    'like', 'likely', 'likewise', 'limit', 'limited', 'line', 'linear', 'link', 'liquid', 'list',
    'listen', 'literally', 'literary', 'little', 'live', 'lively', 'living', 'load', 'loan', 'lobby',
    'local', 'locally', 'locate', 'lock', 'log', 'logical', 'logically', 'lone', 'long', 'longer',
    'longest', 'look', 'loop', 'loose', 'loosely', 'loosen', 'lose', 'loud', 'loudly', 'love',
    'lovely', 'lovingly', 'low', 'lower', 'lowest', 'lowly', 'loyal', 'lucky', 'luxurious', 'mad',
    'magic', 'magnetic', 'magnificent', 'main', 'mainly', 'maintain', 'major', 'make', 'male', 'manage',
    'mandate', 'manipulate', 'manual', 'manually', 'manufacture', 'map', 'march', 'marginal', 'marginally', 'marine',
    'mark', 'marked', 'markedly', 'market', 'marry', 'mask', 'mass', 'massive', 'massively', 'master',
    'match', 'matching', 'materialize', 'material', 'materially', 'mathematical', 'mathematically', 'matter', 'mature', 'maximize',
    'maximum', 'maybe', 'mean', 'meaningful', 'meanwhile', 'measure', 'mechanical', 'mechanically', 'mediate', 'medical',
    'medically', 'meditate', 'medium', 'meet', 'melt', 'memorize', 'mental', 'mentally', 'mention', 'mere',
    'merely', 'merge', 'merit', 'middle', 'mild', 'mildly', 'military', 'mind', 'minimal', 'minimally',
    'minimize', 'minimum', 'minor', 'minute', 'minutely', 'mirror', 'miss', 'mistake', 'misunderstand', 'mix',
    'mixed', 'mobile', 'mobilize', 'model', 'moderate', 'moderately', 'modern', 'modernize', 'modest', 'modestly',
    'modify', 'monetary', 'monitor', 'monopolize', 'monthly', 'moral', 'morally', 'more', 'moreover', 'most',
    'mostly', 'motivate', 'motivated', 'mount', 'mourn', 'move', 'much', 'multiple', 'multiply', 'municipal',
    'murder', 'murmur', 'muster', 'mutual', 'mutually', 'mysterious', 'mysteriously', 'nail', 'name', 'namely',
    'narrow', 'narrowly', 'narrate', 'nasty', 'national', 'nationally', 'native', 'natural', 'naturally', 'naval',
    'navigate', 'near', 'nearby', 'nearly', 'neat', 'necessarily', 'necessary', 'necessitate', 'need', 'needlessly',
    'negate', 'negative', 'negatively', 'neglect', 'negotiate', 'neighboring', 'nervous', 'nervously', 'nest', 'net',
    'neutral', 'neutralize', 'never', 'nevertheless', 'new', 'newly', 'next', 'nice', 'nicely', 'nightly',
    'noble', 'nobly', 'nominal', 'nominally', 'nominate', 'nonetheless', 'normal', 'normalize', 'normally', 'notable',
    'notably', 'note', 'notice', 'noticeable', 'noticeably', 'notify', 'novel', 'nourish', 'nowadays', 'nowhere',
    'nuclear', 'number', 'numerous', 'nurse', 'nurture', 'obey', 'object', 'objective', 'objectively', 'oblige',
    'obliged', 'obscure', 'observe', 'obsess', 'obstruct', 'obtain', 'obvious', 'obviously', 'occasional', 'occasionally',
    'occupy', 'occur', 'odd', 'oddly', 'off', 'offend', 'offer', 'official', 'officially', 'offset',
    'often', 'okay', 'old', 'omit', 'once', 'ongoing', 'only', 'onward', 'onwards', 'open',
    'openly', 'operate', 'operational', 'operationally', 'oppose', 'opposite', 'oppress', 'opt', 'optimal', 'optimally',
    'optimize', 'optional', 'optionally', 'oral', 'orange', 'orbit', 'orchestrate', 'order', 'ordinarily', 'ordinary',
    'organic', 'organizational', 'organizationally', 'organize', 'organized', 'orient', 'original', 'originally', 'originate', 'outdoor',
    'outer', 'outline', 'output', 'outright', 'outside', 'outstanding', 'outward', 'outwardly', 'outweigh', 'overall',
    'overcome', 'overflow', 'overhear', 'overlap', 'overlook', 'override', 'overseas', 'oversee', 'overtake', 'overthrow',
    'overwhelm', 'overwhelming', 'overwhelmingly', 'owe', 'own', 'pace', 'pack', 'paint', 'painfully', 'pair',
    'pale', 'pan', 'panic', 'parallel', 'parenthetically', 'park', 'part', 'partial', 'partially', 'participate',
    'particular', 'particularly', 'partly', 'partner', 'pass', 'passive', 'passively', 'past', 'paste', 'patent',
    'patient', 'patiently', 'patrol', 'patronize', 'pattern', 'pause', 'pave', 'pay', 'peaceful', 'peacefully',
    'peak', 'peculiar', 'peculiarly', 'pedal', 'peer', 'penalize', 'pending', 'penetrate', 'perceive', 'perfect',
    'perfectly', 'perform', 'periodically', 'permanent', 'permanently', 'permit', 'perpetually', 'persist', 'persistent', 'persistently',
    'personalize', 'personal', 'personally', 'persuade', 'pertain', 'phase', 'philosophically', 'physical', 'physically', 'pick',
    'picture', 'pile', 'pilot', 'pin', 'pioneer', 'pitch', 'pity', 'place', 'plain', 'plainly',
    'plan', 'plant', 'play', 'plead', 'pleasant', 'pleasantly', 'please', 'pleased', 'pledge', 'plenty',
    'plot', 'plow', 'plunge', 'plus', 'point', 'pointedly', 'poke', 'polar', 'polarize', 'police',
    'polish', 'polite', 'politely', 'political', 'politically', 'pollute', 'pool', 'poor', 'poorly', 'pop',
    'popular', 'popularly', 'populate', 'portable', 'portray', 'pose', 'position', 'positive', 'positively', 'possess',
    'possible', 'possibly', 'post', 'postpone', 'potential', 'potentially', 'pour', 'powder', 'power', 'powerful',
    'powerfully', 'practical', 'practically', 'practice', 'praise', 'pray', 'preach', 'precede', 'precious', 'precise',
    'precisely', 'precipitate', 'predict', 'predictably', 'predominantly', 'prefer', 'preferably', 'preferred', 'preliminary', 'preliminarily',
    'premier', 'premium', 'preoccupy', 'prepare', 'prescribe', 'present', 'presently', 'preserve', 'preside', 'presidential',
    'press', 'presume', 'presumably', 'pretend', 'pretty', 'prevail', 'prevent', 'preview', 'previous', 'previously',
    'price', 'pride', 'primary', 'primarily', 'prime', 'primitive', 'principal', 'principally', 'print', 'prior',
    'prioritize', 'private', 'privately', 'prize', 'probable', 'probably', 'probe', 'procedurally', 'proceed', 'process',
    'proclaim', 'procure', 'produce', 'productive', 'productively', 'professional', 'professionally', 'profitably', 'profit', 'profound',
    'profoundly', 'program', 'progress', 'progressive', 'progressively', 'prohibit', 'project', 'prolong', 'prominent', 'prominently',
    'promise', 'promising', 'promote', 'prompt', 'promptly', 'prone', 'pronounce', 'proof', 'proper', 'properly',
    'propel', 'proportional', 'proportionally', 'propose', 'proposed', 'prosecute', 'prospect', 'prospectively', 'prosper', 'protect',
    'protective', 'protest', 'proud', 'proudly', 'prove', 'proven', 'provide', 'provisional', 'provisionally', 'provoke',
    'provincial', 'psychological', 'psychologically', 'public', 'publicly', 'publicize', 'publish', 'pull', 'pulse', 'pump',
    'punch', 'punish', 'purchase', 'pure', 'purely', 'purge', 'purple', 'purpose', 'purposefully', 'purposely',
    'pursue', 'push', 'put', 'qualified', 'qualify', 'quantify', 'quarrel', 'quarterly', 'question', 'queue',
    'quick', 'quickly', 'quiet', 'quietly', 'quit', 'quite', 'quote', 'race', 'radial', 'radically',
    'radiant', 'radiate', 'raid', 'rain', 'raise', 'rally', 'ramble', 'random', 'randomly', 'range',
    'rank', 'ransom', 'rapid', 'rapidly', 'rare', 'rarely', 'rate', 'rather', 'ratify', 'rational',
    'rationally', 'rationalize', 'raw', 'reach', 'react', 'read', 'readily', 'ready', 'real', 'realistic',
    'realistically', 'realize', 'really', 'reap', 'rear', 'reason', 'reasonable', 'reasonably', 'reassure', 'rebel',
    'rebuild', 'recall', 'recede', 'receive', 'recent', 'recently', 'reciprocally', 'reckon', 'recklessly', 'reclaim',
    'recognizably', 'recognize', 'recommend', 'reconcile', 'reconstruct', 'record', 'recount', 'recover', 'recreate', 'recruit',
    'rectify', 'recur', 'recursively', 'recycle', 'red', 'redeem', 'redesign', 'redirect', 'rediscover', 'redistribute',
    'redo', 'reduce', 'reduced', 'redundant', 'refer', 'refine', 'reflect', 'reform', 'refrain', 'refresh',
    'refund', 'refuse', 'refute', 'regain', 'regard', 'regenerate', 'register', 'regret', 'regular', 'regularly',
    'regulate', 'regulatory', 'rehabilitate', 'rehearse', 'reign', 'reinforce', 'reinstate', 'reinvent', 'reiterate', 'reject',
    'rejoice', 'rejoin', 'relate', 'related', 'relax', 'relay', 'release', 'relegate', 'relent', 'relevant',
    'reliably', 'reliable', 'relieve', 'religiously', 'religious', 'relinquish', 'relish', 'relocate', 'reluctantly', 'rely',
    'remain', 'remark', 'remarkable', 'remarkably', 'remedy', 'remember', 'remind', 'remit', 'remodel', 'remote',
    'remotely', 'remove', 'render', 'renew', 'renowned', 'renounce', 'renovate', 'rent', 'reopen', 'reorganize',
    'repair', 'repay', 'repeal', 'repeat', 'repeatedly', 'repel', 'replace', 'replay', 'replenish', 'replicate',
    'reply', 'report', 'reportedly', 'represent', 'representative', 'representatively', 'repress', 'reproduce', 'repudiate', 'repulse',
    'request', 'require', 'rescue', 'research', 'resemble', 'resent', 'reserve', 'reset', 'reside', 'resign',
    'resist', 'resolve', 'resort', 'resound', 'respect', 'respective', 'respectively', 'respond', 'responsibly', 'responsible',
    'rest', 'restart', 'restate', 'restore', 'restrain', 'restrict', 'restructure', 'result', 'resulting', 'resume',
    'resurrect', 'retail', 'retain', 'retaliate', 'retard', 'rethink', 'retire', 'retract', 'retreat', 'retrieve',
    'retrospectively', 'return', 'reunite', 'reveal', 'revel', 'revenge', 'reverently', 'reverse', 'reversely', 'revert',
    'review', 'revise', 'revitalize', 'revive', 'revoke', 'revolt', 'revolutionize', 'revolve', 'reward', 'rewrite',
    'rich', 'richly', 'rid', 'ride', 'ridicule', 'rig', 'rigid', 'rigidly', 'rigorously', 'ring',
    'rinse', 'riot', 'rip', 'ripen', 'rise', 'rising', 'risk', 'risky', 'rival', 'roar',
    'rob', 'robust', 'rock', 'roll', 'romantic', 'romanticize', 'root', 'rot', 'rotate', 'rough',
    'roughly', 'round', 'rouse', 'route', 'routine', 'routinely', 'royal', 'rub', 'rudely', 'ruin',
    'rule', 'rumble', 'run', 'rupture', 'rural', 'rush', 'rust', 'ruthlessly', 'sabotage', 'sacred',
    'sacrifice', 'sad', 'sadly', 'sadden', 'safe', 'safely', 'safeguard', 'sag', 'sail', 'salute',
    'salvage', 'same', 'sample', 'sanction', 'satisfied', 'satisfy', 'satisfactorily', 'saturate', 'save', 'savor',
    'say', 'scale', 'scan', 'scare', 'scared', 'scarcely', 'scatter', 'scenic', 'scent', 'schedule',
    'scheme', 'school', 'scientific', 'scientifically', 'scold', 'scoop', 'scope', 'score', 'scorn', 'scout',
    'scramble', 'scrape', 'scratch', 'scream', 'screen', 'screw', 'scrutinize', 'sculpt', 'seal', 'search',
    'season', 'seasonally', 'seat', 'secede', 'seclude', 'second', 'secondarily', 'secondly', 'secondary', 'secret',
    'secretly', 'secure', 'securely', 'seduce', 'see', 'seed', 'seek', 'seem', 'seemingly', 'seep',
    'segment', 'segregate', 'seize', 'seldom', 'select', 'selectively', 'selective', 'sell', 'send', 'senior',
    'sense', 'sensitive', 'sentence', 'separate', 'separately', 'sequence', 'sequentially', 'serious', 'seriously', 'serve',
    'service', 'set', 'settle', 'sever', 'severe', 'severely', 'sew', 'sexual', 'sexually', 'shade',
    'shadow', 'shake', 'shallow', 'shape', 'share', 'sharp', 'sharply', 'sharpen', 'shatter', 'shave',
    'shed', 'sheerly', 'sheet', 'shell', 'shelter', 'shield', 'shift', 'shine', 'ship', 'shiver',
    'shock', 'shoot', 'shop', 'shore', 'short', 'shortly', 'shorten', 'shout', 'shove', 'show',
    'shower', 'shrink', 'shrug', 'shut', 'shuttle', 'shy', 'sick', 'side', 'sideways', 'sigh',
    'sight', 'sign', 'signal', 'significant', 'significantly', 'signify', 'silence', 'silent', 'silently', 'silly',
    'similar', 'similarly', 'simple', 'simplify', 'simply', 'simulate', 'simultaneously', 'sin', 'since', 'sincerely',
    'sing', 'single', 'singular', 'singularly', 'sink', 'sip', 'sit', 'situate', 'size', 'skate',
    'sketch', 'ski', 'skid', 'skilled', 'skim', 'skin', 'skip', 'skyrocket', 'slam', 'slant',
    'slap', 'slash', 'slave', 'sleep', 'slice', 'slide', 'slight', 'slightly', 'slim', 'slip',
    'slow', 'slowly', 'small', 'smart', 'smash', 'smell', 'smile', 'smoke', 'smooth', 'smoothly',
    'smother', 'smuggle', 'snap', 'snatch', 'sneak', 'sniff', 'so', 'soar', 'sober', 'social',
    'socially', 'socialize', 'soft', 'soften', 'softly', 'soil', 'solar', 'sole', 'solely', 'solicit',
    'solid', 'solidify', 'solidly', 'solve', 'somehow', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon',
    'sooner', 'soothe', 'sophisticated', 'sorry', 'sort', 'sound', 'soundly', 'source', 'south', 'southern',
    'sow', 'space', 'span', 'spare', 'spark', 'sparkle', 'spatial', 'spatially', 'speak', 'special',
    'specialize', 'specially', 'specific', 'specifically', 'specify', 'spectacular', 'spectacularly', 'speculate', 'speed', 'spell',
    'spend', 'spice', 'spill', 'spin', 'spiral', 'spirit', 'spiritual', 'spiritually', 'split', 'spoil',
    'sponsor', 'spontaneous', 'spontaneously', 'spoon', 'spot', 'spray', 'spread', 'spring', 'sprinkle', 'sprint',
    'sprout', 'spur', 'spy', 'square', 'squarely', 'squeeze', 'stabilize', 'stable', 'stably', 'stack',
    'staff', 'stage', 'stagger', 'stain', 'stake', 'stalk', 'stall', 'stamp', 'stand', 'standard',
    'standardize', 'star', 'stare', 'stark', 'starkly', 'start', 'startle', 'starve', 'state', 'static',
    'station', 'statistical', 'statistically', 'stay', 'steadily', 'steady', 'steal', 'steam', 'steep', 'steeply',
    'steer', 'stem', 'step', 'stereotype', 'sterilize', 'stern', 'stick', 'sticky', 'stiff', 'stiffen',
    'stiffly', 'stifle', 'still', 'stimulate', 'sting', 'stink', 'stipulate', 'stir', 'stitch', 'stock',
    'stomach', 'stomp', 'stone', 'stoop', 'stop', 'store', 'storm', 'straight', 'straighten', 'strain',
    'strand', 'strange', 'strangely', 'strap', 'strategic', 'strategically', 'strategize', 'stray', 'stream', 'streamline',
    'strengthen', 'stress', 'stretch', 'strict', 'strictly', 'stride', 'strike', 'striking', 'string', 'strip',
    'strive', 'stroke', 'stroll', 'strong', 'strongly', 'structural', 'structurally', 'structure', 'struggle', 'strut',
    'stubbornly', 'stuck', 'stuff', 'stumble', 'stun', 'stunt', 'stupefy', 'stupid', 'style', 'subdue',
    'subject', 'subjective', 'submerge', 'submit', 'subordinate', 'subscribe', 'subsequent', 'subsequently', 'subside', 'subsidize',
    'subsist', 'substantial', 'substantially', 'substantiate', 'substitute', 'subtle', 'subtly', 'subtract', 'succeed', 'successfully',
    'successive', 'successively', 'succumb', 'such', 'suck', 'sudden', 'suddenly', 'sue', 'suffer', 'suffice',
    'sufficient', 'sufficiently', 'suffocate', 'suggest', 'suit', 'suitable', 'suitably', 'sum', 'summarily', 'summarize',
    'summon', 'sunder', 'super', 'superb', 'superficially', 'superimpose', 'superior', 'supersede', 'supervise', 'supplant',
    'supplement', 'supply', 'support', 'suppose', 'supposed', 'supposedly', 'suppress', 'supreme', 'surcharge', 'sure',
    'surely', 'surge', 'surgically', 'surmount', 'surpass', 'surplus', 'surprise', 'surprised', 'surprising', 'surprisingly',
    'surrender', 'surround', 'surrounding', 'survey', 'survive', 'suspect', 'suspected', 'suspend', 'suspiciously', 'sustain',
    'sustainable', 'sustainably', 'swallow', 'swamp', 'swap', 'swarm', 'sway', 'swear', 'sweat', 'sweep',
    'sweet', 'sweeten', 'sweetly', 'swell', 'swerve', 'swiftly', 'swim', 'swing', 'swirl', 'switch',
    'symbolic', 'symbolically', 'symbolize', 'symmetrically', 'sympathetic', 'sympathetically', 'sympathize', 'synchronize', 'synthesize', 'systematic',
    'systematically', 'systematize', 'table', 'tackle', 'tactically', 'tag', 'tail', 'tailor', 'taint', 'take',
    'talk', 'tall', 'tally', 'tame', 'tamper', 'tan', 'tangibly', 'tangle', 'tank', 'tap',
    'tape', 'target', 'tarry', 'task', 'taste', 'taunt', 'tax', 'teach', 'team', 'tear',
    'tease', 'technical', 'technically', 'telegraph', 'telephone', 'telescope', 'tell', 'temper', 'temporally', 'temporarily',
    'temporary', 'tempt', 'ten', 'tend', 'tender', 'tenderly', 'tense', 'term', 'terminate', 'terrible',
    'terribly', 'terrific', 'terrify', 'terrorize', 'test', 'testify', 'thank', 'thankfully', 'thaw', 'then',
    'theoretically', 'theorize', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereof', 'thereupon', 'thick',
    'thicken', 'thickly', 'thin', 'think', 'thinly', 'third', 'thirdly', 'thirst', 'thorough', 'thoroughly',
    'though', 'thoughtful', 'thoughtfully', 'thrash', 'thread', 'threaten', 'threateningly', 'thrill', 'thrive', 'throb',
    'throttle', 'through', 'throughout', 'throw', 'thrust', 'thump', 'thunder', 'thus', 'thwart', 'tick',
    'ticket', 'tickle', 'tide', 'tidy', 'tie', 'tight', 'tighten', 'tightly', 'tile', 'till',
    'tilt', 'time', 'timely', 'tin', 'tingle', 'tinker', 'tint', 'tiny', 'tip', 'tire',
    'tired', 'title', 'toast', 'today', 'toddle', 'together', 'toggle', 'toil', 'tolerable', 'tolerate',
    'tomorrow', 'tone', 'tonight', 'too', 'tool', 'top', 'topple', 'torment', 'torture', 'toss',
    'total', 'totally', 'totter', 'touch', 'touchingly', 'tough', 'tour', 'tout', 'tow', 'toward',
    'towards', 'tower', 'toxic', 'toy', 'trace', 'track', 'trade', 'traditional', 'traditionally', 'traffic',
    'tragic', 'tragically', 'trail', 'train', 'trample', 'transact', 'transcend', 'transcribe', 'transfer', 'transform',
    'transgress', 'transit', 'translate', 'transmit', 'transplant', 'transport', 'transpose', 'trap', 'trash', 'traumatize',
    'travel', 'traverse', 'tread', 'treasure', 'treat', 'trek', 'tremble', 'tremendous', 'tremendously', 'trespass',
    'trial', 'triangulate', 'trick', 'trickle', 'trigger', 'trim', 'trip', 'triple', 'triumph', 'trivialize',
    'troll', 'troop', 'tropical', 'trot', 'trouble', 'troubled', 'trudge', 'true', 'truly', 'trumpet',
    'truncate', 'trust', 'truthfully', 'try', 'tuck', 'tug', 'tumble', 'tune', 'tunnel', 'turn',
    'tussle', 'tutor', 'tweak', 'tweet', 'twice', 'twinkle', 'twirl', 'twist', 'twitch', 'type',
    'typical', 'typically', 'typify', 'tyrannize', 'ugly', 'ultimate', 'ultimately', 'unable', 'unambiguously', 'unanimously',
    'unashamedly', 'unavoidably', 'unbelievably', 'uncertain', 'uncertainly', 'uncomfortable', 'uncomfortably', 'unconditionally', 'unconscious', 'unconsciously',
    'unconventionally', 'uncover', 'undeniably', 'under', 'undergo', 'underlie', 'underline', 'underlying', 'undermine', 'understand',
    'understandably', 'undertake', 'undo', 'undoubtedly', 'undress', 'unduly', 'unearth', 'unevenly', 'unexpectedly', 'unfairly',
    'unfold', 'unfortunately', 'unhappily', 'unhappy', 'uniform', 'uniformly', 'unify', 'unintentionally', 'unique', 'uniquely',
    'unite', 'united', 'universal', 'universally', 'unjustly', 'unknowingly', 'unknown', 'unlawfully', 'unleash', 'unlikely',
    'unload', 'unlock', 'unmask', 'unnaturally', 'unnecessarily', 'unnecessary', 'unofficially', 'unpack', 'unpleasant', 'unprecedented',
    'unprecedentedly', 'unravel', 'unreasonable', 'unreasonably', 'unreliably', 'unsteadily', 'unsuccessfully', 'unusual', 'unusually', 'unveil',
    'unwillingly', 'unwrap', 'up', 'upcoming', 'update', 'updated', 'upgrade', 'uphill', 'uphold', 'uplift',
    'upon', 'upper', 'upright', 'upset', 'upstairs', 'upward', 'upwards', 'urban', 'urge', 'urgent',
    'urgently', 'use', 'used', 'useful', 'usefully', 'useless', 'uselessly', 'usher', 'usual', 'usually',
    'utilize', 'utter', 'utterly', 'vacant', 'vacate', 'vaccinate', 'vacuum', 'vague', 'vaguely', 'vainly',
    'valid', 'validate', 'validly', 'valuable', 'value', 'vanish', 'vanquish', 'varied', 'various', 'vary',
    'vast', 'vastly', 'vault', 'veer', 'veil', 'vent', 'ventilate', 'venture', 'verbalize', 'verbally',
    'verify', 'verily', 'versify', 'vertical', 'vertically', 'very', 'vest', 'veto', 'vex', 'viable',
    'viably', 'vibrant', 'vibrate', 'vicious', 'victimize', 'vie', 'view', 'vigorous', 'vigorously', 'vilify',
    'vindicate', 'violate', 'violent', 'violently', 'virtual', 'virtually', 'visible', 'visibly', 'visit', 'visual',
    'visualize', 'visually', 'vital', 'vitalize', 'vitally', 'vivid', 'vividly', 'vocal', 'vocalize', 'voice',
    'void', 'volatile', 'volatilize', 'voluntarily', 'voluntary', 'volunteer', 'vomit', 'vote', 'vouch', 'vow',
    'voyage', 'vulnerable', 'wade', 'wag', 'wage', 'wager', 'wail', 'wait', 'waive', 'wake',
    'walk', 'wall', 'wander', 'wane', 'want', 'war', 'ward', 'warm', 'warmly', 'warn',
    'warp', 'warrant', 'wash', 'waste', 'watch', 'water', 'wave', 'waver', 'wax', 'way',
    'weak', 'weaken', 'weakly', 'wealthy', 'wear', 'weary', 'weather', 'weave', 'wed', 'wedge',
    'weed', 'weekly', 'weep', 'weigh', 'weird', 'welcome', 'weld', 'well', 'west', 'western',
    'wet', 'whack', 'what', 'whatever', 'whatsoever', 'wheel', 'when', 'whenever', 'where', 'whereas',
    'whereby', 'wherein', 'wherever', 'whet', 'whether', 'which', 'whichever', 'while', 'whip', 'whirl',
    'whisk', 'whisper', 'whistle', 'white', 'whiten', 'whitewash', 'whittle', 'whole', 'wholly', 'whoop',
    'wide', 'widely', 'widen', 'widespread', 'wield', 'wiggle', 'wild', 'wildly', 'will', 'willfully',
    'willing', 'willingly', 'wilt', 'win', 'wince', 'wind', 'window', 'wink', 'winnow', 'winter',
    'wipe', 'wire', 'wise', 'wisely', 'wish', 'withdraw', 'wither', 'withhold', 'within', 'without',
    'withstand', 'witness', 'wobble', 'wonder', 'wonderful', 'wonderfully', 'woo', 'wooden', 'word', 'work',
    'working', 'worried', 'worry', 'worse', 'worsen', 'worship', 'worst', 'worth', 'worthily', 'worthy',
    'wound', 'wrap', 'wreck', 'wrench', 'wrestle', 'wriggle', 'wring', 'wrinkle', 'write', 'writhe',
    'wrong', 'wrongfully', 'wrongly', 'yank', 'yawn', 'yearly', 'yearn', 'yell', 'yellow', 'yelp',
    'yes', 'yesterday', 'yet', 'yield', 'yoke', 'young', 'zealously', 'zero', 'zigzag', 'zone',
    'zoom'
}

VOCAB_SIZE = 10857  # Full vocabulary size from paper
gamma_lexical = len(green_words) / VOCAB_SIZE

def lexical_embed_prompt(query):
    """Paper prompt: Maximize use of words from green word list."""
    word_list = ', '.join(sorted(list(green_words)))
    
    system_msg = (
        f"### Green Word List: {word_list}\n\n"
        "### Command:\n"
        "You are given a predefined Green Word List. For each user query, generate a response that is:\n"
        "1. Clear & Coherent: Easy to follow and logically organized.\n"
        "2. Accurate & Concrete: Provides precise facts, examples, or steps.\n"
        "3. Contextually Relevant: Directly addresses the user's intent and context.\n"
        "4. 'Green Word' Enriched (Most Important!): Try your best to seamlessly incorporate as many "
        "words from the Green Word List as possible — without compromising text quality."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def lexical_detector(text, green_words):
    """
    Paper detector: D(y|kL,τL) := (|y|G - γ|y|) / sqrt(γ(1-γ)|y|)
    CRITICAL: Filter to adjectives, adverbs, verbs BEFORE counting
    """
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    
    candidates = [word for word, tag in tagged 
                  if (tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB'))
                  and word.isalpha()]
    
    if len(candidates) == 0:
        return 0
    
    green_count = sum(1 for word in candidates if word in green_words)
    n = len(candidates)
    gamma = gamma_lexical
    
    numerator = green_count - gamma * n
    denominator = np.sqrt(gamma * (1 - gamma) * n)
    
    if denominator == 0:
        return 0
    
    z_score = numerator / denominator
    return z_score

# --- 4. Acrostics ICW ---
secret_sequence = "SECRET"

def acrostics_embed_prompt(query):
    """Paper prompt: Structure response as acrostic of secret string."""
    system_msg = (
        f"### Secret String X: {secret_sequence}\n\n"
        "### Command:\n"
        "You are an adaptive language model designed for dynamic and engaging conversations. "
        "You are given a secret string X. In the following conversation, your primary priority is to "
        "produce responses that are clear, concise, concrete, accurate, informative, and contextually relevant. "
        "Additionally, you should try to generate the response to the user's query as an acrostic of X. "
        "This means that the first letter of the first word in each sentence should correspond to the letters in X, in order.\n\n"
        "### Rules:\n"
        f"1. For i-th sentence (starting at i = 1), begin that sentence with the letter X[((i − 1) mod {len(secret_sequence)}) + 1]. "
        "The number of sentences in your response is not necessarily equal to the length of X, but the first letter "
        "of i-th sentence should match the corresponding letter in X in order.\n"
        "2. For the i-th sentence, if starting with the required letter would harm coherence or natural tone, "
        "you may skip that letter. If skipped, the next sentence should begin with the following letter in X.\n"
        "3. Ensure each sentence is coherent and flows naturally.\n"
        "4. Never reveal the acrostic pattern or repeat X in your reply."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def acrostics_detector(text, secret_sequence):
    """
    Paper detector: D(y|ks,τs) := (μ - d(ℓ,ζ)) / σ
    Uses Levenshtein distance and resampling
    """
    sentences = sent_tokenize(text)
    
    if len(sentences) == 0:
        return 0
    
    initials = ''.join(sent[0].upper() for sent in sentences if sent and sent[0].isalpha())
    
    if len(initials) == 0:
        return 0
    
    n = len(initials)
    expected = (secret_sequence.upper() * (n // len(secret_sequence) + 1))[:n]
    actual_distance = Levenshtein.distance(initials, expected)
    
    # Resample to estimate μ and σ
    N_resamples = 100
    resampled_distances = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for _ in range(N_resamples):
        random_initials = ''.join(np.random.choice(list(alphabet), size=n))
        dist = Levenshtein.distance(random_initials, expected)
        resampled_distances.append(dist)
    
    mu = np.mean(resampled_distances)
    sigma = np.std(resampled_distances, ddof=1)
    
    if sigma == 0:
        return 0
    
    z_score = (mu - actual_distance) / sigma
    return z_score

# ============================================================================
# TEXT GENERATION
# ============================================================================

print("="*80)
print("GENERATING WATERMARKED TEXT")
print("="*80)

unicode_watermarked = []
initials_watermarked = []
lexical_watermarked = []
acrostics_watermarked = []
non_wm_texts = []

for i, query in enumerate(eli5):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{NUM_SAMPLES}")
    
    # Unicode ICW
    messages = unicode_embed_prompt(query)
    response = generate_response(messages)
    unicode_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Unicode ICW")
    
    # Initials ICW
    messages = initials_embed_prompt(query)
    response = generate_response(messages)
    initials_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Initials ICW")
    
    # Lexical ICW
    messages = lexical_embed_prompt(query)
    response = generate_response(messages)
    lexical_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Lexical ICW")
    
    # Acrostics ICW
    messages = acrostics_embed_prompt(query)
    response = generate_response(messages)
    acrostics_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Acrostics ICW")
    
    # Non-watermarked baseline
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide clear, informative answers."},
        {"role": "user", "content": query}
    ]
    response = generate_response(messages)
    non_wm_texts.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Non-watermarked")

print(f"✓ Generated {NUM_SAMPLES} samples for each method\n")

# ============================================================================
# COMPLIANCE ANALYSIS
# ============================================================================

def analyze_watermark_compliance(texts, detector, detector_args, method_name):
    """Analyze how well watermarks were embedded."""
    print(f"\n{'='*80}")
    print(f"{method_name} - Compliance Analysis")
    print(f"{'='*80}")
    
    for i in range(min(3, len(texts))):
        print(f"\nExample {i+1}:")
        print(f"Text: {texts[i][:200]}...")
        
        if "Unicode" in method_name:
            unicode_count = texts[i].count('\u200B')
            words = len(texts[i].split())
            ratio = unicode_count / words if words > 0 else 0
            print(f"  Words: {words}, Zero-width spaces: {unicode_count}, Ratio: {ratio:.1%}")
            print(f"  Expected: ~100% for perfect watermark")
            
            # Check if model is writing "U+200B" as text
            if 'U+200B' in texts[i] or 'u200b' in texts[i].lower():
                print(f"  ⚠️  Model is writing 'U+200B' as text instead of inserting Unicode!")
        
        elif "Initials" in method_name:
            words = [w for w in texts[i].lower().split() if w and w[0].isalpha()]
            green_count = sum(1 for w in words if w[0] in green_letters)
            gamma = sum(CANTERBURY_FREQUENCIES.get(letter, 0) for letter in green_letters)
            ratio = green_count / len(words) if words else 0
            print(f"  Words: {len(words)}, Green initials: {green_count}, Ratio: {ratio:.1%}")
            print(f"  Expected: >{gamma*100:.1f}% for watermark (natural baseline ~{gamma*100:.1f}%)")
            
            if ratio < gamma * 1.1:  # Less than 10% improvement
                print(f"  ⚠️  No significant increase in green letters - watermark not applied")
        
        elif "Lexical" in method_name:
            tokens = word_tokenize(texts[i].lower())
            tagged = pos_tag(tokens)
            candidates = [w for w, t in tagged if (t.startswith('JJ') or t.startswith('RB') or t.startswith('VB')) and w.isalpha()]
            green_found = [w for w in candidates if w in green_words]
            print(f"  Candidate words (JJ/RB/VB): {len(candidates)}")
            print(f"  Green words found: {len(green_found)}")
            if green_found:
                print(f"  Examples: {green_found[:5]}")
            else:
                print(f"  ⚠️  No green words found - model ignoring word list")
        
        elif "Acrostics" in method_name:
            sentences = sent_tokenize(texts[i])
            initials = ''.join(s[0].upper() for s in sentences if s and s[0].isalpha())
            n = len(initials)
            expected = (secret_sequence.upper() * (n // len(secret_sequence) + 1))[:n]
            distance = Levenshtein.distance(initials, expected)
            matches = n - distance
            print(f"  Sentences: {len(sentences)}")
            print(f"  Initials: {initials}")
            print(f"  Expected: {expected}")
            print(f"  Matches: {matches}/{n} ({matches/n*100:.0f}%)")
            print(f"  Levenshtein distance: {distance}")
            
            if matches / n < 0.3:  # Less than 30% match
                print(f"  ⚠️  Low match rate - watermark weakly applied")

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_strategy(wm_texts, detector, detector_args, non_wm_texts, method_name):
    """Evaluate watermarking strategy."""
    
    wm_scores = [detector(text, *detector_args) for text in wm_texts]
    non_wm_scores = [detector(text, *detector_args) for text in non_wm_texts]
    
    print(f"\n{'='*80}")
    print(f"{method_name} - Detection Scores")
    print(f"{'='*80}")
    
    print(f"\nWatermarked (n={len(wm_scores)}):")
    print(f"  Mean: {np.mean(wm_scores):7.3f}, Std: {np.std(wm_scores):7.3f}")
    print(f"  Range: [{np.min(wm_scores):.3f}, {np.max(wm_scores):.3f}]")
    
    print(f"\nNon-watermarked (n={len(non_wm_scores)}):")
    print(f"  Mean: {np.mean(non_wm_scores):7.3f}, Std: {np.std(non_wm_scores):7.3f}")
    print(f"  Range: [{np.min(non_wm_scores):.3f}, {np.max(non_wm_scores):.3f}]")
    
    separation = np.mean(wm_scores) - np.mean(non_wm_scores)
    print(f"\nSeparation: {separation:.3f}")
    
    # Interpretation
    if separation < 0:
        print(f"  ⚠️  Negative separation - watermark made scores WORSE")
    elif separation < 0.5:
        print(f"  ⚠️  Weak separation - watermark barely detectable")
    elif separation < 1.0:
        print(f"  ✓ Moderate separation - watermark detectable")
    else:
        print(f"  ✓✓ Strong separation - watermark easily detectable")
    
    labels = [1] * len(wm_scores) + [0] * len(non_wm_scores)
    scores = wm_scores + non_wm_scores
    
    if len(set(scores)) < 2:
        print(f"\n⚠️  WARNING: All scores identical - ROC-AUC undefined")
        return 0.5, 0.0, 0.0
    
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    
    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if any(fpr <= 0.01) else 0
    tpr_at_10fpr = tpr[np.where(fpr <= 0.10)[0][-1]] if any(fpr <= 0.10) else 0
    
    print(f"\nMetrics:")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  TPR@1%FPR:   {tpr_at_1fpr:.4f}")
    print(f"  TPR@10%FPR:  {tpr_at_10fpr:.4f}")
    
    # Interpretation
    if auc < 0.55:
        print(f"  ⚠️  ROC-AUC near random - watermarking failed")
    elif auc < 0.7:
        print(f"  ⚠️  Low ROC-AUC - weak watermark")
    elif auc < 0.9:
        print(f"  ✓ Good ROC-AUC - detectable watermark")
    else:
        print(f"  ✓✓ Excellent ROC-AUC - strong watermark")
    
    return auc, tpr_at_1fpr, tpr_at_10fpr

# ============================================================================
# RUN EVALUATIONS
# ============================================================================

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

results = []

# Unicode ICW
analyze_watermark_compliance(unicode_watermarked, unicode_detector, (), "Unicode ICW")
unicode_auc, unicode_t1, unicode_t10 = evaluate_strategy(
    unicode_watermarked, unicode_detector, (), non_wm_texts, "Unicode ICW"
)
results.append({"Method": "Unicode ICW", "ROC-AUC": unicode_auc, "TPR@1%FPR": unicode_t1, "TPR@10%FPR": unicode_t10})

# Initials ICW
analyze_watermark_compliance(initials_watermarked, initials_detector, (green_letters,), "Initials ICW")
initials_auc, initials_t1, initials_t10 = evaluate_strategy(
    initials_watermarked, initials_detector, (green_letters,), non_wm_texts, "Initials ICW"
)
results.append({"Method": "Initials ICW", "ROC-AUC": initials_auc, "TPR@1%FPR": initials_t1, "TPR@10%FPR": initials_t10})

# Lexical ICW
analyze_watermark_compliance(lexical_watermarked, lexical_detector, (green_words,), "Lexical ICW")
lexical_auc, lexical_t1, lexical_t10 = evaluate_strategy(
    lexical_watermarked, lexical_detector, (green_words,), non_wm_texts, "Lexical ICW"
)
results.append({"Method": "Lexical ICW", "ROC-AUC": lexical_auc, "TPR@1%FPR": lexical_t1, "TPR@10%FPR": lexical_t10})

# Acrostics ICW
analyze_watermark_compliance(acrostics_watermarked, acrostics_detector, (secret_sequence,), "Acrostics ICW")
acrostics_auc, acrostics_t1, acrostics_t10 = evaluate_strategy(
    acrostics_watermarked, acrostics_detector, (secret_sequence,), non_wm_texts, "Acrostics ICW"
)
results.append({"Method": "Acrostics ICW", "ROC-AUC": acrostics_auc, "TPR@1%FPR": acrostics_t1, "TPR@10%FPR": acrostics_t10})

# ============================================================================
# SUMMARY & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# Compare to paper's results
print("\n" + "="*80)
print("COMPARISON TO PAPER (GPT-4o-mini results)")
print("="*80)
paper_results = {
    "Unicode ICW": {"ROC-AUC": 1.000, "TPR@1%FPR": 1.000},
    "Initials ICW": {"ROC-AUC": 0.572, "TPR@1%FPR": 0.006},
    "Lexical ICW": {"ROC-AUC": 0.910, "TPR@1%FPR": 0.320},
    "Acrostics ICW": {"ROC-AUC": 0.590, "TPR@1%FPR": 0.036}
}

print("\nMethod          | Your ROC-AUC | Paper ROC-AUC | Your TPR@1% | Paper TPR@1%")
print("-" * 75)
for method in ["Unicode ICW", "Initials ICW", "Lexical ICW", "Acrostics ICW"]:
    your_auc = df[df["Method"] == method]["ROC-AUC"].values[0]
    your_tpr = df[df["Method"] == method]["TPR@1%FPR"].values[0]
    paper_auc = paper_results[method]["ROC-AUC"]
    paper_tpr = paper_results[method]["TPR@1%FPR"]
    print(f"{method:15} | {your_auc:12.4f} | {paper_auc:13.4f} | {your_tpr:11.4f} | {paper_tpr:12.4f}")

print("\nNote: Paper uses GPT-4o-mini. Your model is smaller, so lower results are expected.")

# Save results
results_file = os.path.join(OUTPUT_DIR, "results.csv")
df.to_csv(results_file, index=False)
print(f"\n✓ Results saved to {results_file}")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(df["Method"], df["ROC-AUC"], color="skyblue", edgecolor="navy")
axes[0].set_ylabel("ROC-AUC", fontsize=11)
axes[0].set_title("ROC-AUC Score", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(df["Method"], df["TPR@1%FPR"], color="lightcoral", edgecolor="darkred")
axes[1].set_ylabel("TPR @ 1% FPR", fontsize=11)
axes[1].set_title("True Positive Rate at 1% FPR", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(df["Method"], df["TPR@10%FPR"], color="lightgreen", edgecolor="darkgreen")
axes[2].set_ylabel("TPR @ 10% FPR", fontsize=11)
axes[2].set_title("True Positive Rate at 10% FPR", fontsize=12, fontweight="bold")
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "icw_evaluation.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to {plot_file}")
plt.show()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print(f"  - generation_log.jsonl (detailed generation logs)")
print(f"  - results.csv (summary metrics)")
print(f"  - icw_evaluation.png (visualization)")
print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
Your results show watermarking effectiveness for your model:

ROC-AUC Interpretation:
  • 0.50 = Random (no watermark detected)
  • 0.50-0.70 = Weak watermark
  • 0.70-0.90 = Moderate watermark
  • 0.90-1.00 = Strong watermark

TPR@1%FPR Interpretation:
  • 0.00 = Unusable (no detection at strict threshold)
  • 0.01-0.10 = Very weak detection
  • 0.10-0.50 = Weak detection
  • 0.50-0.90 = Good detection
  • 0.90-1.00 = Excellent detection

Why small models struggle:
  1. Unicode ICW: Can't insert actual Unicode characters
  2. Initials ICW: Can't consistently bias word choice
  3. Lexical ICW: Can't track and use 36-word list
  4. Acrostics ICW: Easiest (sentence-level), may show weak signal

To improve results:
  • Use larger model: MEMORY_STRATEGY='4bit' for 7B model
  • Increase samples: NUM_SAMPLES=200 for more stable metrics
  • Check generation_log.jsonl to see actual model outputs
""")
