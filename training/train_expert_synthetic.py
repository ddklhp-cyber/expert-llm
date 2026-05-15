"""
Train ExpertLLM with domain-specialized experts.
Uses synthetic data generation for each domain.
Self-contained — no external data files needed.

Domains:
  0: reading    (stories, vocabulary, simple sentences)
  1: math       (arithmetic, numbers, word problems)
  2: nature     (animals, weather, plants, seasons)
  3: science    (physics, chemistry, biology facts)
  4: history    (events, people, cause/effect)
  5: code       (programming patterns, logic)

Usage: python training/train_expert_synthetic.py
"""
import sys, time, math, random, re, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.expert_transformer import ExpertLLM

# ===== Domain Data Generation =====

DOMAINS = {
    0: "reading",
    1: "math",
    2: "nature",
    3: "science",
    4: "history",
    5: "code",
}

READING_TEMPLATES = [
    "once upon a time there was a {adj} {animal} named {name}",
    "the {adj} {animal} loved to {verb} in the {place}",
    "{name} found a {adj} {color} {object} near the {place}",
    "{name} was so {emotion} that he {verb} all day",
    "the {animal} and the {animal} became best friends",
    "every morning {name} would {verb} before going to school",
    "{name} said i want to {verb} with my friends today",
    "the {adj} little {animal} ran to the {place} and {verb}",
    "one day {name} discovered a secret {place} behind the {object}",
    "they lived happily in the {adj} {place} near the {object}",
]

MATH_TEMPLATES = [
    "{name} had {n1} {object} and got {n2} more now he has {n3} {object}",
    "there are {n1} {animal} in the {place} and {n2} more came total is {n3}",
    "if you have {n1} {object} and give away {n2} you have {n4} left",
    "{name} counted {n1} {object} plus {n2} {object} equals {n3} {object}",
    "the {place} has {n1} rows with {n2} {object} in each row that is {n5} total",
    "{n1} plus {n2} equals {n3}",
    "{n3} minus {n2} equals {n1}",
    "{name} shared {n5} {object} equally among {n2} friends each got {n6} {object}",
    "half of {n5} is {n7}",
    "double {n1} is {n8}",
]

NATURE_TEMPLATES = [
    "the {animal} lives in the {habitat} and eats {food}",
    "in {season} the trees {season_action} and the weather is {weather}",
    "the {animal} has {feature} to help it {ability}",
    "{flower} grows in {season} and needs {need} to survive",
    "the {weather} comes from {weather_source} in the sky",
    "birds fly {direction} in {season} to find {food}",
    "the {animal} sleeps during {time} and hunts at {time2}",
    "water flows from the {water_source} to the {water_dest}",
    "the {animal} is a {animal_type} that can {ability}",
    "plants need {need} and {need2} to grow tall and strong",
]

SCIENCE_TEMPLATES = [
    "water boils at {temp1} degrees and freezes at {temp2} degrees",
    "the earth goes around the sun in {days} days",
    "gravity pulls objects {direction} toward the ground",
    "light travels faster than sound that is why we see {event} before we hear it",
    "magnets have two poles {pole1} and {pole2} that attract or repel",
    "plants make food from sunlight through {process}",
    "the heart pumps {substance} through the body",
    "electricity flows through {material} but not through {insulator}",
    "the moon reflects light from the {source}",
    "atoms are made of {particle1} {particle2} and {particle3}",
]

HISTORY_TEMPLATES = [
    "long ago people lived in {dwelling} and hunted {animal} for food",
    "the {people} built {structure} to {purpose}",
    "in the past people traveled by {transport} because there were no {modern}",
    "the invention of {invention} changed how people {activity}",
    "{leader} led the people to {achievement}",
    "ancient {people} discovered how to make {material} from {source}",
    "the first {invention} was created {timeago} years ago",
    "people learned to {skill} which helped them {benefit}",
    "trade between {people} and {people2} brought new {goods}",
    "writing was invented so people could {purpose}",
]

CODE_TEMPLATES = [
    "to add numbers use x plus y equals z",
    "a loop repeats the same action {n1} times",
    "if the value is greater than {n1} then do action a else do action b",
    "a list stores {n1} items in order from first to last",
    "a function takes input and returns output",
    "to sort numbers put the smallest first and largest last",
    "a variable named {var} stores the value {n1}",
    "count from {n1} to {n2} by adding one each step",
    "if {var} equals true then run the code block",
    "to find the largest number compare each number to the current maximum",
]

WORDS = {
    'animal': ['cat', 'dog', 'bird', 'fish', 'bear', 'fox', 'rabbit', 'frog', 'deer', 'owl'],
    'name': ['tom', 'lily', 'max', 'emma', 'sam', 'anna', 'ben', 'mia', 'jack', 'ruby'],
    'adj': ['little', 'big', 'happy', 'brave', 'kind', 'fast', 'clever', 'gentle', 'strong', 'quiet'],
    'verb': ['play', 'run', 'jump', 'sing', 'dance', 'swim', 'read', 'draw', 'climb', 'explore'],
    'place': ['garden', 'forest', 'park', 'school', 'river', 'hill', 'beach', 'library', 'cave', 'lake'],
    'color': ['red', 'blue', 'green', 'yellow', 'golden', 'silver', 'purple', 'orange', 'white', 'pink'],
    'object': ['ball', 'book', 'stone', 'star', 'flower', 'shell', 'apple', 'coin', 'key', 'box'],
    'emotion': ['happy', 'excited', 'proud', 'surprised', 'grateful', 'amazed', 'curious', 'brave', 'calm', 'cheerful'],
    'habitat': ['forest', 'ocean', 'desert', 'mountain', 'river', 'cave', 'tree', 'pond', 'field', 'jungle'],
    'food': ['fish', 'berries', 'grass', 'insects', 'seeds', 'fruit', 'leaves', 'nuts', 'worms', 'mice'],
    'season': ['spring', 'summer', 'autumn', 'winter'],
    'season_action': ['bloom', 'grow tall', 'lose leaves', 'rest under snow'],
    'weather': ['warm', 'hot', 'cool', 'cold', 'rainy', 'sunny', 'windy', 'snowy'],
    'flower': ['rose', 'daisy', 'tulip', 'sunflower', 'lily'],
    'need': ['water', 'sunlight', 'soil', 'air', 'warmth'],
    'need2': ['nutrients', 'space', 'time', 'care', 'rain'],
    'feature': ['wings', 'sharp claws', 'thick fur', 'long legs', 'big eyes', 'a long tail'],
    'ability': ['fly', 'hunt', 'stay warm', 'run fast', 'see at night', 'swim'],
    'direction': ['south', 'north', 'up', 'down'],
    'time': ['day', 'winter', 'summer', 'morning'],
    'time2': ['night', 'dawn', 'dusk', 'evening'],
    'water_source': ['mountain', 'spring', 'rain', 'glacier'],
    'water_dest': ['ocean', 'lake', 'river', 'sea'],
    'animal_type': ['mammal', 'reptile', 'bird', 'amphibian', 'insect'],
    'weather_source': ['clouds', 'the ocean', 'warm air', 'cold fronts'],
    'temp1': ['100', 'one hundred'],
    'temp2': ['0', 'zero'],
    'days': ['365', 'three hundred sixty five'],
    'process': ['photosynthesis'],
    'substance': ['blood'],
    'material': ['metal', 'copper', 'wire', 'water'],
    'insulator': ['rubber', 'plastic', 'wood', 'glass'],
    'source': ['sun'],
    'particle1': ['protons'], 'particle2': ['neutrons'], 'particle3': ['electrons'],
    'pole1': ['north'], 'pole2': ['south'],
    'event': ['lightning'],
    'dwelling': ['caves', 'huts', 'tents', 'villages'],
    'people': ['egyptians', 'romans', 'greeks', 'chinese', 'vikings'],
    'people2': ['persians', 'indians', 'africans', 'europeans'],
    'structure': ['pyramids', 'temples', 'walls', 'roads', 'bridges'],
    'purpose': ['protect their land', 'honor their gods', 'remember events', 'share knowledge'],
    'transport': ['horse', 'boat', 'walking', 'camel'],
    'modern': ['cars', 'trains', 'planes', 'buses'],
    'invention': ['wheel', 'fire', 'printing', 'compass', 'paper'],
    'activity': ['travel', 'communicate', 'build', 'farm', 'learn'],
    'leader': ['the king', 'the queen', 'the chief', 'the general'],
    'achievement': ['build a great city', 'win freedom', 'discover new lands', 'create peace'],
    'skill': ['farm', 'write', 'build tools', 'make fire', 'sail'],
    'benefit': ['grow more food', 'share stories', 'stay warm', 'explore far places'],
    'goods': ['spices', 'silk', 'gold', 'knowledge', 'tools'],
    'timeago': ['5000', '3000', '2000', '500', '200'],
    'var': ['x', 'count', 'total', 'result', 'flag'],
}


def fill_template(template):
    """Fill a template with random words and compute math values."""
    n1 = random.randint(1, 9)
    n2 = random.randint(1, 9)
    n3 = n1 + n2
    n4 = max(n1, n2) - min(n1, n2)
    n5 = n1 * n2
    n6 = n5 // n2 if n2 > 0 else 1
    n7 = n5 // 2
    n8 = n1 * 2

    sent = template
    sent = sent.replace('{n1}', str(n1)).replace('{n2}', str(n2))
    sent = sent.replace('{n3}', str(n3)).replace('{n4}', str(n4))
    sent = sent.replace('{n5}', str(n5)).replace('{n6}', str(n6))
    sent = sent.replace('{n7}', str(n7)).replace('{n8}', str(n8))

    for key, vals in WORDS.items():
        while '{' + key + '}' in sent:
            sent = sent.replace('{' + key + '}', random.choice(vals), 1)
    return sent


def generate_domain_data(num_per_domain=20000):
    """Generate training sentences for each domain."""
    all_templates = [READING_TEMPLATES, MATH_TEMPLATES, NATURE_TEMPLATES,
                     SCIENCE_TEMPLATES, HISTORY_TEMPLATES, CODE_TEMPLATES]
    data = []
    for domain_id, templates in enumerate(all_templates):
        for _ in range(num_per_domain):
            tmpl = random.choice(templates)
            sent = fill_template(tmpl)
            data.append((sent, domain_id))
    random.shuffle(data)
    return data


def build_vocab_from_data(data, max_vocab=3000):
    word_count = {}
    for sent, _ in data:
        for w in sent.split():
            word_count[w] = word_count.get(w, 0) + 1
    sorted_words = sorted(word_count.items(), key=lambda x: -x[1])[:max_vocab - 3]
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2}
    for w, _ in sorted_words:
        vocab[w] = len(vocab)
    return vocab


def tokenize_data(data, vocab, max_len=32):
    bos, eos = vocab['<BOS>'], vocab['<EOS>']
    examples = []
    for sent, domain in data:
        ids = [vocab.get(w, 0) for w in sent.split() if w in vocab]
        if 3 <= len(ids) <= max_len - 2:
            examples.append(([bos] + ids + [eos], domain))
    return examples


def make_batch(examples, indices, max_len=32):
    batch_x, batch_y, domains = [], [], []
    for idx in indices:
        ids, domain = examples[idx]
        x = ids[:-1]
        y = ids[1:]
        pad = max_len - len(x)
        if pad < 0: continue
        batch_x.append(x + [0] * pad)
        batch_y.append(y + [-100] * pad)
        domains.append(domain)
    return torch.tensor(batch_x), torch.tensor(batch_y), domains


# ===== Training =====

def eval_model(model, val_examples, vocab_size, num_experts):
    model.eval()
    domain_losses = {i: [] for i in range(num_experts)}
    with torch.no_grad():
        for vi in range(0, min(len(val_examples), 2000), 64):
            indices = list(range(vi, min(vi + 64, len(val_examples))))
            x, y, domains = make_batch(val_examples, indices)
            if len(x) == 0: continue
            for did in set(domains):
                mask = [i for i, d in enumerate(domains) if d == did]
                if not mask: continue
                logits = model(x[mask], active_experts=[did])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y[mask].reshape(-1), ignore_index=-100)
                domain_losses[did].append(loss.item())
    model.train()
    return {k: sum(v)/len(v) if v else 0 for k, v in domain_losses.items()}


def train():
    random.seed(42)
    torch.manual_seed(42)

    NUM_EXPERTS = 6
    BATCH_SIZE = 64
    STEPS = 50000
    LR = 5e-4

    print("Generating domain data...", flush=True)
    data = generate_domain_data(num_per_domain=20000)
    vocab = build_vocab_from_data(data)
    examples = tokenize_data(data, vocab)
    random.shuffle(examples)

    val_size = 3000
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    print(f"Vocab: {len(vocab)}, Train: {len(train_examples)}, Val: {len(val_examples)}", flush=True)
    domain_counts = [0] * NUM_EXPERTS
    for _, d in train_examples:
        domain_counts[d] += 1
    for i, name in DOMAINS.items():
        print(f"  {name}: {domain_counts[i]:,}", flush=True)

    model = ExpertLLM(vocab_size=len(vocab), num_experts=NUM_EXPERTS)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val = float('inf')
    checkpoint_dir = ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    t0 = time.time()

    for step in range(1, STEPS + 1):
        warmup = STEPS // 20
        if step < warmup:
            lr = LR * step / warmup
        else:
            decay = (step - warmup) / (STEPS - warmup)
            lr = 1e-5 + (LR - 1e-5) * 0.5 * (1 + math.cos(math.pi * decay))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        optimizer.zero_grad()

        indices = random.sample(range(len(train_examples)), BATCH_SIZE)
        x, y, domains = make_batch(train_examples, indices)

        total_loss = 0.0
        for did in set(domains):
            mask = [i for i, d in enumerate(domains) if d == did]
            if not mask: continue
            logits = model(x[mask], active_experts=[did])
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y[mask].reshape(-1), ignore_index=-100)
            (loss * len(mask) / BATCH_SIZE).backward()
            total_loss += loss.item() * len(mask)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2000 == 0:
            domain_losses = eval_model(model, val_examples, len(vocab), NUM_EXPERTS)
            val_avg = sum(domain_losses.values()) / len(domain_losses)
            saved = ''
            if val_avg < best_val:
                best_val = val_avg
                torch.save(model.state_dict(), checkpoint_dir / 'expert_synthetic.pt')
                saved = ' *saved*'

            elapsed = (time.time() - t0) / 60
            eta = elapsed / step * (STEPS - step)
            dl = " | ".join(f"{DOMAINS[i]}={domain_losses[i]:.3f}" for i in range(NUM_EXPERTS))
            print(f"step {step}/{STEPS} | val {val_avg:.4f}{saved} | {dl} | {elapsed:.0f}min eta={eta:.0f}min", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone! Best val: {best_val:.4f} ({elapsed:.1f} min)", flush=True)
    print(f"RESULT:{json.dumps({'best_val': best_val, 'minutes': elapsed, 'params': params})}", flush=True)


if __name__ == '__main__':
    train()
